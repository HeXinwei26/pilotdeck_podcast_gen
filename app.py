#!/usr/bin/env python3
"""
播客工坊 · 一体化服务（Web UI + VoxCPM2/VoxCPM-0.5B TTS）
=============================================
启动: python app.py
访问: http://localhost:8080
"""

import argparse
import gc
import json
import os
import platform
import random
import re
import ssl
import struct
import sys
import threading
import traceback
import urllib.request
import urllib.error
import tempfile
import wave
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("HF_HUB_DISABLE_SSL_VERIFICATION", "1")
os.environ.setdefault("TORCH_DYNAMO_DISABLE", "1")

PORT = 8080
MODEL_CACHE_DIR = None
FORCE_CPU = False
MODEL_SIZE = "auto"
ACTIVE_MODEL_ID = "openbmb/VoxCPM2"
MODEL_LABEL = "VoxCPM2"

_model = None
_model_lock = threading.Lock()
_model_loading = False
_model_error = ""
_model_progress = ""
_model_ready = threading.Event()


def get_available_ram_gb() -> float:
    try:
        import psutil;
        return psutil.virtual_memory().available / 1024 ** 3
    except:
        pass
    try:
        import ctypes
        if sys.platform == "win32":
            class M(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                            ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                            ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                            ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                            ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
            m = M();
            m.dwLength = ctypes.sizeof(M)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
            return m.ullAvailPhys / 1024 ** 3
    except:
        pass
    try:
        if sys.platform == "darwin":
            import subprocess
            out = subprocess.check_output(["sysctl", "-n", "hw.memsize"])
            return int(out.strip()) / 1024 ** 3
    except:
        pass
    try:
        # Linux 兜底：读 /proc/meminfo 的 MemAvailable
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) / 1024 ** 2
    except:
        pass
    return 0.0


def _load_model_background():
    global _model, _model_loading, _model_error, _model_progress, ACTIVE_MODEL_ID, MODEL_LABEL
    try:
        from voxcpm import VoxCPM
        kwargs = dict(hf_model_id=ACTIVE_MODEL_ID, load_denoiser=False, optimize=False)
        if MODEL_CACHE_DIR: kwargs["cache_dir"] = MODEL_CACHE_DIR
        if FORCE_CPU: kwargs["device"] = "cpu"
        _model_progress = f"正在加载 {MODEL_LABEL}..."
        print(f"\n  ⏳ {_model_progress}")
        _model = VoxCPM.from_pretrained(**kwargs)
        try:
            import torch
            if torch.cuda.is_available(): dev_str = f"CUDA ({torch.cuda.get_device_name(0)})"
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available(): dev_str = "MPS (Apple Silicon)"
            else: dev_str = "CPU"
        except: dev_str = "?"
        _model_progress = f"{MODEL_LABEL} 加载完成 ✓  推理设备: {dev_str}"
        print(f"  ✓ {MODEL_LABEL} 加载完成  推理设备: {dev_str}")
    except Exception as e:
        _model_error = f"模型加载失败：{e}"
        traceback.print_exc()
    finally:
        _model_loading = False
        _model_ready.set()


def start_model_loading():
    global _model_loading, _model_progress
    with _model_lock:
        if _model is not None or _model_loading: return
        _model_loading = True
        _model_progress = "正在准备加载模型..."
        t = threading.Thread(target=_load_model_background, daemon=True)
        t.start()


def wait_for_model(timeout=600):
    if _model is not None: return _model, ""
    if _model_error: return None, _model_error
    _model_ready.wait(timeout=timeout)
    if _model is not None: return _model, ""
    if _model_error: return None, _model_error
    return None, "模型加载超时"


_INTROS = ["各位听众朋友大家好，欢迎收听本期播客节目。", "大家好，欢迎来到今天的播客时间。"]
_REACTIONS = ["确实，我也这么觉得。", "对，你说得很有道理。", "嗯，这个角度很好。", "没错，而且我觉得更关键的是——", "对，这一点确实值得注意。"]
_EXPANDS = ["而且我还注意到一个细节，", "除此之外，还有一个关键信息，", "其实这件事背后的原因也很有意思，", "说到底，这件事反映了什么呢？", "从更宏观的角度来看，"]
_CLOSINGS = ["时间关系，今天的播客就聊到这里。", "好，那我们今天就先聊到这儿。"]
_OUTRO_A = ["感谢各位听众的收听，我们下期再见！", "谢谢大家，下次节目再见！"]
_OUTRO_B = ["下期再见！", "拜拜！"]
_TOPIC_MAP = {"ai":"人工智能","model":"模型","llm":"大语言模型","gpt":"GPT","robot":"机器人","chip":"芯片","data":"数据","cloud":"云","startup":"创业","fund":"融资","ipo":"上市","health":"健康","medical":"医疗","space":"航天","climate":"气候","energy":"能源","game":"游戏"}

def _detect_topic(t): t=t.lower(); return next((v for k,v in _TOPIC_MAP.items() if k in t), "科技")
def _clean(text):
    if not text: return ""
    text = re.sub(r'\[\+\d+\s*chars?\]', '', text)
    for p in [r'作者\s*[|｜]\s*\S+', r'编者按[：:].*?(?=\n|$)', r'本文约\d+字.*?(?:\n|$)',
              r'来源[：:]\S+', r'原标题[：:]\S+', r'^摘编\S*', r'公众号\S*', r'特别策划\S*',
              r'^文/\S+', r'^图/\S+', r'编辑[：:]\S*', r'^\d{4}[-/]\d{1,2}[-/]\d{1,2}']:
        text = re.sub(p, '', text, flags=re.MULTILINE)
    text = re.sub(r'^[\s\u3000，。！？、；：""''【】《》\n\r]+', '', text)
    return text.strip()

def generate_script(article):
    if not article: return "[Host A] 暂无新闻内容。"
    title = article.get("title",""); desc = _clean(article.get("description","") or "")
    content = _clean(article.get("content","") or ""); full = content if len(content) > len(desc) else desc
    if not full: full = f"一条关于{_detect_topic(title)}的新闻"
    if len(full) > 200: full = full[-200:]
    src = article.get("source",{}).get("name","媒体"); tp = _detect_topic(title)
    lines = [
        f"[Host A] {random.choice(_INTROS)}今天我们来聊一条关于{tp}的动态。来自{src}的消息——{title}",
        f"[Host B] 这条消息确实值得关注。我觉得关键在于，{full[:100]}",
        f"[Host A] {random.choice(_REACTIONS)}{random.choice(_EXPANDS)}{full[40:120] if len(full)>120 else '从整体来看，这对行业有不小的影响。'}",
        f"[Host B] 对，而且从长远来看，{full[60:140] if len(full)>140 else '这可能会改变现有的格局。'}你觉得呢？",
        f"[Host A] 我同意。{full[30:100] if len(full)>100 else '这种进展的影响是不可忽视的。'}",
        f"[Host B] {random.choice(_CLOSINGS)}",
        f"[Host A] {random.choice(_OUTRO_A)}",
        f"[Host B] {random.choice(_OUTRO_B)}",
    ]
    return "\n".join(lines)


DEEPSEEK_URL = "https://api.deepseek.com/v1/chat/completions"

def generate_script_with_llm(article, api_key):
    title=article.get("title",""); d=(article.get("description")or""); c=(article.get("content")or"")
    s=article.get("source",{}).get("name",""); news=f"标题：{title}\n来源：{s}\n简介：{d}\n正文：{c}"
    prompt="你是一个播客文案撰写专家。请根据以下新闻内容，生成一段双人中文播客对话文案。\n\n要求：\n1. 每行以 [Host A] 或 [Host B] 开头\n2. 两人围绕新闻重点内容展开自然讨论（提问、回应、补充观点）\n3. 包含开场介绍和结尾总结\n4. 语言口语化、自然流畅\n5. 只输出文案内容\n\n新闻内容：\n"+news[:2000]
    payload=json.dumps({"model":"deepseek-chat","messages":[{"role":"user","content":prompt}],"temperature":0.8,"max_tokens":2048}).encode("utf-8")
    req=urllib.request.Request(DEEPSEEK_URL,data=payload,headers={"Content-Type":"application/json","Authorization":f"Bearer {api_key}"},method="POST")
    try:
        with urllib.request.urlopen(req,timeout=120) as r: result=json.loads(r.read().decode("utf-8"))
        script=result["choices"][0]["message"]["content"].strip()
        if "[Host A]" not in script: script = f"[Host A] {script}"
        return script
    except urllib.error.HTTPError as e:
        b=e.read().decode("utf-8",errors="replace")[:200]; raise RuntimeError(f"DeepSeek API 错误 ({e.code}): {b}")
    except Exception as e: raise RuntimeError(f"DeepSeek 调用失败: {e}")


# ── 新闻摘要生成（用于 VoxCPM-0.5B 单人朗读）──

def generate_summary(article):
    """生成新闻要点总结"""
    if not article: return "暂无新闻内容。"
    title = article.get("title","")
    desc = _clean(article.get("description","") or "")
    content = _clean(article.get("content","") or "")
    full = content if len(content) > len(desc) else desc
    if not full: full = f"一条关于{_detect_topic(title)}的新闻"
    if len(full) > 300: full = full[-300:]
    src = article.get("source",{}).get("name","媒体")
    tp = _detect_topic(title)
    lines = [
        f"各位听众朋友大家好，今天我们来聊一条关于{tp}的新闻动态。",
        f"这条消息来自{src}，标题是——{title}。",
        f"根据报道，{full[:150]}",
        f"以上就是今天的新闻要点，感谢您的收听，我们下期再见。",
    ]
    text = "\n\n".join(lines)
    # 缩短标点停顿：句号、感叹号、问号、破折号改为逗号
    text = text.replace("！", "，").replace("？", "，").replace("——", "，").replace("；", "，")
    return text

def generate_summary_with_llm(article, api_key):
    """调用 DeepSeek 生成新闻总结"""
    title=article.get("title",""); d=(article.get("description")or""); c=(article.get("content")or"")
    s=article.get("source",{}).get("name",""); news=f"标题：{title}\n来源：{s}\n简介：{d}\n正文：{c}"
    prompt="请用中文总结以下新闻的核心要点，要求简洁明了，200字以内，适合语音播报。\n\n新闻内容：\n"+news[:2000]
    payload=json.dumps({"model":"deepseek-chat","messages":[{"role":"user","content":prompt}],"temperature":0.5,"max_tokens":512}).encode("utf-8")
    req=urllib.request.Request(DEEPSEEK_URL,data=payload,headers={"Content-Type":"application/json","Authorization":f"Bearer {api_key}"},method="POST")
    try:
        with urllib.request.urlopen(req,timeout=120) as r: result=json.loads(r.read().decode("utf-8"))
        return result["choices"][0]["message"]["content"].strip().replace("。", "，").replace("！", "，").replace("？", "，")
    except urllib.error.HTTPError as e:
        b=e.read().decode("utf-8",errors="replace")[:200]; raise RuntimeError(f"DeepSeek API 错误 ({e.code}): {b}")
    except Exception as e: raise RuntimeError(f"DeepSeek 调用失败: {e}")


TTS_CFG=1.5; TTS_STEPS=4; TTS_MAX_LEN=1024; TTS_MIN_LEN=1; TTS_RETRY=False
TTS_GEN_KWARGS = dict(cfg_value=TTS_CFG, inference_timesteps=TTS_STEPS, min_len=TTS_MIN_LEN, max_len=TTS_MAX_LEN, retry_badcase=TTS_RETRY)

def _tts_gen(model, text, **kw):
    import torch, numpy
    with torch.inference_mode(): audio = model.generate(text=text, **kw)
    gc.collect()
    if hasattr(audio,'numpy'): audio=audio.numpy()
    if hasattr(audio,'reshape'): audio=audio.reshape(-1)
    # 简单低通滤波，抑制高频啸声
    if len(audio) > 100:
        window = 7
        kernel = numpy.ones(window) / window
        audio = numpy.convolve(audio, kernel, mode='same')
    return audio

def generate_tts(text, voice_desc):
    model,err=wait_for_model(); 
    if err: raise RuntimeError(err)
    print(f"  🎤 TTS... {TTS_STEPS} steps")
    return _wav_encode(_tts_gen(model, text), int(model.tts_model.sample_rate))

def _segments(script):
    segs=[]
    for line in script.split("\n"):
        line=line.strip()
        if not line: continue
        m=re.match(r'^\[Host\s*([AB])\]\s*(.*)',line)
        if m: segs.append({"speaker":m.group(1),"text":m.group(2).strip()})
        elif segs: segs[-1]["text"]+="。"+line
    return segs

def generate_dual_tts(script, voice_a, voice_b, ref_a=None, ref_b=None, text_a="", text_b=""):
    """双人 TTS
    VoxCPM2: 逐段生成，第2段起以第1段为参考音频
    VoxCPM-0.5B: 逐段生成，每段用上传的参考音频零样本克隆音色
    """
    model,err=wait_for_model()
    if err: raise RuntimeError(err)
    segs=_segments(script)
    if not segs: raise RuntimeError("脚本格式错误")
    import torch; sr=int(model.tts_model.sample_rate); full_audio=[]; tmp_files=[]
    ref_wavs={"A":None,"B":None}; total=len(segs)

    for idx, s in enumerate(segs, 1):
        sp,t=s["speaker"],s["text"]
        if not t: continue
        k=dict(TTS_GEN_KWARGS)
        print(f"  [{idx}/{total}] Host {sp}...")

        if MODEL_SIZE=="full":
            user_ref = ref_a if sp=="A" else ref_b
            k["reference_wav_path"] = user_ref or ref_wavs[sp] or None
        elif (sp=="A" and ref_a) or (sp=="B" and ref_b):
            print(f"  ⚠ 0.5B 不支持参考音频，使用默认音色")

        audio = _tts_gen(model, t, **k)
        full_audio.append(audio)

        # VoxCPM2: 保存首段作为后续参考
        if MODEL_SIZE=="full":
            if not ref_wavs[sp]:
                tmp=tempfile.NamedTemporaryFile(suffix=".wav",delete=False); tmp.close()
                with wave.open(tmp.name,"wb") as wf:
                    wf.setnchannels(1);wf.setsampwidth(2);wf.setframerate(sr)
                    wf.writeframes((audio*32767).astype("int16").tobytes())
                ref_wavs[sp]=tmp.name; tmp_files.append(tmp.name)

    gc.collect()
    import numpy as np
    combined=np.concatenate(full_audio) if len(full_audio)>1 else full_audio[0]
    wav=_wav_encode(combined,sr)
    for f in tmp_files:
        try: os.unlink(f)
        except: pass
    return wav


def _wav_encode(samples, sr):
    if hasattr(samples,'numpy'): samples=samples.numpy()
    if hasattr(samples,'tolist'): samples=samples.tolist()
    if hasattr(samples,'reshape'): samples=samples.reshape(-1)
    n=len(samples); buf=bytearray(44+n*2)
    def w16(o,v): struct.pack_into("<h",buf,o,int(v))
    def w32(o,v): struct.pack_into("<I",buf,o,v)
    buf[0:4]=b"RIFF"; w32(4,36+n*2); buf[8:12]=b"WAVE"; buf[12:16]=b"fmt "; w32(16,16)
    w16(20,1); w16(22,1); w32(24,sr); w32(28,sr*2); w16(32,2); w16(34,16)
    buf[36:40]=b"data"; w32(40,n*2)
    for i in range(n):
        v=max(-1.0,min(1.0,float(samples[i])))*32767.0; w16(44+i*2,int(v))
    return bytes(buf)


class AppHandler(SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Access-Control-Allow-Origin","*")
        self.send_header("Access-Control-Allow-Methods","GET,POST,OPTIONS")
        self.send_header("Access-Control-Allow-Headers","Content-Type")
        super().end_headers()
    def do_OPTIONS(self): self.send_response(204); self.end_headers()

    def do_GET(self):
        p=self.path.split("?")[0]
        if p=="/status":
            st="loading" if _model_loading else ("ready" if _model else "stopped")
            b=json.dumps({"status":st,"error":_model_error,"progress":_model_progress,"model":MODEL_LABEL},ensure_ascii=False).encode("utf-8")
            self.send_response(200); self.send_header("Content-Type","application/json"); self.send_header("Content-Length",str(len(b)))
            self.end_headers(); self.wfile.write(b); return
        return super().do_GET()

    def do_POST(self):
        p=self.path.split("?")[0]
        if p=="/tts": self._handle_tts()
        elif p=="/generate_script": self._handle_generate_script()
        elif p=="/generate_script_v2": self._handle_generate_script_v2()
        elif p=="/generate_summary": self._handle_generate_summary()
        elif p=="/generate_dual_tts": self._handle_dual_tts()
        else: self.send_error(404,"Not found")

    def _read_json(self):
        return json.loads(self.rfile.read(int(self.headers.get("Content-Length",0))))

    def _handle_tts(self):
        try: d=self._read_json()
        except: self._json_error("无效的 JSON",400); return
        t=(d.get("text")or"").strip()
        if not t: self._json_error("缺少 text",400); return
        try: w=generate_tts(t,(d.get("voice_desc")or"").strip())
        except RuntimeError as e: self._json_error(str(e),500); return
        self._send_wav(w)

    def _handle_generate_script(self):
        try: d=self._read_json()
        except: self._json_error("无效的 JSON",400); return
        if not d.get("article"): self._json_error("缺少 article",400); return
        s=generate_script(d["article"]); b=json.dumps({"script":s},ensure_ascii=False).encode("utf-8")
        self.send_response(200); self.send_header("Content-Type","application/json"); self.send_header("Content-Length",str(len(b)))
        self.end_headers(); self.wfile.write(b)

    def _handle_generate_script_v2(self):
        try: d=self._read_json()
        except: self._json_error("无效的 JSON",400); return
        art=d.get("article"); key=(d.get("api_key")or"").strip()
        if not art: self._json_error("缺少 article",400); return
        try:
            s=generate_script_with_llm(art,key) if key else generate_script(art)
        except RuntimeError as e:
            msg=str(e)
            if "402" in msg or "Insufficient Balance" in msg: msg="DeepSeek 余额不足，清除 Key 可回退模板"
            self._json_error(msg,402); return
        b=json.dumps({"script":s},ensure_ascii=False).encode("utf-8")
        self.send_response(200); self.send_header("Content-Type","application/json"); self.send_header("Content-Length",str(len(b)))
        self.end_headers(); self.wfile.write(b)

    def _handle_generate_summary(self):
        try: d=self._read_json()
        except: self._json_error("无效的 JSON",400); return
        art=d.get("article"); key=(d.get("api_key")or"").strip()
        if not art: self._json_error("缺少 article",400); return
        try:
            s=generate_summary_with_llm(art,key) if key else generate_summary(art)
        except RuntimeError as e:
            if "402" in e.args[0] or "Insufficient" in e.args[0]:
                s=generate_summary(art)
            else: self._json_error(str(e),500); return
        b=json.dumps({"script":s},ensure_ascii=False).encode("utf-8")
        self.send_response(200); self.send_header("Content-Type","application/json"); self.send_header("Content-Length",str(len(b)))
        self.end_headers(); self.wfile.write(b)

    def _handle_dual_tts(self):
        """支持 JSON 和 multipart/form-data 两种请求格式"""
        ct=self.headers.get("Content-Type","")
        if "multipart" in ct:
            # 文件上传模式（VoxCPM-0.5B 参考音频）
            import cgi
            form=cgi.FieldStorage(fp=self.rfile,headers=self.headers,environ={"REQUEST_METHOD":"POST","CONTENT_TYPE":ct})
            script=(form.getvalue("script")or"").strip()
            va=(form.getvalue("voice_a")or"普通话").strip(); vb=(form.getvalue("voice_b")or"普通话").strip()
            ta=(form.getvalue("text_a")or"").strip(); tb=(form.getvalue("text_b")or"").strip()
            ref_a=form["ref_a"].file if "ref_a" in form and form["ref_a"].filename else None
            ref_b=form["ref_b"].file if "ref_b" in form and form["ref_b"].filename else None
            if not script: self._json_error("缺少 script",400); return
            ref_paths=[]
            for f,fn in [(ref_a,"ref_a.wav"),(ref_b,"ref_b.wav")]:
                if f:
                    tmp=tempfile.NamedTemporaryFile(suffix=".wav",delete=False); tmp.close()
                    with open(tmp.name,"wb") as out: out.write(f.read())
                    ref_paths.append(tmp.name)
                else: ref_paths.append(None)
            try:
                w=generate_dual_tts(script,va,vb,ref_a=ref_paths[0],ref_b=ref_paths[1],text_a=ta,text_b=tb)
            except RuntimeError as e: self._json_error(str(e),500); return
            finally:
                for p in ref_paths:
                    if p: os.unlink(p)
            self._send_wav(w,"podcast_dual.wav")
        else:
            # JSON 模式
            try: d=self._read_json()
            except: self._json_error("无效的 JSON",400); return
            script=(d.get("script")or"").strip(); va=(d.get("voice_a")or"普通话").strip(); vb=(d.get("voice_b")or"普通话").strip()
            if not script: self._json_error("缺少 script",400); return
            try:
                w=generate_dual_tts(script,va,vb)
            except RuntimeError as e: self._json_error(str(e),500); return
            self._send_wav(w,"podcast_dual.wav")

    def _send_wav(self,w,fn="podcast.wav"):
        self.send_response(200); self.send_header("Content-Type","audio/wav")
        self.send_header("Content-Length",str(len(w)))
        self.send_header("Content-Disposition",f'attachment; filename="{fn}"')
        self.end_headers(); self.wfile.write(w)

    def _json_error(self,msg,st=400):
        b=json.dumps({"error":msg},ensure_ascii=False).encode("utf-8")
        self.send_response(st); self.send_header("Content-Type","application/json"); self.send_header("Content-Length",str(len(b)))
        self.end_headers(); self.wfile.write(b)


def main():
    config_file=Path(__file__).parent/"config.json"; sc={}
    if config_file.exists():
        try: sc=json.loads(config_file.read_text(encoding="utf-8"))
        except: pass
    parser=argparse.ArgumentParser(description="播客工坊")
    parser.add_argument("--port",type=int,default=8080)
    parser.add_argument("--model-dir",type=str,default=sc.get("model_dir"))
    parser.add_argument("--host",type=str,default="0.0.0.0")
    parser.add_argument("--cpu",action="store_true")
    parser.add_argument("--model-size",choices=["auto","full","lite"],default="auto")
    args=parser.parse_args()
    global PORT,MODEL_CACHE_DIR,FORCE_CPU,MODEL_SIZE,ACTIVE_MODEL_ID,MODEL_LABEL
    PORT=args.port
    if args.model_dir: MODEL_CACHE_DIR=args.model_dir
    if args.cpu: FORCE_CPU=True
    MODEL_SIZE=args.model_size
    if args.model_dir:
        config_file.write_text(json.dumps({"model_dir":args.model_dir},ensure_ascii=False,indent=2),encoding="utf-8")

    print("\n  ╔══════════════════════════════════════╗")
    print("  ║     播客工坊 · Podcast Studio        ║")
    print("  ╚══════════════════════════════════════╝")
    print(f"  OS: {platform.system()} {platform.release()}")
    print(f"  Python: {sys.version.split()[0]}")
    ram=get_available_ram_gb(); print(f"  可用内存: {ram:.1f} GB")
    if MODEL_SIZE=="auto":
        if ram>=5.0: MODEL_SIZE="full"; print("  ✓ 自动选择 VoxCPM2（完整版）")
        else: MODEL_SIZE="lite"; print(f"  ⚠ 内存仅 {ram:.1f}GB，自动选择 VoxCPM-0.5B")
    if MODEL_SIZE=="full": ACTIVE_MODEL_ID="openbmb/VoxCPM2"; MODEL_LABEL="VoxCPM2"; print("  ✓ 模型: VoxCPM2（支持参考音频锚定）")
    else: ACTIVE_MODEL_ID="openbmb/VoxCPM-0.5B"; MODEL_LABEL="VoxCPM-0.5B"; print("  ✓ 模型: VoxCPM-0.5B")
    try: import voxcpm; print(f"  ✓ voxcpm {getattr(voxcpm,'__version__','?')}")
    except ImportError: print("  ❌ pip install voxcpm"); sys.exit(1)
    try: import torch; print(f"  ✓ PyTorch {torch.__version__} {'CPU' if not torch.cuda.is_available() else 'CUDA'}")
    except ImportError: print("  ❌ pip install torch"); sys.exit(1)
    if not MODEL_CACHE_DIR: print(f"  ✓ 模型缓存: {Path.home()/'.cache'/'huggingface'/'hub'}")
    print(f"  ⏳ 后台加载 {MODEL_LABEL} 模型..."); start_model_loading()
    print(f"  🌐 http://localhost:{PORT}"); print("  Ctrl+C 停止\n")
    os.chdir(Path(__file__).parent)
    server=ThreadingHTTPServer((args.host,PORT),AppHandler)
    try: server.serve_forever()
    except KeyboardInterrupt: print("\n  已停止。"); server.server_close()

if __name__=="__main__": main()
