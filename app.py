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
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")  # MPS 未实现算子回退 CPU，避免崩溃

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

# ── 参考音频配置 ──
_BASE_DIR = Path(__file__).parent
REF_AUDIO_A = _BASE_DIR / "example._womanmp3.mp3"  # Host A 女声参考
REF_AUDIO_B = _BASE_DIR / "example_man.mp3"        # Host B 男声参考
# 可选：参考音频对应的文字稿（若模型 generate 支持 prompt/参考文本，可提升克隆质量；留空则不传）
REF_TEXT_A = ""
REF_TEXT_B = ""
OUTPUT_DIR = _BASE_DIR / "outputs"

_ref_wav_cache = {}  # 源音频路径 -> 转换后的 wav 临时路径


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
            out = subprocess.check_output(["vm_stat"]).decode()
            page = 4096
            mp = re.search(r"page size of (\d+)", out)
            if mp: page = int(mp.group(1))
            def _pages(name):
                m = re.search(rf"{name}:\s+(\d+)\.", out)
                return int(m.group(1)) if m else 0
            # 可立即使用 ≈ 空闲 + 不活跃 + 预测缓存
            free_pages = _pages("Pages free") + _pages("Pages inactive") + _pages("Pages speculative")
            return free_pages * page / 1024 ** 3
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
        # 设备保持 auto（cuda→mps→cpu），不显式指定。
        # 仅当实际会落到 MPS 时强制 float32，规避低精度杂音；CUDA / CPU 不受影响。
        use_mps = False
        try:
            import torch
            use_mps = (not FORCE_CPU) and (not torch.cuda.is_available()) \
                      and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
        except Exception:
            pass
        if use_mps:
            os.environ["VOXCPM_MPS_DTYPE"] = "float32"
            os.environ.setdefault("VOXCPM_FORCE_EAGER_ATTENTION", "1")
            print("  ⚙ 检测到 MPS：强制 float32")
        from voxcpm import VoxCPM
        kwargs = dict(hf_model_id=ACTIVE_MODEL_ID, load_denoiser=False, optimize=False)
        if MODEL_CACHE_DIR: kwargs["cache_dir"] = MODEL_CACHE_DIR
        if FORCE_CPU: kwargs["device"] = "cpu"  # 否则保持 auto
        _model_progress = f"正在加载 {MODEL_LABEL}..."
        print(f"\n  ⏳ {_model_progress}")
        _model = VoxCPM.from_pretrained(**kwargs)
        try:
            import torch
            if torch.cuda.is_available(): dev_str = f"CUDA ({torch.cuda.get_device_name(0)})"
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available(): dev_str = "MPS (Apple Silicon)"
            else: dev_str = "CPU"
        except: dev_str = "?"
        # 预热一次：把首次 Metal/内核编译开销挪到启动期，避免落在用户第一个请求上
        try:
            import torch
            warm = _supported_gen_kwargs(_model, dict(TTS_GEN_KWARGS, max_len=64))
            with torch.inference_mode():
                _model.generate(text="预热。", **warm)
            print("  ✓ 预热完成")
        except Exception as e:
            print(f"  ⚠ 预热跳过: {e}")
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

def _ensure_ref_wav(src_path, sample_rate):
    """把参考音频（mp3/其他）转成模型采样率的单声道 wav，返回临时 wav 路径；失败返回 None。"""
    src_path = str(src_path)
    cached = _ref_wav_cache.get(src_path)
    if cached and os.path.exists(cached):
        return cached
    if not os.path.exists(src_path):
        print(f"  ⚠ 参考音频不存在: {src_path}")
        return None
    try:
        import librosa, numpy as np
        y, _ = librosa.load(src_path, sr=sample_rate, mono=True)
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False); tmp.close()
        with wave.open(tmp.name, "wb") as wf:
            wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(sample_rate)
            wf.writeframes((np.clip(y, -1.0, 1.0) * 32767).astype("<i2").tobytes())
        _ref_wav_cache[src_path] = tmp.name
        print(f"  ✓ 参考音频就绪: {os.path.basename(src_path)}")
        return tmp.name
    except Exception as e:
        print(f"  ⚠ 参考音频转换失败 {src_path}: {e}")
        return None


def _supported_gen_kwargs(model, candidate):
    """只保留 model.generate 真正接受的参数，避免传入不支持的关键字导致报错。"""
    try:
        import inspect
        params = inspect.signature(model.generate).parameters
        if any(p.kind == p.VAR_KEYWORD for p in params.values()):
            return dict(candidate)
        return {k: v for k, v in candidate.items() if k in params}
    except Exception:
        return dict(candidate)


def _save_output_wav(w, prefix):
    """把生成的 wav 字节先落盘到 outputs/，返回路径；失败不影响主流程。"""
    try:
        import time
        OUTPUT_DIR.mkdir(exist_ok=True)
        path = OUTPUT_DIR / f"{prefix}_{time.strftime('%Y%m%d_%H%M%S')}.wav"
        path.write_bytes(w)
        print(f"  💾 已保存: {path}")
        return str(path)
    except Exception as e:
        print(f"  ⚠ 落盘失败: {e}")
        return None


def _tts_gen(model, text, **kw):
    import torch, numpy
    kw = _supported_gen_kwargs(model, kw)
    with torch.inference_mode(): audio = model.generate(text=text, **kw)
    gc.collect()
    try:
        if hasattr(torch, "mps") and torch.backends.mps.is_available():
            torch.mps.empty_cache()  # MPS 缓存不易释放，长脚本逐段清理防显存堆积
    except Exception:
        pass
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

def _split_long_text(text, max_chars=150):
    """把长文本按中英文标点切成不超过 max_chars 的小块。
    降低 MPS 长句风险（如 'Output channels > 65536' 报错）并提升稳定性。"""
    text=(text or "").strip()
    if not text: return []
    if len(text)<=max_chars: return [text]
    parts=re.split(r'(?<=[。！？!?；;\.\n])', text)
    chunks, cur=[], ""
    def _hardcut(s):
        out=[]
        while len(s)>max_chars: out.append(s[:max_chars]); s=s[max_chars:]
        if s: out.append(s)
        return out
    for p in parts:
        p=p.strip()
        if not p: continue
        if len(cur)+len(p)<=max_chars:
            cur+=p; continue
        if cur: chunks.append(cur); cur=""
        if len(p)<=max_chars:
            cur=p; continue
        # 单句仍超长：按逗号再切，最后兜底硬切
        for x in re.split(r'(?<=[，,、])', p):
            x=x.strip()
            if not x: continue
            if len(cur)+len(x)<=max_chars: cur+=x
            else:
                if cur: chunks.append(cur); cur=""
                seg=_hardcut(x)
                cur=seg.pop() if seg else ""
                chunks.extend(seg)
    if cur: chunks.append(cur)
    return chunks

def generate_dual_tts(script, voice_a, voice_b, ref_a=None, ref_b=None, text_a="", text_b=""):
    """双人 TTS：Host A / Host B 各用一份参考音频锚定音色。
    - VoxCPM2(full): 用户为某主播上传参考音频时，该主播每一段都以此音频为参考克隆；
      未上传则该主播使用模型默认音色。
    - VoxCPM-0.5B(lite): 不支持参考音频，使用默认音色。
    """
    model, err = wait_for_model()
    if err: raise RuntimeError(err)
    segs = _segments(script)
    if not segs: raise RuntimeError("脚本格式错误")
    import torch, numpy as np
    sr = int(model.tts_model.sample_rate)
    full_audio = []

    # 每个主播的参考音频：仅使用用户上传的音频；未上传则该主播使用默认音色。
    # 上传文件先归一化到模型采样率的单声道 wav（_ensure_ref_wav 兼容 mp3/wav 等）。
    ref_for = {"A": None, "B": None}
    ref_text = {"A": (text_a or "").strip(), "B": (text_b or "").strip()}
    if MODEL_SIZE == "full":
        if ref_a: ref_for["A"] = _ensure_ref_wav(ref_a, sr)
        if ref_b: ref_for["B"] = _ensure_ref_wav(ref_b, sr)
    elif ref_a or ref_b:
        print("  ⚠ 0.5B 不支持参考音频，使用默认音色")

    total = len(segs)
    try:
        for idx, s in enumerate(segs, 1):
            sp, t = s["speaker"], s["text"]
            if not t: continue
            k = dict(TTS_GEN_KWARGS)
            ref = ref_for.get(sp)
            if ref:
                # 存在参考音频：每一段都以该参考音频为参考生成
                k["reference_wav_path"] = ref
                if ref_text.get(sp):
                    # 提供文本时启用高保真克隆（reference + prompt 必须成对，否则 generate 报错）
                    k["prompt_wav_path"] = ref
                    k["prompt_text"] = ref_text[sp]
            print(f"  [{idx}/{total}] Host {sp}  参考音频={'是' if ref else '默认音色'}")
            for ci, chunk in enumerate(_split_long_text(t), 1):
                audio = _tts_gen(model, chunk, **k)
                full_audio.append(audio)
        if not full_audio: raise RuntimeError("没有可合成的内容")
        combined = np.concatenate(full_audio) if len(full_audio) > 1 else full_audio[0]
        return _wav_encode(combined, sr)
    finally:
        gc.collect()


def _wav_encode(samples, sr):
    import numpy as np
    a = samples
    if hasattr(a, 'numpy'): a = a.numpy()
    a = np.asarray(a, dtype=np.float32).reshape(-1)
    a = np.clip(a, -1.0, 1.0)
    pcm = (a * 32767.0).astype('<i2').tobytes()
    n = len(a)
    header = struct.pack("<4sI4s4sIHHIIHH4sI",
                         b"RIFF", 36 + n * 2, b"WAVE", b"fmt ", 16, 1, 1,
                         sr, sr * 2, 2, 16, b"data", n * 2)
    return header + pcm


class AppHandler(SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Access-Control-Allow-Origin","*")
        self.send_header("Access-Control-Allow-Methods","GET,POST,OPTIONS")
        self.send_header("Access-Control-Allow-Headers","Content-Type")
        try:
            super().end_headers()
        except (BrokenPipeError, ConnectionResetError):
            pass
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
                    if p:
                        try: os.unlink(p)
                        except: pass
            _save_output_wav(w, "podcast_dual")
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
            _save_output_wav(w, "podcast_dual")
            self._send_wav(w,"podcast_dual.wav")

    def _send_wav(self,w,fn="podcast.wav"):
        try:
            self.send_response(200); self.send_header("Content-Type","audio/wav")
            self.send_header("Content-Length",str(len(w)))
            self.send_header("Content-Disposition",f'attachment; filename="{fn}"')
            self.end_headers(); self.wfile.write(w)
        except (BrokenPipeError, ConnectionResetError):
            print("  ⚠ 客户端已断开，音频已生成并保存到 outputs 目录，本次未能回传")

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
    try:
        import torch
        if torch.cuda.is_available(): _dev="CUDA"
        elif hasattr(torch.backends,'mps') and torch.backends.mps.is_available(): _dev="MPS"
        else: _dev="CPU"
        print(f"  ✓ PyTorch {torch.__version__} {_dev}")
    except ImportError: print("  ❌ pip install torch"); sys.exit(1)
    if not MODEL_CACHE_DIR: print(f"  ✓ 模型缓存: {Path.home()/'.cache'/'huggingface'/'hub'}")
    print(f"  ⏳ 后台加载 {MODEL_LABEL} 模型..."); start_model_loading()
    print(f"  🌐 http://localhost:{PORT}"); print("  Ctrl+C 停止\n")
    os.chdir(Path(__file__).parent)
    server=ThreadingHTTPServer((args.host,PORT),AppHandler)
    try: server.serve_forever()
    except KeyboardInterrupt: print("\n  已停止。"); server.server_close()

if __name__=="__main__": main()
