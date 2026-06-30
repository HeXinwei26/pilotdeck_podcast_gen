#!/usr/bin/env python3
"""
播客工坊 · 一体化服务（Web UI + VoxCPM2/VoxCPM-0.5B TTS）
=============================================
"""

import argparse, gc, json, os, platform, random, re, ssl, struct, sys, threading
import traceback, urllib.request, urllib.error, tempfile, wave
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
from pathlib import Path

try: ssl._create_default_https_context = ssl._create_unverified_context
except: pass
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING","1")
os.environ.setdefault("HF_HUB_DISABLE_SSL_VERIFICATION","1")
os.environ.setdefault("TORCH_DYNAMO_DISABLE","1")

_BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = _BASE_DIR / "outputs"
PORT = 8080; MODEL_CACHE_DIR = None; FORCE_CPU = False
MODEL_SIZE = "auto"; ACTIVE_MODEL_ID = "openbmb/VoxCPM2"; MODEL_LABEL = "VoxCPM2"
_model = None; _model_lock = threading.Lock(); _model_loading = False
_model_error = ""; _model_progress = ""; _model_ready = threading.Event()
_ref_wav_cache = {}


def get_available_ram_gb():
    try: import psutil; return psutil.virtual_memory().available / 1024**3
    except: pass
    try:
        import ctypes
        if sys.platform == "win32":
            class M(ctypes.Structure):
                _fields_ = [("dwLength",ctypes.c_ulong),("dwMemoryLoad",ctypes.c_ulong),
                            ("ullTotalPhys",ctypes.c_ulonglong),("ullAvailPhys",ctypes.c_ulonglong),
                            ("ullTotalPageFile",ctypes.c_ulonglong),("ullAvailPageFile",ctypes.c_ulonglong),
                            ("ullTotalVirtual",ctypes.c_ulonglong),("ullAvailVirtual",ctypes.c_ulonglong),
                            ("ullAvailExtendedVirtual",ctypes.c_ulonglong)]
            m=M(); m.dwLength=ctypes.sizeof(M)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
            return m.ullAvailPhys/1024**3
    except: pass
    try:
        import subprocess
        if sys.platform=="darwin":
            r=subprocess.run(["vm_stat"],capture_output=True,text=True)
            for l in r.stdout.split("\n"):
                if "free" in l and "page" not in l: free_pages=int(l.split()[1].rstrip(".")); break
            return free_pages*16384/1024**3
    except: pass
    return 0.0


def _load_model_background():
    global _model,_model_loading,_model_error,_model_progress
    try:
        from voxcpm import VoxCPM
        kwargs=dict(hf_model_id=ACTIVE_MODEL_ID,load_denoiser=False,optimize=False)
        if MODEL_CACHE_DIR: kwargs["cache_dir"]=MODEL_CACHE_DIR
        if FORCE_CPU: kwargs["device"]="cpu"
        _model_progress=f"正在加载 {MODEL_LABEL}..."
        print(f"\n  ⏳ {_model_progress}")
        _model=VoxCPM.from_pretrained(**kwargs)
        try:
            import torch
            if torch.cuda.is_available(): dev=f"CUDA ({torch.cuda.get_device_name(0)})"
            elif hasattr(torch.backends,'mps') and torch.backends.mps.is_available(): dev="MPS (Apple Silicon)"
            else: dev="CPU"
        except: dev="?"
        _model_progress=f"{MODEL_LABEL} 加载完成 ✓  推理设备: {dev}"
        print(f"  ✓ {MODEL_LABEL} 加载完成  推理设备: {dev}")
    except Exception as e:
        _model_error=f"模型加载失败：{e}"; traceback.print_exc()
    finally: _model_loading=False; _model_ready.set()

def start_model_loading():
    global _model_loading,_model_progress
    with _model_lock:
        if _model is not None or _model_loading: return
        _model_loading=True; _model_progress="正在准备加载模型..."
        t=threading.Thread(target=_load_model_background,daemon=True); t.start()

def wait_for_model(timeout=600):
    if _model is not None: return _model,""
    if _model_error: return None,_model_error
    _model_ready.wait(timeout=timeout)
    if _model is not None: return _model,""
    if _model_error: return None,_model_error
    return None,"模型加载超时"


_INTROS=["各位听众朋友大家好，欢迎收听本期播客节目。","大家好，欢迎来到今天的播客时间。"]
_REACTIONS=["确实，我也这么觉得。","对，你说得很有道理。","嗯，这个角度很好。","没错，而且我觉得更关键的是——","对，这一点确实值得注意。"]
_EXPANDS=["而且我还注意到一个细节，","除此之外，还有一个关键信息，","其实这件事背后的原因也很有意思，","说到底，这件事反映了什么呢？","从更宏观的角度来看，"]
_CLOSINGS=["时间关系，今天的播客就聊到这里。","好，那我们今天就先聊到这儿。"]
_OUTRO_A=["感谢各位听众的收听，我们下期再见！","谢谢大家，下次节目再见！"]
_OUTRO_B=["下期再见！","拜拜！"]
_TOPIC_MAP={"ai":"人工智能","model":"模型","llm":"大语言模型","gpt":"GPT","robot":"机器人","chip":"芯片","data":"数据","cloud":"云","startup":"创业","fund":"融资","ipo":"上市","health":"健康","medical":"医疗","space":"航天","climate":"气候","energy":"能源","game":"游戏"}

def _detect_topic(t): t=t.lower(); return next((v for k,v in _TOPIC_MAP.items() if k in t),"科技")
def _clean(text):
    if not text: return ""
    text=re.sub(r'\[\+\d+\s*chars?\]','',text)
    for p in [r'作者\s*[|｜]\s*\S+',r'编者按[：:].*?(?=\n|$)',r'本文约\d+字.*?(?:\n|$)',
              r'来源[：:]\S+',r'原标题[：:]\S+',r'^摘编\S*',r'公众号\S*',r'特别策划\S*',
              r'^文/\S+',r'^图/\S+',r'编辑[：:]\S*',r'^\d{4}[-/]\d{1,2}[-/]\d{1,2}']:
        text=re.sub(p,'',text,flags=re.MULTILINE)
    return re.sub(r'^[\s\u3000，。！？、；：""''【】《》\n\r]+','',text).strip()

def generate_script(article):
    if not article: return "[Host A] 暂无新闻内容。"
    title=article.get("title",""); desc=_clean(article.get("description","")or""); content=_clean(article.get("content","")or"")
    full=content if len(content)>len(desc) else desc
    if not full: full=f"一条关于{_detect_topic(title)}的新闻"
    if len(full)>200: full=full[-200:]
    src=article.get("source",{}).get("name","媒体"); tp=_detect_topic(title)
    return "\n".join([
        f"[Host A] {random.choice(_INTROS)}今天我们来聊一条关于{tp}的动态。来自{src}的消息——{title}",
        f"[Host B] 这条消息确实值得关注。我觉得关键在于，{full[:100]}",
        f"[Host A] {random.choice(_REACTIONS)}{random.choice(_EXPANDS)}{full[40:120] if len(full)>120 else '从整体来看，这对行业有不小的影响。'}",
        f"[Host B] 对，而且从长远来看，{full[60:140] if len(full)>140 else '这可能会改变现有的格局。'}你觉得呢？",
        f"[Host A] 我同意。{full[30:100] if len(full)>100 else '这种进展的影响是不可忽视的。'}",
        f"[Host B] {random.choice(_CLOSINGS)}",
        f"[Host A] {random.choice(_OUTRO_A)}",
        f"[Host B] {random.choice(_OUTRO_B)}",
    ])

DEEPSEEK_URL="https://api.deepseek.com/v1/chat/completions"

def generate_script_with_llm(article,api_key):
    title=article.get("title",""); d=(article.get("description")or""); c=(article.get("content")or"")
    s=article.get("source",{}).get("name",""); news=f"标题：{title}\n来源：{s}\n简介：{d}\n正文：{c}"
    prompt="你是一个播客文案撰写专家。请根据以下新闻内容，生成一段双人中文播客对话文案。\n\n要求：\n1. 每行以 [Host A] 或 [Host B] 开头\n2. 两人围绕新闻重点内容展开自然讨论（提问、回应、补充观点）\n3. 包含开场介绍和结尾总结\n4. 语言口语化、自然流畅\n5. 只输出文案内容\n\n新闻内容：\n"+news[:2000]
    payload=json.dumps({"model":"deepseek-chat","messages":[{"role":"user","content":prompt}],"temperature":0.8,"max_tokens":2048}).encode("utf-8")
    req=urllib.request.Request(DEEPSEEK_URL,data=payload,headers={"Content-Type":"application/json","Authorization":f"Bearer {api_key}"},method="POST")
    try:
        with urllib.request.urlopen(req,timeout=120) as r: result=json.loads(r.read().decode("utf-8"))
        script=result["choices"][0]["message"]["content"].strip()
        if "[Host A]" not in script: script=f"[Host A] {script}"
        return script
    except urllib.error.HTTPError as e:
        b=e.read().decode("utf-8",errors="replace")[:200]; raise RuntimeError(f"DeepSeek API 错误 ({e.code}): {b}")
    except Exception as e: raise RuntimeError(f"DeepSeek 调用失败: {e}")


def generate_summary(article):
    if not article: return "暂无新闻内容。"
    title=article.get("title",""); desc=_clean(article.get("description","")or""); content=_clean(article.get("content","")or"")
    full=content if len(content)>len(desc) else desc
    if not full: full=f"一条关于{_detect_topic(title)}的新闻"
    if len(full)>300: full=full[-300:]
    src=article.get("source",{}).get("name","媒体"); tp=_detect_topic(title)
    text="\n\n".join([
        f"各位听众朋友大家好，今天我们来聊一条关于{tp}的新闻动态",
        f"这条消息来自{src}，标题是{title}",
        f"根据报道，{full[:150]}",
        f"以上就是今天的新闻要点，感谢您的收听，我们下期再见",
    ])
    return text.replace("。","，").replace("！","，").replace("？","，").replace("——","，").replace("；","，")

def generate_summary_with_llm(article,api_key):
    title=article.get("title",""); d=(article.get("description")or""); c=(article.get("content")or"")
    s=article.get("source",{}).get("name",""); news=f"标题：{title}\n来源：{s}\n简介：{d}\n正文：{c}"
    prompt="请用中文总结以下新闻的核心要点，要求简洁明了，200字以内，适合语音播报。\n\n新闻内容：\n"+news[:2000]
    payload=json.dumps({"model":"deepseek-chat","messages":[{"role":"user","content":prompt}],"temperature":0.5,"max_tokens":512}).encode("utf-8")
    req=urllib.request.Request(DEEPSEEK_URL,data=payload,headers={"Content-Type":"application/json","Authorization":f"Bearer {api_key}"},method="POST")
    try:
        with urllib.request.urlopen(req,timeout=120) as r: result=json.loads(r.read().decode("utf-8"))
        return result["choices"][0]["message"]["content"].strip().replace("。","，").replace("！","，").replace("？","，")
    except urllib.error.HTTPError as e:
        b=e.read().decode("utf-8",errors="replace")[:200]; raise RuntimeError(f"DeepSeek API 错误 ({e.code}): {b}")
    except Exception as e: raise RuntimeError(f"DeepSeek 调用失败: {e}")


TTS_CFG=1.5; TTS_STEPS=4; TTS_MAX_LEN=1024; TTS_MIN_LEN=1; TTS_RETRY=False
TTS_GEN_KWARGS = dict(cfg_value=TTS_CFG, inference_timesteps=TTS_STEPS, min_len=TTS_MIN_LEN, max_len=TTS_MAX_LEN, retry_badcase=TTS_RETRY)

def _ensure_ref_wav(src_path,sr):
    src_path=str(src_path); cached=_ref_wav_cache.get(src_path)
    if cached and os.path.exists(cached): return cached
    if not os.path.exists(src_path): print(f"  ⚠ 参考音频不存在: {src_path}"); return None
    try:
        import librosa,numpy as np
        y,_=librosa.load(src_path,sr=sr,mono=True)
        tmp=tempfile.NamedTemporaryFile(suffix=".wav",delete=False);tmp.close()
        with wave.open(tmp.name,"wb") as wf:
            wf.setnchannels(1);wf.setsampwidth(2);wf.setframerate(sr)
            wf.writeframes((np.clip(y,-1.0,1.0)*32767).astype("<i2").tobytes())
        _ref_wav_cache[src_path]=tmp.name
        print(f"  ✓ 参考音频就绪: {os.path.basename(src_path)}")
        return tmp.name
    except Exception as e: print(f"  ⚠ 参考音频转换失败: {e}"); return None

def _supported_gen_kwargs(model,candidate):
    try:
        import inspect
        params=inspect.signature(model.generate).parameters
        return dict(candidate) if any(p.kind==p.VAR_KEYWORD for p in params.values()) else {k:v for k,v in candidate.items() if k in params}
    except: return dict(candidate)

def _tts_gen(model,text,**kw):
    import torch,numpy
    kw=_supported_gen_kwargs(model,kw)
    with torch.inference_mode(): audio=model.generate(text=text,**kw)
    gc.collect()
    try:
        if hasattr(torch,"mps") and torch.backends.mps.is_available(): torch.mps.empty_cache()
    except: pass
    if hasattr(audio,'numpy'): audio=audio.numpy()
    if hasattr(audio,'reshape'): audio=audio.reshape(-1)
    if len(audio)>100:
        w=7; k=numpy.ones(w)/w; audio=numpy.convolve(audio,k,mode='same')
    return audio

def _split_long_text(text,max_chars=150):
    if len(text)<=max_chars: yield text; return
    for i in range(0,len(text),max_chars): yield text[i:i+max_chars]

def generate_tts(text,voice_desc):
    model,err=wait_for_model()
    if err: raise RuntimeError(err)
    print(f"  🎤 TTS... {TTS_STEPS} steps")
    return _wav_encode(_tts_gen(model,text),int(model.tts_model.sample_rate))

def _segments(script):
    segs=[]
    for line in script.split("\n"):
        line=line.strip()
        if not line: continue
        m=re.match(r'^\[Host\s*([AB])\]\s*(.*)',line)
        if m: segs.append({"speaker":m.group(1),"text":m.group(2).strip()})
        elif segs: segs[-1]["text"]+="。"+line
    return segs

def generate_dual_tts(script,voice_a,voice_b,ref_a=None,ref_b=None,text_a="",text_b=""):
    model,err=wait_for_model()
    if err: raise RuntimeError(err)
    segs=_segments(script)
    if not segs: raise RuntimeError("脚本格式错误")
    import torch,numpy as np
    sr=int(model.tts_model.sample_rate); full_audio=[]
    ref_for={"A":None,"B":None}; anchor={"A":None,"B":None}
    if MODEL_SIZE=="full":
        if ref_a: ref_for["A"]=_ensure_ref_wav(ref_a,sr)
        if ref_b: ref_for["B"]=_ensure_ref_wav(ref_b,sr)
    elif ref_a or ref_b: print("  ⚠ 0.5B 不支持参考音频")
    total=len(segs)
    try:
        for idx,s in enumerate(segs,1):
            sp,t=s["speaker"],s["text"]
            if not t: continue
            k=dict(TTS_GEN_KWARGS); ref=ref_for.get(sp)
            if ref:
                k["reference_wav_path"]=ref
                print(f"  [{idx}/{total}] Host {sp}  参考音频锚定")
            elif anchor[sp]:
                k["reference_wav_path"]=anchor[sp]
                print(f"  [{idx}/{total}] Host {sp}  首段锚定")
            else:
                print(f"  [{idx}/{total}] Host {sp}  首段（后续以此为参考）")
            for chunk in _split_long_text(t):
                audio=_tts_gen(model,chunk,**k); full_audio.append(audio)
            if not ref and anchor[sp] is None:
                tmp=tempfile.NamedTemporaryFile(suffix=".wav",delete=False);tmp.close()
                with wave.open(tmp.name,"wb") as wf:
                    wf.setnchannels(1);wf.setsampwidth(2);wf.setframerate(sr)
                    wf.writeframes((np.asarray(audio,dtype="float32")*32767).astype("int16").tobytes())
                anchor[sp]=tmp.name
        if not full_audio: raise RuntimeError("没有可合成的内容")
        return _wav_encode(np.concatenate(full_audio) if len(full_audio)>1 else full_audio[0],sr)
    finally: gc.collect()

def _wav_encode(a,sr):
    if hasattr(a,'numpy'): a=a.numpy()
    import numpy as np
    a=np.asarray(a,dtype=np.float32).reshape(-1); a=np.clip(a,-1.0,1.0)
    pcm=(a*32767.0).astype("<i2").tobytes(); n=len(a)
    h=struct.pack("<4sI4s4sIHHIIHH4sI",b"RIFF",36+n*2,b"WAVE",b"fmt ",16,1,1,sr,sr*2,2,16,b"data",n*2)
    return h+pcm


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
            b=json.dumps({"status":st,"error":_model_error,"progress":_model_progress,"model":MODEL_LABEL},ensure_ascii=False).encode()
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

    def _read_json(self): return json.loads(self.rfile.read(int(self.headers.get("Content-Length",0))))

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
        s=generate_script(d["article"]); b=json.dumps({"script":s},ensure_ascii=False).encode()
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
            self._json_error("DeepSeek 余额不足，清除 Key 可回退模板" if "402" in str(e) else str(e),402); return
        b=json.dumps({"script":s},ensure_ascii=False).encode()
        self.send_response(200); self.send_header("Content-Type","application/json"); self.send_header("Content-Length",str(len(b)))
        self.end_headers(); self.wfile.write(b)

    def _handle_generate_summary(self):
        try: d=self._read_json()
        except: self._json_error("无效的 JSON",400); return
        art=d.get("article"); key=(d.get("api_key")or"").strip()
        if not art: self._json_error("缺少 article",400); return
        try: s=generate_summary_with_llm(art,key) if key else generate_summary(art)
        except RuntimeError as e:
            if "402" in str(e): s=generate_summary(art)
            else: self._json_error(str(e),500); return
        b=json.dumps({"script":s},ensure_ascii=False).encode()
        self.send_response(200); self.send_header("Content-Type","application/json"); self.send_header("Content-Length",str(len(b)))
        self.end_headers(); self.wfile.write(b)

    def _handle_dual_tts(self):
        ct=self.headers.get("Content-Type","")
        if "multipart" in ct:
            import cgi
            form=cgi.FieldStorage(fp=self.rfile,headers=self.headers,environ={"REQUEST_METHOD":"POST","CONTENT_TYPE":ct})
            script=(form.getvalue("script")or"").strip(); va=(form.getvalue("voice_a")or"").strip(); vb=(form.getvalue("voice_b")or"").strip()
            ta=(form.getvalue("text_a")or"").strip(); tb=(form.getvalue("text_b")or"").strip()
            ref_a=form["ref_a"].file if "ref_a" in form and form["ref_a"].filename else None
            ref_b=form["ref_b"].file if "ref_b" in form and form["ref_b"].filename else None
            if not script: self._json_error("缺少 script",400); return
            rp=[]
            for f,fn in [(ref_a,"a.wav"),(ref_b,"b.wav")]:
                if f:
                    tmp=tempfile.NamedTemporaryFile(suffix=".wav",delete=False);tmp.close()
                    with open(tmp.name,"wb") as out: out.write(f.read())
                    rp.append(tmp.name)
                else: rp.append(None)
            try:
                w=generate_dual_tts(script,va,vb,ref_a=rp[0],ref_b=rp[1],text_a=ta,text_b=tb)
            except RuntimeError as e: self._json_error(str(e),500); return
            finally:
                for p in rp:
                    if p: os.unlink(p)
            self._send_wav(w,"podcast_dual.wav")
        else:
            try: d=self._read_json()
            except: self._json_error("无效的 JSON",400); return
            script=(d.get("script")or"").strip(); va=(d.get("voice_a")or"").strip(); vb=(d.get("voice_b")or"").strip()
            if not script: self._json_error("缺少 script",400); return
            try: w=generate_dual_tts(script,va,vb)
            except RuntimeError as e: self._json_error(str(e),500); return
            self._send_wav(w,"podcast_dual.wav")

    def _send_wav(self,w,fn="podcast.wav"):
        self.send_response(200); self.send_header("Content-Type","audio/wav"); self.send_header("Content-Length",str(len(w)))
        self.send_header("Content-Disposition",f'attachment; filename="{fn}"')
        self.end_headers(); self.wfile.write(w)

    def _json_error(self,msg,st=400):
        b=json.dumps({"error":msg},ensure_ascii=False).encode()
        self.send_response(st); self.send_header("Content-Type","application/json"); self.send_header("Content-Length",str(len(b)))
        self.end_headers(); self.wfile.write(b)


def main():
    config_file=_BASE_DIR/"config.json"; sc={}
    if config_file.exists():
        try: sc=json.loads(config_file.read_text(encoding="utf-8"))
        except: pass
    import argparse
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
    print(f"  ⏳ 后台加载 {MODEL_LABEL}..."); start_model_loading()
    print(f"  🌐 http://localhost:{PORT}"); print("  Ctrl+C 停止\n")
    os.chdir(_BASE_DIR)
    server=ThreadingHTTPServer((args.host,PORT),AppHandler)
    try: server.serve_forever()
    except KeyboardInterrupt: print("\n  已停止。"); server.server_close()

if __name__=="__main__": main()
