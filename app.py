#!/usr/bin/env python3
"""
播客工坊 · 一体化服务（Web UI + VoxCPM2/VoxCPM-0.5B TTS）
=============================================
"""

import argparse, atexit, gc, json, os, platform, random, re, ssl, struct, sys, threading, time
import traceback, urllib.request, urllib.error, urllib.parse, tempfile, wave
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
from pathlib import Path

try: ssl._create_default_https_context = ssl._create_unverified_context
except: pass
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING","1")
os.environ.setdefault("HF_HUB_DISABLE_SSL_VERIFICATION","1")
os.environ.setdefault("TORCH_DYNAMO_DISABLE","1")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")  # MPS 未实现算子回退 CPU，避免崩溃

_BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = _BASE_DIR / "outputs"
PORT = 8080; MODEL_CACHE_DIR = None; FORCE_CPU = False
MODEL_SIZE = "auto"; ACTIVE_MODEL_ID = "openbmb/VoxCPM2"; MODEL_LABEL = "VoxCPM2"
_model = None; _model_lock = threading.Lock(); _model_loading = False
_model_error = ""; _model_progress = ""; _model_ready = threading.Event()
# 推理锁：ThreadingHTTPServer 每请求一线程，model.generate 不可并发（MPS/显存竞争），
# 用独立锁串行化推理；_model_lock 只管加载。
_infer_lock = threading.Lock()
_ref_wav_cache = {}

def _cleanup_ref_cache():
    for p in _ref_wav_cache.values():
        try: os.unlink(p)
        except OSError: pass
atexit.register(_cleanup_ref_cache)


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
            ps=subprocess.run(["sysctl","-n","hw.pagesize"],capture_output=True,text=True,timeout=5)
            page_size=int(ps.stdout.strip())
            vm=subprocess.run(["vm_stat"],capture_output=True,text=True,timeout=5)
            # macOS 的"可用内存"不能只看 Pages free：系统会把大量空闲内存用作缓存，
            # 计入 inactive/speculative/purgeable（可随时回收）。只算 free 会严重低估，
            # 导致明明内存充足却误判为不足、自动降级到 lite 模型。
            pages={}
            for l in vm.stdout.split("\n"):
                m=re.match(r"\s*(.+?):\s+(\d+)\.", l)
                if m: pages[m.group(1).strip()]=int(m.group(2))
            avail=(pages.get("Pages free",0)+pages.get("Pages inactive",0)
                   +pages.get("Pages speculative",0)+pages.get("Pages purgeable",0))
            if avail>0:
                return avail*page_size/1024**3
    except: pass
    try:
        # Linux 兜底：读 /proc/meminfo 的 MemAvailable（单位 kB）
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1])/1024**2
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

DEEPSEEK_URL = "https://api.deepseek.com/v1/chat/completions"

# ── VoxCPM 文本规范（依据 VoxCPM 2 官方 cookbook「最佳实践」提炼）──
# 作为 system prompt 注入 DeepSeek，让其直接产出「适合 VoxCPM 朗读」的文本。
# 这份规则是固定常量，每次调用只带这一份精炼文本，不让模型去“读链接”，
# 既避免联网依赖，也避免重复传入整篇文档。
# 核心依据（cookbook 原文要点）：
#   - 直接用干净的目标语言正文，通常不必加语言标签；
#   - 想要方言就用地道方言书写（如粤语/四川话），不要用普通话硬套；
#   - 可插入英文方括号「非语言标签」让语音更生动（如 [laughing]/[sigh]/[Uhm]），
#     全小写更稳定，一句话别叠太多。
VOXCPM_TTS_GUIDE = (
    "你撰写的文本将直接交给 VoxCPM 2 语音合成模型朗读。请遵循官方最佳实践：\n"
    "1. 写干净、自然的口语正文：像真人聊天那样，多用「对」「确实」「我觉得」「你看」"
    "这类口语词，句子短一些、一句一个意思，读起来更顺。\n"
    "   要多用语气词，让对话更口语、更有真人感：大部分句子都应带上语气词，"
    "尤其是句末的「啊」「呢」「吧」「嘛」「呀」「哦」「啦」「嘞」，"
    "以及句首/句中的「嗯」「诶」「哎」「哈」「其实吧」「你别说」等口头语。"
    "两位主播一来一往时，用「是吧」「对吧」「可不是嘛」这类接话词自然衔接。"
    "整体宁可偏口语化，也不要写得像书面播报。\n"
    "2. 可适度使用「非语言标签」让语气更生动（点到为止，一句最多一个，全小写）：\n"
    "   - [laughing] 笑、[sigh] 叹气、[Uhm] 迟疑停顿、[Shh] 安静\n"
    "   - [Question-ah]/[Question-en] 疑问、[Surprise-wa] 惊讶\n"
    "   例如：「这个数据是真的吗 [Question-ah]」「确实没想到 [Surprise-wa]」。\n"
    "3. 不要输出 Markdown 标记（井号、星号、反引号等）、表情符号 emoji、网址链接。\n"
    "4. 不要保留新闻里的「编者按」「来源」「作者」「点击查看」「[+123 chars]」等导语残留。\n"
    "5. 较大的数字和英文缩写尽量按口语写，便于朗读：例如「2025年」可写「二零二五年」，"
    "「3.5%」可写「百分之三点五」；中英文之间留一个空格。\n"
    "6. 只输出朗读文本本身，不要任何解释、标题或额外说明。"
)

DUAL_SCRIPT_SYSTEM = (
    "你是一位专业的中文播客文案撰写专家。请根据用户提供的新闻，生成一段两位主播的"
    "自然对话文案。\n\n"
    "结构要求：\n"
    "- 每行以 [Host A] 或 [Host B] 开头，两者交替发言。\n"
    "- 包含：开场寒暄与话题引入 → 围绕新闻重点的来回讨论（提问、回应、补充、追问）→ 收束总结与道别。\n"
    "- 两人观点要有互动感，不是各说各话。\n"
    "- 注意：[Host A]/[Host B] 是说话人标记，与下文提到的「非语言标签」是两回事，都要保留。\n\n"
    "【语气词是硬性要求】几乎每一句都要自然地带上语气词或口头语，这是本任务最重要的风格要求，"
    "不要写成书面播报腔。请严格模仿下面这段范例的口语密度（注意几乎每句都有语气词）：\n"
    "[Host A] 哎，大家好啊，今天这条新闻我一看呐，还真挺有意思的。\n"
    "[Host B] 是吧？我也觉得诶。你说这事儿吧，其实早就有苗头了，对不对。\n"
    "[Host A] 对对对，可不是嘛。那你觉得啊，这里头最关键的点是啥呢？\n"
    "[Host B] 嗯……我寻思吧，关键还是在这个数据上头哈，你别说，还真挺出乎意料的。\n"
    "[Host A] 哈，那咱们今天就聊到这儿啦，谢谢各位收听啊，下期再见咯！\n"
    "（范例仅示范语气词密度和口语感，实际内容请紧扣用户提供的新闻。）\n\n"
    + VOXCPM_TTS_GUIDE
)

SUMMARY_SYSTEM = (
    "你是一位新闻总结和播报文案撰写专家。请把用户提供的新闻总结成一段适合单人语音播报的"
    "核心要点，二百字以内。"
    "要求："
    "- 只输出客观的新闻要点总结，不要对话、不要评论"
    "- 不要使用任何括号语气标签如 [Uhm] [laughing] 等"
    "- 语言干净自然，适合直接朗读"
)


# ── TTS 文本兜底清理 ──
_EMOJI_RE = re.compile(
    "[" "\U0001F300-\U0001FAFF" "\U00002600-\U000027BF"
    "\U0001F000-\U0001F0FF" "\U00002190-\U000021FF" "\U00002B00-\U00002BFF" "️" "]",
    flags=re.UNICODE,
)

def normalize_for_tts(text):
    """对最终要朗读的文本做保守清理，降低 VoxCPM 合成异常（杂音/乱读）的概率。
    对所有生成路径（模板 + LLM）统一兜底，保留 [Host A]/[Host B] 行首标记。
    刻意保守：不做激进改写，避免破坏语义或对话结构。"""
    if not text:
        return text
    out_lines = []
    for line in text.split("\n"):
        m = re.match(r'^(\[Host\s*[AB]\]\s*)(.*)$', line)
        prefix, body = (m.group(1), m.group(2)) if m else ("", line)
        # 去 Markdown 残留标记
        body = re.sub(r'[*#`>_~]', '', body)
        # 去网址
        body = re.sub(r'https?://\S+', '', body)
        # 去表情符号
        body = _EMOJI_RE.sub('', body)
        # 圆括号 / 【】 及其内容（多为旁注，朗读时是噪声）
        body = re.sub(r'[（(【][^）)】]*[）)】]', '', body)
        # 方括号 [..]：保留 cookbook 推荐的英文「非语言标签」（如 [laughing]/[Question-ah]），
        # 仅删除含中文的方括号注释（如 [来源：路透社]）和 [+123 chars] 残留
        body = re.sub(r'\[[^\]]*[一-鿿][^\]]*\]', '', body)
        body = re.sub(r'\[\+?\d[^\]]*\]', '', body)
        # 省略号 / 破折号 → 逗号（保留停顿但去掉模型易卡的符号）
        body = body.replace('……', '，').replace('…', '，')
        body = re.sub(r'[—–-]{2,}', '，', body)
        body = body.replace('——', '，')
        # 去成对引号符号，保留内容
        body = re.sub(r'[“”"《》「」『』]', '', body)
        # 压缩重复标点与空白
        body = re.sub(r'[，,]{2,}', '，', body)
        body = re.sub(r'[。.]{2,}', '。', body)
        body = body.replace('　', ' ')  # 全角空格 → 半角
        body = re.sub(r'[ \t]{2,}', ' ', body).strip()
        # 清理因删除内容（如 URL/括号）残留的悬空标点：标点前是空白
        body = re.sub(r'\s+([，,。.、；;：:])', r'\1', body)
        body = re.sub(r'([，,。.、；;：:])\s*([，,。.、；;：:])', r'\2', body)
        # 行首/行尾多余标点清理
        body = re.sub(r'^[，,。.、；;：:\s]+', '', body)
        body = re.sub(r'[，,、；;：:\s]+$', '', body).strip()
        if prefix or body:
            out_lines.append((prefix + body).rstrip())
    # 去掉清理后产生的空行
    return "\n".join(l for l in out_lines if l.strip())


def generate_script_with_llm(article, api_key):
    title=article.get("title",""); d=(article.get("description")or""); c=(article.get("content")or"")
    s=article.get("source",{}).get("name",""); news=f"标题：{title}\n来源：{s}\n简介：{d}\n正文：{c}"
    user_msg = ("请根据以下新闻生成双人播客对话文案。切记：几乎每一句都要带语气词或口头语"
                "（啊/呢/吧/嘛/呀/哦/啦/嗯/诶/哈/是吧/可不是嘛等），口语感要拉满，别写成书面播报。\n\n"
                + news[:2000])
    payload=json.dumps({"model":"deepseek-chat","messages":[
        {"role":"system","content":DUAL_SCRIPT_SYSTEM},
        {"role":"user","content":user_msg},
    ],"temperature":0.9,"max_tokens":2048}).encode("utf-8")
    req=urllib.request.Request(DEEPSEEK_URL,data=payload,headers={"Content-Type":"application/json","Authorization":f"Bearer {api_key}"},method="POST")
    try:
        with urllib.request.urlopen(req,timeout=120) as r: result=json.loads(r.read().decode("utf-8"))
        script=result["choices"][0]["message"]["content"].strip()

        if "[Host A]" not in script: script = f"[Host A] {script}"
        return normalize_for_tts(script)

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
    text = "\n\n".join([
        f"各位听众朋友大家好，今天我们来聊一条关于{tp}的新闻动态",
        f"这条消息来自{src}，标题是{title}",
        f"根据报道，{full[:150]}",
        f"以上就是今天的新闻要点，感谢您的收听，我们下期再见。",
    ])
    
    # 缩短标点停顿：句号、感叹号、问号、破折号改为逗号
    text = text.replace("！", "，").replace("？", "，").replace("——", "，").replace("；", "，")
    return normalize_for_tts(text)

def generate_summary_with_llm(article, api_key):
    """调用 DeepSeek 生成新闻总结"""
    title=article.get("title",""); d=(article.get("description")or""); c=(article.get("content")or"")
    s=article.get("source",{}).get("name",""); news=f"标题：{title}\n来源：{s}\n简介：{d}\n正文：{c}"
    user_msg = "请总结以下新闻的核心要点：\n\n" + news[:2000]
    payload=json.dumps({"model":"deepseek-chat","messages":[
        {"role":"system","content":SUMMARY_SYSTEM},
        {"role":"user","content":user_msg},
    ],"temperature":0.5,"max_tokens":512}).encode("utf-8")
    req=urllib.request.Request(DEEPSEEK_URL,data=payload,headers={"Content-Type":"application/json","Authorization":f"Bearer {api_key}"},method="POST")
    try:
        with urllib.request.urlopen(req,timeout=120) as r: result=json.loads(r.read().decode("utf-8"))

        text=result["choices"][0]["message"]["content"].strip()
        return normalize_for_tts(text)

    except urllib.error.HTTPError as e:
        b=e.read().decode("utf-8",errors="replace")[:200]; raise RuntimeError(f"DeepSeek API 错误 ({e.code}): {b}")
    except Exception as e: raise RuntimeError(f"DeepSeek 调用失败: {e}")


NEWSAPI_URL = "https://newsapi.org/v2/everything"

def search_news(query, api_key):
    """后端代理 NewsAPI：Key 只留服务端（走 X-Api-Key 头，不进浏览器 URL）。
    返回 NewsAPI 原样 JSON dict，供前端沿用现有渲染逻辑。"""
    if not query: raise RuntimeError("缺少搜索关键词")
    if not api_key: raise RuntimeError("请先配置 NewsAPI Key")
    url=f"{NEWSAPI_URL}?q={urllib.parse.quote(query)}&pageSize=3&sortBy=publishedAt"
    req=urllib.request.Request(url,headers={"X-Api-Key":api_key},method="GET")
    try:
        with urllib.request.urlopen(req,timeout=30) as r: return json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        b=e.read().decode("utf-8",errors="replace")[:300]
        try: msg=json.loads(b).get("message",b)
        except Exception: msg=b
        raise RuntimeError(f"NewsAPI 错误 ({e.code}): {msg}")
    except Exception as e: raise RuntimeError(f"NewsAPI 调用失败: {e}")


# 推理参数（依据 VoxCPM 官方使用指南 / API 参考核实）：
# - inference_timesteps 官方默认 10、建议 4-30；取默认 10 兼顾质量与速度。
# - cfg_value 官方默认 2.0；长音频发闷/嗡鸣时官方建议降至 1.5-1.6 更稳，取 1.6。
# - min_len 官方默认 2；过短输入（<1s 训练下限）易发虚，不再下调为 1。
# - retry_badcase 开启：音频异常偏短/偏长时自动重试；官方默认最多重试 3 次，
#   播客长文本场景把 retry_badcase_max_times 封顶为 1，避免长尾耗时（最坏 4x→2x）。
# - normalize=True：官方数字/日期展开，兜底模板路径的裸数字朗读。
TTS_CFG=1.6; TTS_STEPS=10; TTS_MAX_LEN=1024; TTS_MIN_LEN=2; TTS_RETRY=True
TTS_GEN_KWARGS = dict(cfg_value=TTS_CFG, inference_timesteps=TTS_STEPS, min_len=TTS_MIN_LEN, max_len=TTS_MAX_LEN,
                      retry_badcase=TTS_RETRY, retry_badcase_max_times=1, normalize=True)

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
    import torch
    kw=_supported_gen_kwargs(model,kw)
    # 推理串行化；GC/清缓存不在此处逐块做（固定开销随 chunk 数线性累积），
    # 改为每次请求结束时 _release_memory() 统一做一次。
    with _infer_lock:
        with torch.inference_mode(): audio=model.generate(text=text,**kw)
    if hasattr(audio,'numpy'): audio=audio.numpy()
    if hasattr(audio,'reshape'): audio=audio.reshape(-1)

    # 注意：低通滤波不在此处逐块进行（会在每个 chunk 边界造成音量塌陷），
    # 改为所有片段拼接完成后用 _lowpass 整段处理一次。
    return audio


def _release_memory():
    """每次 TTS 请求结束后统一回收：全量 GC + MPS 缓存清理。"""
    gc.collect()
    try:
        import torch
        if hasattr(torch,"mps") and torch.backends.mps.is_available(): torch.mps.empty_cache()
    except Exception: pass


def _lowpass(audio, window=7):
    """整段移动平均低通，抑制 VoxCPM 长文本累积的高频啸声。
    用边缘填充 + valid 卷积，避免首尾及（旧实现的）chunk 边界出现音量塌陷。"""
    import numpy as np
    a = np.asarray(audio, dtype=np.float32).reshape(-1)
    if len(a) <= window or window < 2:
        return a
    kernel = np.ones(window, dtype=np.float32) / window
    left = (window - 1) // 2
    right = (window - 1) - left
    padded = np.pad(a, (left, right), mode='edge')
    return np.convolve(padded, kernel, mode='valid').astype(np.float32)


def _fade_edges(audio, sr, fade_ms=8):
    """给单段音频首尾加极短淡入淡出，消除片段拼接处的"咔哒"声。返回副本，不改原数组。"""
    import numpy as np
    a = np.array(audio, dtype=np.float32).reshape(-1)  # np.array 复制，避免影响锚点原始音频
    n = int(sr * fade_ms / 1000)
    if n > 0 and len(a) > 2 * n:
        ramp = np.linspace(0.0, 1.0, n, dtype=np.float32)
        a[:n] *= ramp
        a[-n:] *= ramp[::-1]
    return a


def _save_anchor_wav(audio, sr):
    """把一段生成音频存成临时 wav，用作后续同一主播段落的音色锚点。失败返回 None。"""
    try:
        import numpy as np
        a = np.clip(np.asarray(audio, dtype=np.float32).reshape(-1), -1.0, 1.0)
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False); tmp.close()
        with wave.open(tmp.name, "wb") as wf:
            wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(sr)
            wf.writeframes((a * 32767).astype("<i2").tobytes())
        return tmp.name
    except Exception as e:
        print(f"  ⚠ 音色锚点保存失败: {e}")
        return None

def _sanitize_voice_desc(desc, max_len=60):
    """清洗用户音色描述，用作 Control Instruction：去括号/换行/首尾空白、限长，
    避免破坏 '(desc)text' 的括号结构。空输入返回空串。"""
    if not desc: return ""
    d = re.sub(r'[()（）\[\]【】\r\n]', ' ', str(desc))
    d = re.sub(r'\s{2,}', ' ', d).strip()
    return d[:max_len]


def generate_tts(text, voice_desc):
    model,err=wait_for_model();
    if err: raise RuntimeError(err)
    print(f"  🎤 TTS... {TTS_STEPS} steps")
    desc=_sanitize_voice_desc(voice_desc)
    if desc: text=f"({desc}){text}"  # 无参考音频时用文字描述设计音色（Control Instruction）
    t0=time.perf_counter()                       # 生成开始
    try:
        audio=_tts_gen(model, text, **TTS_GEN_KWARGS)
    finally:
        _release_memory()
    t1=time.perf_counter()                        # 生成结束（推理耗时 = t1 - t0）
    print(f"  ⏱ 推理耗时: {t1-t0:.2f}s")
    return _wav_encode(_lowpass(audio), int(model.tts_model.sample_rate))

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
    - VoxCPM2(full): 用户为某主播上传参考音频时，该主播每一段都以此音频为参考克隆。
    - VoxCPM-0.5B(lite): 不支持参考音频，使用默认音色。
    片段拼接处加极短淡入淡出，说话人切换时插入短停顿；最后整段统一低通滤波。
    """
    model, err = wait_for_model()

    if err: raise RuntimeError(err)
    segs=_segments(script)
    if not segs: raise RuntimeError("脚本格式错误")

    import torch, numpy as np
    sr = int(model.tts_model.sample_rate)
    full_audio = []
    anchor_tmps = []  # 自动生成的锚点临时文件，结束时清理

    # 每个主播的参考音频：用户上传。
    # 上传文件先归一化到模型采样率的单声道 wav（_ensure_ref_wav 兼容 mp3/wav 等）。
    ref_for = {"A": None, "B": None}
    ref_text = {"A": (text_a or "").strip(), "B": (text_b or "").strip()}
    desc_for = {"A": _sanitize_voice_desc(voice_a), "B": _sanitize_voice_desc(voice_b)}  # Control Instruction
    auto_anchor = {"A": None, "B": None}  # 自动锚点 wav 路径（无用户参考时启用）
    is_full = (MODEL_SIZE == "full")
    if is_full:
        if ref_a: ref_for["A"] = _ensure_ref_wav(ref_a, sr)
        if ref_b: ref_for["B"] = _ensure_ref_wav(ref_b, sr)
    elif ref_a or ref_b:
        print("  ⚠ 0.5B 不支持参考音频，使用默认音色")

    sep = np.zeros(int(sr * 0.18), dtype=np.float32)  # 说话人切换间的短停顿
    total = len(segs)
    prev_sp = None
    t0=time.perf_counter()                            # 生成开始

    try:
        for idx,s in enumerate(segs,1):
            sp,t=s["speaker"],s["text"]
            if not t: continue

            k = dict(TTS_GEN_KWARGS)
            ref = ref_for.get(sp) or auto_anchor.get(sp)
            hifi = False
            if ref:
                # 存在参考音频（用户上传或自动锚点）：以该音频为参考生成
                k["reference_wav_path"] = ref
                if ref_for.get(sp) and ref_text.get(sp):
                    # 用户参考且提供文本时启用高保真克隆（reference + prompt 必须成对）
                    k["prompt_wav_path"] = ref
                    k["prompt_text"] = ref_text[sp]
                    hifi = True
            # 音色描述前缀：Hi-Fi 段官方明示忽略 Control Instruction，不拼
            desc = "" if hifi else desc_for.get(sp, "")
            src = "用户参考" if ref_for.get(sp) else ("自动锚点" if auto_anchor.get(sp) else ("文字音色" if desc else "默认音色"))
            print(f"  [{idx}/{total}] Host {sp}  音色来源={src}")
            seg_audio = []
            for ci, chunk in enumerate(_split_long_text(t), 1):
                if desc: chunk = f"({desc}){chunk}"
                seg_audio.append(_tts_gen(model, chunk, **k))
            if not seg_audio: continue
            raw = np.concatenate(seg_audio) if len(seg_audio) > 1 else seg_audio[0]
            # 无用户参考时，用本主播首段原始音频作为后续锚点（在 fade 前保存）
            if is_full and not ref_for.get(sp) and not auto_anchor.get(sp):
                ap = _save_anchor_wav(raw, sr)
                if ap: auto_anchor[sp] = ap; anchor_tmps.append(ap)
            # 说话人切换处插入短停顿
            if prev_sp is not None and prev_sp != sp:
                full_audio.append(sep)
            full_audio.append(_fade_edges(raw, sr))
            prev_sp = sp
        if not full_audio: raise RuntimeError("没有可合成的内容")
        t1=time.perf_counter()                        # 生成结束（推理耗时 = t1 - t0）
        print(f"  ⏱ 推理耗时: {t1-t0:.2f}s（共 {total} 句）")
        combined = np.concatenate(full_audio) if len(full_audio) > 1 else full_audio[0]
        return _wav_encode(_lowpass(combined), sr)
    finally:
        for p in anchor_tmps:
            try: os.unlink(p)
            except: pass
        _release_memory()


def _wav_encode(a,sr):
    if hasattr(a,'numpy'): a=a.numpy()
    import numpy as np
    a=np.asarray(a,dtype=np.float32).reshape(-1); a=np.clip(a,-1.0,1.0)
    pcm=(a*32767.0).astype("<i2").tobytes(); n=len(a)
    h=struct.pack("<4sI4s4sIHHIIHH4sI",b"RIFF",36+n*2,b"WAVE",b"fmt ",16,1,1,sr,sr*2,2,16,b"data",n*2)
    return h+pcm


def _parse_multipart(body, content_type):
    """极简 multipart/form-data 解析（替代 Py3.13 移除的 cgi.FieldStorage）。
    返回 (fields: {name: str}, files: {name: bytes})。只处理本应用自己的表单，
    不求覆盖 RFC 全部边角（如嵌套 multipart）。"""
    fields, files = {}, {}
    m = re.search(r'boundary="?([^";]+)"?', content_type)
    if not m: return fields, files
    boundary = b"--" + m.group(1).encode()
    for part in body.split(boundary):
        part = part.strip(b"\r\n")
        if not part or part == b"--": continue
        if b"\r\n\r\n" not in part: continue
        raw_headers, data = part.split(b"\r\n\r\n", 1)
        headers = raw_headers.decode("utf-8", errors="replace")
        dm = re.search(r'Content-Disposition:[^\r\n]*?\bname="([^"]*)"', headers, re.I)
        if not dm: continue
        name = dm.group(1)
        fm = re.search(r'\bfilename="([^"]*)"', headers, re.I)
        if fm and fm.group(1):
            files[name] = data
        else:
            fields[name] = data.decode("utf-8", errors="replace")
    return fields, files


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
        elif p=="/search_news": self._handle_search_news()
        elif p=="/generate_script": self._handle_generate_script()
        elif p=="/generate_script_v2": self._handle_generate_script_v2()
        elif p=="/generate_summary": self._handle_generate_summary()
        elif p=="/generate_dual_tts": self._handle_dual_tts()
        else: self.send_error(404,"Not found")

    def _read_json(self):
        n=int(self.headers.get("Content-Length",0))
        if n > 5*1024*1024:  # JSON 请求体上限 5MB，防止超大 body 占满内存
            raise ValueError("请求体过大")
        return json.loads(self.rfile.read(n))

    def _handle_search_news(self):
        try: d=self._read_json()
        except: self._json_error("无效的 JSON",400); return
        q=(d.get("q")or"").strip(); key=(d.get("api_key")or"").strip()
        try: result=search_news(q,key)
        except RuntimeError as e: self._json_error(str(e),502); return
        b=json.dumps(result,ensure_ascii=False).encode()
        self.send_response(200); self.send_header("Content-Type","application/json"); self.send_header("Content-Length",str(len(b)))
        self.end_headers(); self.wfile.write(b)

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
        s=normalize_for_tts(generate_script(d["article"])); b=json.dumps({"script":s},ensure_ascii=False).encode()
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

            msg=str(e)
            if "402" in msg or "Insufficient" in msg:
                s=generate_summary(art)
            else: self._json_error(msg,500); return
        b=json.dumps({"script":s},ensure_ascii=False).encode("utf-8")

        self.send_response(200); self.send_header("Content-Type","application/json"); self.send_header("Content-Length",str(len(b)))
        self.end_headers(); self.wfile.write(b)

    def _handle_dual_tts(self):
        ct=self.headers.get("Content-Type","")
        if "multipart" in ct:
            n=int(self.headers.get("Content-Length",0))
            if n > 50*1024*1024:  # 上传体上限 50MB（两段参考音频 + 表单）
                self._json_error("请求体过大",413); return
            fields,files=_parse_multipart(self.rfile.read(n),ct)
            script=(fields.get("script")or"").strip(); va=(fields.get("voice_a")or"").strip(); vb=(fields.get("voice_b")or"").strip()
            ta=(fields.get("text_a")or"").strip(); tb=(fields.get("text_b")or"").strip()
            if not script: self._json_error("缺少 script",400); return
            rp=[]
            for key in ("ref_a","ref_b"):
                data=files.get(key)
                if data:
                    tmp=tempfile.NamedTemporaryFile(suffix=".wav",delete=False);tmp.close()
                    with open(tmp.name,"wb") as out: out.write(data)
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
