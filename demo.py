#!/usr/bin/env python3
"""
播客工坊 · 命令行演示
===================
无需浏览器，直接合成语音。

用法:
  python demo.py --text "你好世界"
  python demo.py --text "今天天气不错" --output hello.wav
  python demo.py --text "你好" --ref-audio reference.wav
  python demo.py --model-dir /path/to/models
"""

import argparse, os, platform, struct, sys, time, re
from pathlib import Path


def normalize_for_tts(text):
    """清除 VoxCPM 音效标签和 emoji，保持文本干净"""
    if not text: return text
    text = re.sub(r'\[[^\]]*\]', '', text)  # 清除所有 [xxx] 标签
    text = re.sub(r'[\U0001F300-\U0001FAFF\U00002600-\U000027BF\U0001F000-\U0001F0FF\u2190-\u21FF\u2B00-\u2BFF\uFE0F]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def main():
    parser = argparse.ArgumentParser(description="播客工坊 · 语音合成演示")
    parser.add_argument("--text", default="你好，欢迎使用播客工坊。这是一个基于 VoxCPM2 的语音合成演示。",
                        help="要合成的文本")
    parser.add_argument("--output", default="demo_output.wav",
                        help="输出音频文件路径")
    parser.add_argument("--model-dir", default=None,
                        help="模型缓存目录（默认 ~/.cache/huggingface/hub）")
    parser.add_argument("--ref-audio", default=None,
                        help="参考音频路径，用于音色克隆（VoxCPM2 模式）")
    args = parser.parse_args()

    print("")
    print("  ╔══════════════════════════════════════╗")
    print("  ║     播客工坊 · 语音合成演示           ║")
    print("  ╚══════════════════════════════════════╝")
    print(f"  OS: {platform.system()} {platform.release()}")
    print(f"  文本: {args.text}")
    print(f"  输出: {args.output}")
    print("")

    # 清理文本
    text = normalize_for_tts(args.text)

    # ── 加载模型 ──
    print("  ⏳ 正在加载 VoxCPM2 模型...")
    t0 = time.time()

    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    os.environ.setdefault("TORCH_DYNAMO_DISABLE", "1")

    try:
        from voxcpm import VoxCPM
    except ImportError:
        print("  ❌ 需要安装: pip install voxcpm")
        sys.exit(1)

    try:
        kwargs = dict(
            hf_model_id="openbmb/VoxCPM2",
            load_denoiser=False,
            optimize=False,
        )
        if args.model_dir:
            kwargs["cache_dir"] = args.model_dir
        model = VoxCPM.from_pretrained(**kwargs)
        print(f"  ✓ 模型加载完成（{time.time()-t0:.1f}s）")
    except Exception as e:
        print(f"  ❌ 模型加载失败: {e}")
        sys.exit(1)

    # ── 合成 ──
    print(f"  🎤 合成中...")
    gen_kw = dict(cfg_value=1.5, inference_timesteps=4, max_len=1024)
    if args.ref_audio:
        if os.path.exists(args.ref_audio):
            import librosa, tempfile, wave, numpy as np
            y, _ = librosa.load(args.ref_audio, sr=int(model.tts_model.sample_rate), mono=True)
            tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False); tmp.close()
            with wave.open(tmp.name, "wb") as wf:
                wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(int(model.tts_model.sample_rate))
                wf.writeframes((np.clip(y, -1.0, 1.0) * 32767).astype("<i2").tobytes())
            gen_kw["reference_wav_path"] = tmp.name
            print(f"  ✓ 参考音频已加载: {args.ref_audio}")
        else:
            print(f"  ⚠ 参考音频不存在: {args.ref_audio}，使用默认音色")
    t1 = time.time()
    try:
        audio = model.generate(text=text, **gen_kw)
        print(f"  ✓ 合成完成（{time.time()-t1:.1f}s）")
    except Exception as e:
        print(f"  ❌ 合成失败: {e}")
        sys.exit(1)

    # ── 保存 WAV ──
    sr = int(model.tts_model.sample_rate)
    if hasattr(audio, 'numpy'): audio = audio.numpy()
    if hasattr(audio, 'reshape'): audio = audio.reshape(-1)

    # 低通滤波（同 app.py _lowpass）
    import numpy as np
    if len(audio) > 7:
        w=7;k=np.ones(w,dtype=np.float32)/w
        l=(w-1)//2;r=(w-1)-l
        p=np.pad(audio,(l,r),mode='edge')
        audio=np.convolve(p,k,mode='valid')

    n = len(audio)
    h = struct.pack("<4sI4s4sIHHIIHH4sI",
                    b"RIFF", 36 + n * 2, b"WAVE", b"fmt ", 16, 1, 1,
                    sr, sr * 2, 2, 16, b"data", n * 2)
    pcm = (np.clip(audio, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()

    Path(args.output).write_bytes(h + pcm)
    size_mb = os.path.getsize(args.output) / 1024 / 1024
    print(f"  ✓ 已保存: {args.output}（{size_mb:.1f} MB，{n/sr:.1f}s）")
    print("")
    print(f"  播放: {'start' if sys.platform=='win32' else 'open'} {args.output}")
    print("  Web UI: python app.py")


if __name__ == "__main__":
    main()
