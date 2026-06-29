#!/usr/bin/env python3
"""
播客工坊 · 命令行演示
===================
无需浏览器，直接合成语音。

用法:
  python demo.py
  python demo.py --text "你好世界" --voice "粤语" --output hello.wav
  python demo.py --model-dir /path/to/models
"""

import argparse
import os
import platform
import struct
import sys
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="播客工坊 · 语音合成演示")
    parser.add_argument("--text", default="你好，欢迎使用播客工坊。这是一个基于 VoxCPM2 的语音合成演示。",
                        help="要合成的文本")
    parser.add_argument("--voice", default="普通话",
                        help="音色描述，如：普通话、粤语中年男性、四川话年轻女性")
    parser.add_argument("--output", default="demo_output.wav",
                        help="输出音频文件路径")
    parser.add_argument("--model-dir", default=None,
                        help="模型缓存目录（默认 ~/.cache/huggingface/hub）")
    args = parser.parse_args()

    print("")
    print("  ╔══════════════════════════════════════╗")
    print("  ║     播客工坊 · 语音合成演示           ║")
    print("  ╚══════════════════════════════════════╝")
    print(f"  OS: {platform.system()} {platform.release()}")
    print(f"  文本: {args.text}")
    print(f"  音色: {args.voice}")
    print(f"  输出: {args.output}")
    print("")

    # ── 加载模型 ──
    print("  ⏳ 正在加载 VoxCPM2 模型...")
    t0 = time.time()

    # 环境变量（跨平台）
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
    control = (args.voice or "").strip() or "默认"
    final_text = f"({control}){args.text}" if control else args.text
    print(f"  🎤 合成中... control={control}")

    t1 = time.time()
    try:
        audio = model.generate(
            text=final_text,
            cfg_value=2.0,
            inference_timesteps=10,
            max_len=4096,
        )
        print(f"  ✓ 合成完成（{time.time()-t1:.1f}s）")
    except Exception as e:
        print(f"  ❌ 合成失败: {e}")
        sys.exit(1)

    # ── 保存 WAV ──
    sr = int(model.tts_model.sample_rate)

    if hasattr(audio, 'numpy'):
        audio = audio.numpy()
    if hasattr(audio, 'tolist'):
        audio = audio.tolist()
    if hasattr(audio, 'reshape'):
        audio = audio.reshape(-1)

    n = len(audio)
    buf = bytearray(44 + n * 2)

    def w16(o, v): struct.pack_into("<h", buf, o, int(v))
    def w32(o, v): struct.pack_into("<I", buf, o, v)

    buf[0:4] = b"RIFF"
    w32(4, 36 + n * 2)
    buf[8:12] = b"WAVE"
    buf[12:16] = b"fmt "
    w32(16, 16)
    w16(20, 1)
    w16(22, 1)
    w32(24, sr)
    w32(28, sr * 2)
    w16(32, 2)
    w16(34, 16)
    buf[36:40] = b"data"
    w32(40, n * 2)

    for i in range(n):
        v = max(-1.0, min(1.0, float(audio[i]))) * 32767.0
        w16(44 + i * 2, int(v))

    Path(args.output).write_bytes(bytes(buf))
    size_mb = os.path.getsize(args.output) / 1024 / 1024

    print(f"  ✓ 已保存: {args.output}（{size_mb:.1f} MB，{n/sr:.1f}s）")
    print("")
    print(f"  播放: {'start' if sys.platform == 'win32' else 'open'} {args.output}")
    print("  Web UI: python app.py")


if __name__ == "__main__":
    main()
