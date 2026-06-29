# 🎙 Podcast Studio · 播客工坊

> **智能播客全流程自动化引擎**
> 语义搜索 → 对话式文案生成 → 多角色语音合成

---

## 主要技术

- **PilotDeck**：GitHub仓库 https://github.com/OpenBMB/PilotDeck
- **VoxCPM**：GitHub仓库 https://github.com/OpenBMB/VoxCPM/

欢迎进入相关主页并点击 Star**⭐️** ！

## 核心能力矩阵

| 维度          | 能力层级           | 技术实现                        |
| ----------- | -------------- | --------------------------- |
| **语义检索**    | 跨语言实时新闻召回      | NewsAPI + 向量式关键词匹配          |
| **对话生成**    | 基于模板引擎的交互式文案编排 | Python 规则引擎 + 动态话术组合        |
| **多角色 TTS** | 独立音色控制的语音合成    | OpenBMB VoxCPM2（30 语种 + 方言） |
| **音频渲染**    | 逐句分片合成 → 无缝拼接  | 流式推理 + 实时 WAV 编码            |

### 🔧 技术改进

- **啸声抑制**：针对 VoxCPM 长文本合成时高频噪声累积的问题，在音频输出端增加了低通滤波（移动平均），有效抑制逐渐出现的啸声，同时保留语音清晰度。
- **标点停顿优化**（0.5B 模式）：自动将句号、感叹号等长停顿标点替换为逗号，减少朗读停顿感，语流更连贯。
- **音色一致性**（VoxCPM2 模式）：首段音频自动保存为参考样本，后续段落以其为锚定，确保同一主播的每句话音色完全一致。

## 工作流

**VoxCPM2（完整版，内存 ≥5 GB）** — 双人播客

```
[用户输入] 关键词
    ↓
[语义召回] NewsAPI → 全球最新 3 条新闻
    ↓
[文案编排] 基于选中新闻 → 生成双人互动对话脚本
    ├─ 引入话题 → 观点交流 → 收束总结
    └─ 格式: [Host A] ... / [Host B] ...
    ↓
[语音合成] 逐句推理
    ├─ Host A: 可指定音色（如 粤语中年男性）
    ├─ Host B: 可指定音色（如 普通话女声）
    └─ 参考音频锚定 → 音色一致 → WAV 拼接
    ↓
[输出] 在线播放 + 文件下载
```

**VoxCPM-0.5B（轻量版，内存不足时自动启用）** — 单人新闻播报

```
[用户输入] 关键词
    ↓
[语义召回] NewsAPI → 全球最新 3 条新闻
    ↓
[摘要生成] 基于选中新闻 → 生成新闻核心要点总结
    ├─ 简洁明了，适合语音播报
    └─ 可编辑修改
    ↓
[语音合成] 单人朗读
    ├─ 默认音色（不对音色做任何定义）
    └─ 缩短标点停顿，语流更连贯
    ↓
[输出] 在线播放 + 文件下载
```

## 前置条件

**VoxCPM2 方案（双人播客，需较高配置）**

| 组件      | 要求                            | 说明                            |
| ------- | ----------------------------- | ----------------------------- |
| Python  | 3.10+                         | 推荐 3.10-3.12                  |
| 内存      | 建议 **16 GB**（Mac 统一内存 12 GB+） | 模型约 2B 参数 + AudioVAE ≈ 4.5 GB |
| 磁盘      | 6 GB 可用                       | 模型缓存 + 临时文件                   |
| 网络      | 首次需联网                         | 从镜像站下载模型（约 4.5 GB）            |
| GPU（可选） | NVIDIA 或 Apple Silicon        | 可加速 10-50 倍                   |

**VoxCPM-0.5B 方案（单人新闻播报，低配可用）**

| 组件   | 要求          | 说明                 |
| ---- | ----------- | ------------------ |
| 内存   | **2 GB 即可** | 模型仅 0.5B 参数        |
| 磁盘   | 2 GB 可用     | 模型缓存 + 临时文件        |
| 网络   | 首次需联网       | 从镜像站下载模型（约 0.5 GB） |
| GPU  | 非必需         | CPU 即可流畅运行         |

## 部署

```bash
# 1. 安装运行时依赖
pip install voxcpm

# 注：不同平台 PyTorch 配置
# Windows + NVIDIA GPU:
#   pip install torch --index-url https://download.pytorch.org/whl/cu124
# macOS (Apple Silicon):
#   pip install torch  # 默认包含 MPS 加速
# macOS (Intel) / Linux CPU:
#   pip install torch  # CPU 推理，速度较慢

# 2. 启动进程（模型自动选择）
cd podcast_studio
python app.py                       # auto: 内存≥5GB → VoxCPM2，否则→VoxCPM-0.5B
python app.py --model-size full     # 强制 VoxCPM2（完整版，需 5GB+ 内存）
python app.py --model-size lite     # 强制 VoxCPM-0.5B（轻量版，约 0.5GB）
python app.py --model-dir D:\models # 首次指定目录（自动保存，后续无需重复输入）

# 3. 等待终端显示「✓ 模型加载完成」后再操作
# 4. 访问控制面板
# → http://localhost:8080

> ⏳ 首次启动会自动下载模型（VoxCPM2 约 4.5GB / VoxCPM-0.5B 约 0.5GB），
>     请等待终端出现「✓ 模型加载完成」字样后再点击「生成播客文案」或「生成播客音频」。
>     后续启动直接从缓存加载，秒级完成。

> ✏️ 生成播客文案后可直接在文本框中编辑修改，修改后点击「下载文案 .txt」将保存编辑后的版本。

> 💾 `--model-dir` 只需首次指定，程序会自动保存到 `config.json`，后续直接 `python app.py` 即可。
>     如需更换目录，重新传入 `--model-dir` 即可覆盖。
```
## 命令行演示

```bash
# 基本用法（默认使用 VoxCPM2，自动下载模型）
python demo.py

# 指定文本和音色
python demo.py --text "你好世界" --voice "粤语中年男性" --output hello.wav

# 指定模型缓存目录
python demo.py --model-dir /data/models

# 播放生成的音频
start demo_output.wav    # Windows
open demo_output.wav     # macOS
```

demo 脚本自动加载 VoxCPM2 模型，合成指定文本并保存为 WAV 文件，无需启动 Web 界面。

## 文案生成方案对比

系统提供两种播客文案生成方式，在界面下方 API 配置中填入 DeepSeek Key 即可启用 AI 生成：

| 维度        | 模板生成（默认，无需 Key） | DeepSeek AI 生成          |
| --------- | --------------- | ----------------------- |
| **费用**    | 免费              | DeepSeek API 按 token 计费 |
| **内容质量**  | 固定模板，可能含导语残留    | AI 理解新闻后生成自然对话          |
| **对话自然度** | 固定话术，略显机械       | 真正互动（提问→回应→补充→反问）       |
| **配置方式**  | 无需配置            | 在 API 配置中填入 Key         |

> 建议首次先用模板走通流程，再配置 Key 体验 AI 生成。

## 语音模型对比

系统根据可用内存自动选择模型，两种模式工作方式不同：

| 维度         | VoxCPM2（完整版）                  | VoxCPM-0.5B（轻量版） |
| ---------- | ----------------------------- | ---------------- |
| **模型参数量**  | ~2B（LM）+ AudioVAE / 下载 4.5 GB | 0.5B / 下载 0.5 GB |
| **最低内存**   | 建议 16 GB（Mac 统一内存需 12 GB+）    | 2 GB 即可          |
| **CPU 速度** | 全流程 1-2 小时                    | 全流程 5-10 分钟      |
| **GPU 速度** | 全流程 2-3 分钟                    | 全流程 1-2 分钟       |
| **播客模式**   | 双人对话（可分别指定音色）                 | 单人新闻摘要朗读         |
| **音色控制**   | 支持自定义音色描述                     | 默认音色，不可自定义       |
| **文案生成**   | 双人互动对话脚本                      | 新闻核心要点总结（可编辑）    |
| **音频后处理**  | 低通滤波抑制高频啸声                    | 低通滤波抑制高频啸声       |

- 自动：`python app.py`（内存 ≥5 GB 用 VoxCPM2，否则 VoxCPM-0.5B）
- 完整版：`python app.py --model-size full`（双人播客，需 5 GB+ 空闲内存）
- 轻量版：`python app.py --model-size lite`（单人新闻播报，2 GB 即可）

## 接口规范

### `POST /generate_script`
**语义对话文案生成**
```json
// Request
{ "article": { "title": "...", "description": "...", "source": {...} } }

// Response
{ "script": "[Host A] ...\n[Host B] ..." }
```

### `POST /generate_dual_tts`
**双角色语音合成**
```json
// Request
{ "script": "...", "voice_a": "粤语中年男性", "voice_b": "普通话" }

// Response → audio/wav (binary)
```

### `POST /tts`
**单角色语音合成（兼容旧版）**
```json
{ "text": "...", "voice_desc": "普通话" }
```

```
## 音色控制参数

VoxCPM2 的 `control_instruction` 参数接受自然语言音色描述，理论支持 30 种语言及方言的隐式表达。

​```python
# 示例控制向量
"粤语中年男性"       → 方言 + 性别 + 年龄
"温柔知性的女声"      → 风格 + 性别
"新闻播报，正式严肃"   → 场景 + 语气
"四川话年轻女性，活泼" → 方言 + 年龄 + 性别 + 风格
```

## 高级参数

```bash
python app.py \
  --port 9090              # 绑定端口（默认 8080）
  --host 0.0.0.0           # 监听地址（默认 0.0.0.0）
  --model-dir /data/models # 显式指定模型缓存根目录
```

## 诊断

| 现象                         | 根因分析               | 处置                                       |
| -------------------------- | ------------------ | ---------------------------------------- |
| 进程崩溃（exit code 0xC0000005） | 系统内存不足，模型权重无法完成映射  | 释放内存或增加物理内存                              |
| SSL 证书验证失败                 | 本地 CA 证书链不完整       | 已内置 `ssl._create_unverified_context` 兼容层 |
| NewsAPI 返回 426             | 非 localhost 来源请求被限 | 确认通过 `http://localhost:8080` 访问          |
| 模型加载卡在 safetensors         | 磁盘 I/O 瓶颈或大文件读取延迟  | 首次加载约 1-3 分钟，属正常范围                       |

---

*Powered by OpenBMB VoxCPM2 · 镜像站: hf-mirror.com*
