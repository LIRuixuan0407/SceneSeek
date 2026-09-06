# SceneSeek

SceneSeek 是一个本地优先的多模态图片与视频检索系统。你可以用自然语言或参考图片搜索本地媒体库；视频结果会返回相关短窗口的起止时间，而不是只返回整段视频。

当前版本完成设计文档的 P0 可用闭环，并为 Temporal Adapter、Moment Head、Hard Negative 与 Reranker 保留了清晰模块边界。它不会把尚未训练或评测的模型包装成完成结果。

## 已实现

- 递归扫描图片和视频，使用 SHA-256 内容哈希去重。
- SQLite 保存路径、尺寸、时长、FPS、修改时间、内容哈希与索引版本。
- 文件变化后自动失效旧向量；文件删除时级联清理 clip 与 embedding。
- FFmpeg / ffprobe 视频探测与批量抽帧；重叠窗口复用 frame embedding cache，避免重复解码与编码。
- 默认 4 秒 window、2 秒 stride、1 FPS 的视频短窗口索引；采样配置变化会自动刷新 clip 并失效旧索引。
- 文本搜图片/视频、以图搜图/视频、组合查询 API。
- CLIP/SigLIP 兼容的 Transformers 编码器，以及无需下载模型的 CPU Lite 降级模式。
- FAISS Flat/HNSW（安装 ML extra 后）与精确 NumPy inner-product fallback。
- 同一视频相邻候选合并，每个视频限制重复 moment，返回 start/end/thumbnail 时间。
- FastAPI 状态、媒体、视频片段与 relevance feedback API；搜索 query 与候选 impression 会一并记录，为后续 reranker/eval 留下训练上下文。
- 响应式 React 前端，支持索引管理、搜索筛选、视频片段预览与结果反馈。
- Railway 容器配置与 GitHub Pages 前端工作流。
- 检索指标、moment IoU、增量扫描、索引一致性和 API 闭环测试。

## 架构

```text
媒体目录
  ├─ 图片 ── 解码/EXIF ───────────────┐
  └─ 视频 ── ffprobe ── 短窗口/抽帧 ──┤
                                      ├─ Encoder ── SQLite Embeddings ── Vector Index
查询文本/图片 ── Query Encoder ───────┘                                  │
                                                                          └─ Top-K
                                                                               └─ 视频候选聚合
                                                                                   └─ 图片 + [视频, start, end]
```

索引阶段承担视频解码和特征抽取；在线阶段只编码 query 并查询少量候选。

## 本地启动

要求 Python 3.10+、Node.js 18+ 和可从命令行调用的 FFmpeg。

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'

cd web
npm install
npm run build
cd ..

sceneseek serve --host 0.0.0.0 --port 8000
```

打开 `http://localhost:8000`。首次使用时在“管理索引”中输入服务端可访问的绝对媒体目录，先扫描，再构建索引。

前后端分离开发：

```bash
# 终端 1
sceneseek serve --port 8000

# 终端 2
cd web
npm run dev
```

Vite 会把 `/api` 代理到 `http://localhost:8000`。

## 启用 CLIP 或 SigLIP

默认 `SCENESEEK_ENCODER=auto`。未安装 PyTorch/Transformers 时系统使用 `lite-color-text-v1`，方便在 CPU 环境验证完整产品链路；它不是设计文档中的正式 benchmark baseline。

```bash
pip install -e '.[ml,dev]'
export SCENESEEK_ENCODER=transformers
export SCENESEEK_MODEL_ID=openai/clip-vit-base-patch32
export SCENESEEK_DEVICE=auto
sceneseek build --rebuild
```

切换模型、采样 FPS、window/stride 或每窗口最大帧数都会产生新的 `model_version`，已有媒体会进入待重建状态。frame embedding cache 只绑定编码器与图像预处理版本，因此仅调整窗口布局时可继续复用已有帧特征。若使用多语言 SigLIP 模型，只需把 `SCENESEEK_MODEL_ID` 改成对应的 Hugging Face model id 并重建。

## 命令行

```bash
sceneseek scan /absolute/path/to/media
sceneseek build
sceneseek status
sceneseek serve --port 8000
```

常用环境变量见 [.env.example](.env.example)。生产环境建议设置 `SCENESEEK_ALLOWED_ROOTS`，用逗号分隔允许扫描的根目录。

## API

```text
POST /api/index/scan
POST /api/index/build
GET  /api/index/status
POST /api/search/text
POST /api/search/image
POST /api/search/composed
GET  /api/media/{media_id}
GET  /api/media/{media_id}/thumbnail
GET  /api/video/{media_id}/clip?start=&end=
POST /api/feedback/relevance
```

启动后可在 `/docs` 查看 OpenAPI 文档。

## 测试

```bash
pytest
ruff check src tests
cd web && npm run build
```

测试覆盖重复扫描、文件修改失效、删除/重复内容清理、采样配置触发 clip 刷新、frame embedding cache 复用、向量检索、Recall/MRR/nDCG/IoU，以及扫描到搜索并记录 query/impression 的 API 路径。

## 部署

### Railway

仓库根目录包含 `Dockerfile` 与 `railway.toml`。挂载持久化 volume 到 `/data`，并按需设置：

```text
SCENESEEK_ALLOWED_ROOTS=/media
SCENESEEK_CORS_ORIGINS=https://your-name.github.io
```

注意：Railway 容器无法直接读取你个人电脑上的目录。云端部署需要把授权媒体挂载到容器，或只把 Railway 用作可公开演示的样例库。

### GitHub Pages

仓库包含前端 Pages workflow。把 Railway API 地址保存为仓库 Actions secret `RAILWAY_API_URL`，然后在 GitHub Pages 设置中选择 GitHub Actions 作为来源。若 API 不公开，Pages 前端无法访问你的本地服务。

## 隐私边界

- 默认不上传原始媒体。
- 媒体接口只允许访问已进入 SQLite 的文件记录，不能用 URL 任意读取服务器路径。
- 可通过 `SCENESEEK_ALLOWED_ROOTS` 限制索引目录。
- `data/` 保存数据库、缩略图、视频预览片段与索引，已从 Git 排除。
- 删除源文件并重新扫描对应根目录后，元数据、clip、embedding、缩略图与预览缓存会一致性清理。

## 下一阶段

P1–P4 仍需继续用真实数据与 GPU 实验推进：Temporal Adapter/Hard Negative 的重复实验与消融、QVHighlights Moment Head、标准 moment 指标和 Reranker 质量/延迟曲线。`encoders/`、`temporal/`、`retrieval/` 与 `eval/` 已按这些阶段分离，后续实现不需要推倒 P0 产品链路。

## P1：本地训练 Temporal Adapter

SceneSeek 的训练设计默认把昂贵的视觉/文本 encoder 冻结，只预计算一次 frame/text embedding，再训练轻量 Temporal Adapter。这样训练阶段只读取小尺寸特征，不需要每个 epoch 重复跑 CLIP/SigLIP，适合单张消费级 GPU。

### 1. 准备 benchmark manifest

训练工具接受通用 JSONL，每行格式：

```json
{"sample_id":"train:video0:0","video_id":"video0","video_path":"/data/MSRVTT/video0.mp4","caption":"a person is walking","split":"train"}
```

如果使用常见的 MSR-VTT `sentences` JSON，可以先转换：

```bash
sceneseek make-msrvtt-manifest annotations/train.json /data/MSRVTT/videos manifests/train.jsonl --split train
sceneseek make-msrvtt-manifest annotations/val.json   /data/MSRVTT/videos manifests/val.jsonl   --split val
sceneseek make-msrvtt-manifest annotations/test.json  /data/MSRVTT/videos manifests/test.jsonl  --split test
cat manifests/train.jsonl manifests/val.jsonl manifests/test.jsonl > manifests/all.jsonl
```

不同 MSR-VTT 发布版本的 split 约定并不完全一致，因此 SceneSeek 要求显式提供 split，不自行猜测。

### 2. 预计算冻结 encoder 特征

建议正式实验使用 CLIP/SigLIP，而不是 LiteEncoder：

```bash
pip install -e '.[ml,dev]'
export SCENESEEK_ENCODER=transformers
export SCENESEEK_MODEL_ID=openai/clip-vit-base-patch32
export SCENESEEK_DEVICE=cuda

sceneseek prepare-temporal manifests/all.jsonl data/msrvtt-features \
  --sample-fps 1.0 \
  --max-frames 16 \
  --batch-size 32
```

输出目录包含共享的视频 frame embeddings、批量 text embeddings、split manifest 与 encoder metadata。一个视频有多条 caption 时不会重复保存视频特征。

### 3. 先跑 mean-pooling baseline

```bash
sceneseek benchmark-temporal data/msrvtt-features --split test
```

至少记录 `R@1 / R@5 / R@10 / MRR / Median Rank`。这组数字是 Temporal Adapter 必须打败的基线。

### 4. 在本地 GPU 训练 Temporal Adapter

```bash
sceneseek train-temporal data/msrvtt-features artifacts/temporal \
  --epochs 10 \
  --batch-size 64 \
  --lr 3e-4 \
  --layers 2 \
  --heads 8 \
  --max-frames 16 \
  --device cuda
```

训练只更新一个小型 Temporal Transformer；CLIP/SigLIP 已经被冻结为离线特征。loss 使用 symmetric multi-positive contrastive objective，同一视频的多条 caption 不会被错误当作负样本。默认按 validation MRR 保存 `best.pt`，并在连续 3 个 epoch 没有提升后 early stop；可以用 `--selection-metric` 和 `--early-stopping-patience` 调整。每轮记录 train loss 与 validation retrieval metrics，保存 `temporal-adapter.pt`、`best.pt`、`history.json` 和 `summary.json`。

#### Hard Negative v2

先用冻结的 CLIP text/video 特征在完整 train 候选池中挖掘每条 caption 最相似的错误视频：

```bash
sceneseek mine-hard-negatives data/msrvtt-features \
  --split train \
  --top-k 16 \
  --device cuda
```

然后让每个 batch 的一半样本由这些全局困难负样本补入；总 batch size 不变：

```bash
sceneseek train-temporal data/msrvtt-features artifacts/temporal-v2 \
  --epochs 10 \
  --batch-size 64 \
  --lr 3e-4 \
  --layers 2 \
  --heads 8 \
  --max-frames 16 \
  --device cuda \
  --selection-metric mrr \
  --early-stopping-patience 3 \
  --hard-negatives data/msrvtt-features/hard-negatives-train.npz \
  --hard-negative-pool 16
```

Hard Negative index 只基于冻结 encoder 特征生成，可以跨多次 Temporal Adapter 训练复用。它的目标是把训练重点从随机容易负样本转向语义相近但视频错误的候选；是否改善 `R@1` 仍必须以 validation/test 实验为准。

### 5. 评测训练后的 checkpoint

```bash
sceneseek benchmark-temporal data/msrvtt-features \
  --split test \
  --checkpoint artifacts/temporal/best.pt \
  --device cuda
```

只有真实 test split 指标超过 mean-pooling baseline 后，才应该在 README 或简历里声称 Temporal Adapter 带来提升。

### 6. 把 checkpoint 接回 SceneSeek 产品索引

```bash
export SCENESEEK_TEMPORAL_CHECKPOINT=$PWD/artifacts/temporal/best.pt
sceneseek build --rebuild
```

checkpoint 内容会进入索引版本 fingerprint；更换 Temporal Adapter 会自动要求重建 clip embedding，但仍复用冻结 encoder 的 frame cache。未设置 `SCENESEEK_TEMPORAL_CHECKPOINT` 时行为与 P0.5 一致，继续使用 mean pooling。

> 当前 P1 已覆盖 text-to-video representation learning 与离线 Hard Negative mining；QVHighlights Moment Head 和 Reranker 仍属于后续阶段。Hard Negative 是否带来指标提升，在完成 v2 实验前不包装成已验证结果。
