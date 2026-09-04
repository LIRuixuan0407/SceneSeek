# SceneSeek

SceneSeek 是一个本地优先的多模态图片与视频检索系统。你可以用自然语言或参考图片搜索本地媒体库；视频结果会返回相关短窗口的起止时间，而不是只返回整段视频。

当前版本完成设计文档的 P0 可用闭环，并为 Temporal Adapter、Moment Head、Hard Negative 与 Reranker 保留了清晰模块边界。它不会把尚未训练或评测的模型包装成完成结果。

## 已实现

- 递归扫描图片和视频，使用 SHA-256 内容哈希去重。
- SQLite 保存路径、尺寸、时长、FPS、修改时间、内容哈希与索引版本。
- 文件变化后自动失效旧向量；文件删除时级联清理 clip 与 embedding。
- FFmpeg / ffprobe 视频探测与抽帧。
- 默认 4 秒 window、2 秒 stride、1 FPS 的视频短窗口索引。
- 文本搜图片/视频、以图搜图/视频、组合查询 API。
- CLIP/SigLIP 兼容的 Transformers 编码器，以及无需下载模型的 CPU Lite 降级模式。
- FAISS Flat/HNSW（安装 ML extra 后）与精确 NumPy inner-product fallback。
- 同一视频相邻候选合并，每个视频限制重复 moment，返回 start/end/thumbnail 时间。
- FastAPI 状态、媒体、视频片段与 relevance feedback API。
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

切换模型会产生新的 `model_version`，已有媒体会进入待重建状态。若使用多语言 SigLIP 模型，只需把 `SCENESEEK_MODEL_ID` 改成对应的 Hugging Face model id 并重建。

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

测试覆盖文档要求的重复扫描、文件修改失效、删除清理、clip 边界、向量检索与 brute-force 一致性、Recall/MRR/nDCG/IoU，以及扫描到搜索的 API 路径。

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

P1–P4 仍需真实数据与 GPU 实验完成：MSR-VTT/Flickr baseline、Temporal Adapter 训练、Hard Negative mining、QVHighlights Moment Head、标准 moment 指标和 Reranker 质量/延迟曲线。`encoders/`、`temporal/`、`retrieval/` 与 `eval/` 已按这些阶段分离，后续实现不需要推倒 P0 产品链路。
