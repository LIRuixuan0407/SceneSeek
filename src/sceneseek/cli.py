from __future__ import annotations

import argparse
import json
from pathlib import Path

from sceneseek.config import Settings
from sceneseek.encoders import create_encoder
from sceneseek.ingestion import MediaIndexer, MediaScanner
from sceneseek.retrieval import VectorIndex
from sceneseek.storage import Database


def _print(payload: object) -> None:
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(prog="sceneseek")
    subparsers = parser.add_subparsers(dest="command", required=True)

    serve = subparsers.add_parser("serve", help="启动 FastAPI 服务")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)

    scan = subparsers.add_parser("scan", help="扫描媒体目录")
    scan.add_argument("path", type=Path)

    build = subparsers.add_parser("build", help="构建向量索引")
    build.add_argument("--rebuild", action="store_true")
    subparsers.add_parser("status", help="显示媒体库状态")

    msrvtt = subparsers.add_parser("make-msrvtt-manifest", help="转换 MSR-VTT 标注为 SceneSeek JSONL")
    msrvtt.add_argument("annotations", type=Path)
    msrvtt.add_argument("video_root", type=Path)
    msrvtt.add_argument("output", type=Path)
    msrvtt.add_argument("--split", required=True, choices=("train", "val", "test"))
    msrvtt.add_argument("--extension", default=".mp4")

    prepare = subparsers.add_parser("prepare-temporal", help="预计算视频帧与文本特征，供 Temporal Adapter 训练")
    prepare.add_argument("manifest", type=Path)
    prepare.add_argument("output", type=Path)
    prepare.add_argument("--sample-fps", type=float, default=1.0)
    prepare.add_argument("--max-frames", type=int, default=16)
    prepare.add_argument("--batch-size", type=int, default=32)
    prepare.add_argument("--overwrite", action="store_true")

    benchmark = subparsers.add_parser("benchmark-temporal", help="评测 mean-pooling baseline 或 Temporal Adapter checkpoint")
    benchmark.add_argument("features", type=Path)
    benchmark.add_argument("--split", default="test")
    benchmark.add_argument("--checkpoint", type=Path)
    benchmark.add_argument("--device", default="auto")
    benchmark.add_argument("--batch-size", type=int, default=64)

    train = subparsers.add_parser("train-temporal", help="训练轻量 Temporal Adapter")
    train.add_argument("features", type=Path)
    train.add_argument("output", type=Path)
    train.add_argument("--epochs", type=int, default=10)
    train.add_argument("--batch-size", type=int, default=64)
    train.add_argument("--lr", type=float, default=3e-4)
    train.add_argument("--weight-decay", type=float, default=0.01)
    train.add_argument("--temperature", type=float, default=0.07)
    train.add_argument("--layers", type=int, default=2)
    train.add_argument("--heads", type=int, default=8)
    train.add_argument("--dropout", type=float, default=0.1)
    train.add_argument("--max-frames", type=int, default=32)
    train.add_argument("--device", default="auto")
    train.add_argument("--seed", type=int, default=7)
    train.add_argument("--num-workers", type=int, default=0)

    args = parser.parse_args()

    if args.command == "serve":
        import uvicorn

        uvicorn.run(
            "sceneseek.service.app:create_app",
            host=args.host,
            port=args.port,
            reload=False,
            factory=True,
        )
        return

    if args.command == "make-msrvtt-manifest":
        from sceneseek.benchmarks import create_msrvtt_manifest

        _print(
            create_msrvtt_manifest(
                args.annotations,
                args.video_root,
                args.output,
                split=args.split,
                extension=args.extension,
            )
        )
        return

    if args.command == "prepare-temporal":
        from sceneseek.training import prepare_temporal_features

        settings = Settings()
        encoder = create_encoder(settings)
        _print(
            prepare_temporal_features(
                args.manifest,
                args.output,
                encoder,
                sample_fps=args.sample_fps,
                max_frames=args.max_frames,
                batch_size=args.batch_size,
                overwrite=args.overwrite,
            )
        )
        return

    if args.command == "benchmark-temporal":
        from sceneseek.training import evaluate_mean_pooling, evaluate_temporal_checkpoint

        if args.checkpoint:
            result = evaluate_temporal_checkpoint(
                args.features,
                args.checkpoint,
                split=args.split,
                device=args.device,
                batch_size=args.batch_size,
            )
        else:
            result = evaluate_mean_pooling(args.features, split=args.split)
        _print(result)
        return

    if args.command == "train-temporal":
        from sceneseek.training.train import train_temporal_adapter

        _print(
            train_temporal_adapter(
                args.features,
                args.output,
                epochs=args.epochs,
                batch_size=args.batch_size,
                learning_rate=args.lr,
                weight_decay=args.weight_decay,
                temperature=args.temperature,
                num_layers=args.layers,
                num_heads=args.heads,
                dropout=args.dropout,
                max_frames=args.max_frames,
                device=args.device,
                seed=args.seed,
                num_workers=args.num_workers,
            )
        )
        return

    settings = Settings()
    database = Database(settings.database_path)
    if args.command == "scan":
        result = MediaScanner(settings, database).scan(args.path)
        _print(result.to_dict())
    elif args.command == "build":
        encoder = create_encoder(settings)
        indexer = MediaIndexer(settings, database, encoder)
        result = indexer.build(rebuild=args.rebuild)
        index = VectorIndex(settings.index_path, settings.index_backend)
        result["vectors"] = index.rebuild(database, indexer.model_version)
        _print(result)
    else:
        _print(database.counts())


if __name__ == "__main__":
    main()
