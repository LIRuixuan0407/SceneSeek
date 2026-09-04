from __future__ import annotations

import argparse
import json
from pathlib import Path

from sceneseek.config import Settings
from sceneseek.encoders import create_encoder
from sceneseek.ingestion import MediaIndexer, MediaScanner
from sceneseek.retrieval import VectorIndex
from sceneseek.storage import Database


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
    args = parser.parse_args()

    if args.command == "serve":
        import uvicorn

        uvicorn.run("sceneseek.service.app:app", host=args.host, port=args.port, reload=False)
        return

    settings = Settings()
    database = Database(settings.database_path)
    if args.command == "scan":
        result = MediaScanner(settings, database).scan(args.path)
        print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
    elif args.command == "build":
        encoder = create_encoder(settings)
        indexer = MediaIndexer(settings, database, encoder)
        result = indexer.build(rebuild=args.rebuild)
        index = VectorIndex(settings.index_path, settings.index_backend)
        result["vectors"] = index.rebuild(database, indexer.model_version)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(json.dumps(database.counts(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
