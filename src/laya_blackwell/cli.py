"""Command line entry points for hardware inspection, inference, and serving."""

import argparse
import json
import logging
from pathlib import Path


def _engine_arguments(parser):
    parser.add_argument(
        "--mode",
        choices=("balanced", "fast"),
        default="balanced",
        help="balanced is default; fast uses compiled SM120 kernels and extra GPU tables",
    )
    parser.add_argument(
        "--backend",
        choices=("fused", "eager", "fp8"),
        default="fused",
        help="fused BF16 is default; fp8 is an experimental accuracy tradeoff",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--model", default="convaiinnovations/laya")
    parser.add_argument(
        "--revision", help="Model revision; defaults to the engine's pinned checkpoint"
    )


def _engine_options(args):
    options = {"model": args.model, "backend": args.backend, "device": args.device}
    if args.mode != "balanced":
        options["mode"] = args.mode
    if args.revision is not None:
        options["revision"] = args.revision
    return options


def main(argv=None):
    parser = argparse.ArgumentParser(prog="laya-blackwell")
    commands = parser.add_subparsers(dest="command", required=True)

    info = commands.add_parser(
        "info", help="Print the Blackwell GPU and runtime information"
    )
    info.add_argument("--device", default="cuda:0")

    build = commands.add_parser(
        "build-fast", help="Build the pinned native extensions for fast mode"
    )
    build.add_argument("--cuda-home", type=Path, help="CUDA toolkit containing nvcc")
    build.add_argument("--cutlass", type=Path, help="Local pinned CUTLASS checkout")
    build.add_argument(
        "--flash-attention", type=Path, help="Local pinned FlashAttention checkout"
    )
    build.add_argument(
        "--cache-dir",
        type=Path,
        help="Build cache; use LAYA_FAST_CACHE when running inference",
    )
    build.add_argument(
        "--offline",
        action="store_true",
        help="Use cached or supplied headers without network access",
    )

    predict = commands.add_parser("predict", help="Run one JSON request")
    _engine_arguments(predict)
    predict.add_argument("--request", type=Path, required=True)

    serve = commands.add_parser("serve", help="Serve the model over HTTP")
    _engine_arguments(serve)
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument(
        "--api-key", help="Bearer key; otherwise use LAYA_API_KEY if set"
    )
    serve.add_argument(
        "--no-warmup", action="store_true", help="Skip representative startup warmup"
    )
    args = parser.parse_args(argv)

    if args.command == "build-fast":
        from .fast.build import build_all

        try:
            built = build_all(
                cuda_home=args.cuda_home,
                cutlass=args.cutlass,
                flash_attention=args.flash_attention,
                cache_dir=args.cache_dir,
                offline=args.offline,
            )
        except RuntimeError as error:
            parser.exit(1, f"{error}\n")
        print(
            json.dumps(
                {
                    "ready": True,
                    "cache": built["directory"],
                    "native_libraries": len(built["artifacts"]),
                },
                indent=2,
            )
        )
        return 0

    if args.command == "info":
        from .engine import hardware_info

        print(json.dumps(hardware_info(args.device), indent=2))
        return 0

    if args.command == "predict":
        from .modes import create_engine
        from .server import SystemOneRequest

        try:
            with args.request.open(encoding="utf-8") as handle:
                request = SystemOneRequest.model_validate(json.load(handle))
        except (OSError, ValueError) as exc:
            parser.error(f"invalid request: {exc}")
        with create_engine(**_engine_options(args)) as engine:
            result = engine.predict(request.state, request.questions)
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0

    if not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535")
    import uvicorn

    from .server import create_app

    app = create_app(
        **_engine_options(args), api_key=args.api_key, warmup=not args.no_warmup
    )
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    uvicorn.run(app, host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
