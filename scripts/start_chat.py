#!/usr/bin/env python
"""Start the RAG Chat UI server.

Usage:
    python scripts/start_chat.py
    python scripts/start_chat.py --port 9000
    python scripts/start_chat.py --host 127.0.0.1 --port 8080 --config config/settings.yaml
"""

import argparse
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Start the RAG Chat UI server.")
    parser.add_argument("--host", default="0.0.0.0", help="Bind host (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8080, help="Bind port (default: 8080)")
    parser.add_argument(
        "--config",
        default=str(_ROOT / "config" / "settings.yaml"),
        help="Path to settings.yaml (default: config/settings.yaml)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        print(f"[ERROR] Config file not found: {config_path}")
        sys.exit(1)

    # Pass config path to the FastAPI app via environment variable
    os.environ["CHAT_CONFIG"] = str(config_path.resolve())

    display_host = "localhost" if args.host == "0.0.0.0" else args.host
    print(f"[*] Starting RAG Chat UI")
    print(f"[*] Config: {config_path}")
    print(f"[*] Access at: http://{display_host}:{args.port}")

    import uvicorn
    uvicorn.run(
        "app.chat_api:app",
        host=args.host,
        port=args.port,
        reload=False,
    )


if __name__ == "__main__":
    main()
