"""
Development entry point that forces SelectorEventLoop on Windows.
Production (Linux) doesn't need this — uvicorn already uses the right loop.

Usage:
  python run.py                  # defaults: host=0.0.0.0  port=8083
  python run.py --port 9000
"""

import asyncio
import platform
import selectors
import sys


def _windows_loop_factory() -> asyncio.AbstractEventLoop:
    """Create a SelectorEventLoop — required by psycopg async on Windows."""
    selector = selectors.SelectSelector()
    return asyncio.SelectorEventLoop(selector)


def main() -> None:
    import uvicorn

    port = 8083
    host = "0.0.0.0"
    for i, arg in enumerate(sys.argv[1:], 1):
        if arg == "--port" and i < len(sys.argv) - 1:
            port = int(sys.argv[i + 1])
        elif arg == "--host" and i < len(sys.argv) - 1:
            host = sys.argv[i + 1]

    loop_kw = {}
    if platform.system() == "Windows":
        loop_kw["loop"] = _windows_loop_factory

    uvicorn.run(
        "app.main:app",
        host=host,
        port=port,
        workers=1,
        log_level="info",
        **loop_kw,
    )


if __name__ == "__main__":
    main()
