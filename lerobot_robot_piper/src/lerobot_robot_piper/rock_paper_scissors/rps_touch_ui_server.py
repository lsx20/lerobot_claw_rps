#!/usr/bin/env python3
"""Local touch-screen UI server for the RPS prize flow."""

from __future__ import annotations

import argparse
import json
import mimetypes
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote


ROOT = Path(__file__).resolve().parent
DEFAULT_STATE = ROOT / "rps_touch_ui_state.json"
DEFAULT_COMMAND = ROOT / "rps_touch_ui_command.json"
DEFAULT_HTML = ROOT / "rps_touch_ui.html"
TACTILE_DIR = ROOT / "ball_tactile_classifier"


def read_json(path: Path, fallback: dict[str, object]) -> dict[str, object]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return fallback


def atomic_write_json(path: Path, data: dict[str, object]) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp_path.replace(path)


def make_handler(state_file: Path, command_file: Path, html_file: Path) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: object) -> None:
            print(f"{self.address_string()} - {format % args}")

        def send_json(self, data: dict[str, object], status: int = 200) -> None:
            body = json.dumps(data, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def send_file(self, path: Path) -> None:
            if not path.exists() or not path.is_file():
                self.send_error(404)
                return
            body = path.read_bytes()
            content_type = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
            if path.suffix == ".html":
                content_type = "text/html; charset=utf-8"
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            path = unquote(self.path.split("?", 1)[0])
            if path in {"/", "/index.html"}:
                self.send_file(html_file)
                return
            if path == "/api/state":
                self.send_json(read_json(state_file, {"stage": "offline", "prompt": "等待猜拳程序连接"}))
                return
            if path.startswith("/tactile/"):
                requested = (TACTILE_DIR / path.removeprefix("/tactile/")).resolve()
                if TACTILE_DIR.resolve() not in requested.parents and requested != TACTILE_DIR.resolve():
                    self.send_error(403)
                    return
                self.send_file(requested)
                return
            self.send_error(404)

        def do_POST(self) -> None:
            if self.path != "/api/command":
                self.send_error(404)
                return
            length = int(self.headers.get("Content-Length", "0") or 0)
            try:
                payload = json.loads(self.rfile.read(length).decode("utf-8") or "{}")
            except json.JSONDecodeError:
                self.send_json({"ok": False, "error": "bad json"}, 400)
                return
            command = str(payload.get("command", "")).strip()
            if command not in {"start_game", "draw_prize", "quit"}:
                self.send_json({"ok": False, "error": "bad command"}, 400)
                return
            previous = read_json(command_file, {"seq": 0})
            seq = int(previous.get("seq", 0) or 0) + 1
            atomic_write_json(command_file, {"seq": seq, "command": command, "created_at": time.time()})
            self.send_json({"ok": True, "seq": seq})

    return Handler


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--state-file", type=Path, default=DEFAULT_STATE)
    parser.add_argument("--command-file", type=Path, default=DEFAULT_COMMAND)
    parser.add_argument("--html-file", type=Path, default=DEFAULT_HTML)
    args = parser.parse_args()

    handler = make_handler(args.state_file, args.command_file, args.html_file)
    server = ThreadingHTTPServer((args.host, args.port), handler)
    print(f"RPS touch UI: http://{args.host}:{args.port}/")
    print(f"state file: {args.state_file}")
    print(f"command file: {args.command_file}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        return 0
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
