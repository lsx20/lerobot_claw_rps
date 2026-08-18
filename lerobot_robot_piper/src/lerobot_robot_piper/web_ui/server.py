"""User-facing claw/RPS web UI controller.

The frontend is the old multi-page UI promoted to a package-level controller.
RPS and remote claw control are separate hardware programs. This web layer
does not send DisableArm or any disable-exit command.

* RPS mode talks to rps_yolo_pick.py through the existing JSON state/command
  protocol.
* Remote mode launches the claw-machine teleop backend, not the RPS backend.
"""
from __future__ import annotations

import argparse
import json
import mimetypes
import os
import signal
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse


ROOT = Path(__file__).resolve().parent
PACKAGE_ROOT = ROOT.parent
SRC_ROOT = PACKAGE_ROOT.parent
RPS_ROOT = PACKAGE_ROOT / "rock_paper_scissors"
REPO_ROOT = ROOT.parents[3]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from lerobot_robot_piper.piper_follower import load_piper_interface_v2  # noqa: E402
from lerobot_robot_piper.web_ui.piper_boot import (  # noqa: E402
    DEFAULT_DISABLE_EXIT_DURATION_S,
    DEFAULT_DISABLE_EXIT_SETTLE_S,
    prepare_piper_can_control,
    run_d_disable_flow,
)

DEFAULT_STATE = RPS_ROOT / "rps_touch_ui_state.json"
DEFAULT_COMMAND = RPS_ROOT / "rps_touch_ui_command.json"
DEFAULT_HTML = ROOT / "index.html"
DEFAULT_RPS_BACKEND = RPS_ROOT / "run_rps_homography_grasp_touch_ui.sh"
DEFAULT_REMOTE_BACKEND = REPO_ROOT / "run_claw_machine.sh"
DASHBOARD = RPS_ROOT / "ball_tactile_classifier" / "live_dashboard.html"
DASHBOARD_ROOT = DASHBOARD.parent
TACTILE_OUTPUT = RPS_ROOT / "ball_tactile_classifier" / "live_predictions.csv"
TACTILE_PENDING_HTML = (
    '<header>'
    '<div>'
    '<h1>RH56F2 tactile live dashboard</h1>'
    '<p class="muted">Auto refreshes every 0.5s. Heatmap color and area show finger force in N.</p>'
    '</div>'
    '<div class="prediction">stage1 ? final ? confidence - status <strong>classifying</strong></div>'
    '</header>'
    '<section class="panel ball-panel">'
    '<div class="current-layout">'
    '<div class="ball-art-shell">'
    '<div class="ball-art-meta">'
    '<div class="ball-art-line">?</div>'
    '<div class="ball-art-title">classifying</div>'
    '<div class="ball-art-line"><strong>正在抓取/判断</strong></div>'
    '<div class="ball-art-line">结果出来后会立刻刷新到这里</div>'
    '</div>'
    '</div>'
    '<div class="heatmap-card">'
    '<div class="heatmap-head">'
    '<div>'
    '<h3>Hand Force delta (N)</h3>'
    '<p class="muted">forceAct delta / 100; green ring = thumb proximity</p>'
    '</div>'
    '<div class="heatmap-total"><strong>0.0</strong><span>TOTAL N</span></div>'
    '</div>'
    '<div class="force-heatmap">'
    '<div class="heatmap-value" style="left:17%;top:31%;width:46px;height:46px;">0.0N</div>'
    '<div class="heatmap-label" style="left:17%;top:42%;">little</div>'
    '<div class="heatmap-value" style="left:34%;top:23%;width:46px;height:46px;">0.0N</div>'
    '<div class="heatmap-label" style="left:34%;top:34%;">ring</div>'
    '<div class="heatmap-value" style="left:50%;top:18%;width:46px;height:46px;">0.0N</div>'
    '<div class="heatmap-label" style="left:50%;top:29%;">middle</div>'
    '<div class="heatmap-value" style="left:67%;top:25%;width:46px;height:46px;">0.0N</div>'
    '<div class="heatmap-label" style="left:67%;top:36%;">index</div>'
    '<div class="heatmap-value" style="left:80%;top:56%;width:46px;height:46px;">0.0N</div>'
    '<div class="heatmap-label" style="left:80%;top:67%;">thumb</div>'
    '</div>'
    '<div class="heatmap-foot">'
    '<div><span>max </span><strong>little</strong><br>No hover proximity yet.</div>'
    '<div class="heatmap-legend"><span><i></i>low</span><span><i></i>mid</span><span><i></i>high</span></div>'
    '</div>'
    '</div>'
    '</div>'
    '</section>'
)

COMPAT_STAGE_MAP = {
    "boot": {"busy": False, "mode": "rps", "last_result": None, "lottery_started": False, "remote_result": None},
    "wait_start": {"busy": False, "mode": "rps", "last_result": None, "lottery_started": False, "remote_result": None},
    "align_hand": {"busy": True, "mode": "rps", "last_result": None, "lottery_started": False, "remote_result": None},
    "result": {"busy": False, "mode": "rps", "last_result": None, "lottery_started": False, "remote_result": None},
    "tactile": {"busy": True, "mode": "rps", "last_result": None, "lottery_started": True, "remote_result": None},
    "done": {"busy": False, "mode": "rps", "last_result": None, "lottery_started": True, "remote_result": None},
    "pick_failed": {"busy": False, "mode": "rps", "last_result": None, "lottery_started": True, "remote_result": None},
    "remote_running": {"busy": True, "mode": "remote", "last_result": None, "lottery_started": False, "remote_result": None},
    "remote_done": {"busy": False, "mode": "remote", "last_result": None, "lottery_started": False, "remote_result": "SUCCESS"},
    "remote_failed": {"busy": False, "mode": "remote", "last_result": None, "lottery_started": False, "remote_result": "FAILED"},
    "device_not_enabled": {"busy": False, "mode": None, "last_result": None, "lottery_started": False, "remote_result": None},
}

IDLE_STAGES = {"boot", "wait_start", "done", "pick_failed", "remote_done", "remote_failed", "device_not_enabled"}
RPS_STAGES = {"wait_start", "align_hand", "result", "tactile", "done", "pick_failed"}
REMOTE_STAGES = {"remote_running", "remote_done", "remote_failed"}
DISABLE_EXIT_COMMAND = "__disable_exit__"


def queue_disable_command(command_file: Path) -> None:
    previous = read_json(command_file, {"seq": 0})
    seq = int(previous.get("seq", 0) or 0) + 1
    atomic_write_json(
        command_file,
        {"seq": seq, "command": DISABLE_EXIT_COMMAND, "created_at": time.time()},
    )


def run_web_ui_d_disable(
    can_port: str,
    command_file: Path,
    backend_manager: BackendManager | None,
    enable_keeper: PiperEnableKeeper | None,
    wait_for_backend: bool = True,
) -> None:
    if enable_keeper is not None:
        enable_keeper.stop()
    queue_disable_command(command_file)
    if not wait_for_backend:
        return
    finished = False
    if backend_manager is not None and backend_manager._running():
        timeout = DEFAULT_DISABLE_EXIT_DURATION_S + DEFAULT_DISABLE_EXIT_SETTLE_S + 2.0
        print(f"Waiting up to {timeout:.1f}s for backend D-flow disable.")
        finished = backend_manager.wait_for_exit(timeout)
    if not finished:
        print("Running D-flow disable from Web UI process.")
        run_d_disable_flow(can_port)


class PiperEnableKeeper:
    def __init__(self, can_port: str, interval_s: float = 0.05):
        self.can_port = can_port
        self.interval_s = interval_s
        self.piper = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._last_status: list[bool] = []
        self._last_error: str | None = None

    def start(self) -> None:
        C_PiperInterface_V2 = load_piper_interface_v2()
        self.piper = C_PiperInterface_V2(
            self.can_port,
            judge_flag=False,
            can_auto_init=False,
            dh_is_offset=1,
            start_sdk_fk_cal=True,
        )
        self.piper.ConnectPort()
        time.sleep(0.2)
        self._thread = threading.Thread(target=self._run, name="piper-enable-keeper", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        assert self.piper is not None
        while not self._stop.is_set():
            try:
                self.piper.EnableArm(7, 0x02)
                self._last_status = list(self.piper.GetArmEnableStatus())
                self._last_error = None
            except Exception as exc:
                self._last_error = str(exc)
            time.sleep(self.interval_s)

    def status(self) -> dict[str, object]:
        return {"can": self.can_port, "enable": self._last_status, "error": self._last_error}

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        if self.piper is not None:
            self.piper.DisconnectPort()
            self.piper = None


def read_json(path: Path, fallback: dict[str, object]) -> dict[str, object]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return fallback


def atomic_write_json(path: Path, data: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp_path.replace(path)


def publish_state(state_file: Path, stage: str, prompt: str, **payload: object) -> None:
    atomic_write_json(state_file, {"stage": stage, "prompt": prompt, "updated_at": time.time(), **payload})


def clear_tactile_live_outputs() -> None:
    for path in (TACTILE_OUTPUT, DASHBOARD):
        try:
            path.unlink()
        except FileNotFoundError:
            pass


def read_raw_state(state_file: Path) -> dict[str, object]:
    return read_json(state_file, {"stage": "boot", "prompt": "程序正在启动，请稍候"})


def compat_status(raw_state: dict[str, object]) -> dict[str, object]:
    stage = str(raw_state.get("stage", "boot") or "boot")
    data = dict(COMPAT_STAGE_MAP.get(stage, COMPAT_STAGE_MAP["boot"]))
    data.update(
        stage=stage,
        prompt=raw_state.get("prompt", ""),
        updated_at=raw_state.get("updated_at"),
        player=raw_state.get("player"),
        system=raw_state.get("system"),
        result=raw_state.get("result"),
        stable_count=raw_state.get("stable_count"),
        stable_frames=raw_state.get("stable_frames"),
        pick_ok=raw_state.get("pick_ok"),
        enabled=raw_state.get("enabled"),
    )
    if stage in {"result", "tactile", "done", "pick_failed"} and raw_state.get("result"):
        data["last_result"] = raw_state.get("result")
    if stage == "remote_done":
        data["remote_result"] = "SUCCESS"
    if stage == "remote_failed":
        data["remote_result"] = "FAILED"
    return data


def stage_of(raw_state: dict[str, object]) -> str:
    return str(raw_state.get("stage", "boot") or "boot")


class DemoSession:
    def __init__(self) -> None:
        self.last_result: str | None = None
        self.rounds = iter(("SYSTEM_WIN", "TIE", "PLAYER_WIN"))
        self.remote_results = iter(("FAILED", "SUCCESS"))
        self.state: dict[str, object] = {"stage": "boot", "prompt": "程序正在启动，请稍候", "updated_at": time.time()}

    def read_state(self) -> dict[str, object]:
        return dict(self.state)

    def publish(self, stage: str, prompt: str, **payload: object) -> dict[str, object]:
        self.state = {"stage": stage, "prompt": prompt, "updated_at": time.time(), **payload}
        return self.read_state()

    def start_game(self) -> dict[str, object]:
        self.publish("align_hand", "请伸手对准摄像头与灵巧手进行猜拳", stable_count=0, stable_frames=5)
        result = next(self.rounds, "PLAYER_WIN")
        self.last_result = result
        if result == "PLAYER_WIN":
            return self.publish("result", "恭喜获胜", player="Rock", system="Scissors", result=result)
        return self.publish("result", "再试一次", player="Paper", system="Scissors", result=result)

    def start_lottery(self) -> dict[str, object]:
        if self.last_result != "PLAYER_WIN":
            return self.publish("pick_failed", "请先在猜拳中获胜", pick_ok=False)
        return self.publish("tactile", "抓取触觉界面", pick_ok=True)

    def finish_lottery(self) -> dict[str, object]:
        return self.publish("done", "已完成抽奖画面", pick_ok=True)

    def run_remote(self) -> dict[str, object]:
        result = next(self.remote_results, "SUCCESS")
        stage = "remote_done" if result == "SUCCESS" else "remote_failed"
        return self.publish(stage, "遥控抓取结束", remote_ok=result == "SUCCESS")


class BackendManager:
    def __init__(self, state_file: Path, rps_script: Path, remote_script: Path, auto_rps: bool):
        self.state_file = state_file
        self.rps_script = rps_script
        self.remote_script = remote_script
        self.auto_rps = auto_rps
        self.active_mode: str | None = None
        self.process: subprocess.Popen | None = None

    def _running(self) -> bool:
        return self.process is not None and self.process.poll() is None

    def status(self) -> dict[str, object]:
        if self.process is not None and self.process.poll() is not None:
            rc = self.process.returncode
            mode = self.active_mode
            self.process = None
            self.active_mode = None
            if mode == "remote":
                if rc == 0:
                    publish_state(self.state_file, "remote_done", "遥控抓取流程已结束", remote_ok=True)
                else:
                    publish_state(self.state_file, "remote_failed", f"遥控抓取流程异常退出：{rc}", remote_ok=False)
        return {"mode": self.active_mode, "running": self._running()}

    def stop(self, timeout_s: float = 8.0) -> None:
        if not self._running():
            self.process = None
            self.active_mode = None
            return
        assert self.process is not None
        self.process.send_signal(signal.SIGINT)
        try:
            self.process.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            self.process.terminate()
            self.process.wait(timeout=timeout_s)
        self.process = None
        self.active_mode = None

    def wait_for_exit(self, timeout_s: float) -> bool:
        if not self._running():
            self.process = None
            self.active_mode = None
            return True
        assert self.process is not None
        try:
            self.process.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            return False
        self.process = None
        self.active_mode = None
        return True

    def start(self, mode: str) -> dict[str, object]:
        self.status()
        if self.active_mode == mode and self._running():
            return {"mode": mode, "running": True, "already_running": True}
        if self._running():
            self.stop()
        script = self.rps_script if mode == "rps" else self.remote_script
        if not script.exists():
            raise FileNotFoundError(str(script))
        popen_kwargs: dict[str, object] = {"start_new_session": False}
        if mode == "rps":
            if not sys.stdin.isatty():
                raise RuntimeError("统一硬件后端需要 Web UI server 运行在真实终端中，以便遥控模式读取手柄输入")
        self.process = subprocess.Popen(
            [str(script)],
            cwd=str(script.parent),
            **popen_kwargs,
        )
        self.active_mode = mode
        if mode == "remote":
            publish_state(self.state_file, "remote_running", "遥控抓取后端运行中，请使用手柄操作")
        else:
            publish_state(self.state_file, "boot", "猜拳自主抓取后端启动中，请稍候")
        return {"mode": mode, "running": True, "pid": self.process.pid}

    def ensure_rps(self) -> dict[str, object]:
        if not self.auto_rps:
            return self.status()
        return self.start("rps")

    def ensure_remote(self) -> dict[str, object]:
        return self.start("remote")


def can_start_rps(raw_state: dict[str, object]) -> tuple[bool, str]:
    stage = stage_of(raw_state)
    if stage in REMOTE_STAGES and stage not in {"remote_done", "remote_failed"}:
        return False, "遥控抓取正在运行，不能同时开始猜拳自主抓取"
    if stage not in IDLE_STAGES and stage != "result":
        return False, "当前流程正在运行，请等待结束后再切换模式"
    return True, ""


def can_draw_prize(raw_state: dict[str, object]) -> tuple[bool, str]:
    stage = stage_of(raw_state)
    if stage != "result" or raw_state.get("result") != "PLAYER_WIN":
        return False, "请先在猜拳中获胜，再启动自主抓取"
    return True, ""


def can_start_remote(raw_state: dict[str, object]) -> tuple[bool, str]:
    stage = stage_of(raw_state)
    if stage in RPS_STAGES and stage not in {"wait_start", "done", "pick_failed"}:
        return False, "猜拳自主抓取流程正在运行，不能同时开始遥控抓取"
    if stage == "remote_running":
        return False, "遥控抓取正在运行"
    if stage not in IDLE_STAGES:
        return False, "当前流程正在运行，请等待结束后再切换模式"
    return True, ""


def make_handler(
    state_file: Path,
    command_file: Path,
    html_file: Path,
    backend_manager: BackendManager | None,
    enable_keeper: PiperEnableKeeper | None,
    disable_password: str,
    can_port: str,
) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        demo = False
        session = DemoSession()

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
            elif content_type.startswith("text/"):
                content_type = f"{content_type}; charset=utf-8"
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def read_state(self) -> dict[str, object]:
            if self.demo:
                return self.session.read_state()
            if backend_manager is not None:
                backend_manager.status()
            return read_raw_state(state_file)

        def wait_for_state(self, previous_updated_at: object, predicate, timeout_s: float) -> dict[str, object]:
            deadline = time.monotonic() + timeout_s
            while time.monotonic() < deadline:
                current = self.read_state()
                updated_at = current.get("updated_at")
                if updated_at != previous_updated_at and predicate(current):
                    return current
                time.sleep(0.05)
            raise TimeoutError("timeout waiting for backend state update")

        def queue_command(self, command: str) -> int:
            if self.demo:
                if command == "start_game":
                    self.session.start_game()
                elif command == "draw_prize":
                    self.session.start_lottery()
                    self.session.finish_lottery()
                elif command == "quit":
                    self.session.publish("boot", "程序正在启动，请稍候")
                return int(self.session.state.get("updated_at", time.time()))
            previous = read_json(command_file, {"seq": 0})
            seq = int(previous.get("seq", 0) or 0) + 1
            atomic_write_json(command_file, {"seq": seq, "command": command, "created_at": time.time()})
            return seq

        def do_GET(self) -> None:  # noqa: N802
            path = unquote(urlparse(self.path).path)
            if path in {"/", "/index.html"}:
                self.send_file(html_file)
                return
            if path == "/api/status":
                state = compat_status(self.read_state())
                if backend_manager is not None:
                    state["backend"] = backend_manager.status()
                self.send_json(state)
                return
            if path == "/api/state":
                self.send_json(self.read_state())
                return
            if path == "/api/tactile/summary":
                try:
                    html = tactile_summary_html()
                except OSError:
                    self.send_json({"html": TACTILE_PENDING_HTML, "pending": True})
                    return
                except Exception as exc:
                    self.send_json({"error": str(exc)}, 500)
                    return
                self.send_json({"html": html})
                return
            if path == "/dashboard":
                self.send_file(DASHBOARD)
                return
            if path.startswith("/dashboard_assets/"):
                file_path = (DASHBOARD_ROOT / path.removeprefix("/dashboard_assets/")).resolve()
                if file_path.is_file() and DASHBOARD_ROOT.resolve() in file_path.parents:
                    self.send_file(file_path)
                    return
                self.send_error(404)
                return
            file_path = (ROOT / path.lstrip("/")).resolve()
            if file_path.is_file() and ROOT.resolve() in file_path.parents:
                self.send_file(file_path)
                return
            self.send_error(404)

        def do_POST(self) -> None:  # noqa: N802
            path = unquote(urlparse(self.path).path)
            if path == "/api/disable":
                length = int(self.headers.get("Content-Length", "0") or 0)
                try:
                    payload = json.loads(self.rfile.read(length).decode("utf-8") or "{}")
                except json.JSONDecodeError:
                    self.send_json({"ok": False, "error": "bad json"}, 400)
                    return
                if str(payload.get("password", "")) != disable_password:
                    self.send_json({"ok": False, "error": "密码错误"}, 403)
                    return
                try:
                    run_web_ui_d_disable(
                        can_port,
                        command_file,
                        backend_manager,
                        enable_keeper,
                        wait_for_backend=False,
                    )
                except Exception as exc:
                    self.send_json({"ok": False, "error": f"失能失败：{exc}"}, 500)
                    return
                self.send_json({"ok": True})
                return
            if path == "/api/mode/rps":
                if self.demo:
                    self.session.publish("wait_start", "点击开始游戏")
                    self.send_json({"mode": "rps"})
                    return
                ok, error = can_start_rps(self.read_state())
                if not ok:
                    self.send_json({"error": error}, 409)
                    return
                try:
                    status = backend_manager.ensure_rps() if backend_manager is not None else {}
                except Exception as exc:
                    self.send_json({"error": str(exc)}, 500)
                    return
                if backend_manager is not None:
                    current_state = self.read_state()
                    if stage_of(current_state) == "boot":
                        try:
                            current_state = self.wait_for_state(
                                current_state.get("updated_at"),
                                lambda state: stage_of(state) == "wait_start",
                                120.0,
                            )
                        except TimeoutError as exc:
                            self.send_json({"error": str(exc)}, 504)
                            return
                    before = current_state.get("updated_at")
                    self.queue_command("prepare_rps")
                    try:
                        self.wait_for_state(
                            before,
                            lambda state: stage_of(state) == "wait_start" and state.get("mode") == "rps_ready",
                            120.0,
                        )
                    except TimeoutError as exc:
                        self.send_json({"error": str(exc)}, 504)
                        return
                self.send_json({"mode": "rps", "backend": status})
                return
            if path == "/api/mode/remote":
                if self.demo:
                    self.session.publish("remote_running", "当前为遥控抓取模式")
                    self.send_json({"mode": "remote"})
                    return
                try:
                    status = backend_manager.ensure_rps() if backend_manager is not None else {}
                except Exception as exc:
                    self.send_json({"error": str(exc)}, 500)
                    return
                if backend_manager is not None:
                    current_state = self.read_state()
                    if stage_of(current_state) == "wait_start":
                        self.queue_command("prepare_claw")
                self.send_json({"mode": "remote", "backend": status})
                return
            if path == "/api/command":
                length = int(self.headers.get("Content-Length", "0") or 0)
                try:
                    payload = json.loads(self.rfile.read(length).decode("utf-8") or "{}")
                except json.JSONDecodeError:
                    self.send_json({"ok": False, "error": "bad json"}, 400)
                    return
                command = str(payload.get("command", "")).strip()
                if command not in {"prepare_claw", "prepare_rps", "start_game", "draw_prize", "remote_run", DISABLE_EXIT_COMMAND, "quit"}:
                    self.send_json({"ok": False, "error": "bad command"}, 400)
                    return
                seq = self.queue_command(command)
                self.send_json({"ok": True, "seq": seq})
                return
            if path == "/api/game/start":
                if self.demo:
                    current = self.session.start_game()
                    self.send_json({"result": current.get("result"), "player": current.get("player"), "system": current.get("system"), "stage": current.get("stage")})
                    return
                try:
                    if backend_manager is not None:
                        backend_manager.ensure_rps()
                except Exception as exc:
                    self.send_json({"error": str(exc)}, 500)
                    return
                current_state = self.read_state()
                if stage_of(current_state) == "boot":
                    try:
                        current_state = self.wait_for_state(
                            current_state.get("updated_at"),
                            lambda state: stage_of(state) in {"wait_start", "result", "done", "pick_failed"},
                            120.0,
                        )
                    except TimeoutError as exc:
                        self.send_json({"error": str(exc)}, 504)
                        return
                ok, error = can_start_rps(current_state)
                if not ok:
                    self.send_json({"error": error}, 409)
                    return
                before = current_state.get("updated_at")
                self.queue_command("start_game")
                try:
                    current = self.wait_for_state(
                        before,
                        lambda state: state.get("result") is not None or str(state.get("stage", "")) in {"result", "tactile", "done", "pick_failed"},
                        60.0,
                    )
                except TimeoutError as exc:
                    self.send_json({"error": str(exc)}, 504)
                    return
                self.send_json({"result": current.get("result"), "player": current.get("player"), "system": current.get("system"), "stage": current.get("stage")})
                return
            if path == "/api/lottery/start":
                if self.demo:
                    current = self.session.start_lottery()
                    self.send_json({"dashboard": "/dashboard", "stage": current.get("stage"), "pick_ok": current.get("pick_ok")})
                    return
                ok, error = can_draw_prize(self.read_state())
                if not ok:
                    self.send_json({"error": error}, 409)
                    return
                before = self.read_state().get("updated_at")
                clear_tactile_live_outputs()
                self.queue_command("draw_prize")
                try:
                    current = self.wait_for_state(before, lambda state: str(state.get("stage", "")) in {"tactile", "done", "pick_failed"}, 180.0)
                except TimeoutError as exc:
                    self.send_json({"error": str(exc)}, 504)
                    return
                self.send_json({"dashboard": "/dashboard", "stage": current.get("stage"), "pick_ok": current.get("pick_ok")})
                return
            if path == "/api/remote/run":
                if self.demo:
                    current = self.session.run_remote()
                    self.send_json({"result": "SUCCESS" if current.get("remote_ok") else "FAILED", "stage": current.get("stage")})
                    return
                current_state = self.read_state()
                ok, error = can_start_remote(current_state)
                if not ok:
                    self.send_json({"error": error}, 409)
                    return
                if backend_manager is None:
                    self.send_json({"error": "遥控抓取需要 Web UI server 管理统一硬件后端"}, 500)
                    return
                try:
                    status = backend_manager.ensure_rps()
                except Exception as exc:
                    self.send_json({"error": str(exc)}, 500)
                    return
                clear_tactile_live_outputs()
                current_state = self.read_state()
                if stage_of(current_state) == "boot":
                    try:
                        current_state = self.wait_for_state(
                            current_state.get("updated_at"),
                            lambda state: stage_of(state) in {"wait_start", "remote_running", "remote_done", "remote_failed"},
                            120.0,
                        )
                    except TimeoutError as exc:
                        self.send_json({"error": str(exc)}, 504)
                        return
                before = current_state.get("updated_at")
                self.queue_command("remote_run")
                try:
                    current = self.wait_for_state(
                        before,
                        lambda state: str(state.get("stage", "")) in {"remote_running", "remote_done", "remote_failed"},
                        120.0,
                    )
                except TimeoutError as exc:
                    self.send_json({"error": str(exc)}, 504)
                    return
                self.send_json({"result": "RUNNING", "stage": current.get("stage", "remote_running"), "backend": status})
                return
            self.send_error(404)

    return Handler


def tactile_summary_html() -> str:
    html = DASHBOARD.read_text(encoding="utf-8")
    panel_start = html.find('<section class="panel ball-panel"')
    panel_end = html.find("</section>", panel_start)
    if panel_start < 0 or panel_end < 0:
        raise RuntimeError("live_dashboard.html does not contain the expected tactile summary section")
    panel = html[panel_start : panel_end + len("</section>")]
    heatmap = extract_balanced_div(panel, '<div class="heatmap-card"')
    summary = f'<section class="panel ball-panel heatmap-only"><div class="current-layout">{heatmap}</div></section>'
    return summary.replace('src="ball_assets/', 'src="/dashboard_assets/ball_assets/')


def extract_balanced_div(html: str, marker: str) -> str:
    start = html.find(marker)
    if start < 0:
        raise RuntimeError("live_dashboard.html does not contain the expected tactile heatmap section")
    depth = 0
    index = start
    while index < len(html):
        next_open = html.find("<div", index)
        next_close = html.find("</div>", index)
        if next_close < 0:
            raise RuntimeError("live_dashboard.html tactile heatmap section is incomplete")
        if 0 <= next_open < next_close:
            depth += 1
            index = next_open + len("<div")
            continue
        depth -= 1
        index = next_close + len("</div>")
        if depth == 0:
            return html[start:index]
    raise RuntimeError("live_dashboard.html tactile heatmap section is incomplete")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--demo", action="store_true", help="run the user UI without the robot backend")
    parser.add_argument("--manage-backends", action="store_true", help="let the web UI switch independent RPS/remote backend scripts")
    parser.add_argument("--no-auto-rps-backend", action="store_true", help="do not start the RPS backend automatically from the web UI")
    parser.add_argument("--can", default="can0", help="Piper CAN interface used by the UI-level enable keeper")
    parser.add_argument("--enable-keeper", action="store_true", help="debug only: keep sending EnableArm from the Web UI process")
    parser.add_argument("--disable-password", default=os.environ.get("WEB_UI_DISABLE_PASSWORD", "piper"), help="password for the top-right debug disable button")
    parser.add_argument("--state-file", type=Path, default=DEFAULT_STATE)
    parser.add_argument("--command-file", type=Path, default=DEFAULT_COMMAND)
    parser.add_argument("--html-file", type=Path, default=DEFAULT_HTML)
    parser.add_argument("--rps-backend", type=Path, default=DEFAULT_RPS_BACKEND)
    parser.add_argument("--remote-backend", type=Path, default=DEFAULT_REMOTE_BACKEND)
    args = parser.parse_args()

    keeper = None
    if not args.demo:
        try:
            print("Checking Piper teach/CAN mode after setup_can0.")
            prepare_piper_can_control(args.can)
        except Exception as exc:
            print(f"[warn] teach/CAN restore failed: {exc}")
            try:
                run_d_disable_flow(args.can)
            except Exception as disable_exc:
                print(f"[warn] D-flow disable after boot failure failed: {disable_exc}")
            raise

    if not args.demo and args.enable_keeper:
        keeper = PiperEnableKeeper(args.can)
        keeper.start()

    manager = None
    if not args.demo and args.manage_backends:
        os.environ["WEB_UI_URL"] = f"http://{args.host}:{args.port}/"
        manager = BackendManager(args.state_file, args.rps_backend, args.remote_backend, not args.no_auto_rps_backend)
        publish_state(args.state_file, "boot", "请选择抓取模式")
        manager.ensure_rps()

    handler = make_handler(args.state_file, args.command_file, args.html_file, manager, keeper, args.disable_password, args.can)
    handler.demo = args.demo
    server = ThreadingHTTPServer((args.host, args.port), handler)
    print(f"Claw/RPS web UI: http://{args.host}:{args.port}/")
    if args.demo:
        print("Demo mode: the UI simulates game flow without the robot backend.")
    elif manager is not None:
        print(f"unified hardware backend: {args.rps_backend}")
        print("remote mode: unified backend remote_run -> ClawMachineController gamepad loop")
    else:
        print("Frontend-only mode: RPS backend must already be running; remote backend cannot be started.")
    print(f"state file: {args.state_file}")
    print(f"command file: {args.command_file}")
    if keeper is not None:
        print(f"enable keeper: {keeper.status()}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nWeb UI interrupted; running D-flow disable.")
        return 0
    except Exception as exc:
        print(f"\n[warn] Web UI crashed: {exc}")
        return 1
    finally:
        try:
            if not args.demo:
                run_web_ui_d_disable(args.can, args.command_file, manager, keeper)
        except Exception as disable_exc:
            print(f"[warn] D-flow disable on Web UI exit failed: {disable_exc}")
        server.server_close()
        if manager is not None:
            manager.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
