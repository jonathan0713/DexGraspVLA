#!/usr/bin/env python3
import argparse
import os
import shlex
import signal
import socket
import subprocess
import sys
import threading
import time
import shutil
from pathlib import Path
from multiprocessing.connection import Client

import zmq


REPO_ROOT = Path(__file__).resolve().parents[1]
TRACKER_DIR = Path("/home/jonathan/Documents/ITRI_Project/Vision/sam3")
TRACKER_SCRIPT = REPO_ROOT / "tools" / "hot_tracker_server.py"
INFERENCE_SCRIPT = REPO_ROOT / "vla_inference_mycobot_copy_2.py"
DEFAULT_INFERENCE_RESET_FILE = "/tmp/vla_inference_reset.flag"
DEFAULT_INFERENCE_CONTROL_SOCKET = "/tmp/vla_inference_control.sock"
VLA_KILL_TIMEOUT = 5.0
ROS_SETUP_FILES = [
    "/opt/ros/humble/setup.bash",
    "/home/jonathan/Documents/ITRI_Project/ros2_ws/install/setup.bash",
]

DEFAULT_OBJECTS = {
    "1": ("blue bottle", "blue bottle"),
    "2": ("purple bottle", "purple bottle"),
    "3": ("green bottle", "green bottle"),
    "4": ("black bottle", "black bottle"),
    "5": ("custom", None),
}


def source_prefix(paths):
    existing = [path for path in paths if Path(path).exists()]
    if not existing:
        return ""
    return " && ".join(f"source {shlex.quote(path)}" for path in existing) + " && "


def conda_command(env_name, inner_command):
    if not env_name:
        return inner_command
    return f"conda run --no-capture-output -n {shlex.quote(env_name)} bash -lc {shlex.quote(inner_command)}"


def uv_command(project_dir, inner_command):
    return f"cd {shlex.quote(str(project_dir))} && uv run bash -lc {shlex.quote(inner_command)}"


def venv_command(project_dir, inner_command):
    python_path = project_dir / ".venv" / "bin" / "python"
    return f"cd {shlex.quote(str(project_dir))} && export PATH={shlex.quote(str(python_path.parent))}:$PATH && {inner_command}"


def system_command(project_dir, inner_command):
    return f"cd {shlex.quote(str(project_dir))} && {inner_command}"


def runner_command(runner, project_dir, inner_command, conda_env=None):
    if runner == "conda":
        return conda_command(conda_env, f"cd {shlex.quote(str(project_dir))} && {inner_command}")
    if runner == "uv":
        return uv_command(project_dir, inner_command)
    if runner == "venv":
        return venv_command(project_dir, inner_command)
    if runner == "system":
        return system_command(project_dir, inner_command)

    has_uv_project = (project_dir / "pyproject.toml").exists()
    has_venv = (project_dir / ".venv" / "bin" / "python").exists()
    has_uv = shutil.which("uv") is not None
    has_conda = shutil.which("conda") is not None

    if has_uv_project and has_uv:
        return uv_command(project_dir, inner_command)
    if has_venv:
        return venv_command(project_dir, inner_command)
    if conda_env and has_conda:
        return conda_command(conda_env, f"cd {shlex.quote(str(project_dir))} && {inner_command}")
    return system_command(project_dir, inner_command)


def build_tracker_command(args, prompt):
    if args.tracker_command:
        return args.tracker_command.format(prompt=shlex.quote(prompt), host=args.host, port=args.port)

    inner = (
        f"python {shlex.quote(str(TRACKER_SCRIPT))} "
        f"--prompt {shlex.quote(prompt)} "
        f"--host {shlex.quote(args.host)} "
        f"--port {args.port}"
    )
    if getattr(args, "tracker_no_debug_window", False):
        inner += " --no-debug-window"
    debug_frame_path = getattr(args, "tracker_debug_frame_path", None)
    if debug_frame_path:
        inner += f" --debug-frame-path {shlex.quote(debug_frame_path)}"
    return runner_command(args.tracker_runner, TRACKER_DIR, inner, args.tracker_env)


def build_inference_command(args):
    if args.inference_command:
        return args.inference_command

    inner = (
        source_prefix(ROS_SETUP_FILES)
        + f"export VLA_INFERENCE_RESET_FILE={shlex.quote(args.inference_reset_file)} && "
        + f"export VLA_INFERENCE_CONTROL_SOCKET={shlex.quote(args.inference_control_socket)} && "
        + f"python {shlex.quote(str(INFERENCE_SCRIPT))}"
    )
    return runner_command(args.inference_runner, REPO_ROOT, inner, args.inference_env)


def choose_prompt(args):
    if args.prompt:
        return args.prompt
    if args.object:
        lowered = args.object.strip().lower()
        for _, (name, prompt) in DEFAULT_OBJECTS.items():
            if lowered == name.lower() and prompt:
                return prompt
        return args.object

    return choose_prompt_interactive()


def choose_prompt_interactive():
    print("Select tracking target:")
    for key, (name, _) in DEFAULT_OBJECTS.items():
        print(f"  {key}. {name}")
    selection = input("Object [1]: ").strip() or "1"
    name, prompt = DEFAULT_OBJECTS.get(selection, ("custom", None))
    if prompt:
        return prompt
    return input("Tracker prompt: ").strip()


def prefix_output(name, stream):
    for line in iter(stream.readline, ""):
        print(f"[{name}] {line}", end="", flush=True)


class ManagedProcess:
    def __init__(self, name, command):
        self.name = name
        self.command = command
        self.process = None
        self._threads = []

    def start(self):
        print(f"\n==> starting {self.name}", flush=True)
        print(f"    {self.command}", flush=True)
        self.process = subprocess.Popen(
            ["bash", "-lc", self.command],
            cwd=str(REPO_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            start_new_session=True,
        )
        thread = threading.Thread(target=prefix_output, args=(self.name, self.process.stdout), daemon=True)
        thread.start()
        self._threads.append(thread)

    def stop(self, timeout=8.0):
        if not self.process or self.process.poll() is not None:
            return
        print(f"\n==> stopping {self.name}", flush=True)
        try:
            os.killpg(self.process.pid, signal.SIGINT)
            self.process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(self.process.pid, signal.SIGTERM)
            try:
                self.process.wait(timeout=3.0)
            except subprocess.TimeoutExpired:
                os.killpg(self.process.pid, signal.SIGKILL)
                self.process.wait()

    def kill(self, timeout=2.0):
        if not self.process or self.process.poll() is not None:
            return
        print(f"\n==> killing {self.name}", flush=True)
        try:
            os.killpg(self.process.pid, signal.SIGKILL)
            self.process.wait(timeout=timeout)
        except ProcessLookupError:
            return
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait()

    def poll(self):
        if not self.process:
            return None
        return self.process.poll()


def wait_for_managed_port(process, host, port, timeout):
    deadline = time.time() + timeout
    while time.time() < deadline:
        exit_code = process.poll()
        if exit_code is not None:
            raise RuntimeError(f"{process.name} exited with code {exit_code} before opening {host}:{port}")
        if port_is_open(host, port):
            return True
        time.sleep(0.25)
    return False


def port_is_open(host, port):
    try:
        with socket.create_connection((host, port), timeout=0.3):
            return True
    except OSError:
        return False


def print_controls():
    print("\nControls:")
    print("  r          return to object menu, kill VLA, then restart VLA")
    print("  p <prompt> switch prompt, kill VLA, then restart VLA")
    print("  s          soft reset menu: keep VLA model loaded, cannot interrupt CUDA already running")
    print("  sp <prompt> soft switch: keep VLA model loaded, cannot interrupt CUDA already running")
    print("  q          stop both processes and exit")


def command_loop(state):
    print_controls()
    while not state["stop"]:
        try:
            command = input().strip()
        except EOFError:
            state["stop"] = True
            return
        if command == "q":
            state["stop"] = True
        elif command == "r":
            state["kill_vla_now"] = True
            prompt = choose_prompt_interactive()
            if prompt:
                state["pending_prompt"] = ("hard", prompt)
            else:
                print("Prompt is empty.")
        elif command == "s":
            state["reset_vla_now"] = True
            prompt = choose_prompt_interactive()
            if prompt:
                state["pending_prompt"] = ("soft", prompt)
            else:
                print("Prompt is empty.")
        elif command.startswith("p "):
            prompt = command[2:].strip()
            if prompt:
                state["kill_vla_now"] = True
                state["pending_prompt"] = ("hard", prompt)
            else:
                print("Prompt is empty.")
        elif command.startswith("sp "):
            prompt = command[3:].strip()
            if prompt:
                state["reset_vla_now"] = True
                state["pending_prompt"] = ("soft", prompt)
            else:
                print("Prompt is empty.")
        elif command:
            print_controls()


def start_stack(args, prompt):
    if port_is_open(args.host, args.port):
        raise RuntimeError(
            f"{args.host}:{args.port} is already in use. Stop the old tracker first, "
            "otherwise inference may connect to the wrong prompt."
        )

    tracker = ManagedProcess("tracker", build_tracker_command(args, prompt))
    inference = ManagedProcess("inference", build_inference_command(args))

    tracker.start()
    if not wait_for_managed_port(tracker, args.host, args.port, args.tracker_timeout):
        tracker.stop()
        raise RuntimeError(f"tracker did not open {args.host}:{args.port} within {args.tracker_timeout:.0f}s")

    kill_vla_processes()
    inference.start()
    return tracker, inference


def start_inference(args):
    inference = ManagedProcess("inference", build_inference_command(args))
    inference.start()
    return inference


def find_vla_pids():
    pattern = str(INFERENCE_SCRIPT)
    try:
        result = subprocess.run(
            ["pgrep", "-f", pattern],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except FileNotFoundError:
        return []

    current_pid = os.getpid()
    parent_pid = os.getppid()
    pids = []
    for line in result.stdout.splitlines():
        try:
            pid = int(line.strip())
        except ValueError:
            continue
        if pid in (current_pid, parent_pid):
            continue
        pids.append(pid)
    return sorted(set(pids))


def kill_vla_processes(managed_process=None):
    if managed_process is not None:
        managed_process.kill()

    deadline = time.time() + VLA_KILL_TIMEOUT
    while True:
        pids = find_vla_pids()
        if not pids:
            return

        print(f"==> force killing VLA process ids: {', '.join(str(pid) for pid in pids)}", flush=True)
        for pid in pids:
            try:
                os.killpg(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            except PermissionError:
                raise RuntimeError(f"permission denied while killing VLA process group {pid}")
            except OSError:
                try:
                    os.kill(pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                except PermissionError:
                    raise RuntimeError(f"permission denied while killing VLA process {pid}")

        time.sleep(0.2)
        if time.time() >= deadline:
            remaining = find_vla_pids()
            if remaining:
                raise RuntimeError(
                    "VLA process did not exit after SIGKILL: "
                    + ", ".join(str(pid) for pid in remaining)
                )
            return


def set_tracker_prompt(args, prompt, timeout_ms=10000):
    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    socket.setsockopt(zmq.LINGER, 0)
    socket.setsockopt(zmq.RCVTIMEO, timeout_ms)
    socket.setsockopt(zmq.SNDTIMEO, timeout_ms)
    try:
        socket.connect(f"tcp://{args.host}:{args.port}")
        socket.send_multipart([b"set_prompt", prompt.encode("utf-8")])
        reply = socket.recv()
    finally:
        socket.close(0)
        context.term()

    if reply != b"OK":
        raise RuntimeError(f"tracker rejected prompt switch: {reply!r}")


def signal_inference_reset(args):
    deadline = time.time() + args.inference_reset_timeout
    last_error = None
    while time.time() < deadline:
        try:
            conn = Client(args.inference_control_socket, family="AF_UNIX", authkey=b"vla_control")
            conn.send("reset")
            reply = conn.recv()
            conn.close()
            if reply != "OK":
                raise RuntimeError(f"inference rejected reset: {reply!r}")
            return
        except (ConnectionRefusedError, FileNotFoundError, OSError, EOFError) as exc:
            last_error = exc
            time.sleep(0.1)

    reset_path = Path(args.inference_reset_file)
    reset_path.parent.mkdir(parents=True, exist_ok=True)
    reset_path.touch()
    print(f"Warning: used reset file fallback because control socket was unavailable: {last_error}")


def main():
    parser = argparse.ArgumentParser(description="Start tracker_server.py and VLA inference as one managed stack.")
    parser.add_argument("--prompt", help="Text prompt passed to tracker_server.py.")
    parser.add_argument("--object", help="Preset object name, or any text to use as the prompt.")
    parser.add_argument(
        "--tracker-runner",
        choices=("auto", "uv", "venv", "conda", "system"),
        default="auto",
        help="How to run tracker. auto prefers uv projects, then .venv, then conda.",
    )
    parser.add_argument(
        "--inference-runner",
        choices=("auto", "uv", "venv", "conda", "system"),
        default="auto",
        help="How to run inference. auto prefers uv projects, then .venv, then conda.",
    )
    parser.add_argument("--tracker-env", default="sam3", help="Conda env for tracker when --tracker-runner=conda.")
    parser.add_argument("--inference-env", default="dexgraspvla", help="Conda env for inference when --inference-runner=conda.")
    parser.add_argument("--tracker-command", help="Full shell command for tracker. Use {prompt} where the prompt should go.")
    parser.add_argument("--inference-command", help="Full shell command for inference.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5555)
    parser.add_argument("--tracker-no-debug-window", action="store_true")
    parser.add_argument("--tracker-debug-frame-path")
    parser.add_argument(
        "--inference-reset-file",
        default=DEFAULT_INFERENCE_RESET_FILE,
        help="File touched by the launcher to ask the running inference process to clear runtime state.",
    )
    parser.add_argument(
        "--inference-control-socket",
        default=DEFAULT_INFERENCE_CONTROL_SOCKET,
        help="Unix socket used for immediate VLA reset commands.",
    )
    parser.add_argument("--inference-reset-timeout", type=float, default=3.0)
    parser.add_argument("--tracker-timeout", type=float, default=90.0)
    parser.add_argument("--no-interactive", action="store_true", help="Disable stdin controls.")
    parser.add_argument("--list-objects", action="store_true", help="Show built-in object presets and exit.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without launching processes.")
    args = parser.parse_args()

    if args.list_objects:
        for _, (name, prompt) in DEFAULT_OBJECTS.items():
            if prompt:
                print(f"{name}: {prompt}")
        return 0

    prompt = choose_prompt(args)
    if not prompt:
        print("No prompt provided.", file=sys.stderr)
        return 2

    print(f"Tracking prompt: {prompt!r}")
    print("Tip: use --tracker-runner / --inference-runner, or full --tracker-command / --inference-command, if needed.")
    if args.dry_run:
        print("\nTracker command:")
        print(build_tracker_command(args, prompt))
        print("\nInference command:")
        print(build_inference_command(args))
        return 0

    state = {
        "stop": False,
        "reset_vla_now": False,
        "kill_vla_now": False,
        "pending_prompt": None,
        "prompt": prompt,
    }
    if not args.no_interactive:
        threading.Thread(target=command_loop, args=(state,), daemon=True).start()

    tracker = inference = None
    try:
        tracker, inference = start_stack(args, state["prompt"])
        while not state["stop"]:
            if state["reset_vla_now"]:
                state["reset_vla_now"] = False
                print("\nResetting VLA immediately; model stays loaded.", flush=True)
                if inference is not None:
                    signal_inference_reset(args)

            if state["kill_vla_now"]:
                state["kill_vla_now"] = False
                print("\nStopping VLA inference before object selection finishes.", flush=True)
                kill_vla_processes(inference)
                inference = None

            tracker_code = tracker.poll()
            if tracker_code is not None:
                print(f"\ntracker exited with code {tracker_code}; restarting the whole stack.")
                if inference:
                    inference.stop()
                tracker, inference = start_stack(args, state["prompt"])
                continue
            if inference is not None:
                inference_code = inference.poll()
                if inference_code is not None:
                    print(f"\ninference exited with code {inference_code}; stopping tracker and exiting.")
                    state["stop"] = True
                    break

            next_prompt = state["pending_prompt"]
            if next_prompt:
                state["pending_prompt"] = None
                reset_mode, prompt_text = next_prompt
                if reset_mode == "hard":
                    print(f"\nHard restarting VLA inference with prompt: {prompt_text!r}")
                    kill_vla_processes(inference)
                    set_tracker_prompt(args, prompt_text)
                    state["prompt"] = prompt_text
                    inference = start_inference(args)
                else:
                    print(f"\nSwitching prompt without reloading VLA: {prompt_text!r}")
                    set_tracker_prompt(args, prompt_text)
                    signal_inference_reset(args)
                    state["prompt"] = prompt_text

            time.sleep(0.05)

    except RuntimeError as exc:
        print(f"\nLauncher error: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        state["stop"] = True
    finally:
        if inference:
            inference.stop()
        if tracker:
            tracker.stop()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
