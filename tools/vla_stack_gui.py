#!/usr/bin/env python3
import argparse
import os
import queue
import subprocess
import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import ttk
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "tools"))

from vla_stack_launcher import (  # noqa: E402
    DEFAULT_INFERENCE_CONTROL_SOCKET,
    DEFAULT_INFERENCE_RESET_FILE,
    DEFAULT_OBJECTS,
    build_inference_command,
    build_tracker_command,
    kill_vla_processes,
    port_is_open,
    set_tracker_prompt,
    signal_inference_reset,
    wait_for_managed_port,
)


DEBUG_FRAME_PATH = "/tmp/vla_tracker_debug.png"
LOG_KEYWORDS = (
    "==>",
    "ERROR",
    "Warning",
    "started",
    "stopped",
    "killed",
    "switch",
    "reset",
    "ready",
    "loaded",
    "Prompt changed",
    "Initializing tracker",
    "exited",
)


class GuiProcess:
    def __init__(self, name, command, log_queue):
        self.name = name
        self.command = command
        self.log_queue = log_queue
        self.process = None

    def start(self):
        self.log(f"==> starting {self.name}")
        self.log(f"    {self.command}")
        self.process = subprocess.Popen(
            ["bash", "-lc", self.command],
            cwd=str(REPO_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            start_new_session=True,
        )
        threading.Thread(target=self._stream_output, daemon=True).start()

    def _stream_output(self):
        if not self.process or not self.process.stdout:
            return
        for line in iter(self.process.stdout.readline, ""):
            text = line.rstrip()
            if should_show_process_log(text):
                self.log(f"[{self.name}] {text}")

    def stop(self):
        from vla_stack_launcher import ManagedProcess

        proxy = ManagedProcess(self.name, self.command)
        proxy.process = self.process
        proxy.stop()

    def kill(self):
        from vla_stack_launcher import ManagedProcess

        proxy = ManagedProcess(self.name, self.command)
        proxy.process = self.process
        proxy.kill()

    def poll(self):
        if not self.process:
            return None
        return self.process.poll()

    def log(self, message):
        self.log_queue.put(message)


def should_show_process_log(message):
    return any(keyword in message for keyword in LOG_KEYWORDS)


class VLAStackGui:
    def __init__(self, root):
        self.root = root
        self.root.title("DexGraspVLA Stack")
        self.root.geometry("920x980")

        self.log_queue = queue.Queue()
        self.tracker = None
        self.inference = None
        self.worker_running = False
        self.current_prompt = tk.StringVar(value="blue bottle")
        self.custom_prompt = tk.StringVar()
        self.status = tk.StringVar(value="Stopped")
        self.feed_image = None
        self.feed_mtime = 0.0

        self.args = self.make_args()
        self.build_ui()
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.root.after(100, self.drain_logs)
        self.root.after(200, self.refresh_status)
        self.root.after(150, self.refresh_tracker_feed)

    def make_args(self):
        return SimpleNamespace(
            tracker_command=None,
            tracker_runner="auto",
            tracker_env="sam3",
            tracker_no_debug_window=True,
            tracker_debug_frame_path=DEBUG_FRAME_PATH,
            inference_command=None,
            inference_runner="auto",
            inference_env="dexgraspvla",
            inference_reset_file=DEFAULT_INFERENCE_RESET_FILE,
            inference_control_socket=DEFAULT_INFERENCE_CONTROL_SOCKET,
            inference_reset_timeout=3.0,
            host="127.0.0.1",
            port=5555,
            tracker_timeout=90.0,
        )

    def build_ui(self):
        root_frame = ttk.Frame(self.root, padding=10)
        root_frame.pack(fill=tk.BOTH, expand=True)

        controls = ttk.LabelFrame(root_frame, text="Control", padding=10)
        controls.pack(fill=tk.X)

        ttk.Label(controls, text="Target").grid(row=0, column=0, sticky="w")
        choices = [name for name, prompt in DEFAULT_OBJECTS.values() if prompt is not None]
        self.target_combo = ttk.Combobox(
            controls,
            textvariable=self.current_prompt,
            values=choices,
            state="readonly",
            width=20,
        )
        self.target_combo.grid(row=0, column=1, sticky="ew", padx=6)
        controls.columnconfigure(1, weight=1)

        ttk.Label(controls, text="Custom").grid(row=0, column=2, sticky="w", padx=(12, 0))
        ttk.Entry(controls, textvariable=self.custom_prompt).grid(
            row=0,
            column=3,
            columnspan=3,
            sticky="ew",
            padx=6,
        )
        controls.columnconfigure(3, weight=2)

        ttk.Button(controls, text="Start", command=self.start_stack).grid(row=1, column=0, sticky="ew", pady=(10, 0))
        ttk.Button(controls, text="Stop", command=self.stop_stack).grid(row=1, column=1, sticky="ew", padx=6, pady=(10, 0))
        ttk.Button(controls, text="Kill VLA", command=self.kill_vla).grid(row=1, column=2, sticky="ew", pady=(10, 0))

        ttk.Button(controls, text="Hard Switch", command=self.hard_switch).grid(row=1, column=3, sticky="ew", padx=6, pady=(10, 0))
        ttk.Button(controls, text="Soft Switch", command=self.soft_switch).grid(row=1, column=4, sticky="ew", padx=6, pady=(10, 0))
        ttk.Button(controls, text="Soft Reset", command=self.soft_reset).grid(row=1, column=5, sticky="ew", pady=(10, 0))

        quick = ttk.LabelFrame(root_frame, text="Quick Hard Switch", padding=10)
        quick.pack(fill=tk.X, pady=(10, 0))
        for index, (_, prompt) in enumerate(DEFAULT_OBJECTS.values()):
            if prompt is None:
                continue
            ttk.Button(
                quick,
                text=prompt,
                command=lambda value=prompt: self.hard_switch(value),
            ).grid(row=0, column=index, sticky="ew", padx=4, pady=4)
            quick.columnconfigure(index, weight=1)

        status_box = ttk.LabelFrame(root_frame, text="Status", padding=8)
        status_box.pack(fill=tk.X, pady=(10, 0))
        ttk.Label(status_box, textvariable=self.status).pack(anchor="w")

        feed_box = ttk.LabelFrame(root_frame, text="SAM3 Tracker Feed", padding=8)
        feed_box.pack(fill=tk.BOTH, expand=True)
        self.feed_label = ttk.Label(feed_box, text="No SAM3 frame yet", anchor=tk.CENTER)
        self.feed_label.pack(fill=tk.BOTH, expand=True)

        log_box = ttk.LabelFrame(root_frame, text="Events", padding=8)
        log_box.pack(fill=tk.X, pady=(10, 0))
        self.log_text = tk.Text(log_box, height=6, wrap=tk.WORD, state=tk.DISABLED)
        log_scroll = ttk.Scrollbar(log_box, orient=tk.VERTICAL, command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=log_scroll.set)
        self.log_text.pack(side=tk.LEFT, fill=tk.X, expand=True)
        log_scroll.pack(side=tk.RIGHT, fill=tk.Y)

    def selected_prompt(self):
        custom = self.custom_prompt.get().strip()
        if custom:
            return custom
        return self.current_prompt.get().strip()

    def run_worker(self, target):
        if self.worker_running:
            self.log("Busy: another operation is still running.")
            return

        def wrapped():
            self.worker_running = True
            try:
                target()
            except Exception as exc:
                self.log(f"ERROR: {exc}")
            finally:
                self.worker_running = False

        threading.Thread(target=wrapped, daemon=True).start()

    def start_stack(self):
        self.run_worker(self._start_stack)

    def _start_stack(self):
        prompt = self.selected_prompt()
        if not prompt:
            self.log("ERROR: prompt is empty")
            return
        if self.tracker and self.tracker.poll() is None:
            self.log("Stack is already running.")
            return
        if port_is_open(self.args.host, self.args.port):
            raise RuntimeError(f"{self.args.host}:{self.args.port} is already in use")

        self.clear_debug_frame()
        self.tracker = GuiProcess("tracker", build_tracker_command(self.args, prompt), self.log_queue)
        self.tracker.start()
        if not wait_for_managed_port(self.tracker, self.args.host, self.args.port, self.args.tracker_timeout):
            self.tracker.stop()
            raise RuntimeError("tracker did not become ready")

        kill_vla_processes()
        self.inference = GuiProcess("inference", build_inference_command(self.args), self.log_queue)
        self.inference.start()
        self.current_prompt.set(prompt)
        self.log(f"Stack started with prompt: {prompt!r}")

    def stop_stack(self):
        self.run_worker(self._stop_stack)

    def _stop_stack(self):
        if self.inference:
            self.inference.stop()
            self.inference = None
        if self.tracker:
            self.tracker.stop()
            self.tracker = None
        self.log("Stack stopped.")

    def kill_vla(self):
        self.run_worker(lambda: self._kill_vla(set_inference_none=True))

    def _kill_vla(self, set_inference_none=False):
        kill_vla_processes(self.inference)
        if set_inference_none:
            self.inference = None
        self.log("VLA process killed.")

    def hard_switch(self, prompt=None):
        if prompt is not None:
            self.current_prompt.set(prompt)
            self.custom_prompt.set("")
        self.run_worker(self._hard_switch)

    def _hard_switch(self):
        prompt = self.selected_prompt()
        if not prompt:
            self.log("ERROR: prompt is empty")
            return
        if not self.tracker or self.tracker.poll() is not None:
            self.log("Tracker is not running; starting full stack.")
            self._start_stack()
            return

        self.log(f"Hard switch to {prompt!r}: killing VLA, keeping tracker loaded.")
        kill_vla_processes(self.inference)
        set_tracker_prompt(self.args, prompt)
        self.inference = GuiProcess("inference", build_inference_command(self.args), self.log_queue)
        self.inference.start()
        self.current_prompt.set(prompt)

    def soft_switch(self):
        self.run_worker(self._soft_switch)

    def _soft_switch(self):
        prompt = self.selected_prompt()
        if not prompt:
            self.log("ERROR: prompt is empty")
            return
        if not self.tracker or self.tracker.poll() is not None:
            self.log("Tracker is not running; starting full stack.")
            self._start_stack()
            return

        self.log(f"Soft switch to {prompt!r}; VLA model stays loaded.")
        signal_inference_reset(self.args)
        set_tracker_prompt(self.args, prompt)
        signal_inference_reset(self.args)
        self.current_prompt.set(prompt)

    def soft_reset(self):
        self.run_worker(self._soft_reset)

    def _soft_reset(self):
        self.log("Soft resetting VLA runtime; VLA model stays loaded.")
        signal_inference_reset(self.args)

    def refresh_status(self):
        tracker_state = self.process_state(self.tracker)
        inference_state = self.process_state(self.inference)
        mode = "busy" if self.worker_running else "ready"
        self.status.set(
            f"prompt={self.current_prompt.get()!r} | tracker={tracker_state} | inference={inference_state} | {mode}"
        )
        self.root.after(300, self.refresh_status)

    def process_state(self, process):
        if process is None:
            return "off"
        code = process.poll()
        return "running" if code is None else f"exited({code})"

    def refresh_tracker_feed(self):
        path = Path(DEBUG_FRAME_PATH)
        try:
            mtime = path.stat().st_mtime
            if mtime != self.feed_mtime:
                self.feed_mtime = mtime
                self.feed_image = tk.PhotoImage(file=str(path))
                self.feed_label.configure(image=self.feed_image, text="")
        except tk.TclError as exc:
            self.feed_label.configure(text=f"Could not load tracker frame: {exc}", image="")
        except OSError:
            if self.feed_image is None:
                self.feed_label.configure(text="No SAM3 frame yet", image="")
        self.root.after(150, self.refresh_tracker_feed)

    def clear_debug_frame(self):
        try:
            os.remove(DEBUG_FRAME_PATH)
        except OSError:
            pass
        self.feed_image = None
        self.feed_mtime = 0.0
        self.feed_label.configure(text="No SAM3 frame yet", image="")

    def drain_logs(self):
        while True:
            try:
                message = self.log_queue.get_nowait()
            except queue.Empty:
                break
            self.append_log(message)
        self.root.after(100, self.drain_logs)

    def append_log(self, message):
        self.log_text.configure(state=tk.NORMAL)
        self.log_text.insert(tk.END, message + "\n")
        self.log_text.see(tk.END)
        self.log_text.configure(state=tk.DISABLED)

    def log(self, message):
        self.log_queue.put(message)

    def on_close(self):
        def close_worker():
            try:
                self._stop_stack()
            finally:
                self.root.after(0, self.root.destroy)

        threading.Thread(target=close_worker, daemon=True).start()


def main():
    parser = argparse.ArgumentParser(description="GUI launcher for DexGraspVLA tracker + inference.")
    parser.parse_args()
    root = tk.Tk()
    VLAStackGui(root)
    root.mainloop()


if __name__ == "__main__":
    main()
