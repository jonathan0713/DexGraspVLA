#!/usr/bin/env python3
import argparse
import os
import sys
import time

import cv2
import numpy as np
import zmq

sys.path.insert(0, os.getcwd())

from vision_tracker import VisionTracker


def main():
    parser = argparse.ArgumentParser(description="Vision Tracker Server with hot prompt switching.")
    parser.add_argument(
        "--prompt",
        type=str,
        required=True,
        help="Initial object prompt, for example: 'red bottle'.",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5555)
    parser.add_argument("--no-debug-window", action="store_true")
    parser.add_argument("--debug-frame-path", help="Write latest tracker overlay frame to this PNG path.")
    parser.add_argument("--debug-frame-interval", type=float, default=0.1)
    args = parser.parse_args()

    prompt_text = args.prompt
    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind(f"tcp://{args.host}:{args.port}")

    tracker = VisionTracker(device="cuda")
    is_init = False
    last_debug_frame_time = 0.0

    print(f"[AI Server] Tracker loaded once and listening on {args.host}:{args.port}.", flush=True)
    print(f"[AI Server] Current tracking prompt: '{prompt_text}'", flush=True)

    while True:
        msg_parts = socket.recv_multipart()
        command = msg_parts[0].decode("utf-8")

        if command == "set_prompt":
            if len(msg_parts) < 2:
                socket.send(b"ERROR: missing prompt")
                continue

            prompt_text = msg_parts[1].decode("utf-8")
            is_init = False
            socket.send(b"OK")
            print(f"[AI Server] Prompt changed to '{prompt_text}'. Tracker memory cleared.", flush=True)
            continue

        if command == "reset":
            if len(msg_parts) >= 2 and msg_parts[1]:
                prompt_text = msg_parts[1].decode("utf-8")
                print(f"[AI Server] Prompt changed to '{prompt_text}' by reset.", flush=True)
            is_init = False
            socket.send(b"OK")
            print("[AI Server] Tracker memory cleared.", flush=True)
            continue

        if command != "cam1":
            print(f"[AI Server] Unexpected request: {command}", flush=True)
            socket.send(b"ERROR")
            continue

        if len(msg_parts) < 2:
            socket.send(b"ERROR: missing image")
            continue

        img_bytes = msg_parts[1]
        nparr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is None:
            socket.send(b"ERROR: bad image")
            continue

        if not is_init:
            print(f"[Front Cam] Initializing tracker with prompt: '{prompt_text}'...", flush=True)
            mask = tracker.start_tracking(frame, prompt_text)
            is_init = True
        else:
            mask = tracker.track(frame)

        socket.send(mask.tobytes())

        if not args.no_debug_window or args.debug_frame_path:
            now = time.time()
            should_write_frame = (
                args.debug_frame_path
                and now - last_debug_frame_time >= args.debug_frame_interval
            )
            if not args.no_debug_window or should_write_frame:
                color_mask = np.zeros_like(frame)
                color_mask[mask == 1] = [0, 255, 0]
                display_frame = cv2.addWeighted(frame, 1.0, color_mask, 0.5, 0)

        if args.debug_frame_path and should_write_frame:
            debug_frame_dir = os.path.dirname(args.debug_frame_path)
            if debug_frame_dir:
                os.makedirs(debug_frame_dir, exist_ok=True)
            temp_path = args.debug_frame_path + ".tmp.png"
            cv2.imwrite(temp_path, display_frame)
            os.replace(temp_path, args.debug_frame_path)
            last_debug_frame_time = now

        if not args.no_debug_window:
            cv2.imshow("Tracker Debug Feed", display_frame)
            cv2.waitKey(1)


if __name__ == "__main__":
    main()
