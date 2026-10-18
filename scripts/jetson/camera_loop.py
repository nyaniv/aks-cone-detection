#!/usr/bin/env python3
"""Live cone detection from a USB webcam on the Jetson, with honest timing.

    # inside the Ultralytics JetPack 6 container (see run_container.sh)
    python3 /cone/scripts/jetson/camera_loop.py --show              # boxes on screen
    python3 /cone/scripts/jetson/camera_loop.py --seconds 30        # headless: time it, then summarise

Why not `yolo predict source=0`: that opens the camera at OpenCV's default,
often 640x480 YUYV, which shrinks distant cones below the size the model can
find. This asks for MJPG at the camera's full mode (Brio 105: 1920x1080 @ 30)
and prints what the camera actually delivered.

--imgsz must match the engine export_engine.sh built (default 736 1280 for a
16:9 camera). 1920x1080 then scales to 1280x720 - the same scale as the
square 1280 eval, so the bucketed recall in results/ still applies.
"""
import argparse
import statistics
import sys
import time

import cv2


def open_camera(dev, w, h, fps):
    src = int(dev) if str(dev).isdigit() else dev
    cap = cv2.VideoCapture(src, cv2.CAP_V4L2)
    if not cap.isOpened():
        sys.exit(f"could not open camera {dev} - check `ls /dev/video*`")
    # FOURCC must be set before the size: at 1080p, YUYV over USB usually
    # drops to ~5 fps, MJPG holds 30.
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
    cap.set(cv2.CAP_PROP_FPS, fps)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # newest frame, not a queued stale one
    gw, gh = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    gfps = cap.get(cv2.CAP_PROP_FPS)
    four = int(cap.get(cv2.CAP_PROP_FOURCC)).to_bytes(4, "little").decode(errors="replace")
    print(f"camera: {gw}x{gh} @ {gfps:.0f} fps, {four}")
    if (gw, gh) != (w, h):
        print(f"[warn] asked for {w}x{h}, got {gw}x{gh} - small cones will suffer. "
              f"Check the modes with: v4l2-ctl -d {dev} --list-formats-ext")
    return cap


def pct(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(p / 100 * (len(xs) - 1))))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/cone/models/base_p2_1280.engine")
    ap.add_argument("--cam", default="/dev/video0")
    ap.add_argument("--width", type=int, default=1920)
    ap.add_argument("--height", type=int, default=1080)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--imgsz", type=int, nargs=2, default=[736, 1280],
                    metavar=("H", "W"), help="must match the engine")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--show", action="store_true", help="draw boxes in a window")
    ap.add_argument("--seconds", type=float, default=0, help="stop after N s (0 = until q / Ctrl+C)")
    args = ap.parse_args()

    from ultralytics import YOLO
    model = YOLO(args.model, task="detect")
    cap = open_camera(args.cam, args.width, args.height, args.fps)

    # warm-up: the first TensorRT calls allocate and are not representative
    ok, frame = cap.read()
    if not ok:
        sys.exit("camera returned no frame")
    for _ in range(5):
        model.predict(frame, imgsz=args.imgsz, conf=args.conf, verbose=False)

    model_ms, loop_ms, wait_ms, counts = [], [], [], []
    t_start = t_prev = time.perf_counter()
    t_print = t_start
    try:
        while True:
            t0 = time.perf_counter()
            ok, frame = cap.read()          # blocks until the camera has a frame
            t1 = time.perf_counter()
            if not ok:
                print("camera stopped delivering frames")
                break
            r = model.predict(frame, imgsz=args.imgsz, conf=args.conf, verbose=False)[0]
            t2 = time.perf_counter()

            wait_ms.append((t1 - t0) * 1e3)
            model_ms.append((t2 - t1) * 1e3)    # pre + inference + post, as bench_latency.py
            loop_ms.append((t2 - t_prev) * 1e3)
            counts.append(len(r.boxes))
            t_prev = t2

            if args.show:
                img = r.plot(line_width=2)
                hz = 1e3 / statistics.mean(loop_ms[-30:])
                cv2.putText(img, f"{hz:4.1f} Hz  model {model_ms[-1]:4.1f} ms  cones {counts[-1]}",
                            (16, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
                cv2.imshow("cones", img)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            if t2 - t_print >= 1.0:
                n = min(len(loop_ms), 30)
                print(f"{1e3 / statistics.mean(loop_ms[-n:]):5.1f} Hz | model "
                      f"{statistics.median(model_ms[-n:]):5.1f} ms | waiting for camera "
                      f"{statistics.median(wait_ms[-n:]):5.1f} ms | cones {counts[-1]}")
                t_print = t2
            if args.seconds and t2 - t_start >= args.seconds:
                break
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        if args.show:
            cv2.destroyAllWindows()

    if len(model_ms) > 1:
        print("\n--- summary ---")
        print(f"frames        : {len(model_ms)}")
        print(f"loop rate     : {1e3 / statistics.mean(loop_ms[1:]):.1f} Hz (camera-capped at {args.fps})")
        print(f"model / frame : median {statistics.median(model_ms):.1f} ms, "
              f"p95 {pct(model_ms, 95):.1f} ms  ->  model alone could do "
              f"{1e3 / statistics.median(model_ms):.0f} Hz")
        print(f"cones / frame : median {statistics.median(counts):.0f}")


if __name__ == "__main__":
    main()
