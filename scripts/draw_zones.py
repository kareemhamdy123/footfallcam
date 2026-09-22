"""
Small helper: click points on the first frame of your video to define a
line or polygon zone, then print YAML you can paste into your config.

Usage:
    python scripts/draw_zones.py --source data/sample.mp4 --kind line --name main_entrance
    python scripts/draw_zones.py --source data/sample.mp4 --kind zone --name checkout_queue --zone-kind queue

Controls:
    left click  - add a point
    'u'         - undo last point
    'q' / ENTER - finish and print YAML
"""
from __future__ import annotations

import argparse

import cv2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--kind", choices=["line", "zone"], required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--zone-kind", default="generic",
                         choices=["generic", "queue", "sales", "safe_occupancy", "outside"])
    parser.add_argument("--in-direction", default="top_to_bottom",
                         choices=["left_to_right", "right_to_left", "top_to_bottom", "bottom_to_top"])
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.source)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise SystemExit(f"Could not read a frame from {args.source}")

    points: list[tuple[int, int]] = []

    def on_click(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((x, y))

    window = f"Draw {args.kind}: {args.name} (click points, 'u'=undo, ENTER/q=done)"
    cv2.namedWindow(window)
    cv2.setMouseCallback(window, on_click)

    while True:
        display = frame.copy()
        for i, p in enumerate(points):
            cv2.circle(display, p, 4, (0, 255, 0), -1)
            if i > 0:
                cv2.line(display, points[i - 1], p, (0, 255, 0), 2)
        if args.kind == "zone" and len(points) > 2:
            cv2.line(display, points[-1], points[0], (0, 255, 0), 1)
        cv2.imshow(window, display)
        key = cv2.waitKey(20) & 0xFF
        if key in (ord("q"), 13):
            break
        if key == ord("u") and points:
            points.pop()

    cv2.destroyAllWindows()

    if args.kind == "line":
        if len(points) < 2:
            raise SystemExit("Need at least 2 points for a line.")
        p1, p2 = points[0], points[1]
        print(f"""
  - name: "{args.name}"
    p1: [{p1[0]}, {p1[1]}]
    p2: [{p2[0]}, {p2[1]}]
    in_direction: "{args.in_direction}"
""")
    else:
        if len(points) < 3:
            raise SystemExit("Need at least 3 points for a zone polygon.")
        poly = ", ".join(f"[{x}, {y}]" for x, y in points)
        print(f"""
  - name: "{args.name}"
    kind: "{args.zone_kind}"
    polygon: [{poly}]
""")


if __name__ == "__main__":
    main()
