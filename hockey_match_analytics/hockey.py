import subprocess, sys, os

PKG_TO_MODULE = {
    "opencv-python": "cv2",
    "numpy":         "numpy",
    "matplotlib":    "matplotlib",
    "scipy":         "scipy",
    "ultralytics":   "ultralytics",
    "Pillow":        "PIL",
    "tqdm":          "tqdm",
}

def install_deps():
    print("[SETUP] Checking dependencies...", flush=True)
    for pkg, mod in PKG_TO_MODULE.items():
        try:
            __import__(mod)
        except ImportError:
            print(f"[INSTALL] {pkg} (missing) ...", flush=True)
            try:
                subprocess.check_call(
                    [sys.executable, "-m", "pip", "install", pkg, "-q", "--disable-pip-version-check"]
                )
            except subprocess.CalledProcessError as e:
                print(f"[ERROR] Failed to install {pkg}: {e}", flush=True)
                sys.exit(1)
    print("[SETUP] All dependencies ready ✓", flush=True)

install_deps()

import cv2, numpy as np, warnings
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import defaultdict, deque
from scipy.ndimage import gaussian_filter
from ultralytics import YOLO
from tqdm import tqdm
warnings.filterwarnings("ignore")

C_PUCK   = (0, 220, 255)
C_WHITE  = (255, 255, 255)
C_BLACK  = (0, 0, 0)
C_REF    = (140, 140, 140)

# ---- TEAM COLORS (only these two are ever used for players) ----
TEAM_A_COLOR = (255, 120, 0)     # blue-ish (BGR)
TEAM_B_COLOR = (0, 60, 220)      # red-ish (BGR)

PIXEL_TO_METER   = 0.045
LABEL_FONT       = cv2.FONT_HERSHEY_DUPLEX
LABEL_SCALE      = 0.68
LABEL_THICK      = 2

DET_CONF         = 0.45           # higher = fewer false positives (crowd/clutter)
MATCH_MAX_DIST   = 65             # px, tighter association = less ID swapping
CALIBRATION_FRAMES = 40           # frames used to learn the 2 team jersey colors

# Referee jerseys are typically a bright, distinct hue (e.g. neon yellow/green stripes).
# Tune this HSV hue range (0-179 in OpenCV) to match your footage's ref uniform.
REFEREE_HUE_RANGE = (25, 45)      # yellow-green band; widen/narrow as needed
REFEREE_MIN_SAT   = 90            # must also be fairly saturated (not white/black stripes)

# ---- ROI: polygon marking the actual playing surface (excludes crowd/stands) ----
# Points are FRACTIONS of (width, height), so they scale to any resolution.
# Default is a rough trapezoid for a typical elevated broadcast angle.
# TO CALIBRATE: run once, look at hockey_roi_preview.png it saves, then adjust these points
# so the polygon hugs the boards/ice edge in your footage.
ROI_POINTS = [
    (0.00, 0.20),  # top-left   (just below the boards/crowd line)
    (1.00, 0.20),  # top-right
    (1.00, 1.00),  # bottom-right
    (0.00, 1.00),  # bottom-left
]


def build_roi_mask(W, H):
    pts = np.array([[int(x * W), int(y * H)] for x, y in ROI_POINTS], dtype=np.int32)
    mask = np.zeros((H, W), dtype=np.uint8)
    cv2.fillPoly(mask, [pts], 255)
    return mask, pts


def in_roi(cx, cy, roi_mask):
    h, w = roi_mask.shape
    xi, yi = int(np.clip(cx, 0, w - 1)), int(np.clip(cy, 0, h - 1))
    return roi_mask[yi, xi] > 0


def jersey_hsv_sample(frame, box):
    """Sample the torso region (upper-middle of bbox) and return mean HSV."""
    x1, y1, x2, y2 = box
    h = y2 - y1
    ty1 = int(y1 + h * 0.15)
    ty2 = int(y1 + h * 0.55)
    tx1 = int(x1 + (x2 - x1) * 0.20)
    tx2 = int(x2 - (x2 - x1) * 0.20)
    ty1, ty2 = max(0, ty1), max(ty1 + 1, ty2)
    tx1, tx2 = max(0, tx1), max(tx1 + 1, tx2)
    patch = frame[ty1:ty2, tx1:tx2]
    if patch.size == 0:
        return None
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    return hsv.reshape(-1, 3).mean(axis=0)  # (H, S, V)


def is_referee(hsv_sample):
    if hsv_sample is None:
        return False
    h, s, v = hsv_sample
    return (REFEREE_HUE_RANGE[0] <= h <= REFEREE_HUE_RANGE[1]) and s >= REFEREE_MIN_SAT


class TeamClassifier:
    """Learns 2 jersey-color clusters from the first N frames, then classifies persistently."""
    def __init__(self, calib_frames=CALIBRATION_FRAMES):
        self.calib_frames = calib_frames
        self.samples = []
        self.centers = None      # (2,3) HSV cluster centers
        self.locked_team = {}    # tid -> 'A' or 'B', sticky once decided

    def add_sample(self, hsv):
        if self.centers is None and hsv is not None:
            self.samples.append(hsv)

    def maybe_fit(self, frame_idx):
        if self.centers is not None:
            return
        if frame_idx >= self.calib_frames and len(self.samples) >= 10:
            data = np.array(self.samples, dtype=np.float32)
            crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.5)
            _, labels, centers = cv2.kmeans(data, 2, None, crit, 8, cv2.KMEANS_PP_CENTERS)
            self.centers = centers
            print(f"[TEAMS] Learned 2 jersey color clusters from {len(data)} samples ✓", flush=True)

    def classify(self, tid, hsv):
        if tid in self.locked_team:
            return self.locked_team[tid]
        if self.centers is None or hsv is None:
            return None
        d0 = np.linalg.norm(hsv - self.centers[0])
        d1 = np.linalg.norm(hsv - self.centers[1])
        team = 'A' if d0 <= d1 else 'B'
        self.locked_team[tid] = team   # lock so a player doesn't flicker team each frame
        return team

    def color_for(self, team):
        return TEAM_A_COLOR if team == 'A' else TEAM_B_COLOR


class PlayerTracker:
    def __init__(self, fps=25):
        self.tracks   = {}
        self.next_id  = 0
        self.lost     = defaultdict(int)
        self.max_lost = 15
        self.fps      = fps
        self.dists    = defaultdict(float)
        self.speeds   = defaultdict(float)

    @staticmethod
    def _iou(a, b):
        ax1, ay1, ax2, ay2 = a; bx1, by1, bx2, by2 = b
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
        inter = iw * ih
        if inter == 0:
            return 0.0
        area_a = (ax2 - ax1) * (ay2 - ay1)
        area_b = (bx2 - bx1) * (by2 - by1)
        return inter / float(area_a + area_b - inter)

    def update(self, dets):
        """dets: list of (x1,y1,x2,y2,conf,poly,hsv)"""
        new = {}; used = set()

        for x1, y1, x2, y2, conf, poly, hsv in dets:
            cx = (x1 + x2) / 2; cy = (y1 + y2) / 2
            box = (x1, y1, x2, y2)

            best_id, best_score = None, -1
            for tid, info in self.tracks.items():
                if tid in used: continue
                d = np.hypot(cx - info["cx"], cy - info["cy"])
                if d > MATCH_MAX_DIST:
                    continue
                iou = self._iou(box, info["box"])
                score = iou - (d / MATCH_MAX_DIST) * 0.3   # favor high IoU, penalize distance
                if score > best_score:
                    best_score = score; best_id = tid

            if best_id is None:
                best_id = self.next_id; self.next_id += 1

            if best_id in self.tracks:
                dpx = np.hypot(cx - self.tracks[best_id]["cx"], cy - self.tracks[best_id]["cy"])
                self.dists[best_id] += dpx * PIXEL_TO_METER
                self.speeds[best_id] = dpx * PIXEL_TO_METER * self.fps * 3.6

            new[best_id] = {"box": box, "cx": cx, "cy": cy, "conf": conf, "poly": poly, "hsv": hsv}
            used.add(best_id)

        for tid in list(self.tracks):
            if tid not in used:
                self.lost[tid] += 1
                if self.lost[tid] <= self.max_lost:
                    new[tid] = self.tracks[tid]
            else:
                self.lost[tid] = 0

        self.tracks = new
        return new


class PuckTracker:
    def __init__(self, fps):
        self.fps = fps
        self.trail = deque(maxlen=40)

    def update(self, pos):
        if pos is None:
            return
        self.trail.append(pos)


def txt(img, text, x, y, scale=0.7, color=C_WHITE, thick=2, font=LABEL_FONT):
    cv2.putText(img, text, (x, y), font, scale, (0, 0, 0), thick + 3, cv2.LINE_AA)
    cv2.putText(img, text, (x, y), font, scale, color, thick, cv2.LINE_AA)


def label_block(img, lines, colors, cx, y_start, scale=LABEL_SCALE, thick=LABEL_THICK,
                 bg_color=(20, 20, 20), alpha=0.72):
    font = LABEL_FONT
    pad_x, pad_y, gap = 12, 6, 4
    sizes = [cv2.getTextSize(l, font, scale, thick)[0] for l in lines]
    max_w = max(s[0] for s in sizes)
    total_h = sum(s[1] for s in sizes) + gap * (len(lines) - 1)
    h_img, w_img = img.shape[:2]

    bx1 = max(0, cx - max_w // 2 - pad_x)
    by1 = max(0, y_start - pad_y)
    bx2 = min(w_img - 1, cx + max_w // 2 + pad_x)
    by2 = min(h_img - 1, y_start + total_h + pad_y)

    overlay = img.copy()
    cv2.rectangle(overlay, (bx1, by1), (bx2, by2), bg_color, -1)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

    y_cur = y_start
    for line, color, (tw, th) in zip(lines, colors, sizes):
        lx = cx - tw // 2
        txt(img, line, lx, y_cur + th, scale, color, thick, font)
        y_cur += th + gap


def draw_players(canvas, tracks, tc, speed_smooth):
    overlay = canvas.copy()
    poly_items = []
    for tid, info in tracks.items():
        team = tc.classify(tid, info.get("hsv"))
        color = tc.color_for(team) if team else (90, 90, 90)
        poly = info.get("poly")
        if poly is not None and len(poly) >= 3:
            pts = np.array(poly, dtype=np.int32).reshape(-1, 1, 2)
            cv2.fillPoly(overlay, [pts], color)
            poly_items.append((pts, color))

    if poly_items:
        cv2.addWeighted(overlay, 0.35, canvas, 0.65, 0, canvas)
    for pts, color in poly_items:
        cv2.polylines(canvas, [pts], True, color, 2, cv2.LINE_AA)

    for tid, info in tracks.items():
        x1, y1, x2, y2 = [int(v) for v in info["box"]]
        team = tc.classify(tid, info.get("hsv"))
        color = tc.color_for(team) if team else (90, 90, 90)
        spd_raw = tc  # placeholder unused
        cx_ = (x1 + x2) // 2
        label_block(
            canvas,
            lines=[f"#{tid}  {'TEAM ' + team if team else '...'}"],
            colors=[color],
            cx=cx_,
            y_start=max(0, y1 - 30),
            bg_color=(18, 18, 18),
        )


def draw_puck(canvas, puck):
    if not puck.trail:
        return
    pts = list(puck.trail)
    n = len(pts)
    for i in range(1, n):
        a = i / n
        col = tuple(int(c * a) for c in C_PUCK)
        cv2.line(canvas, tuple(map(int, pts[i - 1])), tuple(map(int, pts[i])), col, 3, cv2.LINE_AA)
    bx, by = map(int, pts[-1])
    for r, a in [(14, 40), (9, 90), (5, 200)]:
        ov = canvas.copy()
        cv2.circle(ov, (bx, by), r, C_PUCK, -1, cv2.LINE_AA)
        cv2.addWeighted(ov, a / 255.0, canvas, 1 - a / 255.0, 0, canvas)
    cv2.circle(canvas, (bx, by), 5, C_WHITE, 2, cv2.LINE_AA)


def draw_roi_debug(canvas, roi_pts):
    cv2.polylines(canvas, [roi_pts], True, (0, 255, 255), 2, cv2.LINE_AA)


def draw_top_banner(canvas, W):
    title = "HOCKEY MATCH ANALYTICS  v2.0  |  2-team mode"
    (tw, th), _ = cv2.getTextSize(title, LABEL_FONT, 0.78, 2)
    p = 12
    cv2.rectangle(canvas, (0, 0), (tw + p * 2, th + p * 2), (110, 0, 90), -1)
    cv2.putText(canvas, title, (p, th + p), LABEL_FONT, 0.78, C_WHITE, 2, cv2.LINE_AA)


def make_writer(path, fps, W, H):
    for fc in ["mp4v", "avc1", "H264", "h264"]:
        w = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*fc), fps, (W, H))
        if w.isOpened():
            print(f"[CODEC] using {fc} -> {path}", flush=True)
            return w
        w.release()
    raise RuntimeError("No working video codec found on this system")


def process(video_path):
    print("╔══════════════════════════════════════════════╗", flush=True)
    print("║   HOCKEY MATCH ANALYTICS  v2.0 (2-team ROI) ║", flush=True)
    print("║   dev: tubakhxn                             ║", flush=True)
    print("╚══════════════════════════════════════════════╝", flush=True)

    if not os.path.isfile(video_path):
        print(f"[ERROR] Video file not found: {video_path}", flush=True)
        return

    print("[YOLO] Loading segmentation model...", flush=True)
    model = YOLO("yolov8n-seg.pt")
    print("[YOLO] loaded ✓", flush=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open {video_path}", flush=True)
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[INFO] {W}x{H} @ {fps:.1f}fps | {total} frames", flush=True)

    roi_mask, roi_pts = build_roi_mask(W, H)

    out_dir = os.path.dirname(os.path.abspath(video_path))
    # save a one-time ROI preview so you can check/tune ROI_POINTS
    ret0, frame0 = cap.read()
    if ret0:
        preview = frame0.copy()
        draw_roi_debug(preview, roi_pts)
        cv2.imwrite(os.path.join(out_dir, "hockey_roi_preview.png"), preview)
        print("[ROI] Saved hockey_roi_preview.png — check the yellow outline hugs the ice, adjust ROI_POINTS if not.", flush=True)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    out_path = os.path.join(out_dir, "hockey_output.mp4")
    writer = make_writer(out_path, fps, W, H)

    pt = PlayerTracker(fps=fps)
    puck = PuckTracker(fps=fps)
    tc = TeamClassifier()
    speed_smooth = defaultdict(lambda: deque(maxlen=6))
    frame_idx = 0

    print("[PROC] Processing...", flush=True)
    with tqdm(total=total, unit="fr", ncols=80, colour="cyan") as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            canvas = frame.copy()

            results = model(frame, verbose=False, conf=DET_CONF, classes=[0, 32])[0]
            dets = []; puck_pos = None

            if results.boxes is not None:
                has_masks = results.masks is not None
                mask_polys = results.masks.xy if has_masks else None
                for i, box in enumerate(results.boxes):
                    cls = int(box.cls[0]); name = model.names[cls]
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    conf = float(box.conf[0])
                    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2

                    if name == "person":
                        if not in_roi(cx, cy, roi_mask):
                            continue  # crowd/stands -> skip
                        hsv = jersey_hsv_sample(frame, (x1, y1, x2, y2))
                        if is_referee(hsv):
                            continue  # referee -> skip
                        poly = mask_polys[i] if (mask_polys is not None and i < len(mask_polys)) else None
                        dets.append((x1, y1, x2, y2, conf, poly, hsv))
                        tc.add_sample(hsv)
                    elif name == "sports ball":
                        if in_roi(cx, cy, roi_mask):
                            puck_pos = (int(cx), int(cy))

            tc.maybe_fit(frame_idx)
            tracks = pt.update(dets)
            puck.update(puck_pos)

            draw_top_banner(canvas, W)
            draw_players(canvas, tracks, tc, speed_smooth)
            draw_puck(canvas, puck)

            writer.write(canvas)
            frame_idx += 1
            pbar.update(1)

    cap.release()
    writer.release()
    print(f"[DONE] {out_path} ✓", flush=True)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python hockey_match_analytics.py video.mp4")
        sys.exit(1)
    process(sys.argv[1])
