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

C_BALL   = (0, 210, 255)
C_WHITE  = (255, 255, 255)
C_SPIKE  = (0, 60, 255)

# darker, richer 2-team colors (BGR) — same fix as the hockey script
TEAM_A_COLOR     = (140, 40, 10)     # deep navy
TEAM_B_COLOR     = (20, 20, 150)     # deep crimson
UNASSIGNED_COLOR = (80, 80, 80)

PIXEL_TO_METER = 0.045
LABEL_FONT     = cv2.FONT_HERSHEY_DUPLEX
LABEL_SCALE    = 0.62
LABEL_THICK    = 2

DET_CONF            = 0.40
MATCH_MAX_DIST      = 65
CALIBRATION_FRAMES  = 60
MIN_CALIB_SAMPLES   = 25

# referee/staff rejection: low saturation (black/white stripes or all-black polos) + high contrast
REF_MAX_MEAN_SAT = 55
REF_MIN_CONTRAST = 42

MIN_SAT_FOR_CLUSTER = 60
VOTE_BUFFER_LEN  = 8
VOTE_LOCK_THRESH = 5

# net line as a fraction of frame height; ball crossing near this y = "near net"
NET_Y_FRAC        = 0.20
NET_BAND_PX       = 90
SPIKE_VEL_THRESH  = 26.0
SPIKE_COOLDOWN    = 15
RALLY_GAP_FRAMES  = 45

# ROI: playing court only, excludes crowd/stands/officials table
ROI_POINTS = [
    (0.00, 0.22),
    (1.00, 0.22),
    (1.00, 1.00),
    (0.00, 1.00),
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


def torso_patch(frame, box):
    x1, y1, x2, y2 = box
    h = y2 - y1
    ty1 = int(y1 + h * 0.12); ty2 = int(y1 + h * 0.50)
    tx1 = int(x1 + (x2 - x1) * 0.22); tx2 = int(x2 - (x2 - x1) * 0.22)
    ty1, ty2 = max(0, ty1), max(ty1 + 1, ty2)
    tx1, tx2 = max(0, tx1), max(tx1 + 1, tx2)
    patch = frame[ty1:ty2, tx1:tx2]
    return patch if patch.size else None


def analyze_patch(patch):
    if patch is None:
        return None
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV).reshape(-1, 3).astype(np.float32)
    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    return (float(np.mean(hsv[:, 0])), float(np.mean(hsv[:, 1])),
            float(np.mean(gray)), float(np.std(gray)))


def is_referee(stats):
    if stats is None:
        return False
    _, sat, _, contrast = stats
    return sat < REF_MAX_MEAN_SAT and contrast > REF_MIN_CONTRAST


class TeamClassifier:
    def __init__(self):
        self.samples = []
        self.centers = None
        self.votes = defaultdict(lambda: deque(maxlen=VOTE_BUFFER_LEN))
        self.locked_team = {}

    def add_sample(self, stats):
        if self.centers is not None or stats is None:
            return
        hue, sat, _, _ = stats
        if sat >= MIN_SAT_FOR_CLUSTER:
            self.samples.append((hue, sat))

    def maybe_fit(self, frame_idx):
        if self.centers is not None:
            return
        if frame_idx >= CALIBRATION_FRAMES and len(self.samples) >= MIN_CALIB_SAMPLES:
            data = np.array(self.samples, dtype=np.float32)
            crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 60, 0.4)
            _, _, centers = cv2.kmeans(data, 2, None, crit, 10, cv2.KMEANS_PP_CENTERS)
            self.centers = centers
            print(f"[TEAMS] Learned 2 jersey clusters from {len(data)} samples ✓ centers={centers.tolist()}", flush=True)

    def classify(self, tid, stats):
        if tid in self.locked_team:
            return self.locked_team[tid]
        if self.centers is None or stats is None:
            return None
        hue, sat, _, _ = stats
        if sat < MIN_SAT_FOR_CLUSTER:
            return None
        v = np.array([hue, sat], dtype=np.float32)
        d0 = np.linalg.norm(v - self.centers[0])
        d1 = np.linalg.norm(v - self.centers[1])
        vote = 'A' if d0 <= d1 else 'B'
        self.votes[tid].append(vote)

        buf = self.votes[tid]
        if len(buf) >= VOTE_LOCK_THRESH:
            a_count = buf.count('A'); b_count = buf.count('B')
            if a_count >= VOTE_LOCK_THRESH:
                self.locked_team[tid] = 'A'; return 'A'
            if b_count >= VOTE_LOCK_THRESH:
                self.locked_team[tid] = 'B'; return 'B'
        return vote

    def color_for(self, team):
        if team == 'A': return TEAM_A_COLOR
        if team == 'B': return TEAM_B_COLOR
        return UNASSIGNED_COLOR


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
        if inter == 0: return 0.0
        area_a = (ax2 - ax1) * (ay2 - ay1); area_b = (bx2 - bx1) * (by2 - by1)
        return inter / float(area_a + area_b - inter)

    def update(self, dets):
        """dets: list of (x1,y1,x2,y2,conf,stats)"""
        new = {}; used = set()

        for x1, y1, x2, y2, conf, stats in dets:
            cx = (x1 + x2) / 2; cy = (y1 + y2) / 2
            box = (x1, y1, x2, y2)

            best_id, best_score = None, -1
            for tid, info in self.tracks.items():
                if tid in used: continue
                d = np.hypot(cx - info["cx"], cy - info["cy"])
                if d > MATCH_MAX_DIST: continue
                iou = self._iou(box, info["box"])
                score = iou - (d / MATCH_MAX_DIST) * 0.3
                if score > best_score: best_score = score; best_id = tid

            if best_id is None:
                best_id = self.next_id; self.next_id += 1

            if best_id in self.tracks:
                dpx = np.hypot(cx - self.tracks[best_id]["cx"], cy - self.tracks[best_id]["cy"])
                self.dists[best_id] += dpx * PIXEL_TO_METER
                self.speeds[best_id] = dpx * PIXEL_TO_METER * self.fps * 3.6

            new[best_id] = {"box": box, "cx": cx, "cy": cy, "conf": conf, "stats": stats}
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


class BallTracker:
    def __init__(self, fps):
        self.fps = fps
        self.trail = deque(maxlen=45)
        self.vy_history = deque(maxlen=5)
        self.last_spike_frame = -999
        self.spikes = []
        self.rally_count = 0
        self.rally_active = False
        self.frames_since_seen = 0
        self.current_rally_len = 0
        self.rally_lengths = []

    def update(self, pos, frame_idx, net_y):
        if pos is None:
            self.frames_since_seen += 1
            if self.rally_active and self.frames_since_seen > RALLY_GAP_FRAMES:
                self.rally_active = False
                if self.current_rally_len > 0:
                    self.rally_lengths.append(self.current_rally_len)
                self.current_rally_len = 0
            return None

        self.frames_since_seen = 0
        if not self.rally_active:
            self.rally_active = True
            self.rally_count += 1
            self.current_rally_len = 0
        self.current_rally_len += 1

        vy = 0.0
        if self.trail:
            px, py = self.trail[-1]
            vy = py - pos[1]
            self.vy_history.append(vy)

        self.trail.append(pos)

        spike_flagged = False
        near_net = abs(pos[1] - net_y) < NET_BAND_PX
        avg_vy = float(np.mean(self.vy_history)) if self.vy_history else 0.0
        if (near_net and avg_vy > SPIKE_VEL_THRESH and
                (frame_idx - self.last_spike_frame) > SPIKE_COOLDOWN):
            self.spikes.append((frame_idx, pos))
            self.last_spike_frame = frame_idx
            spike_flagged = True

        return spike_flagged


def txt(img, text, x, y, scale=0.7, color=C_WHITE, thick=2, font=LABEL_FONT):
    cv2.putText(img, text, (x, y), font, scale, (0, 0, 0), thick + 3, cv2.LINE_AA)
    cv2.putText(img, text, (x, y), font, scale, color, thick, cv2.LINE_AA)


def label_block(img, lines, colors, cx, y_start, scale=LABEL_SCALE, thick=LABEL_THICK,
                 bg_color=(15, 15, 15), alpha=0.75):
    font = LABEL_FONT
    pad_x, pad_y, gap = 10, 5, 3
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
    for tid, info in tracks.items():
        x1, y1, x2, y2 = [int(v) for v in info["box"]]
        team = tc.classify(tid, info.get("stats"))
        color = tc.color_for(team)
        cx_ = (x1 + x2) // 2
        w_box = x2 - x1

        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)

        ell_w = max(24, int(w_box * 0.55))
        cv2.ellipse(canvas, (cx_, y2), (ell_w, 9), 0, 0, 360, color, 3, cv2.LINE_AA)

        spd_raw = tc.__dict__.get("_unused", 0.0)  # no-op guard, speeds pulled below
        label_block(
            canvas,
            lines=[f"#{tid}"],
            colors=[(255, 255, 255)],
            cx=cx_,
            y_start=max(0, y1 - 26),
            bg_color=tuple(int(c * 0.55) for c in color),
        )


def draw_ball(canvas, ball):
    if not ball.trail:
        return
    pts = list(ball.trail)
    n = len(pts)
    for i in range(1, n):
        a = i / n
        col = tuple(int(c * a) for c in C_BALL)
        cv2.line(canvas, tuple(map(int, pts[i - 1])), tuple(map(int, pts[i])), col, 3, cv2.LINE_AA)
    bx, by = map(int, pts[-1])
    for r, a in [(16, 40), (11, 90), (6, 200)]:
        ov = canvas.copy()
        cv2.circle(ov, (bx, by), r, C_BALL, -1, cv2.LINE_AA)
        cv2.addWeighted(ov, a / 255.0, canvas, 1 - a / 255.0, 0, canvas)
    cv2.circle(canvas, (bx, by), 6, C_WHITE, 2, cv2.LINE_AA)


def draw_net_line(canvas, W, net_y):
    ov = canvas.copy()
    cv2.line(ov, (0, net_y), (W, net_y), (255, 255, 255), 2, cv2.LINE_AA)
    cv2.addWeighted(ov, 0.5, canvas, 0.5, 0, canvas)
    txt(canvas, "NET", 10, net_y - 8, 0.55, (255, 255, 255), 1)


def draw_spike_flash(canvas, pos):
    bx, by = map(int, pos)
    ov = canvas.copy()
    cv2.circle(ov, (bx, by), 55, C_SPIKE, -1, cv2.LINE_AA)
    cv2.addWeighted(ov, 0.30, canvas, 0.70, 0, canvas)
    cv2.circle(canvas, (bx, by), 55, C_SPIKE, 3, cv2.LINE_AA)
    label_block(canvas, ["SPIKE!"], [C_SPIKE], bx, max(0, by - 90), scale=0.9, thick=2, bg_color=(10, 10, 10))


def draw_roi_debug(canvas, roi_pts):
    cv2.polylines(canvas, [roi_pts], True, (0, 255, 255), 2, cv2.LINE_AA)


def draw_bottom_hud(canvas, n_players, ball, W, H):
    bar_h = 50; y0 = H - bar_h - 22
    ov = canvas.copy()
    cv2.rectangle(ov, (0, y0), (W, y0 + bar_h), (8, 12, 20), -1)
    cv2.addWeighted(ov, 0.80, canvas, 0.20, 0, canvas)
    cv2.line(canvas, (0, y0), (W, y0), (50, 50, 80), 1)
    font = LABEL_FONT; scale = 0.68; thick = 2

    rally_len = ball.current_rally_len if ball.rally_active else (ball.rally_lengths[-1] if ball.rally_lengths else 0)
    stat1 = f"RALLY #{ball.rally_count}  LEN {rally_len}"
    txt(canvas, stat1, 20, y0 + bar_h // 2 + 8, scale, (255, 200, 0), thick, font)

    stat2 = f"PLAYERS {n_players}   SPIKES {len(ball.spikes)}"
    txt(canvas, stat2, W - 320, y0 + bar_h // 2 + 8, scale, (0, 200, 255), thick, font)


def draw_top_banner(canvas, W):
    title = "VOLLEYBALL RALLY & SPIKE ANALYTICS  v2.0  |  2-team mode"
    (tw, th), _ = cv2.getTextSize(title, LABEL_FONT, 0.78, 2)
    p = 12
    cv2.rectangle(canvas, (0, 0), (tw + p * 2, th + p * 2), (70, 0, 60), -1)
    cv2.putText(canvas, title, (p, th + p), LABEL_FONT, 0.78, C_WHITE, 2, cv2.LINE_AA)


def save_heatmap(all_positions, shape, path):
    H, W = shape[:2]
    fig, ax = plt.subplots(figsize=(10, 7), facecolor="#080c12")
    ax.set_facecolor("#0a1020")
    hm = np.zeros((H // 4, W // 4), dtype=np.float32)
    for px, py in all_positions:
        xi, yi = int(px / 4), int(py / 4)
        if 0 <= xi < hm.shape[1] and 0 <= yi < hm.shape[0]:
            hm[yi, xi] += 1
    hm = gaussian_filter(hm, sigma=8)
    if hm.max() > 0: hm /= hm.max()
    ax.imshow(hm, cmap="turbo", interpolation="bilinear", aspect="auto", origin="upper")
    ax.set_title("ALL-PLAYER POSITION HEATMAP", color="#00c8ff", fontsize=13, fontweight="bold", pad=10)
    ax.axis("off")
    fig.suptitle("VOLLEYBALL ANALYTICS — HEATMAP  |  dev: tubakhxn", color="#00c8ff", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"[DONE] {path} ✓", flush=True)


def save_dashboard(rally_lengths, spike_count, path):
    fig, ax = plt.subplots(figsize=(8, 6), facecolor="#080c12")
    ax.set_facecolor("#0a1220")
    for sp in ax.spines.values(): sp.set_color("#1a2840")
    fig.suptitle("VOLLEYBALL ANALYTICS — REPORT  |  dev: tubakhxn", color="#00c8ff", fontsize=13, fontweight="bold")
    if rally_lengths:
        ax.bar(range(1, len(rally_lengths) + 1), rally_lengths, color="#ffa000")
    ax.set_title(f"Rally Lengths (frames)  |  Spikes: {spike_count}", color="#00c8ff", fontsize=11)
    ax.set_xlabel("Rally #", color="#6a8aaa")
    ax.tick_params(colors="#6a8aaa")
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"[DONE] {path} ✓", flush=True)


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
    print("║  VOLLEYBALL RALLY & SPIKE ANALYTICS v2.0    ║", flush=True)
    print("║  dev: tubakhxn                              ║", flush=True)
    print("╚══════════════════════════════════════════════╝", flush=True)

    if not os.path.isfile(video_path):
        print(f"[ERROR] Video file not found: {video_path}", flush=True)
        return

    print("[YOLO] Loading detection model...", flush=True)
    model = YOLO("yolov8n.pt")
    print("[YOLO] loaded ✓", flush=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open {video_path}", flush=True)
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    net_y = int(H * NET_Y_FRAC)
    print(f"[INFO] {W}x{H} @ {fps:.1f}fps | {total} frames | net_y={net_y}", flush=True)

    roi_mask, roi_pts = build_roi_mask(W, H)
    out_dir = os.path.dirname(os.path.abspath(video_path))

    ret0, frame0 = cap.read()
    if ret0:
        preview = frame0.copy()
        draw_roi_debug(preview, roi_pts)
        cv2.line(preview, (0, net_y), (W, net_y), (0, 0, 255), 2)
        cv2.imwrite(os.path.join(out_dir, "volleyball_roi_preview.png"), preview)
        print("[ROI] Saved volleyball_roi_preview.png — adjust ROI_POINTS/NET_Y_FRAC if off.", flush=True)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    out_path = os.path.join(out_dir, "volleyball_output.mp4")
    writer = make_writer(out_path, fps, W, H)

    pt = PlayerTracker(fps=fps)
    ball = BallTracker(fps=fps)
    tc = TeamClassifier()
    all_positions = []
    speed_smooth = defaultdict(lambda: deque(maxlen=6))
    active_spike_flash = None
    frame_idx = 0

    print("[PROC] Processing...", flush=True)
    with tqdm(total=total, unit="fr", ncols=80, colour="cyan") as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            canvas = frame.copy()

            results = model(frame, verbose=False, conf=DET_CONF, classes=[0, 32])[0]
            dets = []; ball_pos = None; best_ball_conf = 0.0

            if results.boxes is not None:
                for box in results.boxes:
                    cls = int(box.cls[0]); name = model.names[cls]
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    conf = float(box.conf[0])
                    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2

                    if name == "person":
                        if not in_roi(cx, cy, roi_mask):
                            continue  # crowd/officials table -> skip
                        patch = torso_patch(frame, (x1, y1, x2, y2))
                        stats = analyze_patch(patch)
                        if is_referee(stats):
                            continue  # referee/staff -> skip
                        dets.append((x1, y1, x2, y2, conf, stats))
                        tc.add_sample(stats)
                    elif name == "sports ball" and conf > best_ball_conf:
                        best_ball_conf = conf
                        ball_pos = ((x1 + x2) // 2, (y1 + y2) // 2)

            tc.maybe_fit(frame_idx)
            tracks = pt.update(dets)
            spike_flagged = ball.update(ball_pos, frame_idx, net_y)
            if spike_flagged:
                active_spike_flash = [ball.trail[-1], 12]

            draw_top_banner(canvas, W)
            draw_net_line(canvas, W, net_y)
            draw_players(canvas, tracks, tc, speed_smooth)
            draw_ball(canvas, ball)

            if active_spike_flash is not None:
                draw_spike_flash(canvas, active_spike_flash[0])
                active_spike_flash[1] -= 1
                if active_spike_flash[1] <= 0:
                    active_spike_flash = None

            draw_bottom_hud(canvas, len(tracks), ball, W, H)

            writer.write(canvas)

            for tid, info in tracks.items():
                all_positions.append((info["cx"], info["cy"]))

            frame_idx += 1
            pbar.update(1)

    if ball.rally_active and ball.current_rally_len > 0:
        ball.rally_lengths.append(ball.current_rally_len)

    cap.release()
    writer.release()
    print(f"[DONE] {out_path} ✓", flush=True)
    print(f"[STATS] Rallies: {ball.rally_count} | Spikes detected: {len(ball.spikes)}", flush=True)
    save_heatmap(all_positions, (H, W), os.path.join(out_dir, "volleyball_heatmap.png"))
    save_dashboard(ball.rally_lengths, len(ball.spikes), os.path.join(out_dir, "volleyball_dashboard.png"))


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python volleyball_analytics.py video.mp4")
        sys.exit(1)
    process(sys.argv[1])
