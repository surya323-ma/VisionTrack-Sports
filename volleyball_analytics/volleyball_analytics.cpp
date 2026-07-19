// Volleyball Rally & Spike Analytics v2.0 (C++ port) | 

#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <vector>
#include <deque>
#include <map>
#include <set>
#include <cmath>
#include <iostream>
#include <algorithm>

using namespace cv;
using namespace std;

static const Scalar C_BALL(0, 210, 255);
static const Scalar C_WHITE(255, 255, 255);
static const Scalar C_SPIKE(0, 60, 255);
static const Scalar TEAM_A_COLOR(140, 40, 10);
static const Scalar TEAM_B_COLOR(20, 20, 150);
static const Scalar UNASSIGNED_COLOR(80, 80, 80);

static const int FONT = FONT_HERSHEY_DUPLEX;
static const float LABEL_SCALE = 0.62f;
static const int LABEL_THICK = 2;

static const float DET_CONF = 0.40f;
static const float MATCH_MAX_DIST = 65.f;
static const int CALIBRATION_FRAMES = 60;
static const int MIN_CALIB_SAMPLES = 25;
static const float REF_MAX_MEAN_SAT = 55.f;
static const float REF_MIN_CONTRAST = 42.f;
static const float MIN_SAT_FOR_CLUSTER = 60.f;
static const int VOTE_BUFFER_LEN = 8;
static const int VOTE_LOCK_THRESH = 5;

static const float NET_Y_FRAC = 0.20f;
static const float NET_BAND_PX = 90.f;
static const float SPIKE_VEL_THRESH = 26.f;
static const int SPIKE_COOLDOWN = 15;
static const int RALLY_GAP_FRAMES = 45;

static const vector<Point2f> ROI_FRAC = { {0.f,0.22f},{1.f,0.22f},{1.f,1.f},{0.f,1.f} };

struct Stats { float hue, sat, gray, contrast; bool valid; };

Mat buildRoiMask(int W, int H, vector<Point>& outPts) {
    outPts.clear();
    for (auto& p : ROI_FRAC) outPts.push_back(Point((int)(p.x * W), (int)(p.y * H)));
    Mat mask = Mat::zeros(H, W, CV_8UC1);
    vector<vector<Point>> polys{ outPts };
    fillPoly(mask, polys, Scalar(255));
    return mask;
}

bool inRoi(const Mat& mask, float cx, float cy) {
    int xi = clamp((int)cx, 0, mask.cols - 1);
    int yi = clamp((int)cy, 0, mask.rows - 1);
    return mask.at<uchar>(yi, xi) > 0;
}

Stats analyzePatch(const Mat& frame, Rect box) {
    Stats s{0,0,0,0,false};
    int h = box.height;
    int ty1 = box.y + (int)(h * 0.12f), ty2 = box.y + (int)(h * 0.50f);
    int tx1 = box.x + (int)(box.width * 0.22f), tx2 = box.x + box.width - (int)(box.width * 0.22f);
    ty1 = max(0, ty1); ty2 = max(ty1 + 1, min(frame.rows, ty2));
    tx1 = max(0, tx1); tx2 = max(tx1 + 1, min(frame.cols, tx2));
    if (tx2 <= tx1 || ty2 <= ty1) return s;
    Mat patch = frame(Rect(tx1, ty1, tx2 - tx1, ty2 - ty1));
    if (patch.empty()) return s;
    Mat hsv, gray;
    cvtColor(patch, hsv, COLOR_BGR2HSV);
    cvtColor(patch, gray, COLOR_BGR2GRAY);
    Scalar meanHsv = mean(hsv);
    Scalar meanG, stdG;
    meanStdDev(gray, meanG, stdG);
    s.hue = (float)meanHsv[0]; s.sat = (float)meanHsv[1];
    s.gray = (float)meanG[0]; s.contrast = (float)stdG[0];
    s.valid = true;
    return s;
}

bool isReferee(const Stats& s) {
    if (!s.valid) return false;
    return s.sat < REF_MAX_MEAN_SAT && s.contrast > REF_MIN_CONTRAST;
}

struct TeamClassifier {
    vector<Point2f> samples;
    bool fitted = false;
    Mat centers;
    map<int, deque<char>> votes;
    map<int, char> locked;

    void addSample(const Stats& s) {
        if (fitted || !s.valid) return;
        if (s.sat >= MIN_SAT_FOR_CLUSTER) samples.push_back(Point2f(s.hue, s.sat));
    }

    void maybeFit(int frameIdx) {
        if (fitted) return;
        if (frameIdx >= CALIBRATION_FRAMES && (int)samples.size() >= MIN_CALIB_SAMPLES) {
            Mat data((int)samples.size(), 2, CV_32F);
            for (size_t i = 0; i < samples.size(); i++) {
                data.at<float>((int)i, 0) = samples[i].x;
                data.at<float>((int)i, 1) = samples[i].y;
            }
            Mat labels;
            kmeans(data, 2, labels, TermCriteria(TermCriteria::EPS + TermCriteria::MAX_ITER, 60, 0.4),
                   10, KMEANS_PP_CENTERS, centers);
            fitted = true;
            cout << "[TEAMS] Learned 2 jersey clusters from " << samples.size() << " samples" << endl;
        }
    }

    char classify(int tid, const Stats& s) {
        auto it = locked.find(tid);
        if (it != locked.end()) return it->second;
        if (!fitted || !s.valid || s.sat < MIN_SAT_FOR_CLUSTER) return '?';
        float d0 = hypot(s.hue - centers.at<float>(0,0), s.sat - centers.at<float>(0,1));
        float d1 = hypot(s.hue - centers.at<float>(1,0), s.sat - centers.at<float>(1,1));
        char vote = (d0 <= d1) ? 'A' : 'B';
        auto& buf = votes[tid];
        buf.push_back(vote);
        if ((int)buf.size() > VOTE_BUFFER_LEN) buf.pop_front();
        int a = (int)count(buf.begin(), buf.end(), 'A');
        int b = (int)count(buf.begin(), buf.end(), 'B');
        if ((int)buf.size() >= VOTE_LOCK_THRESH) {
            if (a >= VOTE_LOCK_THRESH) { locked[tid] = 'A'; return 'A'; }
            if (b >= VOTE_LOCK_THRESH) { locked[tid] = 'B'; return 'B'; }
        }
        return vote;
    }

    Scalar colorFor(char t) {
        if (t == 'A') return TEAM_A_COLOR;
        if (t == 'B') return TEAM_B_COLOR;
        return UNASSIGNED_COLOR;
    }
};

struct TrackInfo { Rect box; float cx, cy; float conf; Stats stats; };

struct PlayerTracker {
    map<int, TrackInfo> tracks;
    int nextId = 0;
    map<int, int> lost;
    int maxLost = 15;

    static float iou(const Rect& a, const Rect& b) {
        Rect inter = a & b;
        if (inter.area() == 0) return 0.f;
        return (float)inter.area() / (float)(a.area() + b.area() - inter.area());
    }

    map<int, TrackInfo> update(const vector<tuple<Rect,float,Stats>>& dets) {
        map<int, TrackInfo> newTracks;
        set<int> usedIds;

        for (auto& d : dets) {
            Rect box; float conf; Stats stats;
            tie(box, conf, stats) = d;
            float cx = box.x + box.width / 2.f, cy = box.y + box.height / 2.f;

            int bestId = -1; float bestScore = -1.f;
            for (auto& kv : tracks) {
                int tid = kv.first;
                if (usedIds.count(tid)) continue;
                float dd = hypot(cx - kv.second.cx, cy - kv.second.cy);
                if (dd > MATCH_MAX_DIST) continue;
                float iouVal = iou(box, kv.second.box);
                float score = iouVal - (dd / MATCH_MAX_DIST) * 0.3f;
                if (score > bestScore) { bestScore = score; bestId = tid; }
            }
            if (bestId < 0) bestId = nextId++;

            newTracks[bestId] = { box, cx, cy, conf, stats };
            usedIds.insert(bestId);
        }

        for (auto& kv : tracks) {
            int tid = kv.first;
            if (!usedIds.count(tid)) {
                lost[tid]++;
                if (lost[tid] <= maxLost) newTracks[tid] = kv.second;
            } else lost[tid] = 0;
        }
        tracks = newTracks;
        return tracks;
    }
};

struct BallTracker {
    deque<Point2f> trail;
    deque<float> vyHist;
    int lastSpikeFrame = -999;
    int spikeCount = 0;
    int rallyCount = 0;
    bool rallyActive = false;
    int framesSinceSeen = 0;
    int currentRallyLen = 0;
    vector<int> rallyLengths;

    bool update(bool has, Point2f pos, int frameIdx, float netY) {
        if (!has) {
            framesSinceSeen++;
            if (rallyActive && framesSinceSeen > RALLY_GAP_FRAMES) {
                rallyActive = false;
                if (currentRallyLen > 0) rallyLengths.push_back(currentRallyLen);
                currentRallyLen = 0;
            }
            return false;
        }
        framesSinceSeen = 0;
        if (!rallyActive) { rallyActive = true; rallyCount++; currentRallyLen = 0; }
        currentRallyLen++;

        float vy = 0.f;
        if (!trail.empty()) {
            vy = trail.back().y - pos.y;
            vyHist.push_back(vy);
            if (vyHist.size() > 5) vyHist.pop_front();
        }
        trail.push_back(pos);
        if (trail.size() > 45) trail.pop_front();

        bool spikeFlagged = false;
        bool nearNet = fabs(pos.y - netY) < NET_BAND_PX;
        float avgVy = 0.f;
        for (float v : vyHist) avgVy += v;
        if (!vyHist.empty()) avgVy /= vyHist.size();

        if (nearNet && avgVy > SPIKE_VEL_THRESH && (frameIdx - lastSpikeFrame) > SPIKE_COOLDOWN) {
            spikeCount++;
            lastSpikeFrame = frameIdx;
            spikeFlagged = true;
        }
        return spikeFlagged;
    }
};

void putTxt(Mat& img, const string& text, int x, int y, double scale, Scalar color, int thick) {
    putText(img, text, Point(x, y), FONT, scale, Scalar(0,0,0), thick + 3, LINE_AA);
    putText(img, text, Point(x, y), FONT, scale, color, thick, LINE_AA);
}

void labelBlock(Mat& img, const vector<string>& lines, const vector<Scalar>& colors,
                 int cx, int yStart, Scalar bgColor) {
    int padX = 10, padY = 5, gap = 3;
    vector<Size> sizes; int maxW = 0, totalH = 0;
    for (auto& l : lines) {
        int base;
        Size sz = getTextSize(l, FONT, LABEL_SCALE, LABEL_THICK, &base);
        sizes.push_back(sz);
        maxW = max(maxW, sz.width);
        totalH += sz.height + gap;
    }
    totalH -= gap;
    int bx1 = max(0, cx - maxW/2 - padX);
    int by1 = max(0, yStart - padY);
    int bx2 = min(img.cols - 1, cx + maxW/2 + padX);
    int by2 = min(img.rows - 1, yStart + totalH + padY);

    Mat overlay; img.copyTo(overlay);
    rectangle(overlay, Point(bx1, by1), Point(bx2, by2), bgColor, FILLED);
    addWeighted(overlay, 0.75, img, 0.25, 0, img);

    int yCur = yStart;
    for (size_t i = 0; i < lines.size(); i++) {
        int lx = cx - sizes[i].width / 2;
        putTxt(img, lines[i], lx, yCur + sizes[i].height, LABEL_SCALE, colors[i], LABEL_THICK);
        yCur += sizes[i].height + gap;
    }
}

void drawPlayers(Mat& canvas, map<int, TrackInfo>& tracks, TeamClassifier& tc) {
    for (auto& kv : tracks) {
        int tid = kv.first;
        auto& info = kv.second;
        char team = tc.classify(tid, info.stats);
        Scalar color = tc.colorFor(team);
        int cx = info.box.x + info.box.width / 2;
        int y2 = info.box.y + info.box.height;

        rectangle(canvas, info.box, color, 2, LINE_AA);
        int ellW = max(24, (int)(info.box.width * 0.55f));
        ellipse(canvas, Point(cx, y2), Size(ellW, 9), 0, 0, 360, color, 3, LINE_AA);

        Scalar bg(color[0]*0.55, color[1]*0.55, color[2]*0.55);
        labelBlock(canvas, {"#" + to_string(tid)}, {C_WHITE}, cx, max(0, info.box.y - 26), bg);
    }
}

void drawBall(Mat& canvas, BallTracker& ball) {
    if (ball.trail.empty()) return;
    int n = (int)ball.trail.size();
    for (int i = 1; i < n; i++) {
        double a = (double)i / n;
        Scalar col(C_BALL[0]*a, C_BALL[1]*a, C_BALL[2]*a);
        line(canvas, ball.trail[i-1], ball.trail[i], col, 3, LINE_AA);
    }
    Point last = ball.trail.back();
    int radii[] = {16, 11, 6}; int alphas[] = {40, 90, 200};
    for (int i = 0; i < 3; i++) {
        Mat ov; canvas.copyTo(ov);
        circle(ov, last, radii[i], C_BALL, FILLED, LINE_AA);
        addWeighted(ov, alphas[i]/255.0, canvas, 1 - alphas[i]/255.0, 0, canvas);
    }
    circle(canvas, last, 6, C_WHITE, 2, LINE_AA);
}

void drawNetLine(Mat& canvas, int W, int netY) {
    Mat ov; canvas.copyTo(ov);
    line(ov, Point(0, netY), Point(W, netY), C_WHITE, 2, LINE_AA);
    addWeighted(ov, 0.5, canvas, 0.5, 0, canvas);
    putTxt(canvas, "NET", 10, netY - 8, 0.55, C_WHITE, 1);
}

void drawSpikeFlash(Mat& canvas, Point2f pos) {
    Mat ov; canvas.copyTo(ov);
    circle(ov, pos, 55, C_SPIKE, FILLED, LINE_AA);
    addWeighted(ov, 0.30, canvas, 0.70, 0, canvas);
    circle(canvas, pos, 55, C_SPIKE, 3, LINE_AA);
    labelBlock(canvas, {"SPIKE!"}, {C_SPIKE}, (int)pos.x, max(0, (int)pos.y - 90), Scalar(10,10,10));
}

void drawBottomHud(Mat& canvas, int nPlayers, BallTracker& ball, int W, int H) {
    int barH = 50, y0 = H - barH - 22;
    Mat ov; canvas.copyTo(ov);
    rectangle(ov, Point(0,y0), Point(W, y0+barH), Scalar(8,12,20), FILLED);
    addWeighted(ov, 0.80, canvas, 0.20, 0, canvas);
    line(canvas, Point(0,y0), Point(W,y0), Scalar(50,50,80), 1);

    int rallyLen = ball.rallyActive ? ball.currentRallyLen :
                   (ball.rallyLengths.empty() ? 0 : ball.rallyLengths.back());
    string s1 = "RALLY #" + to_string(ball.rallyCount) + "  LEN " + to_string(rallyLen);
    putTxt(canvas, s1, 20, y0 + barH/2 + 8, 0.68, Scalar(255,200,0), 2);

    string s2 = "PLAYERS " + to_string(nPlayers) + "   SPIKES " + to_string(ball.spikeCount);
    putTxt(canvas, s2, W - 320, y0 + barH/2 + 8, 0.68, Scalar(0,200,255), 2);
}

void drawTopBanner(Mat& canvas, int W) {
    string title = "VOLLEYBALL RALLY & SPIKE ANALYTICS  v2.0 (C++)  |  dev: tubakhxn";
    int base;
    Size sz = getTextSize(title, FONT, 0.78, 2, &base);
    int p = 12;
    rectangle(canvas, Point(0,0), Point(sz.width + p*2, sz.height + p*2), Scalar(70,0,60), FILLED);
    putText(canvas, title, Point(p, sz.height + p), FONT, 0.78, C_WHITE, 2, LINE_AA);
}

int main(int argc, char** argv) {
    if (argc < 3) {
        cout << "Usage: ./volleyball video.mp4 yolov8n.onnx" << endl;
        return 1;
    }
    string videoPath = argv[1];
    string modelPath = argv[2];

    cout << "VOLLEYBALL RALLY & SPIKE ANALYTICS v2.0 (C++) | dev: tubakhxn" << endl;

    dnn::Net net = dnn::readNetFromONNX(modelPath);
    net.setPreferableBackend(dnn::DNN_BACKEND_OPENCV);
    net.setPreferableTarget(dnn::DNN_TARGET_CPU);

    VideoCapture cap(videoPath);
    if (!cap.isOpened()) { cerr << "Cannot open " << videoPath << endl; return 1; }

    double fps = cap.get(CAP_PROP_FPS); if (fps <= 0) fps = 25;
    int W = (int)cap.get(CAP_PROP_FRAME_WIDTH);
    int H = (int)cap.get(CAP_PROP_FRAME_HEIGHT);
    int total = (int)cap.get(CAP_PROP_FRAME_COUNT);
    int netY = (int)(H * NET_Y_FRAC);
    cout << "[INFO] " << W << "x" << H << " @ " << fps << "fps | " << total << " frames | net_y=" << netY << endl;

    vector<Point> roiPts;
    Mat roiMask = buildRoiMask(W, H, roiPts);

    string outPath = "volleyball_output.mp4";
    VideoWriter writer(outPath, VideoWriter::fourcc('m','p','4','v'), fps, Size(W, H));
    if (!writer.isOpened()) { cerr << "Cannot open writer" << endl; return 1; }

    PlayerTracker pt;
    BallTracker ball;
    TeamClassifier tc;

    int frameIdx = 0;
    Mat frame;
    const int INPUT_SIZE = 640;
    Point2f flashPos; int flashFrames = 0;

    while (cap.read(frame)) {
        Mat canvas = frame.clone();

        Mat blob;
        dnn::blobFromImage(frame, blob, 1.0/255.0, Size(INPUT_SIZE, INPUT_SIZE), Scalar(), true, false);
        net.setInput(blob);
        vector<Mat> outs;
        net.forward(outs, net.getUnconnectedOutLayersNames());

        Mat out0 = outs[0];
        int rows = out0.size[2];
        int dims = out0.size[1];

        float xScale = (float)frame.cols / INPUT_SIZE;
        float yScale = (float)frame.rows / INPUT_SIZE;

        vector<tuple<Rect,float,Stats>> dets;
        Point2f ballPos; bool hasBall = false; float bestBallConf = 0.f;

        Mat data(dims, rows, CV_32F, out0.ptr<float>());
        for (int i = 0; i < rows; i++) {
            float cx = data.at<float>(0, i);
            float cy = data.at<float>(1, i);
            float w  = data.at<float>(2, i);
            float h  = data.at<float>(3, i);
            int bestCls = -1; float bestConf = 0.f;
            for (int c = 4; c < dims; c++) {
                float v = data.at<float>(c, i);
                if (v > bestConf) { bestConf = v; bestCls = c - 4; }
            }
            if (bestConf < DET_CONF) continue;
            int x1 = (int)((cx - w/2) * xScale), y1 = (int)((cy - h/2) * yScale);
            int bw = (int)(w * xScale), bh = (int)(h * yScale);
            Rect box(max(0,x1), max(0,y1), max(1,bw), max(1,bh));
            box &= Rect(0,0,W,H);
            if (box.area() <= 0) continue;

            float bcx = box.x + box.width/2.f, bcy = box.y + box.height/2.f;

            if (bestCls == 0) {
                if (!inRoi(roiMask, bcx, bcy)) continue;
                Stats stats = analyzePatch(frame, box);
                if (isReferee(stats)) continue;
                dets.push_back(make_tuple(box, bestConf, stats));
                tc.addSample(stats);
            } else if (bestCls == 32) {
                if (bestConf > bestBallConf) {
                    bestBallConf = bestConf;
                    ballPos = Point2f(bcx, bcy);
                    hasBall = true;
                }
            }
        }

        tc.maybeFit(frameIdx);
        auto tracks = pt.update(dets);
        bool spikeFlagged = ball.update(hasBall, ballPos, frameIdx, (float)netY);
        if (spikeFlagged) { flashPos = ball.trail.back(); flashFrames = 12; }

        drawTopBanner(canvas, W);
        drawNetLine(canvas, W, netY);
        drawPlayers(canvas, tracks, tc);
        drawBall(canvas, ball);

        if (flashFrames > 0) {
            drawSpikeFlash(canvas, flashPos);
            flashFrames--;
        }

        drawBottomHud(canvas, (int)tracks.size(), ball, W, H);

        writer.write(canvas);
        frameIdx++;
        if (frameIdx % 30 == 0) cout << "\r[PROC] frame " << frameIdx << "/" << total << flush;
    }

    if (ball.rallyActive && ball.currentRallyLen > 0) ball.rallyLengths.push_back(ball.currentRallyLen);

    cap.release(); writer.release();
    cout << endl << "[DONE] " << outPath << " saved" << endl;
    cout << "[STATS] Rallies: " << ball.rallyCount << " | Spikes: " << ball.spikeCount << endl;
    return 0;
}
