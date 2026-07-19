// Basketball Match Analytics v3.1 (C++ port) | dev: 


#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/video/tracking.hpp>
#include <vector>
#include <deque>
#include <map>
#include <set>
#include <cmath>
#include <iostream>
#include <fstream>
#include <algorithm>

using namespace cv;
using namespace std;

static const Scalar C_TEAM_A(220, 220, 220);
static const Scalar C_TEAM_A_FILL(255, 255, 255);
static const Scalar C_TEAM_A_BG(50, 50, 70);
static const Scalar C_TEAM_B(50, 130, 255);
static const Scalar C_TEAM_B_FILL(30, 90, 220);
static const Scalar C_TEAM_B_BG(10, 25, 70);
static const Scalar C_BALL(0, 140, 255);
static const Scalar C_WHITE(255, 255, 255);

static const float PIXEL_TO_METER = 0.045f;
static const int FONT = FONT_HERSHEY_DUPLEX;
static const float LABEL_SCALE = 0.72f;
static const int LABEL_THICK = 2;
static const float DET_CONF = 0.38f;

struct Stats { float ws, bs; bool valid; };

Stats jerseyScore(const Mat& frame, Rect box) {
    Stats s{0,0,false};
    int h = box.height;
    int ty = box.y + (int)(h*0.15f), by = box.y + (int)(h*0.55f);
    int tx = box.x + (int)(box.width*0.20f), bx = box.x + (int)(box.width*0.80f);
    ty = max(0,ty); by = min(frame.rows, by);
    tx = max(0,tx); bx = min(frame.cols, bx);
    if (bx<=tx || by<=ty) return s;
    Mat crop = frame(Rect(tx,ty,bx-tx,by-ty));
    if (crop.total() < 100) return s;
    Mat hsv; cvtColor(crop, hsv, COLOR_BGR2HSV);
    vector<Mat> ch; split(hsv, ch);
    Mat Hc = ch[0], Sc = ch[1], Vc = ch[2];
    Mat whiteMask = (Sc < 80) & (Vc > 70);
    Mat blueMask  = (Hc > 90) & (Hc < 140) & (Sc > 80);
    s.ws = (float)countNonZero(whiteMask) / (float)whiteMask.total();
    s.bs = (float)countNonZero(blueMask)  / (float)blueMask.total();
    s.valid = true;
    return s;
}

struct TeamClassifier {
    vector<Point2f> buf;
    bool calibrated = false;
    Point2f centerA, centerB;

    void calibrate() {
        if ((int)buf.size() < 20) return;
        Point2f cA(-1e9f, 1e9f), cB(1e9f, -1e9f);
        for (auto&p: buf){ cA.x=max(cA.x,p.x); cA.y=min(cA.y,p.y); cB.x=min(cB.x,p.x); cB.y=max(cB.y,p.y); }
        for (int it=0; it<20; it++) {
            Point2f sumA(0,0), sumB(0,0); int nA=0, nB=0;
            for (auto&p: buf) {
                float dA = norm(p-cA), dB = norm(p-cB);
                if (dA<dB) { sumA+=p; nA++; } else { sumB+=p; nB++; }
            }
            if (nA>0) cA = Point2f(sumA.x/nA, sumA.y/nA);
            if (nB>0) cB = Point2f(sumB.x/nB, sumB.y/nB);
        }
        if (cA.x >= cB.x) { centerA=cA; centerB=cB; } else { centerA=cB; centerB=cA; }
        calibrated = true;
        cout << "[TEAM] Calibrated white_center=(" << centerA.x << "," << centerA.y
             << ") blue_center=(" << centerB.x << "," << centerB.y << ")" << endl;
    }

    char assign(const Stats& s) {
        if (!s.valid) return '?';
        Point2f f(s.ws, s.bs);
        if (!calibrated) {
            buf.push_back(f);
            if ((int)buf.size() >= 40) calibrate();
            return (s.ws > s.bs) ? 'A' : 'B';
        }
        float dA = norm(f-centerA), dB = norm(f-centerB);
        return dA<=dB ? 'A' : 'B';
    }
};

struct TrackInfo { Rect box; float cx,cy; char team='?'; };

struct PlayerTracker {
    map<int,TrackInfo> tracks;
    int nextId=0;
    map<int,int> lost;
    int maxLost=25;
    float fps;
    map<int,float> dists, speeds;
    map<int,char> teams;

    PlayerTracker(float f):fps(f){}

    map<int,TrackInfo> update(const vector<tuple<Rect,Stats>>& dets, TeamClassifier& tc) {
        map<int,TrackInfo> newTracks;
        set<int> used;

        for (auto& d : dets) {
            Rect box; Stats stats; tie(box,stats)=d;
            float cx=box.x+box.width/2.f, cy=box.y+box.height/2.f;
            int bestId=-1; float bestD=100.f;
            for (auto& kv: tracks) {
                if (used.count(kv.first)) continue;
                float dd = hypot(cx-kv.second.cx, cy-kv.second.cy);
                if (dd<bestD) { bestD=dd; bestId=kv.first; }
            }
            char team;
            if (bestId<0) { bestId=nextId++; team=tc.assign(stats); teams[bestId]=team; }
            else { team = tc.calibrated ? tc.assign(stats) : teams[bestId]; teams[bestId]=team; }

            if (tracks.count(bestId)) {
                float dpx = hypot(cx-tracks[bestId].cx, cy-tracks[bestId].cy);
                dists[bestId]+=dpx*PIXEL_TO_METER;
                speeds[bestId]=dpx*PIXEL_TO_METER*fps*3.6f;
            }
            newTracks[bestId]={box,cx,cy,team};
            used.insert(bestId);
        }
        for (auto& kv: tracks) {
            if (!used.count(kv.first)) {
                lost[kv.first]++;
                if (lost[kv.first]<=maxLost) newTracks[kv.first]=kv.second;
            } else lost[kv.first]=0;
        }
        tracks=newTracks;
        return tracks;
    }
};

struct Possession {
    long totalA=0, totalB=0;
    void update(map<int,TrackInfo>& tracks, bool hasBall, Point2f ball) {
        if (!hasBall) return;
        float bestD=90.f; char bestTeam='?';
        for (auto& kv: tracks) {
            float d = hypot(kv.second.cx-ball.x, kv.second.cy-ball.y);
            if (d<bestD) { bestD=d; bestTeam=kv.second.team; }
        }
        if (bestTeam=='A') totalA++;
        else if (bestTeam=='B') totalB++;
    }
    pair<double,double> pct() {
        long tot=totalA+totalB;
        if (tot==0) return {50.0,50.0};
        return { round(1000.0*totalA/tot)/10.0, round(1000.0*totalB/tot)/10.0 };
    }
};

void putTxt(Mat& img, const string& text, int x, int y, double scale, Scalar color, int thick) {
    putText(img, text, Point(x,y), FONT, scale, Scalar(0,0,0), thick+3, LINE_AA);
    putText(img, text, Point(x,y), FONT, scale, color, thick, LINE_AA);
}

void labelBlock(Mat& img, const vector<string>& lines, const vector<Scalar>& colors,
                 int cx, int yStart, Scalar bgColor) {
    int padX=14, padY=7, gap=5;
    vector<Size> sizes; int maxW=0, totalH=0;
    for (auto&l: lines) { int base; Size sz=getTextSize(l,FONT,LABEL_SCALE,LABEL_THICK,&base);
        sizes.push_back(sz); maxW=max(maxW,sz.width); totalH+=sz.height+gap; }
    totalH -= gap;
    int bx1=max(0,cx-maxW/2-padX), by1=max(0,yStart-padY);
    int bx2=min(img.cols-1,cx+maxW/2+padX), by2=min(img.rows-1,yStart+totalH+padY);
    Mat ov; img.copyTo(ov);
    rectangle(ov, Point(bx1,by1), Point(bx2,by2), bgColor, FILLED);
    addWeighted(ov, 0.74, img, 0.26, 0, img);
    int yCur=yStart;
    for (size_t i=0;i<lines.size();i++) {
        int lx=cx-sizes[i].width/2;
        putTxt(img, lines[i], lx, yCur+sizes[i].height, LABEL_SCALE, colors[i], LABEL_THICK);
        yCur+=sizes[i].height+gap;
    }
}

void drawPlayerOverlay(Mat& canvas, Rect box, char team) {
    int ix1=max(0,box.x+4), iy1=max(0,box.y+4);
    int ix2=min(canvas.cols-1,box.x+box.width-4);
    int iy2=min(canvas.rows-1,box.y+(int)(box.height*0.82f));
    if (ix2<=ix1 || iy2<=iy1) return;
    Scalar fill = (team=='A') ? C_TEAM_A_FILL : C_TEAM_B_FILL;
    Mat ov; canvas.copyTo(ov);
    rectangle(ov, Point(ix1,iy1), Point(ix2,iy2), fill, FILLED);
    addWeighted(ov, 0.22, canvas, 0.78, 0, canvas);
}

void drawCamHud(Mat& frame, float dx, float dy, float cumX, float cumY) {
    string b1 = "Camera Movement:     X=" + to_string(dx) + "  Y=" + to_string(dy);
    string b2 = "Camera Displacement: X=" + to_string(cumX) + "  Y=" + to_string(cumY);
    int base; Size sz1=getTextSize(b1,FONT,0.8,2,&base), sz2=getTextSize(b2,FONT,0.8,2,&base);
    int p=12;
    rectangle(frame, Point(0,0), Point(sz1.width+p*2, sz1.height+p*2), Scalar(110,0,150), FILLED);
    putText(frame, b1, Point(p, sz1.height+p), FONT, 0.8, C_WHITE, 2, LINE_AA);
    int y2=sz1.height+p*2+4;
    rectangle(frame, Point(0,y2), Point(sz2.width+p*2, y2+sz2.height+p*2), Scalar(170,15,15), FILLED);
    putText(frame, b2, Point(p, y2+sz2.height+p), FONT, 0.8, C_WHITE, 2, LINE_AA);
}

void drawMinimap(Mat& canvas, map<int,TrackInfo>& tracks, int W, int H, int mw=220, int mh=160) {
    int mx=W-mw-14, my=H-mh-42;
    Mat ov; canvas.copyTo(ov);
    rectangle(ov, Point(mx,my), Point(mx+mw,my+mh), Scalar(20,10,4), FILLED);
    rectangle(ov, Point(mx,my), Point(mx+mw,my+mh), Scalar(0,80,140), 2);
    line(ov, Point(mx,my+mh/2), Point(mx+mw,my+mh/2), Scalar(0,60,100), 1);
    circle(ov, Point(mx+mw/2,my+mh/2), 18, Scalar(0,60,100), 1);
    addWeighted(ov, 0.82, canvas, 0.18, 0, canvas);
    for (auto& kv: tracks) {
        int px = mx + (int)((kv.second.cx/W)*mw), py = my + (int)((kv.second.cy/H)*mh);
        px = clamp(px, mx+2, mx+mw-2); py = clamp(py, my+2, my+mh-2);
        Scalar col = (kv.second.team=='A') ? C_TEAM_A : (kv.second.team=='B') ? C_TEAM_B : Scalar(120,120,120);
        circle(canvas, Point(px,py), 5, col, FILLED, LINE_AA);
        circle(canvas, Point(px,py), 5, Scalar(0,0,0), 1, LINE_AA);
    }
    putText(canvas, "MINIMAP", Point(mx+4,my-5), FONT, 0.42, Scalar(100,170,220), 1, LINE_AA);
}

void drawBottomHud(Mat& canvas, double pa, double pb, double avgS, double act, int nPlayers, int W, int H) {
    int barH=54, y0=H-barH-22;
    Mat ov; canvas.copyTo(ov);
    rectangle(ov, Point(0,y0), Point(W,y0+barH), Scalar(8,12,20), FILLED);
    addWeighted(ov, 0.80, canvas, 0.20, 0, canvas);
    line(canvas, Point(0,y0), Point(W,y0), Scalar(50,50,80), 1);

    int bw=300, bx=W/2-bw/2, by=y0+8, bh2=16;
    int af=(int)(bw*pa/100.0);
    rectangle(canvas, Point(bx,by), Point(bx+bw,by+bh2), Scalar(40,40,40), FILLED);
    if (af>0) rectangle(canvas, Point(bx,by), Point(bx+af,by+bh2), C_TEAM_A, FILLED);
    if (af<bw) rectangle(canvas, Point(bx+af,by), Point(bx+bw,by+bh2), C_TEAM_B, FILLED);
    rectangle(canvas, Point(bx,by), Point(bx+bw,by+bh2), Scalar(80,80,100), 1);

    string pt_ = "A " + to_string((int)pa) + "%  |  POSSESSION  |  " + to_string((int)pb) + "% B";
    int base; Size sz=getTextSize(pt_,FONT,0.58,1,&base);
    putTxt(canvas, pt_, W/2-sz.width/2, by+bh2+20, 0.58, C_WHITE, 1);

    string s1 = "AVG SPD  " + to_string(avgS).substr(0,4) + " km/h";
    int b2; Size sw=getTextSize(s1,FONT,0.72,2,&b2);
    putTxt(canvas, s1, bx-sw.width-50, y0+barH/2+10, 0.72, Scalar(0,220,130), 2);
    putTxt(canvas, "PLAYERS " + to_string(nPlayers) + "   ACT " + to_string((int)act),
           bx+bw+30, y0+barH/2+10, 0.72, Scalar(0,200,255), 2);
}

void drawTopBanner(Mat& canvas, int W) {
    string title = "BASKETBALL MATCH ANALYTICS  v3.1 (C++)  |  dev: tubakhxn";
    int base; Size sz=getTextSize(title,FONT,0.78,2,&base);
    int p=12;
    rectangle(canvas, Point(0,0), Point(sz.width+p*2, sz.height+p*2), Scalar(110,0,90), FILLED);
    putText(canvas, title, Point(p, sz.height+p), FONT, 0.78, C_WHITE, 2, LINE_AA);
}

int main(int argc, char** argv) {
    if (argc < 3) {
        cout << "Usage: ./basketball video.mp4 yolov8n.onnx" << endl;
        return 1;
    }
    string videoPath = argv[1];
    string modelPath = argv[2];

    cout << "BASKETBALL MATCH ANALYTICS v3.1 (C++) | dev: tubakhxn" << endl;

    dnn::Net net = dnn::readNetFromONNX(modelPath);
    net.setPreferableBackend(dnn::DNN_BACKEND_OPENCV);
    net.setPreferableTarget(dnn::DNN_TARGET_CPU);

    VideoCapture cap(videoPath);
    if (!cap.isOpened()) { cerr << "Cannot open " << videoPath << endl; return 1; }

    double fps = cap.get(CAP_PROP_FPS); if (fps <= 0) fps = 25;
    int W = (int)cap.get(CAP_PROP_FRAME_WIDTH);
    int H = (int)cap.get(CAP_PROP_FRAME_HEIGHT);
    int total = (int)cap.get(CAP_PROP_FRAME_COUNT);
    cout << "[INFO] " << W << "x" << H << " @ " << fps << "fps | " << total << " frames" << endl;

    string outPath = "basketball_output.mp4";
    VideoWriter writer(outPath, VideoWriter::fourcc('m','p','4','v'), fps, Size(W, H));
    if (!writer.isOpened()) { cerr << "Cannot open writer" << endl; return 1; }

    ofstream posLog("basketball_positions.csv");
    posLog << "frame,tid,team,cx,cy\n";

    PlayerTracker pt(fps);
    TeamClassifier tc;
    Possession poss;

    Mat prevGray;
    float cumX = 0.f, cumY = 0.f;
    int frameIdx = 0;
    Mat frame;
    const int INPUT_SIZE = 640;
    vector<double> histSpd;

    while (cap.read(frame)) {
        Mat canvas = frame.clone();

        Mat currGray; cvtColor(frame, currGray, COLOR_BGR2GRAY);
        float dx = 0.f, dy = 0.f;
        if (!prevGray.empty()) {
            vector<Point2f> pts, npts;
            goodFeaturesToTrack(prevGray, pts, 150, 0.01, 10);
            if (!pts.empty()) {
                vector<uchar> status; vector<float> err;
                calcOpticalFlowPyrLK(prevGray, currGray, pts, npts, status, err);
                vector<float> fx, fy;
                for (size_t i = 0; i < pts.size(); i++) {
                    if (status[i]) { fx.push_back(npts[i].x - pts[i].x); fy.push_back(npts[i].y - pts[i].y); }
                }
                if (!fx.empty()) {
                    sort(fx.begin(), fx.end()); sort(fy.begin(), fy.end());
                    dx = fx[fx.size()/2]; dy = fy[fy.size()/2];
                }
            }
        }
        cumX += dx; cumY += dy;
        prevGray = currGray;

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

        vector<tuple<Rect,Stats>> dets;
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

            if (bestCls == 0) {
                Stats stats = jerseyScore(frame, box);
                dets.push_back(make_tuple(box, stats));
            } else if (bestCls == 32) {
                if (bestConf > bestBallConf) {
                    bestBallConf = bestConf;
                    ballPos = Point2f(box.x+box.width/2.f, box.y+box.height/2.f);
                    hasBall = true;
                }
            }
        }

        auto tracks = pt.update(dets, tc);
        poss.update(tracks, hasBall, ballPos);

        if (hasBall) {
            int radii[] = {18,12,7,4}; int alphas[] = {35,70,160,255};
            for (int i=0;i<4;i++) {
                Mat ov2; canvas.copyTo(ov2);
                circle(ov2, ballPos, radii[i], C_BALL, FILLED, LINE_AA);
                addWeighted(ov2, alphas[i]/255.0, canvas, 1-alphas[i]/255.0, 0, canvas);
            }
            circle(canvas, ballPos, 7, C_WHITE, 2, LINE_AA);
        }

        for (auto& kv : tracks) {
            int tid = kv.first;
            auto& info = kv.second;
            float spd = pt.speeds.count(tid) ? pt.speeds[tid] : 0.f;
            float dist = pt.dists.count(tid) ? pt.dists[tid] : 0.f;
            int cx = info.box.x + info.box.width/2, cy = info.box.y + info.box.height/2;
            int rx = max(info.box.width/2 + 14, 24);
            int ry = max((int)(rx*0.33), 9);

            Scalar ecol, bgcol;
            if (info.team=='A') { ecol=C_TEAM_A; bgcol=C_TEAM_A_BG; }
            else if (info.team=='B') { ecol=C_TEAM_B; bgcol=C_TEAM_B_BG; }
            else { ecol=Scalar(120,120,120); bgcol=Scalar(30,30,30); }

            drawPlayerOverlay(canvas, info.box, info.team);

            Scalar glow(ecol[0]*0.3, ecol[1]*0.3, ecol[2]*0.3);
            ellipse(canvas, Point(cx,info.box.y+info.box.height), Size(rx+5,ry+4), 0,0,360, glow, 4, LINE_AA);
            ellipse(canvas, Point(cx,info.box.y+info.box.height), Size(rx,ry), 0,0,360, ecol, 2, LINE_AA);

            char buf1[32], buf2[32];
            snprintf(buf1, sizeof(buf1), "%.1f km/h", spd);
            snprintf(buf2, sizeof(buf2), "%.1f m", dist);
            labelBlock(canvas, {buf1, buf2}, {C_WHITE, Scalar(180,180,180)},
                       cx, info.box.y+info.box.height+ry+8, bgcol);

            posLog << frameIdx << "," << tid << "," << info.team << "," << info.cx << "," << info.cy << "\n";
        }

        drawCamHud(canvas, dx, dy, cumX, cumY);
        drawMinimap(canvas, tracks, W, H);

        auto pab = poss.pct();
        double totalSpd = 0.0; int n = 0;
        for (auto& kv : pt.speeds) { totalSpd += kv.second; n++; }
        double avgS = n ? totalSpd / n : 0.0;
        double act = avgS > 0 ? min(100.0, tracks.size()*avgS/10.0) : 0.0;
        histSpd.push_back(avgS);

        drawTopBanner(canvas, W);
        drawBottomHud(canvas, pab.first, pab.second, avgS, act, (int)tracks.size(), W, H);

        writer.write(canvas);
        frameIdx++;
        if (frameIdx % 30 == 0) cout << "\r[PROC] frame " << frameIdx << "/" << total << flush;
    }

    cap.release(); writer.release(); posLog.close();
    cout << endl << "[DONE] " << outPath << " saved" << endl;
    cout << "[DONE] basketball_positions.csv saved (plot in Excel/Python for heatmaps)" << endl;
    return 0;
}
