# --- main.py (Minimal Debug Runner) ---
from pathlib import Path
import sys, cv2, numpy as np, math

# ====== 參數（必要時才調） ======
Y_L1 = (0.25, 0.55)    # L1 ROI 的 y 比例
Y_S1 = (0.72, 1.00)    # S1 ROI 的 y 比例（抓不到時試 0.68~0.75）
ANG_L1 = 35            # 近水平角度門檻（度）
ANG_S1 = 40
# ==============================

BASE_DIR = Path(__file__).resolve().parent
IMG_PATH = Path(sys.argv[1]) if len(sys.argv) > 1 else BASE_DIR / "input.jpg"

def imread_gray(path: Path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"[讀檔失敗] 找不到影像：{path}\nCWD：{Path.cwd()}")
    data = np.fromfile(str(path), dtype=np.uint8)  # 支援中文路徑
    img = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise RuntimeError(f"[解碼失敗] OpenCV 無法解碼：{path}（檔案損毀或副檔名不符）")
    return img

def auto_canny(gray):
    v = np.median(gray); lo = int(max(0, 0.66*v)); hi = int(min(255, 1.33*v))
    return cv2.Canny(gray, lo, hi)

def find_spine_band(gray):
    """穩健版：二值化骨影→形態學→找最大片、最縱長連通元件→x 範圍"""
    h, w = gray.shape
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    g = clahe.apply(gray)
    g = cv2.bilateralFilter(g, 5, 50, 50)

    # 強化骨骼：Otsu 二值 + 形態學閉運算（連續化）
    _, bw = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 21))
    bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel, iterations=1)

    # 只保留「細長且接近中央」的最大連通元件
    num, labels, stats, _ = cv2.connectedComponentsWithStats(bw)
    cand = []
    cx_img = w/2
    for i in range(1, num):
        x, y, ww, hh, area = stats[i]
        if hh < 0.4*h: 
            continue
        if ww < 0.08*w: 
            continue
        aspect = hh / max(1, ww)
        center_pen = abs((x+ww/2) - cx_img) / (w/2)
        score = aspect*2 + (area/(h*w))*3 - center_pen   # 越高越好
        cand.append((score, x, y, ww, hh))
    if not cand:
        # 退而求其次：用中間 60% 寬
        return int(w*0.20), int(w*0.80)

    cand.sort(reverse=True)
    _, x, y, ww, hh = cand[0]
    # 兩側多留一點邊（避免裁太窄）
    pad = int(0.06*w)
    xL = max(0, x - pad)
    xR = min(w, x + ww + pad)

    # 若帶寬仍過窄，擴成至少 0.25*w
    if (xR - xL) < int(0.25*w):
        cx = (xL + xR)//2
        half = max(int(0.125*w), (xR - xL)//2)
        xL = max(0, cx - half)
        xR = min(w, cx + half)
    return xL, xR

def hough_pass(roi, ang_limit_deg, len_ratio_list=(0.35,0.28,0.22), thr_list=(140,110,80,60)):
    H, W = roi.shape
    edges = auto_canny(roi)
    best, best_score = None, -1
    for min_len_ratio in len_ratio_list:
        min_len = int(W * min_len_ratio)
        for hth in thr_list:
            lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=hth,
                                    minLineLength=min_len, maxLineGap=20)
            if lines is None: continue
            for (x1,y1,x2,y2) in lines[:,0,:]:
                ang = abs(math.degrees(math.atan2(y2-y1, x2-x1)))
                ang = ang if ang <= 90 else 180-ang
                length = math.hypot(x2-x1, y2-y1)
                cx = (x1+x2)/2; center_penalty = abs(cx - W/2) / (W/2 + 1e-6)
                score = (max(0, ang_limit_deg - ang)) * 2 + length - 50*center_penalty
                if ang <= ang_limit_deg and score > best_score:
                    best, best_score = (x1,y1,x2,y2), score
        if best is not None: break
    return best, edges

def rotate_point(x, y, angle_deg, center):
    ang = np.deg2rad(angle_deg); ox, oy = center
    xr = ox + (x-ox)*np.cos(ang) - (y-oy)*np.sin(ang)
    yr = oy + (x-ox)*np.sin(ang) + (y-oy)*np.cos(ang)
    return int(round(xr)), int(round(yr))

def fallback_by_small_rotation(roi, base_ang_limit=35):
    H, W = roi.shape; center = (W//2, H//2)
    best, best_meta = None, (-1, 0)
    for a in range(-12, 13, 2):
        M = cv2.getRotationMatrix2D(center, a, 1.0)
        rot = cv2.warpAffine(roi, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        cand, _ = hough_pass(rot, base_ang_limit, len_ratio_list=(0.30,0.24,0.20), thr_list=(120,90,70))
        if cand is None: continue
        (x1,y1,x2,y2) = cand
        ang = abs(math.degrees(math.atan2(y2-y1, x2-x1))); ang = ang if ang<=90 else 180-ang
        length = math.hypot(x2-x1, y2-y1)
        score = (base_ang_limit - ang) * 2 + length
        if score > best_meta[0]:
            p1 = rotate_point(x1, y1, -a, center); p2 = rotate_point(x2, y2, -a, center)
            best, best_meta = (p1[0],p1[1],p2[0],p2[1]), (score, a)
    return best

def find_endplate_line(gray, spine_band, y0_ratio, y1_ratio, ang_limit_deg=35, tag="L1"):
    h, w = gray.shape; xL, xR = spine_band
    y0, y1 = int(h*y0_ratio), int(h*y1_ratio)
    roi = gray[y0:y1, xL:xR]
    # 前處理
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    roi_p = clahe.apply(roi)
    roi_p = cv2.GaussianBlur(roi_p, (5,5), 0)

    # 儲存 ROI 供 debug
    cv2.imwrite(f"debug_roi_{tag}.jpg", roi_p)
    line, edges = hough_pass(roi_p, ang_limit_deg)
    cv2.imwrite(f"debug_edges_{tag}.jpg", edges)

    if line is None:
        line = fallback_by_small_rotation(roi_p, base_ang_limit=ang_limit_deg+5)

    if line is None:
        return None, (y0, y1, xL, xR)
    (x1,y1_,x2,y2_) = line
    return (x1+xL, y1_+y0, x2+xL, y2_+y0), (y0, y1, xL, xR)

def line_angle_deg(p1, p2):
    dy, dx = p2[1]-p1[1], p2[0]-p1[0]
    ang = math.degrees(math.atan2(dy, dx))
    return (ang + 360) % 180

def main():
    gray = imread_gray(IMG_PATH)
    if gray.dtype != np.uint8:
        gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    print(f"[OK] Loaded: {IMG_PATH} shape={gray.shape}")
    cv2.imwrite("debug_gray.jpg", gray)

    # Step 1: 找脊柱垂直帶
    band = find_spine_band(gray)
    xL, xR = band
    print(f"[OK] Spine band: xL={xL}, xR={xR}, width={xR-xL}")
    overlay = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    cv2.rectangle(overlay, (xL, 0), (xR, gray.shape[0]-1), (0,255,255), 2)
    cv2.imwrite("debug_spine_band.jpg", overlay)

    # Step 2: 找 L1/S1 端板線
    L1_line, L1_roi = find_endplate_line(gray, band, Y_L1[0], Y_L1[1], ang_limit_deg=ANG_L1, tag="L1")
    S1_line, S1_roi = find_endplate_line(gray, band, Y_S1[0], Y_S1[1], ang_limit_deg=ANG_S1, tag="S1")
    print(f"[INFO] L1_line={L1_line}, S1_line={S1_line}")

    # 疊圖輸出
    out = overlay.copy()
    def draw_line(img, line, color):
        if line is not None:
            (x1,y1,x2,y2) = line
            cv2.line(img, (x1,y1), (x2,y2), color, 3)

    draw_line(out, L1_line, (0,0,255))
    draw_line(out, S1_line, (0,255,0))
    cv2.imwrite("debug_lines_overlay.jpg", out)

    # Step 3: 計算 LL
    if L1_line is None or S1_line is None:
        print("[WARN] 有端板線未偵測到 → 請查看 debug_roi_*.jpg / debug_edges_*.jpg / debug_spine_band.jpg")
        print("      提示：調整 Y_L1/Y_S1 或 ANG_L1/ANG_S1 後再試。")
        return

    θ_L1 = line_angle_deg((L1_line[0], L1_line[1]), (L1_line[2], L1_line[3]))
    θ_S1 = line_angle_deg((S1_line[0], S1_line[1]), (S1_line[2], S1_line[3]))
    LL = abs(θ_L1 - θ_S1); LL = LL if LL <= 90 else 180 - LL
    print(f"[OK] Lumbar Lordosis (LL) = {LL:.2f}°")

    cv2.putText(out, f"LL={LL:.2f} deg", (20,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    cv2.imwrite("output_overlay.jpg", out)
    print("[OK] 輸出：output_overlay.jpg（含脊柱帶 + 端板線 + LL）")

if __name__ == "__main__":
    main()
