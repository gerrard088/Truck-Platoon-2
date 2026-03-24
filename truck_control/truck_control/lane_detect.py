import cv2
import numpy as np
from scipy.signal import find_peaks

_LAST_KNOWN_POSITIONS = {}

def apply_birds_eye_view(image):
    height, width = image.shape[:2]
    src = np.float32([
        [width * 0.3, height * 0.4], [width * 0.7, height * 0.4],
        [width, height], [0, height]
    ])
    dst = np.float32([[0, 0], [width, 0], [width, height], [0, height]])
    M = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(image, M, (width, height))

def select_strategic_lanes(all_lane_centers, img_width):
    if len(all_lane_centers) < 2:
        return None, None, None, None, []
    mid = img_width // 2
    left_lanes = sorted([x for x in all_lane_centers if x < mid], reverse=True)
    right_lanes = sorted([x for x in all_lane_centers if x > mid])
    if not left_lanes or not right_lanes:
        return None, None, None, None, []
    center_left = left_lanes[0]
    center_right = right_lanes[0]
    adj_left = left_lanes[1] if len(left_lanes) > 1 else None
    adj_right = right_lanes[1] if len(right_lanes) > 1 else None
    return center_left, center_right, adj_left, adj_right, sorted(left_lanes + right_lanes)

def _get_fallback_seeds(edges, prefer_right_bias=0.1):
    h, w = edges.shape[:2]
    roi = edges[int(h * 0.6):, :]
    hist = np.sum(roi, axis=0)
    if hist.max() <= 10: return None, None
    mid = w // 2
    left_hist, right_hist = hist[:mid], hist[mid:]
    if prefer_right_bias > 0: right_hist = right_hist * (1.0 + prefer_right_bias)
    left_seed_x = int(np.argmax(left_hist)) if left_hist.size > 0 else None
    right_seed_x = int(np.argmax(right_hist)) + mid if right_hist.size > 0 else None
    return left_seed_x, right_seed_x

def detect_lane(image, truck_id, last_left_fit, last_right_fit, last_left_slope, last_right_slope, ss_mask=None):
    """
    이미지에서 차선을 검출하고 슬라이딩 윈도우를 통해 위치를 추적합니다.
    ss_mask가 제공되면(semantic RoadLine=24 마스크), 이를 edges로 사용합니다.
    """
    global _LAST_KNOWN_POSITIONS
    height, width = image.shape[:2]

    if ss_mask is not None:
        # ss_mask는 0/255 이진 마스크로 가정
        edges = ss_mask.copy()
        # 안정성 보강(선택)
        # edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((3,3), np.uint8), iterations=1)
    else:
        hls = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
        white_mask = cv2.inRange(hls, np.array([0, 200, 0]), np.array([180, 255, 70]))
        masked_hls = cv2.bitwise_and(image, image, mask=white_mask)
        gray = cv2.cvtColor(masked_hls, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 50, 150)

    histogram = np.sum(edges[height // 2:, :], axis=0)
    peaks, _ = find_peaks(histogram, height=60, distance=width // 8)
    all_lane_centers = peaks.tolist()
    clx, crx, alx, arx, all_sorted = select_strategic_lanes(all_lane_centers, width) or (None, None, None, None, [])

    if truck_id not in _LAST_KNOWN_POSITIONS: _LAST_KNOWN_POSITIONS[truck_id] = {}
    last_pos = _LAST_KNOWN_POSITIONS[truck_id]

    clx = clx or last_pos.get('center_left')
    crx = crx or last_pos.get('center_right')
    alx = alx or last_pos.get('adj_left')
    arx = arx or last_pos.get('adj_right')

    if clx is None or crx is None:
        fallback_l, fallback_r = _get_fallback_seeds(edges, prefer_right_bias=0.1)
        clx = clx or fallback_l
        crx = crx or fallback_r
    if clx is None: clx = int(width * 0.25)
    if crx is None: crx = int(width * 0.75)

    lanes_to_track = {'center_left': clx, 'center_right': crx, 'adj_left': alx, 'adj_right': arx}

    nwindows = 10
    margin = 80
    minpix = 5
    window_height = height // nwindows
    nonzeroy, nonzerox = np.nonzero(edges)
    window_img = np.zeros_like(image)

    lane_positions = {k: v for k, v in lanes_to_track.items() if v is not None}

    for window in range(nwindows - 1, -1, -1):
        win_y_low = window * window_height
        win_y_high = (window + 1) * window_height
        for lane_key, initial_x in list(lane_positions.items()):
            current_x = lane_positions.get(lane_key, initial_x)
            win_x_low, win_x_high = int(current_x - margin), int(current_x + margin)
            color = (255, 255, 255) if lane_key == 'adj_left' else \
                    (0, 255, 0) if lane_key == 'adj_right' else \
                    (255, 0, 0) if lane_key == 'center_left' else (0, 0, 255)
            cv2.rectangle(window_img, (win_x_low, win_y_low), (win_x_high, win_y_high), color, 2)

            good_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                         (nonzerox >= win_x_low) & (nonzerox < win_x_high)).nonzero()[0]

            if len(good_inds) > minpix:
                new_center = int(np.mean(nonzerox[good_inds]))
                if len(good_inds) < minpix * 3:
                    lane_positions[lane_key] = int(0.7 * current_x + 0.3 * new_center)
                else:
                    lane_positions[lane_key] = new_center

    for key, pos in lane_positions.items():
        if pos is not None: _LAST_KNOWN_POSITIONS[truck_id][key] = pos

    lane_overlay = cv2.addWeighted(image, 1, window_img, 0.3, 0)

    # =========================
    # [추가] 시각 오버레이 구간
    # =========================
    # 1) 화면 테두리 (초록)
    border_thickness = 2
    border_color = (0, 0, 255)
    cv2.rectangle(lane_overlay,
                  (0, 0),
                  (width - 1, height - 1),
                  border_color,
                  border_thickness)

    # 2) 트럭 번호 라벨 (좌상단)
    #    - 가독성을 위해 배경 박스 + 외곽선 텍스트
    label = str(truck_id) if truck_id is not None else "Truck"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.8
    text_thickness = 2
    pad = 8
    # 텍스트 크기
    (tw, th), baseline = cv2.getTextSize(label, font, font_scale, text_thickness)
    x, y = 20, 40  # 텍스트 기준점
    # 배경 박스(검정)
    bg_tl = (x - pad, y - th - pad)
    bg_br = (x + tw + pad, y + baseline + pad // 2)
    cv2.rectangle(lane_overlay, bg_tl, bg_br, (0, 0, 0), cv2.FILLED)
    # 외곽선(검정) + 본문(흰색)
    cv2.putText(lane_overlay, label, (x, y), font, font_scale, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(lane_overlay, label, (x, y), font, font_scale, (255, 255, 255), 2, cv2.LINE_AA)
    # 호환: (img, positions, extra list) 형태로 fit/slope 자리 채움


    extra = [last_left_fit, last_right_fit, last_left_slope, last_right_slope]
    return lane_overlay, lane_positions, extra

def calculate_steering(pid_controller, lane_positions, img_width, target_lane='center',
                       transition_factor=0.0, is_lane_changing=False):
    if not lane_positions or lane_positions.get('center_left') is None or lane_positions.get('center_right') is None:
        return 0.0

    img_center = img_width / 2
    current_left, current_right = lane_positions['center_left'], lane_positions['center_right']
    lane_width = current_right - current_left
    target_left, target_right = current_left, current_right
    if target_lane == 'left':
        adj_left = lane_positions.get('adj_left')
        if adj_left is not None: target_left, target_right = adj_left, adj_left + lane_width
        else:                     target_left, target_right = current_left - lane_width, current_right - lane_width
    elif target_lane == 'right':
        adj_right = lane_positions.get('adj_right')
        if adj_right is not None: target_left, target_right = adj_right - lane_width, adj_right
        else:                      target_left, target_right = current_left + lane_width, current_right + lane_width

    blended_left  = current_left  + (target_left  - current_left)  * transition_factor
    blended_right = current_right + (target_right - current_right) * transition_factor
    lane_center = (blended_left + blended_right) / 2
    error = (lane_center - img_center) / img_center

    pid_output = pid_controller.compute(error)

    if is_lane_changing:
        return np.clip(-pid_output * 20.0, -15.0, 15.0)
    else:
        return np.clip(-pid_output * 35.0, -30.0, 30.0)