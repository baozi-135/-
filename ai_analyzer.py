"""
AI视频分析模块 v2.0 — YOLOv8语义检测 + 多维度评分
替换原版硬编码颜色分析，用目标检测识别食物/手/碗/勺等关键元素
不用ffmpeg/ffprobe，全部用OpenCV完成
"""
import os
import json
import time
import numpy as np
from datetime import datetime
from pathlib import Path

# ---- YOLOv8（延迟加载，避免导入时卡住）----
_yolo_model = None

def _get_yolo():
    """延迟加载YOLOv8 nano模型"""
    global _yolo_model
    if _yolo_model is None:
        from ultralytics import YOLO
        _yolo_model = YOLO("yolov8n.pt")
    return _yolo_model


# ---- 食物相关类别（YOLOv8 COCO类别映射）----
# YOLOv8检测到的对我们有用的类别
FOOD_CLASSES = {
    0: "person",
    39: "bottle",
    44: "spoon",       # 勺子
    45: "bowl",        # 碗
    46: "banana",
    47: "apple",
    48: "sandwich",
    49: "orange",
    50: "broccoli",
    51: "carrot",
    52: "hot dog",
    53: "pizza",
    54: "donut",
    55: "cake",
    56: "chair",
    57: "couch",
    58: "potted plant",
    59: "bed",
    60: "dining table",
    61: "toilet",
    62: "tv",
    63: "laptop",
    64: "mouse",
    65: "remote",
    66: "keyboard",
    67: "cell phone",
    68: "microwave",
    69: "oven",
    70: "toaster",
    71: "sink",
    72: "refrigerator",
    73: "book",
    74: "clock",
    75: "vase",
    76: "scissors",
    77: "teddy bear",
    78: "hair drier",
    79: "toothbrush",
}

# 甜品/糖水相关类别权重
DESSERT_RELEVANT = {
    44: 1.5,   # spoon 高权重
    45: 1.5,   # bowl 高权重
    46: 0.5,   # banana
    47: 0.5,   # apple
    48: 0.3,   # sandwich
    52: 0.3,   # hot dog
    55: 0.4,   # cake
    54: 0.3,   # donut
    39: 0.2,   # bottle (可能是饮品)
    60: 0.2,   # dining table
}


def get_video_info(video_path):
    """用OpenCV获取视频时长和帧率，替换ffprobe"""
    import cv2
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {"duration": 0, "fps": 0, "total_frames": 0, "width": 0, "height": 0}
    
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    duration = total_frames / fps if fps > 0 else 0
    cap.release()
    return {"duration": round(duration, 2), "fps": fps, "total_frames": total_frames, "width": width, "height": height}


def extract_frames_opencv(video_path, timestamps):
    """用OpenCV抽帧，替换ffmpeg"""
    import cv2
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return [], [], []
    
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    frames_color = []
    frames_gray = []
    valid_times = []
    
    for ts in timestamps:
        frame_idx = int(ts * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            continue
        
        # 缩放到128x128用于快速分析（和原版一致）
        small = cv2.resize(frame, (128, 128))
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        
        # 跳过低信息帧
        if np.std(gray) < 5:
            continue
        
        frames_color.append(cv2.cvtColor(small, cv2.COLOR_BGR2RGB))
        frames_gray.append(gray)
        valid_times.append(ts)
    
    cap.release()
    return frames_color, frames_gray, valid_times


def _frame_quality(arr_color, arr_gray):
    """人眼级帧质量评分（保留原版逻辑）"""
    r, g, b = arr_color[:,:,0].astype(float), arr_color[:,:,1].astype(float), arr_color[:,:,2].astype(float)
    brightness = 0.299*r + 0.587*g + 0.114*b
    warm = np.mean((r + g) > b * 1.2)
    highlight = np.mean(brightness > 200)
    rg = np.abs(r - g)
    rb = np.abs(r - b)
    gb = np.abs(g - b)
    saturation = np.mean(rg + rb + gb) / 255.0
    contrast = np.std(brightness) / 255.0
    
    h, w = arr_gray.shape
    c = brightness[h//4:3*h//4, w//4:3*w//4]
    e = np.concatenate([brightness[:h//8].flatten(), brightness[7*h//8:].flatten(),
                        brightness[:, :w//8].flatten(), brightness[:, 7*w//8:].flatten()])
    center_ratio = np.mean(c) / max(np.mean(e), 1)
    
    # 食欲色检测
    yellow_mask = ((r > 150) & (g > 100) & (b < 100)).astype(float)
    red_mask = ((r > 150) & (g < 100) & (b < 100)).astype(float)
    green_mask = ((r < 120) & (g > 120) & (b < 120)).astype(float)
    appetite_color_score = (np.mean(yellow_mask) + np.mean(red_mask) + np.mean(green_mask)) * 15
    
    # 手部检测
    skin_like = ((r > 80) & (r < 230) & (g > 60) & (g < 180) &
                 (b > 40) & (b < 140) & (r > g) & (g > b)).astype(float)
    hand_score = np.mean(skin_like) * 20
    
    edge_dark = np.mean((brightness < 50).astype(float))
    bowl_score = edge_dark * 10
    
    return (warm*25 + highlight*20 + saturation*20 + contrast*20 +
            min(center_ratio-1, 1.0)*15 + appetite_color_score + hand_score + bowl_score)


def _grid_motion_concentration(frames_gray):
    """
    v19: 网格运动集中度 — 区分镜头晃动和真实内容动作
    
    原理：
    - 镜头晃动：全画面均匀抖动 → 各网格diff相似 → 集中度低（<0.3）
    - 真实动作：只有特定区域（手/勺/碗）在动 → diff集中在少数网格 → 集中度高（>0.6）
    
    返回值越大=越可能是真动作。
    """
    GRID_H, GRID_W = 6, 8  # 6×8=48格
    all_concentrations = []
    
    for a, b in zip(frames_gray[:-1], frames_gray[1:]):
        h, w = a.shape
        gh, gw = h // GRID_H, w // GRID_W
        if gh < 4 or gw < 4:
            continue  # 画面太小，跳过
        grid_diffs = []
        for i in range(GRID_H):
            for j in range(GRID_W):
                y1, y2 = i * gh, (i + 1) * gh if i < GRID_H - 1 else h
                x1, x2 = j * gw, (j + 1) * gw if j < GRID_W - 1 else w
                patch_diff = np.mean(np.abs(a[y1:y2, x1:x2].astype(float) - b[y1:y2, x1:x2].astype(float)))
                grid_diffs.append(patch_diff)
        grid_diffs = np.array(grid_diffs)
        mean_gd = max(float(np.mean(grid_diffs)), 0.01)
        conc = float(np.std(grid_diffs)) / mean_gd
        all_concentrations.append(conc)
    
    if not all_concentrations:
        return 0.5  # fallback: 中性值
    return float(np.median(all_concentrations))  # 中位数更抗异常帧


def _yolo_analyze_frames(frames_color, frames_gray, frame_times, video_path):
    """用YOLOv8分析帧中的目标物体"""
    try:
        model = _get_yolo()
    except Exception as e:
        print(f"    ⚠️ YOLOv8加载失败: {e}，回退到颜色分析")
        return None
    
    all_detections = []
    
    # 只分析3帧（开头/中间/结尾）减少计算
    sample_indices = [0, len(frames_color)//2, len(frames_color)-1] if len(frames_color) >= 3 else list(range(len(frames_color)))
    sample_indices = [i for i in sample_indices if i < len(frames_color)]
    
    for idx in sample_indices:
        frame = frames_color[idx]
        try:
            results = model(frame, verbose=False)
            dets = []
            for r in results:
                boxes = r.boxes
                if boxes is not None:
                    for box in boxes:
                        cls_id = int(box.cls[0])
                        conf = float(box.conf[0])
                        xywh = box.xywh[0].tolist()
                        dets.append({
                            "class_id": cls_id,
                            "class_name": FOOD_CLASSES.get(cls_id, f"cls_{cls_id}"),
                            "confidence": round(conf, 3),
                            "xywh": [round(x, 1) for x in xywh]
                        })
            all_detections.append({"frame_idx": idx, "timestamp": frame_times[idx], "detections": dets})
        except Exception as e:
            continue
    
    return all_detections


def _compute_yolo_scores(detections_list, duration):
    """根据YOLOv8检测结果计算各类分数"""
    if not detections_list:
        return {"yolo_score": 0, "food_presence": 0, "hand_presence": 0, 
                "spoon_presence": 0, "bowl_presence": 0, "eating_likelihood": 0}
    
    # 汇总所有帧的检测
    spoon_frames = 0
    bowl_frames = 0
    hand_frames = 0
    food_frames = 0
    total_frames = len(detections_list)
    
    for frame_dets in detections_list:
        has_spoon = False
        has_bowl = False
        has_hand = False
        has_food = False
        
        for d in frame_dets["detections"]:
            cid = d["class_id"]
            conf = d["confidence"]
            
            if cid == 44 and conf > 0.3:  # spoon
                has_spoon = True
            if cid == 45 and conf > 0.3:  # bowl
                has_bowl = True
            if cid == 0 and conf > 0.3:   # person (可能包含手)
                has_hand = True
            if cid in DESSERT_RELEVANT and cid not in (0, 44, 45) and conf > 0.3:
                has_food = True
        
        if has_spoon: spoon_frames += 1
        if has_bowl: bowl_frames += 1
        if has_hand: hand_frames += 1
        if has_food: food_frames += 1
    
    spoon_presence = spoon_frames / max(total_frames, 1)
    bowl_presence = bowl_frames / max(total_frames, 1)
    hand_presence = hand_frames / max(total_frames, 1)
    food_presence = food_frames / max(total_frames, 1)
    
    # eating_likelihood: 同时出现 spoon+bowl 或 spoon+hand 或 bowl+hand
    eating_score = 0
    for frame_dets in detections_list:
        cids = {d["class_id"] for d in frame_dets["detections"]}
        s, b, h = 44 in cids, 45 in cids, 0 in cids
        if (s and b) or (s and h) or (b and h):
            eating_score += 1
    eating_likelihood = eating_score / max(total_frames, 1)
    
    # 综合YOLO评分
    yolo_score = (spoon_presence * 30 + bowl_presence * 25 + hand_presence * 20 + 
                  food_presence * 15 + eating_likelihood * 10)
    
    return {
        "yolo_score": round(min(yolo_score, 100), 2),
        "food_presence": round(food_presence, 3),
        "hand_presence": round(hand_presence, 3),
        "spoon_presence": round(spoon_presence, 3),
        "bowl_presence": round(bowl_presence, 3),
        "eating_likelihood": round(eating_likelihood, 3)
    }


def analyze_shot_yolo(video_path):
    """
    镜头综合分析 — YOLOv8语义版
    完全替代原版analyze_shot()，不用ffmpeg/ffprobe
    """
    import cv2
    
    info = get_video_info(video_path)
    duration = info["duration"]
    
    if duration < 1.5:
        return {
            "motion": 0, "appetite": 0, "shot_type": "unknown",
            "score": 0, "peak_score": 0, "peak_timestamp": 0,
            "activity": 0, "stability": 1.0,
            "food_presence": 0, "hand_presence": 0, "spoon_presence": 0,
            "bowl_presence": 0, "eating_likelihood": 0, "yolo_score": 0,
            "duration": duration
        }
    
    # v18: 抽30帧 — 更高时间分辨率捕获动作
    timestamps = [duration * p for p in np.linspace(0.03, 0.97, 30)]
    frames_color, frames_gray, frame_times = extract_frames_opencv(video_path, timestamps)
    
    if len(frames_gray) < 3:
        return {
            "motion": 0, "appetite": 0, "shot_type": "unknown",
            "score": 0, "peak_score": 0, "peak_timestamp": 0,
            "activity": 0, "stability": 1.0,
            "food_presence": 0, "hand_presence": 0, "spoon_presence": 0,
            "bowl_presence": 0, "eating_likelihood": 0, "yolo_score": 0,
            "duration": duration
        }
    
    # ---- 帧间差分 ----
    diffs = [np.mean(np.abs(a.astype(float) - b.astype(float)))
             for a, b in zip(frames_gray[:-1], frames_gray[1:])]
    motion_score = max(diffs) * 0.6 + np.mean(diffs) * 0.3
    
    # ---- 帧质量评分（用于食欲感，不用于peak）----
    frame_scores = [_frame_quality(c, g) for c, g in zip(frames_color, frames_gray)]
    peak_quality = max(frame_scores)
    appetite_score = np.mean(frame_scores)

    # ---- v18: 运动密度peak检测 — 找动作最密集的片段，不是颜色最好看的 ----
    WINDOW = 5  # 滑动窗口（约0.5-1秒的动作段）
    window_motions = []
    for i in range(len(diffs) - WINDOW + 1):
        window_motions.append(sum(diffs[i:i+WINDOW]))
    if window_motions:
        peak_window_idx = int(np.argmax(window_motions))
        peak_idx = min(peak_window_idx + WINDOW // 2, len(frame_times) - 1)
    else:
        peak_idx = len(frame_times) // 2
    peak_score = peak_quality  # 保留峰值画质分
    peak_ts = frame_times[peak_idx] if peak_idx < len(frame_times) else 0

    # ---- v19: 网格运动集中度 — 区分真动作 vs 镜头晃动 ----
    raw_activity = float(np.mean(diffs) / max(np.max(diffs), 0.01))
    concentration = _grid_motion_concentration(frames_gray)
    # 真动作: concentration>0.6 → 放大; 镜头晃: concentration<0.3 → 压制
    real_activity = min(raw_activity * concentration * 3.5, 1.0)
    
    # ---- v18: 镜头稳定性 — diff方差大=晃动 ----
    diff_std = float(np.std(diffs))
    diff_mean = max(float(np.mean(diffs)), 0.1)
    stability = max(0.0, 1.0 - min(diff_std / (diff_mean * 3.0), 1.0))
    
    # ---- YOLOv8语义检测 ----
    yolo_detections = _yolo_analyze_frames(frames_color, frames_gray, frame_times, video_path)
    yolo_results = _compute_yolo_scores(yolo_detections, duration) if yolo_detections else {
        "yolo_score": 0, "food_presence": 0, "hand_presence": 0,
        "spoon_presence": 0, "bowl_presence": 0, "eating_likelihood": 0
    }
    
    # ---- 镜头类型判断（YOLO增强版）----
    brightness_std = np.std(frames_gray[2])
    peak_color = frames_color[peak_idx] if peak_idx < len(frames_color) else frames_color[0]
    r_p, g_p, b_p = peak_color[:,:,0].astype(float), peak_color[:,:,1].astype(float), peak_color[:,:,2].astype(float)
    
    # YOLO判断舀吃动作
    yolo_is_eating = yolo_results["eating_likelihood"] > 0.3
    
    # v19: eating_action需要真实动作 — 端着碗不动不算
    if yolo_is_eating and real_activity < 0.06:
        yolo_is_eating = False  # 有碗有勺但完全不动 → 降级
    
    # 颜色+手部判断
    skin_like = ((r_p > 80) & (r_p < 230) & (g_p > 60) & (g_p < 180) &
                 (b_p > 40) & (b_p < 140) & (r_p > g_p) & (g_p > b_p))
    yellow_food = ((r_p > 150) & (g_p > 100) & (b_p < 100)).astype(float)
    red_food = ((r_p > 150) & (g_p < 100) & (b_p < 100)).astype(float)
    green_food = ((r_p < 120) & (g_p > 120) & (b_p < 120)).astype(float)
    has_hand = np.mean(skin_like) > 0.008
    has_food_color = (np.mean(yellow_food) + np.mean(red_food) + np.mean(green_food)) > 0.02
    
    h, w = frames_gray[2].shape
    center = peak_color[h//4:3*h//4, w//4:3*w//4]
    center_bright = np.mean(center) > 130
    
    # YOLO增强版shot_type判断（v2.1: 收紧假eating_action误判）
    if yolo_is_eating:
        shot_type = "eating_action"
    elif has_hand and has_food_color:
        # 有手+有食物色 → 需要YOLO也看到碗或勺才确认eating_action
        yolo_sees_tableware = (yolo_results.get("bowl_presence", 0) > 0.15 or
                               yolo_results.get("spoon_presence", 0) > 0.15)
        if yolo_sees_tableware:
            shot_type = "eating_action"
        else:
            shot_type = "closeup" if center_bright else "medium"
    elif has_food_color and center_bright:
        # v2.1: 只有食物色+中心亮 → 大概率是成品展示，降级为closeup
        shot_type = "closeup"
    elif brightness_std < 25:
        shot_type = "wide"
    elif brightness_std < 45:
        shot_type = "medium"
    else:
        c_ = frames_gray[2][h//3:2*h//3, w//3:2*w//3]
        e_ = (np.mean(frames_gray[2][:h//4]) + np.mean(frames_gray[2][3*h//4:]) +
             np.mean(frames_gray[2][:,:w//4]) + np.mean(frames_gray[2][:,3*w//4:])) / 4
        shot_type = "closeup" if np.mean(c_) > e_ * 1.15 else "medium"
    
    # ---- 综合评分（原版加权 + YOLO加权）----
    avg_brightness = np.mean(frames_gray[2])
    quality_penalty = 20 if (avg_brightness < 40 or brightness_std < 8) else 0
    
    if shot_type == "eating_action":
        total = peak_score * 0.45 + appetite_score * 0.15 + motion_score * 0.1 + yolo_results["yolo_score"] * 0.3 + 25
    elif shot_type == "closeup":
        total = peak_score * 0.5 + appetite_score * 0.2 + motion_score * 0.2 + yolo_results["yolo_score"] * 0.1 + 8
    elif shot_type == "medium":
        total = peak_score * 0.45 + appetite_score * 0.2 + motion_score * 0.25 + yolo_results["yolo_score"] * 0.1 + 5
    else:
        total = motion_score * 0.5 + peak_score * 0.3 + appetite_score * 0.2
    
    total = max(0, min(total - quality_penalty, 100))
    
    return {
        "motion": round(motion_score, 2),
        "appetite": round(appetite_score, 2),
        "shot_type": shot_type,
        "score": round(total, 2),
        "peak_score": round(peak_score, 2),
        "peak_timestamp": round(peak_ts, 2),
        "activity": round(real_activity, 4),
        "stability": round(stability, 4),
        "raw_activity": round(raw_activity, 4),
        "concentration": round(concentration, 4),
        "yolo_score": yolo_results["yolo_score"],
        "food_presence": yolo_results["food_presence"],
        "hand_presence": yolo_results["hand_presence"],
        "spoon_presence": yolo_results["spoon_presence"],
        "bowl_presence": yolo_results["bowl_presence"],
        "eating_likelihood": yolo_results["eating_likelihood"],
        "duration": duration
    }


def analyze_shot_fallback(video_path):
    """
    回退分析（YOLO不可用时）— 纯颜色+差分分析，OpenCV版
    与原版analyze_shot()逻辑一致，但用OpenCV替代ffmpeg
    """
    import cv2
    
    info = get_video_info(video_path)
    duration = info["duration"]
    
    if duration < 1.5:
        return {"motion": 0, "appetite": 0, "shot_type": "unknown",
                "score": 0, "peak_score": 0, "peak_timestamp": 0,
                "activity": 0, "stability": 1.0,
                "food_presence": 0, "hand_presence": 0, "spoon_presence": 0,
                "bowl_presence": 0, "eating_likelihood": 0, "yolo_score": 0,
                "duration": duration}
    
    timestamps = [duration * p for p in np.linspace(0.03, 0.97, 30)]
    frames_color, frames_gray, frame_times = extract_frames_opencv(video_path, timestamps)
    
    if len(frames_gray) < 3:
        return {"motion": 0, "appetite": 0, "shot_type": "unknown",
                "score": 0, "peak_score": 0, "peak_timestamp": 0,
                "activity": 0, "stability": 1.0,
                "food_presence": 0, "hand_presence": 0, "spoon_presence": 0,
                "bowl_presence": 0, "eating_likelihood": 0, "yolo_score": 0,
                "duration": duration}
    
    diffs = [np.mean(np.abs(a.astype(float) - b.astype(float)))
             for a, b in zip(frames_gray[:-1], frames_gray[1:])]
    motion_score = max(diffs) * 0.6 + np.mean(diffs) * 0.3
    
    frame_scores = [_frame_quality(c, g) for c, g in zip(frames_color, frames_gray)]
    peak_quality = max(frame_scores)
    appetite_score = np.mean(frame_scores)

    # v18: 运动密度peak检测
    WINDOW = 5
    window_motions = []
    for i in range(len(diffs) - WINDOW + 1):
        window_motions.append(sum(diffs[i:i+WINDOW]))
    if window_motions:
        peak_window_idx = int(np.argmax(window_motions))
        peak_idx = min(peak_window_idx + WINDOW // 2, len(frame_times) - 1)
    else:
        peak_idx = len(frame_times) // 2
    peak_score = peak_quality
    peak_ts = frame_times[peak_idx] if peak_idx < len(frame_times) else 0

    # v18: 活跃度和稳定性
    # v19: 网格运动集中度 — basic版本同样区分真动作 vs 镜头晃
    raw_activity = float(np.mean(diffs) / max(np.max(diffs), 0.01))
    concentration = _grid_motion_concentration(frames_gray)
    real_activity = min(raw_activity * concentration * 3.5, 1.0)
    diff_mean = max(float(np.mean(diffs)), 0.1)
    stability = max(0.0, 1.0 - min(float(np.std(diffs)) / (diff_mean * 3.0), 1.0))
    
    brightness_std = np.std(frames_gray[2])
    peak_color = frames_color[peak_idx] if peak_idx < len(frames_color) else frames_color[0]
    r_p, g_p, b_p = peak_color[:,:,0].astype(float), peak_color[:,:,1].astype(float), peak_color[:,:,2].astype(float)
    
    skin_like = ((r_p > 80) & (r_p < 230) & (g_p > 60) & (g_p < 180) &
                 (b_p > 40) & (b_p < 140) & (r_p > g_p) & (g_p > b_p))
    yellow_food = ((r_p > 150) & (g_p > 100) & (b_p < 100)).astype(float)
    red_food = ((r_p > 150) & (g_p < 100) & (b_p < 100)).astype(float)
    green_food = ((r_p < 120) & (g_p > 120) & (b_p < 120)).astype(float)
    has_hand = np.mean(skin_like) > 0.008
    has_food_color = (np.mean(yellow_food) + np.mean(red_food) + np.mean(green_food)) > 0.02
    
    h, w = frames_gray[2].shape
    center = peak_color[h//4:3*h//4, w//4:3*w//4]
    center_bright = np.mean(center) > 130
    
    # v2.1: fallback收紧 — 无YOLO时只靠颜色误判率太高
    if has_hand and has_food_color:
        shot_type = "eating_action"
    elif has_food_color and center_bright:
        shot_type = "closeup"  # v2.1: 降级，只有食物色不算eating_action
    elif brightness_std < 25:
        shot_type = "wide"
    elif brightness_std < 45:
        shot_type = "medium"
    else:
        c_ = frames_gray[2][h//3:2*h//3, w//3:2*w//3]
        e_ = (np.mean(frames_gray[2][:h//4]) + np.mean(frames_gray[2][3*h//4:]) +
             np.mean(frames_gray[2][:,:w//4]) + np.mean(frames_gray[2][:,3*w//4:])) / 4
        shot_type = "closeup" if np.mean(c_) > e_ * 1.15 else "medium"
    
    avg_brightness = np.mean(frames_gray[2])
    quality_penalty = 20 if (avg_brightness < 40 or brightness_std < 8) else 0
    
    if shot_type == "eating_action":
        total = peak_score * 0.65 + appetite_score * 0.25 + motion_score * 0.1 + 25
    elif shot_type == "closeup":
        total = peak_score * 0.6 + appetite_score * 0.2 + motion_score * 0.2 + 8
    elif shot_type == "medium":
        total = peak_score * 0.5 + appetite_score * 0.2 + motion_score * 0.3 + 5
    else:
        total = motion_score * 0.5 + peak_score * 0.3 + appetite_score * 0.2
    total = max(0, min(total - quality_penalty, 100))
    
    return {
        "motion": round(motion_score, 2),
        "appetite": round(appetite_score, 2),
        "shot_type": shot_type,
        "score": round(total, 2),
        "peak_score": round(peak_score, 2),
        "peak_timestamp": round(peak_ts, 2),
        "activity": round(real_activity, 4),
        "stability": round(stability, 4),
        "raw_activity": round(raw_activity, 4),
        "concentration": round(concentration, 4),
        "food_presence": 0, "hand_presence": 0, "spoon_presence": 0,
        "bowl_presence": 0, "eating_likelihood": 0, "yolo_score": 0,
        "duration": duration
    }


# ---- 智能路由：优先YOLO，失败回退 ----
_YOLO_AVAILABLE = None

def is_yolo_available():
    global _YOLO_AVAILABLE
    if _YOLO_AVAILABLE is None:
        try:
            from ultralytics import YOLO
            _YOLO_AVAILABLE = True
        except ImportError:
            _YOLO_AVAILABLE = False
    return _YOLO_AVAILABLE


def analyze_shot(video_path):
    """统一入口：自动选择YOLO模式或回退模式"""
    if is_yolo_available():
        try:
            result = analyze_shot_yolo(video_path)
        except Exception as e:
            print(f"    ⚠️ YOLO分析失败({e})，回退到颜色分析")
            result = analyze_shot_fallback(video_path)
    else:
        result = analyze_shot_fallback(video_path)
    
    # 追加多维度评分
    dims = compute_multi_dim_score(result)
    result["dim_scores"] = dims
    return result


# ============================================================
# 多维度评分体系 v2.0 — 8维评分
# ============================================================

def compute_multi_dim_score(analysis):
    """
    从单镜头分析结果中提取8维评分（0-100）
    
    维度说明：
    1. 食欲感 (appetite)     — YOLO食物检测 + 食欲色 + eating_likelihood
    2. 画面质量 (quality)     — 峰值帧质量 + 运动模糊度
    3. 构图 (composition)    — 中心亮度比 + 主体占比
    4. 动作丰富度 (action)    — 帧间差分 + YOLO动作检测
    5. 色彩 (color)          — 暖色调占比 + 饱和度
    6. 光线 (lighting)       — 亮度分布 + 高光区域
    7. 内容丰富度 (content)   — YOLO检测类别多样性
    8. 稀缺性 (scarcity)     — shot_type稀有度（默认基于类型）
    """
    dims = {}
    
    # 1. 食欲感 (0-100)
    # 来源：YOLO eating_likelihood + 食欲色得分 + food_presence
    eating = analysis.get("eating_likelihood", 0)
    appetite_raw = analysis.get("appetite", 0)
    food_p = analysis.get("food_presence", 0)
    dims["appetite"] = round(min(eating * 40 + appetite_raw * 0.3 + food_p * 30, 100), 1)
    
    # 2. 画面质量 (0-100)
    peak = analysis.get("peak_score", 0)
    motion = analysis.get("motion", 0)
    # 运动太大=模糊，太小=静止（都不好），适中最佳
    motion_quality = max(0, 25 - abs(motion - 15) * 1.2)
    dims["quality"] = round(min(peak * 0.5 + motion_quality, 100), 1)
    
    # 3. 构图 (0-100)
    # 基于已有信号推算：中心亮度比接近1.0最佳
    peak_s = analysis.get("peak_score", 0)
    dims["composition"] = round(min(peak_s * 0.55 + 15, 100), 1)
    
    # 4. 动作丰富度 (0-100)
    # v19: 网格集中度 — 真实动作集中在局部(concentration高)，镜头晃均匀分布(concentration低)
    # real_activity = raw_activity × concentration × 3.5 (capped at 1.0)
    # 典型值: 真动作 0.3-0.8, 镜头晃 0.01-0.06, 中等 0.08-0.25
    raw_action = min(motion * 2.5, 100)
    activity = analysis.get("activity", 0.5)
    stability_val = analysis.get("stability", 1.0)
    
    # v19 thresholds: real_activity < 0.06 = 本质上没真动作
    if activity < 0.06:
        raw_action *= 0.15   # 几乎没动作 → 砍85%
    elif activity < 0.12:
        raw_action *= 0.35   # 轻微动作 → 砍65%
    elif activity < 0.20:
        raw_action *= 0.55   # 中低动作 → 砍45%
    elif activity < 0.30:
        raw_action *= 0.75   # 中等动作 → 砍25%
    
    # stability < 0.4 = 晃动严重 → 额外扣分
    if stability_val < 0.3:
        raw_action *= 0.5
    elif stability_val < 0.5:
        raw_action *= 0.75
    
    dims["action"] = round(min(raw_action, 100), 1)
    
    # 5. 色彩 (0-100)
    # 来自appetite中的色彩成分 + 温暖感
    appetite_sc = analysis.get("appetite", 0)
    yolo_s = analysis.get("yolo_score", 0)
    dims["color"] = round(min(appetite_sc * 0.5 + yolo_s * 0.1 + 10, 100), 1)
    
    # 6. 光线 (0-100)
    # 亮度合适（不太暗不太亮）最佳
    avg_bright_est = analysis.get("appetite", 30) * 1.5
    lighting = max(0, 60 - abs(avg_bright_est - 45) * 0.8)
    dims["lighting"] = round(min(lighting, 100), 1)
    
    # 7. 内容丰富度 (0-100)
    # YOLO检测到的物体种类数量
    spoon = analysis.get("spoon_presence", 0)
    bowl = analysis.get("bowl_presence", 0)
    hand = analysis.get("hand_presence", 0)
    food = analysis.get("food_presence", 0)
    variety = (1 if spoon > 0.2 else 0) + (1 if bowl > 0.2 else 0) + \
              (1 if hand > 0.2 else 0) + (1 if food > 0.2 else 0)
    dims["content"] = round(min(variety * 25 + 10, 100), 1)
    
    # 8. 稀缺性 (0-100)
    # eating_action > closeup > medium > wide
    st = analysis.get("shot_type", "unknown")
    scarcity_map = {"eating_action": 85, "closeup": 55, "medium": 30, "wide": 10, "unknown": 5}
    dims["scarcity"] = scarcity_map.get(st, 10)
    
    # 综合加权分（v2.1: action提权10%→30%，压低静态素材总分）
    weights = {
        "appetite": 0.20,
        "quality": 0.10,
        "composition": 0.05,
        "action": 0.30,
        "color": 0.10,
        "lighting": 0.05,
        "content": 0.05,
        "scarcity": 0.15
    }
    dims["total"] = round(sum(dims.get(k, 0) * w for k, w in weights.items()), 1)
    
    return dims
