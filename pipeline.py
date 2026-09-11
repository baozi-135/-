"""
Pipeline标准化引擎 v1.0 — 6步流水线
每步可独立运行、调试、监控，清晰的输入输出边界

使用方式:
  python pipeline.py                     # 默认生成5条
  python pipeline.py -n 10               # 生成10条
  python pipeline.py --force-scan         # 强制重新扫描
  python pipeline.py -n 3 --step debug --step-idx 2  # 调试单步
"""
import os
import sys
import time
import glob
import random
import json
import argparse
from datetime import datetime

CWD = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, CWD)
from ai_analyzer import analyze_shot, get_video_info, compute_multi_dim_score

# ===== 路径配置 =====
MATERIAL_DIR = r"E:/111"
BGM_DIR      = r"E:/音频"
OUTPUT_DIR   = r"D:/混剪成品/auto"
LEARN_DIR    = r"D:\.workbuddy\skills\ai-video-auto-clip\learning"
CACHE_FILE   = os.path.join(LEARN_DIR, "material_cache.json")
USED_FILE    = os.path.join(LEARN_DIR, "used_videos.json")  # v19: 跨批次去重

# ===== 模板池（全部12个）=====
ALL_TEMPLATES = [
    {"name": "0508昆明",           "duration": 11.82},
    {"name": "0508新化 爷爷泡的茶", "duration": 10.70},
    {"name": "0430 新化氛围感",     "duration": 11.82},
    {"name": "9月25日海阔天空",     "duration": 11.75},
    {"name": "9月25日十面埋伏",     "duration": 11.61},
    {"name": "9月25日晚风心里吹",   "duration": 11.52},
    {"name": "9月26日列车开往春天", "duration": 11.29},
    {"name": "4月11日（1）",        "duration": 11.82},
    {"name": "4月11日（2）",        "duration": 10.52},
    {"name": "4月11日",             "duration": 9.71},
    {"name": "3月6日",              "duration": 10.24},
    {"name": "4月10日(1)",          "duration": 9.01},
]


# ============================================================
# Step 1: 素材预处理 — 收集 + 格式检查
# ============================================================

def step_preprocess(material_dir, validate=False, cache_data=None):
    """收集素材路径（默认只做快速路径收集，不逐文件打开）"""
    import hashlib
    
    videos = []
    for ext in ['*.mov', '*.mp4', '*.MOV', '*.MP4']:
        videos += glob.glob(os.path.join(material_dir, "**", ext), recursive=True)
    videos = list({v.lower(): v for v in videos if os.path.exists(v)}.values())
    
    # v16: 内容去重 — 文件大小+前64KB哈希，防止IMG_6680和IMG_6691这种同名不同号的内容重复
    seen_content = {}
    deduped = []
    dups_removed = 0
    for v in videos:
        try:
            sz = os.path.getsize(v)
            # 快速哈希：只读前64KB + 文件大小作为指纹
            with open(v, 'rb') as f:
                head = f.read(65536)
            fingerprint = hashlib.md5(head + str(sz).encode()).hexdigest()
            if fingerprint in seen_content:
                dups_removed += 1
                continue
            seen_content[fingerprint] = v
        except OSError:
            pass
        deduped.append(v)
    
    if dups_removed > 0:
        print(f"  [去重] 移除{dups_removed}个内容重复文件（共{len(videos)}→{len(deduped)}）")
    videos = deduped

    if not validate:
        # 快速模式：只收集路径，不检查格式
        return {
            "total": len(videos),
            "valid": len(videos),
            "invalid": 0,
            "videos": [{"path": v, "duration": 0, "fps": 0, "total_frames": 0, "width": 0, "height": 0} for v in videos],
            "invalid_list": []
        }

    # 慢速模式：逐个检查格式（仅在 --force-scan 时使用）
    if cache_data is None:
        cache_data = {}

    valid = []
    invalid = []
    for v in videos:
        key = os.path.abspath(v).lower()
        if key in cache_data:
            d = cache_data[key].get("duration", 0)
            if d >= 1.5:
                valid.append({"path": v, "duration": d})
            else:
                invalid.append({"path": v, "duration": d})
        else:
            info = get_video_info(v)
            if info["duration"] >= 1.5 and info["width"] > 0:
                valid.append({"path": v, **info})
            else:
                invalid.append({"path": v, **info})

    return {
        "total": len(videos),
        "valid": len(valid),
        "invalid": len(invalid),
        "videos": valid,
        "invalid_list": invalid
    }


# ============================================================
# Step 2: 素材评分 — 分析 + 多维度评分 + 缓存
# ============================================================

def step_score(videos, cache_data, force=False):
    """对素材运行分析，返回带多维度评分的结果
    
    v18: 旧缓存（缺activity/stability/dim_scores）自动触发重分析
    """
    results = []
    new_analyzed = 0
    total = len(videos)
    
    # v16: 安全网 — basename去重，防止step_preprocess可能漏掉的同名文件
    seen_basenames = set()
    
    for idx, vinfo in enumerate(videos):
        vp = vinfo["path"]
        
        # v16: basename去重（安全网，正常情况step_preprocess已去重）
        basename = os.path.basename(vp)
        if basename in seen_basenames:
            continue
        seen_basenames.add(basename)
        
        key = os.path.abspath(vp).lower()
        
        # v18: 检测旧缓存 — 缺activity字段说明是旧版，需重分析
        cached = cache_data.get(key) if not force else None
        needs_reanalyze = True
        if cached is not None:
            if "activity" in cached and cached.get("dim_scores"):
                # v19: 检查是否已升级到网格集中度版本
                if "concentration" in cached:
                    needs_reanalyze = False
                else:
                    # v18→v19 迁移：估算 concentration，不重跑YOLO
                    old_act = cached.get("activity", 0.5)
                    if old_act > 0.20:
                        conc = 0.55
                    elif old_act > 0.10:
                        conc = 0.35
                    else:
                        conc = 0.15
                    new_act = min(old_act * conc * 3.5, 1.0)
                    cached["raw_activity"] = old_act
                    cached["concentration"] = round(conc, 4)
                    cached["activity"] = round(new_act, 4)
                    needs_reanalyze = False
        
        if needs_reanalyze:
            # 新分析 / 旧缓存升级
            analysis = analyze_shot(vp)
            duration = analysis.get("duration", vinfo.get("duration", 0))
            dims = analysis.get("dim_scores", {})
            results.append({
                "path": vp,
                "score": dims.get("total", analysis.get("score", 0)),
                "duration": duration,
                "shot_type": analysis.get("shot_type", "unknown"),
                "dim_scores": dims,
                "peak_timestamp": analysis.get("peak_timestamp", 0),
                "activity": analysis.get("activity", 0.5),
                "stability": analysis.get("stability", 1.0),
                "cached": False,
                "_raw": analysis  # 用于写入缓存
            })
            new_analyzed += 1
        else:
            # 复用 v18 缓存
            dim_scores = cached.get("dim_scores", {})
            if "action" not in dim_scores and "motion" in cached:
                dim_scores = dict(dim_scores)
                dim_scores["action"] = round(min(cached["motion"] * 2.5, 100), 1)
            results.append({
                "path": vp,
                "score": cached.get("multi_dim_total", cached.get("motion_score", 0)),
                "duration": cached.get("duration", vinfo.get("duration", 0)),
                "shot_type": cached.get("shot_type", "unknown"),
                "dim_scores": dim_scores,
                "peak_timestamp": cached.get("peak_timestamp", 0),
                "activity": cached.get("activity", 0.5),
                "stability": cached.get("stability", 1.0),
                "cached": True
            })
        
        # v18: 进度 — 每50个或最后一个打印
        if (idx + 1) % 50 == 0 or idx == total - 1:
            print(f"\r  [Step2 评分] {idx+1}/{total} ({new_analyzed} 升级)", end="", flush=True)
    
    if total > 0:
        print()  # 换行
    
    return {
        "results": sorted(results, key=lambda x: x["score"], reverse=True),
        "total": len(results),
        "new_analyzed": new_analyzed
    }


# ============================================================
# Step 3: BGM匹配 — 选择 + 节拍分析
# ============================================================

def step_bgm_match(bgm_dir, template_duration):
    """选择BGM并分析节拍结构"""
    bgms = [
        os.path.join(root, f)
        for root, _, files in os.walk(bgm_dir)
        for f in files if f.lower().endswith('.mp3')
    ]
    
    if not bgms:
        return {"error": "无可用BGM", "bgm_path": None, "beat_plan": None}
    
    bgm_path = random.choice(bgms)
    bgm_name = os.path.basename(bgm_path)
    
    # 节拍分析（调用主脚本的函数，这里做简化版）
    beat_plan = _analyze_beats(bgm_path, template_duration)
    
    return {
        "bgm_path": bgm_path,
        "bgm_name": bgm_name,
        "beat_plan": beat_plan,
        "total_bgms": len(bgms)
    }


def _analyze_beats(bgm_path, duration):
    """
    节拍分析 v9 — 严格拍子网格对齐，不破坏整数拍约束。
    
    v1-v8 共同失败根因：librosa 外部算法计算出的节拍位置 ≠ 剪映内部引擎位置。
    剪映橙色节拍线本质是 BPM 均匀网格（每拍一条线）。
    
    v9 策略（本质性改变）：
    1. 只提取 BPM 值，不追踪任何节拍位置
    2. 每个片段时长 = beat_interval × N（严格整数拍）
    3. 不做 0.7-1.1s 约束（这是破坏对齐的根源）
    4. 只调最后一个片段补齐总时长
    
    所有片段边界（除最后一个）精确落在 BPM 拍子网格上。
    """
    import numpy as np
    
    bpm = 0
    
    if bgm_path and os.path.exists(bgm_path):
        try:
            import librosa
            y, sr = librosa.load(bgm_path, sr=None, duration=duration)
            if len(y) > 0:
                tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
                bpm = float(np.asarray(tempo).ravel()[0]) if (tempo is not None and np.asarray(tempo).ravel()[0] > 0) else 0
        except Exception:
            bpm = 0
    
    if bpm <= 0:
        bpm = 120
    
    beat_interval = 60.0 / bpm
    
    # 确定每片段拍数 N（让片长在 0.7~1.2s 之间）
    N_base = max(1, round(0.85 / beat_interval))
    
    # 只对极端情况做调整，不过度约束
    base_dur = beat_interval * N_base
    if base_dur < 0.5:
        N_base += 1
    elif base_dur > 1.3:
        N_base = max(1, N_base - 1)
    
    base_dur = beat_interval * N_base
    
    # 计算片段数：每个片段严格 = beat_interval × N_base（整数拍），末段吸收余数
    base_dur = beat_interval * N_base
    n_full = max(7, int(duration / base_dur))
    
    clip_durations = [round(base_dur, 4)] * n_full
    remainder = round(duration - sum(clip_durations), 4)
    
    # 处理余数（确保总时长精确 = duration，仅最后一个片段可能非整数拍）
    if remainder >= base_dur * 0.4:
        # 余数够一个短片段，追加
        clip_durations.append(round(remainder, 4))
    elif remainder > 0:
        # 余数太小，合并到最后一个满拍片段
        clip_durations[-1] = round(clip_durations[-1] + remainder, 4)
    
    # 确保至少8个
    while len(clip_durations) < 8:
        clip_durations.insert(0, round(base_dur, 4))
        clip_durations[-1] = round(clip_durations[-1] - base_dur, 4)
    
    # 从 clip_durations 构建 beat_times
    beat_times = [0.0]
    for d in clip_durations:
        beat_times.append(round(beat_times[-1] + d, 4))
    
    print(f"    BPM={bpm:.0f} | 拍长={beat_interval:.3f}s | N={N_base}拍/镜({base_dur:.3f}s) | {len(clip_durations)}镜头")
    
    return {
        "method": "beat_grid_v9",
        "bpm": round(bpm),
        "beat_interval": round(beat_interval, 4),
        "shot_count": len(clip_durations),
        "clip_durations": clip_durations,
        "beat_times": beat_times,
    }


# ============================================================
# Step 4: 镜头方案 — 选中素材 + 时长分配
# ============================================================

def step_plan(scored_results, beat_plan, used_videos=None):
    """从高分素材中选择镜头，分配时长
    
    v18 策略：评分排序 + 运动密度peak + 静态/晃动双重惩罚 + 文件名去重 + 类型驱动入点
    - peak_timestamp 基于运动密度而非颜色质量 (ai_analyzer v18)
    - activity/stability 直接惩罚静态和晃动素材 (compute_pref_score v18)
    - 硬动作阈值：dim_scores.action < 8 直接踢出
    - 硬静态阈值：real_activity < 0.05 直接踢出（v19: 网格集中度，镜头晃已被压制）
    - 类型驱动入点：eating_action前移50%捕获完整舀起动作，closeup居中40%
    - 文件名去重：同一源文件（按basename）只能用一次
    - 镜头多样性：每种类型最多占50%
    """
    import os
    
    if used_videos is None:
        used_videos = set()
    
    clip_durations = beat_plan["clip_durations"]
    shots_needed = len(clip_durations)
    min_duration = min(clip_durations) + 0.3
    
    seen_paths = set(used_videos)
    
    # === 仅排除绝对废片 + 同名去重 ===
    seen_basenames = set()
    base_pool = []
    for r in scored_results:
        if r["path"] in seen_paths:
            continue
        if r["duration"] < min(clip_durations) + 0.3:
            continue
        if r.get("shot_type", "unknown") in ("wide", "unknown"):
            continue
        if r.get("score", 0) < 15:  # v13: 提高门槛
            continue
        # v15: 硬阈值 — 动作太少的素材直接踢出，不靠软扣分（仅当有action数据时）
        action_val = r.get("dim_scores", {}).get("action")
        if action_val is not None and action_val < 8:
            continue
        # v18: 硬静态阈值 — activity极低的素材直接踢出
        activity_val = r.get("activity")
        if activity_val is not None and activity_val < 0.05:
            continue
        # v13: 文件名去重 — 同一源文件只允许一次
        basename = os.path.basename(r["path"])
        if basename in seen_basenames:
            continue
        base_pool.append(r)
        seen_basenames.add(basename)
    
    # v21: 素材池为空（跨批次去重用尽）→ 自动回退允许重复用
    if not base_pool:
        print(f"  ⚠ 素材池为空（{len(used_videos)}个已用素材用尽）→ 回退允许重复用")
        seen_paths = set()  # 清空去重限制
        base_pool = []
        for r in scored_results:
            if r["duration"] < min(clip_durations) + 0.3:
                continue
            if r.get("shot_type", "unknown") in ("wide", "unknown"):
                continue
            if r.get("score", 0) < 15:
                continue
            action_val = r.get("dim_scores", {}).get("action")
            if action_val is not None and action_val < 8:
                continue
            activity_val = r.get("activity")
            if activity_val is not None and activity_val < 0.05:
                continue
            base_pool.append(r)
        if not base_pool:
            print(f"  ⚠ 素材池仍为空！{len(scored_results)}个素材全被过滤")
            return {"plan": [], "shots": 0, "total_duration": 0, "used_pool_size": len(used_videos)}
        # v22 回退时全局打乱，确保1522条素材全部有机会被选中
        random.shuffle(base_pool)
        print(f"  ✓ 回退池大小: {len(base_pool)}，已全局打乱")
    
    # === 计算优选分 ===
    # v18: 静态惩罚 + 晃动惩罚 — 基于activity/stability直读，非仅依赖action维度
    def compute_pref_score(r):
        score = r.get("score", 0)
        dims = r.get("dim_scores", {})
        peak = r.get("peak_timestamp", 0)
        activity = r.get("activity", 0.5)
        stability_val = r.get("stability", 1.0)
        
        pref = score
        # 峰值加分/扣分：有峰值加分，无峰值扣分
        if peak >= 0.5:
            pref += 15
        elif peak >= 0.2:
            pref += 5
        else:
            pref -= 20  # v13: 无动作峰值的素材重扣（废片惩罚）
        
        # v19: 网格集中度静态惩罚 — real_activity区分真动作vs镜头晃
        # real_activity尺度: 真动作0.3-0.8, 镜头晃0.01-0.06
        if activity < 0.05:
            pref -= 40   # 纯静态（连镜头晃都没有） → 重扣
        elif activity < 0.08:
            pref -= 25   # 镜头晃为主 → 中扣
        elif activity < 0.12:
            pref -= 12   # 轻微动作
        elif activity < 0.20:
            pref -= 5    # 有动作但不强
        
        # v18: 晃动惩罚 — stability低=抖得厉害
        if stability_val < 0.3:
            pref -= 20   # 严重晃动 → 大幅度扣分
        elif stability_val < 0.5:
            pref -= 8    # 中度晃动
        
        # 动作加分
        action = dims.get("action", 0)
        if action >= 40: pref += 10
        elif action >= 25: pref += 5
        elif action < 10: pref -= 10  # v13: 无动作素材扣分
        # 质量加分
        quality = dims.get("quality", 0)
        if quality >= 30: pref += 5
        elif quality >= 15: pref += 2
        elif quality < 8: pref -= 5  # v13: 低质量素材扣分
        
        return pref
    
    # 按优选分排序
    base_pool.sort(key=compute_pref_score, reverse=True)
    
    # v22: 全局打乱候选池 — 避免批量生成时总是选同一批高分镜头
    # 排序后直接全局shuffle，让全部素材都有机会被选中
    random.shuffle(base_pool)
    
    # === 多样性约束：每种类型最多50% ===
    max_per_type = max(1, int(shots_needed * 0.50))
    
    selected = []
    sel_seen = set()
    sel_basenames = set()
    type_counts = {}
    
    for r in base_pool:
        if r["path"] in sel_seen:
            continue
        basename = os.path.basename(r["path"])
        if basename in sel_basenames:
            continue
        
        st = r.get("shot_type", "unknown")
        if type_counts.get(st, 0) >= max_per_type:
            continue
        
        # 时长必须够
        if r["duration"] < min_duration:
            continue
        
        selected.append(r)
        sel_seen.add(r["path"])
        sel_basenames.add(basename)
        type_counts[st] = type_counts.get(st, 0) + 1
        
        if len(selected) >= shots_needed:
            break
    
    # === v20 fallback: 第一轮不够 → 取消类型上限再选 ===
    if len(selected) < shots_needed:
        print(f"  [选镜] 首轮{len(selected)}/{shots_needed}不足，fallback解除类型上限")
        for r in base_pool:
            if r["path"] in sel_seen:
                continue
            basename = os.path.basename(r["path"])
            if basename in sel_basenames:
                continue
            if r["duration"] < min_duration:
                continue
            
            selected.append(r)
            sel_seen.add(r["path"])
            sel_basenames.add(basename)
            type_counts[r.get("shot_type", "unknown")] = type_counts.get(r.get("shot_type", "unknown"), 0) + 1
            
            if len(selected) >= shots_needed:
                break
    
    # === v20 fallback 2: 还不够 → 连 wide/unknown 也放进来 ===
    if len(selected) < shots_needed:
        print(f"  [选镜] 二轮仍不足{len(selected)}/{shots_needed}，fallback放宽类型（含wide/unknown）")
        wide_unknown_pool = [r for r in scored_results 
                            if r["path"] not in sel_seen 
                            and r["duration"] >= min_duration
                            and r.get("score", 0) >= 10  # 降低门槛
                            and os.path.basename(r["path"]) not in sel_basenames]
        wide_unknown_pool.sort(key=lambda r: r.get("score", 0), reverse=True)
        for r in wide_unknown_pool:
            if r["path"] in sel_seen:
                continue
            basename = os.path.basename(r["path"])
            if basename in sel_basenames:
                continue
            
            selected.append(r)
            sel_seen.add(r["path"])
            sel_basenames.add(basename)
            type_counts[r.get("shot_type", "unknown")] = type_counts.get(r.get("shot_type", "unknown"), 0) + 1
            
            if len(selected) >= shots_needed:
                break
    
    # 报告选镜质量
    type_dist = {st: c for st, c in type_counts.items()}
    avg_score = sum(r.get("score", 0) for r in selected) / max(len(selected), 1)
    avg_peak = sum(r.get("peak_timestamp", 0) for r in selected) / max(len(selected), 1)
    total_basenames = len(set(os.path.basename(r["path"]) for r in selected) if selected else [])
    print(f"  [选镜] 最终{len(selected)}/{shots_needed}个 | 均分{avg_score:.0f} | 峰值{avg_peak:.1f}s | 唯一文件{total_basenames} | 类型{type_dist}")
    
    if len(selected) < shots_needed:
        print(f"  ⚠ 最终选中{len(selected)}/{shots_needed}个镜头（素材不足）")
    
    # === 分配时长和位置 ===
    plan = []
    target_start = 0.0
    for i, r in enumerate(selected):
        dur = clip_durations[i] if i < len(clip_durations) else 0.9
        peak_ts = r.get("peak_timestamp", 0)
        shot_type = r.get("shot_type", "unknown")
        
        # v15: 类型驱动入点定位 — 不同素材类型高光位置不同
        if peak_ts >= 0.3 and r["duration"] > dur + 0.5:
            if shot_type == "eating_action":
                # 舀吃动作：动作发生在中下部，入点前移捕获完整舀起过程
                front_ratio = 0.50
            elif shot_type == "closeup":
                # 展示镜头：以峰值帧为中心
                front_ratio = 0.40
            else:
                # medium等：通用
                front_ratio = 0.35
            
            # 以峰值为参考，入点在峰值前 front_ratio*dur 处
            ideal_start = peak_ts - dur * front_ratio
            ideal_start = max(0.2, min(ideal_start, r["duration"] - dur - 0.2))
            # 峰值附近随机微调
            low = max(0.2, ideal_start - 0.3)
            high = min(r["duration"] - dur - 0.2, ideal_start + 0.3)
            src_start = random.uniform(low, high) if high > low else ideal_start
        elif r["duration"] > dur + 0.8:
            # 无峰值回退：跳过前15%和后10%，选中间75%
            safe_start = r["duration"] * 0.15
            safe_end = r["duration"] - dur - 0.3
            src_start = random.uniform(safe_start, safe_end) if safe_end > safe_start else 0.2
        else:
            src_start = 0.2 if r["duration"] > 0.5 else 0
        
        plan.append({
            "video_path": r["path"],
            "source_start": round(src_start, 4),
            "clip_duration": dur,
            "target_start": round(target_start, 4),
            "video_duration": r["duration"],
            "score": r["score"],
            "shot_type": r.get("shot_type", "unknown"),
            "peak_timestamp": r.get("peak_timestamp", 0)
        })
        used_videos.add(r["path"])
        target_start += dur
    
    return {
        "plan": plan,
        "shots": len(plan),
        "total_duration": round(sum(p["clip_duration"] for p in plan), 2),
        "used_pool_size": len(used_videos)
    }


# ============================================================
# Step 5: 渲染 — 剪映API草稿创建 + 导出
# ============================================================

def step_render(shot_plan, bgm_info, template_name, seq_num, output_dir, duration, grammar_engine=None):
    """用剪映API创建草稿并渲染导出
    
    Args:
        grammar_engine: EditGrammarEngine实例（可选）。为None时使用规则引擎回退。
    """
    # 动态导入剪映模块
    JY_SKILL_ROOT = r"D:\.workbuddy\skills\jianying-editor-skill"
    sys.path.insert(0, os.path.join(JY_SKILL_ROOT, "scripts"))
    
    try:
        from utils.env_setup import setup_env
        setup_env()
    except:
        pass
    
    from jy_wrapper import JyProject
    import pyJianYingDraft as draft
    from pyJianYingDraft import TrackType, JianyingController
    from pyJianYingDraft.jianying_controller import ExportResolution, ExportFramerate
    
    # 草稿名直接用数字编号（如 5, 6, 7...）
    draft_name = str(seq_num)
    
    # 创建草稿
    project = JyProject(draft_name, width=2160, height=3840, overwrite=True)
    project.script.add_track(TrackType.video, "VideoTrack")
    project.script.add_track(TrackType.audio, "BGM")
    project.script.add_track(TrackType.effect, "EffectTrack")
    project.script.add_track(TrackType.filter, "FilterTrack")
    
    # 添加视频片段（不传target_start，让clip自动首尾相接，无缝对齐）
    ok = 0
    fail_reasons = []
    for i, shot in enumerate(shot_plan):
        dur = shot["clip_duration"]
        seg = None
        try:
            seg = project.add_clip(
                media_path=shot["video_path"],
                source_start=shot["source_start"],
                duration=dur,
                track_name="VideoTrack"
            )
        except Exception as e:
            fail_reasons.append(f"{os.path.basename(shot['video_path'])[:20]}:{str(e)[:40]}")
        
        # 失败重试：换一个源起点
        if seg is None:
            try:
                retry_start = max(0.3, shot["video_duration"] - dur - 0.5)
                if retry_start > 0.5 and abs(retry_start - shot["source_start"]) > 0.3:
                    seg = project.add_clip(
                        media_path=shot["video_path"],
                        source_start=round(retry_start, 2),
                        duration=dur,
                        track_name="VideoTrack"
                    )
            except Exception as e2:
                fail_reasons.append(f"retry:{os.path.basename(shot['video_path'])[:20]}:{str(e2)[:40]}")
        
        if seg is not None:
            ok += 1
        else:
            fail_reasons.append(f"{os.path.basename(shot['video_path'])[:30]}")
    
    if fail_reasons:
        print(f"  ⚠ {len(fail_reasons)}个片段添加失败: {fail_reasons[:3]}{'...' if len(fail_reasons)>3 else ''}")
    
    # BGM
    try:
        project.add_audio_safe(bgm_info["bgm_path"], start_time=0.0, duration=duration, track_name="BGM")
        bgm_ok = True
    except Exception:
        bgm_ok = False
    
    # ---- 使用EditGrammar引擎替代硬编码参数 ----
    grammar_stats = {}
    grammar_errors = []
    if grammar_engine is not None:
        try:
            template = {"name": template_name, "duration": duration}
            grammar_json = grammar_engine.generate_grammar(template, shot_plan, bgm_info, duration)
            grammar_stats = grammar_engine.execute(project, grammar_json, shot_plan, bgm_info, duration)
            grammar_errors = grammar_stats.get("errors", [])
            if grammar_errors:
                print(f"  [Grammar] 警告 ({len(grammar_errors)}项):")
                for err in grammar_errors[:3]:
                    print(f"    - {err}")
                if len(grammar_errors) > 3:
                    print(f"    ... 还有{len(grammar_errors)-3}项")
        except Exception as e:
            print(f"  [Grammar] 引擎执行失败，回退硬编码: {e}")
            grammar_stats = _render_hardcoded_fallback(project, duration)
    else:
        grammar_stats = _render_hardcoded_fallback(project, duration)
    
    filter_ok = grammar_stats.get("filter_ok", False)
    effect_ok = grammar_stats.get("effect_ok", False)
    transitions_added = grammar_stats.get("transitions_added", 0)
    keyframes_added = grammar_stats.get("keyframes_added", 0)
    intros_added = grammar_stats.get("intros_added", 0)
    texts_added = grammar_stats.get("texts_added", 0)
    
    # 保存
    result = project.save()
    
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"{seq_num}.mp4")
    
    # 等待剪映文件监控检测到草稿变更（轮询验证）
    # v22: 自动重启剪映 + 延长等待到30秒，超时后触发F5刷新再重试
    print(f"  ⏳ 等待剪映检测草稿「{draft_name}」...", end="", flush=True)
    ctrl = JianyingController()
    detected = False
    for wait_s in range(1, 31):
        time.sleep(1)
        if wait_s % 5 == 0:
            print(f"{wait_s}s", end="", flush=True)
        else:
            print(".", end="", flush=True)
        try:
            ctrl.get_window()
            draft_ctrl = ctrl.app.TextControl(
                searchDepth=3,
                Compare=lambda c, d: d in (2, 3) and str(c.GetPropertyValue(30159)).lower() == f"homepagedrafttitle:{draft_name}"
            )
            if draft_ctrl.Exists(0):
                detected = True
                print(f" ✓ ({wait_s}s)")
                break
        except Exception:
            pass
    
    # 30秒还没检测到 → 触发F5刷新剪映首页
    if not detected:
        print(f" 超时，触发F5刷新...", end="", flush=True)
        try:
            ctrl.get_window()
            ctrl.app.SendKeys("{F5}")
            time.sleep(3)
            for wait_s in range(1, 11):
                time.sleep(1)
                print(".", end="", flush=True)
                try:
                    ctrl.get_window()
                    draft_ctrl = ctrl.app.TextControl(
                        searchDepth=3,
                        Compare=lambda c, d: d in (2, 3) and str(c.GetPropertyValue(30159)).lower() == f"homepagedrafttitle:{draft_name}"
                    )
                    if draft_ctrl.Exists(0):
                        detected = True
                        print(f" ✓ ({wait_s}s)")
                        break
                except Exception:
                    pass
        except Exception as e:
            print(f" 刷新失败: {e}")
    
    if not detected:
        print(f" 最终超时")
    
    # 导出（带UI刷新重试）
    export_ok = False
    export_error = None
    max_retries = 3
    
    for attempt in range(1, max_retries + 1):
        try:
            if attempt > 1:
                print(f"  🔄 导出重试 {attempt}/{max_retries}...")
                # 重试前触发F5刷新
                try:
                    ctrl.get_window()
                    ctrl.app.SendKeys("{F5}")
                    time.sleep(3)
                except:
                    pass
            
            ctrl.get_window()
            # 确保在首页
            if ctrl.app_status == "edit":
                ctrl.switch_to_home()
                time.sleep(1)
                ctrl.get_window()
            elif ctrl.app_status == "pre_export":
                ctrl.app.SendKeys("{Esc}")
                time.sleep(1)
                ctrl.get_window()
            
            ctrl.export_draft(
                draft_name=draft_name,
                output_path=output_file,
                resolution=ExportResolution.RES_4K,
                framerate=ExportFramerate.FR_60,
            )
            
            if os.path.exists(output_file) and os.path.getsize(output_file) > 1000:
                export_ok = True
                break
            else:
                export_error = "导出完成但文件不存在或过小"
                print(f"  ⚠ {export_error}")
                time.sleep(2)
        except Exception as e:
            err_msg = str(e)[:120]
            export_error = err_msg
            print(f"  ⚠ 导出失败: {err_msg}")
            time.sleep(2)
    
    if not export_ok and export_error:
        print(f"  ❌ 导出最终失败: {export_error}")
    
    return {
        "draft_name": draft_name,
        "output_file": output_file,
        "clips_added": ok,
        "clips_total": len(shot_plan),
        "bgm_ok": bgm_ok,
        "effect_ok": effect_ok,
        "filter_ok": filter_ok,
        "transitions_added": transitions_added,
        "keyframes_added": keyframes_added,
        "intros_added": intros_added,
        "texts_added": texts_added,
        "export_ok": export_ok,
        "save_result": str(result),
        "grammar_stats": grammar_stats
    }


def _render_hardcoded_fallback(project, duration):
    """硬编码回退：当语法引擎不可用时的默认渲染"""
    from pyJianYingDraft import FilterType
    from pyJianYingDraft.metadata.video_scene_effect import VideoSceneEffectType
    from pyJianYingDraft.time_util import Timerange
    
    stats = {"filter_ok": False, "effect_ok": False, "transitions_added": 0,
             "keyframes_added": 0, "intros_added": 0, "texts_added": 0}
    
    try:
        tr = Timerange(0, int(duration * 1_000_000))
        project.script.add_filter(FilterType.食光II, tr, track_name="FilterTrack", intensity=40.0)
        stats["filter_ok"] = True
    except Exception:
        pass
    
    try:
        tr = Timerange(0, int(duration * 1_000_000))
        project.script.add_effect(VideoSceneEffectType.电影感画幅, tr, track_name="EffectTrack")
        stats["effect_ok"] = True
    except Exception:
        pass
    
    return stats


# ============================================================
# Step 6: 质检 — 文件验证 + 报告
# ============================================================

def step_qc(render_result):
    """质量检查"""
    issues = []
    
    output_file = render_result.get("output_file", "")
    
    # 检查文件存在
    if os.path.exists(output_file):
        size_mb = os.path.getsize(output_file) / (1024 * 1024)
        if size_mb < 1:
            issues.append(f"输出文件过小({size_mb:.1f}MB)")
    else:
        issues.append("输出文件不存在")
    
    # 检查片段添加率
    ok = render_result.get("clips_added", 0)
    total = render_result.get("clips_total", 0)
    if total > 0 and ok / total < 0.7:
        issues.append(f"片段添加率过低({ok}/{total})")
    
    # 检查BGM/特效/滤镜
    if not render_result.get("bgm_ok"):
        issues.append("BGM添加失败")
    if not render_result.get("effect_ok"):
        issues.append("特效添加失败")
    if not render_result.get("filter_ok"):
        issues.append("滤镜添加失败")
    if not render_result.get("export_ok"):
        issues.append("导出失败")
    
    return {
        "passed": len(issues) == 0,
        "issues": issues,
        "output_file": output_file,
        "file_exists": os.path.exists(output_file),
        "file_size_mb": round(os.path.getsize(output_file) / (1024 * 1024), 1) if os.path.exists(output_file) else 0
    }


# ============================================================
# Pipeline编排器
# ============================================================

class VideoPipeline:
    """6步流水线编排器"""
    
    def __init__(self, material_dir, bgm_dir, output_dir, cache_file, grammar_engine=None):
        self.material_dir = material_dir
        self.bgm_dir = bgm_dir
        self.output_dir = output_dir
        self.cache_file = cache_file
        self.grammar_engine = grammar_engine  # EditGrammarEngine 实例（可选）
        self.cache_data = self._load_cache()
        self.used_videos = self._load_used()  # v19: 跨批次持久化
        self.log = []
    
    def _load_cache(self):
        try:
            if os.path.exists(self.cache_file):
                with open(self.cache_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except:
            pass
        return {}
    
    def _save_cache(self):
        try:
            os.makedirs(os.path.dirname(self.cache_file), exist_ok=True)
            with open(self.cache_file, 'w', encoding='utf-8') as f:
                json.dump(self.cache_data, f, ensure_ascii=False, indent=2)
        except:
            pass
    
    def _load_used(self):
        """v19: 加载跨批次已用素材列表，防止重复"""
        try:
            if os.path.exists(USED_FILE):
                with open(USED_FILE, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                paths = data.get("paths", [])
                print(f"  [去重] 跨批次已用素材: {len(paths)}个")
                return set(paths)
        except:
            pass
        return set()
    
    def _save_used(self):
        """v19: 持久化已用素材列表"""
        try:
            os.makedirs(os.path.dirname(USED_FILE), exist_ok=True)
            with open(USED_FILE, 'w', encoding='utf-8') as f:
                json.dump({"paths": list(self.used_videos), "updated": datetime.now().isoformat()}, f, ensure_ascii=False)
        except:
            pass
    
    def _update_cache(self, scored_results):
        """把新分析结果写入缓存"""
        for r in scored_results:
            if not r.get("cached") and r.get("_raw"):
                key = os.path.abspath(r["path"]).lower()
                raw = r["_raw"]
                self.cache_data[key] = {
                    "path": r["path"],
                    "motion_score": raw.get("score", 0),
                    "motion": raw.get("motion", 0),
                    "appetite": raw.get("appetite", 0),
                    "peak_score": raw.get("peak_score", 0),
                    "peak_timestamp": raw.get("peak_timestamp", 0),
                    "shot_type": raw.get("shot_type", "unknown"),
                    "duration": raw.get("duration", 0),
                    "yolo_score": raw.get("yolo_score", 0),
                    "food_presence": raw.get("food_presence", 0),
                    "hand_presence": raw.get("hand_presence", 0),
                    "spoon_presence": raw.get("spoon_presence", 0),
                    "bowl_presence": raw.get("bowl_presence", 0),
                    "eating_likelihood": raw.get("eating_likelihood", 0),
                    "activity": raw.get("activity", 0),
                    "stability": raw.get("stability", 1.0),
                    "raw_activity": raw.get("raw_activity", raw.get("activity", 0)),
                    "concentration": raw.get("concentration", 0.5),
                    "dim_scores": raw.get("dim_scores", {}),
                    "multi_dim_total": raw.get("dim_scores", {}).get("total", 0),
                    "cached_at": datetime.now().isoformat()
                }
        self._save_cache()
    
    def run(self, template, seq_num):
        """执行完整6步流水线"""
        template_name = template["name"]
        duration = template["duration"]
        log_entry = {"template": template_name, "seq": seq_num, "steps": {}}
        
        # Step 1: 素材预处理
        t0 = time.time()
        pre_result = step_preprocess(self.material_dir)
        log_entry["steps"]["preprocess"] = {"time": round(time.time()-t0, 1), **pre_result}
        if pre_result["valid"] < 6:
            log_entry["error"] = f"素材不足({pre_result['valid']}<6)"
            self.log.append(log_entry)
            return None
        
        # Step 2: 素材评分
        t0 = time.time()
        score_result = step_score(pre_result["videos"], self.cache_data)
        self._update_cache(score_result["results"])
        log_entry["steps"]["score"] = {"time": round(time.time()-t0, 1),
                                        "total": score_result["total"],
                                        "new_analyzed": score_result["new_analyzed"]}
        
        # Step 3: BGM匹配
        t0 = time.time()
        bgm_result = step_bgm_match(self.bgm_dir, duration)
        log_entry["steps"]["bgm"] = {"time": round(time.time()-t0, 1),
                                      "bgm": bgm_result.get("bgm_name", "N/A"),
                                      "beat_method": bgm_result.get("beat_plan", {}).get("method", "N/A"),
                                      "shots": bgm_result.get("beat_plan", {}).get("shot_count", 0)}
        if bgm_result.get("error"):
            log_entry["error"] = bgm_result["error"]
            self.log.append(log_entry)
            return None
        
        # Step 4: 镜头方案
        t0 = time.time()
        plan_result = step_plan(score_result["results"], bgm_result["beat_plan"], self.used_videos)
        log_entry["steps"]["plan"] = {"time": round(time.time()-t0, 1),
                                       "shots": plan_result["shots"],
                                       "pool_size": plan_result["used_pool_size"]}
        
        # Step 5: 渲染
        t0 = time.time()
        render_result = step_render(plan_result["plan"], bgm_result, template_name, seq_num, 
                                    self.output_dir, duration, grammar_engine=self.grammar_engine)
        log_entry["steps"]["render"] = {"time": round(time.time()-t0, 1),
                                         "clips": f"{render_result['clips_added']}/{render_result['clips_total']}",
                                         "transitions": render_result.get("transitions_added", 0),
                                         "keyframes": render_result.get("keyframes_added", 0),
                                         "intros": render_result.get("intros_added", 0),
                                         "texts": render_result.get("texts_added", 0),
                                         "export": render_result["export_ok"]}
        
        # Step 6: 质检
        t0 = time.time()
        qc_result = step_qc(render_result)
        log_entry["steps"]["qc"] = {"time": round(time.time()-t0, 1),
                                     "passed": qc_result["passed"],
                                     "issues": qc_result["issues"]}
        
        log_entry["total_time"] = round(sum(s.get("time", 0) for s in log_entry["steps"].values()), 1)
        self.log.append(log_entry)
        self._save_used()  # v19: 每条跑完就持久化已用素材
        
        return {"render": render_result, "qc": qc_result}
    
    def print_report(self):
        """打印Pipeline执行报告"""
        if not self.log:
            print("无执行记录")
            return
        
        print("\n" + "=" * 70)
        print("Pipeline执行报告")
        print("=" * 70)
        
        total_time = 0
        success = 0
        
        for entry in self.log:
            status = "✅" if "error" not in entry else "❌"
            print(f"\n{status} #{entry['seq']} 「{entry['template']}」({entry.get('total_time', 0)}s)")
            
            if "error" in entry:
                print(f"   错误: {entry['error']}")
                continue
            
            success += 1
            for step_name, step_data in entry.get("steps", {}).items():
                step_label = {"preprocess": "素材收集", "score": "素材评分", "bgm": "BGM匹配",
                              "plan": "镜头方案", "render": "渲染导出", "qc": "质检"}
                extra = ""
                if step_name == "render":
                    extra = f" 转场{step_data.get('transitions', 0)} KF{step_data.get('keyframes', 0)} 动画{step_data.get('intros', 0)}"
                print(f"   [{step_label.get(step_name, step_name)}] {step_data.get('time', 0)}s{extra}")
            
            total_time += entry.get("total_time", 0)
        
        print(f"\n{'='*70}")
        print(f"总计: {len(self.log)}条视频, 成功{success}条, 总耗时{total_time:.0f}s")
        print("=" * 70)


# ============================================================
# 工具函数
# ============================================================

def get_next_number(output_dir):
    """扫描输出目录已有编号文件，找下一个可用编号"""
    if not os.path.exists(output_dir):
        return 1
    nums = []
    for f in os.listdir(output_dir):
        name = os.path.splitext(f)[0]
        try:
            n = int(name)
            if 1 <= n <= 10000:
                nums.append(n)
        except:
            pass
    if not nums:
        return 1
    used = set(nums)
    for n in range(1, max(used) + 2):
        if n not in used:
            return n


# ============================================================
# 剪映进程管理 — 防止内存膨胀导致卡死
# ============================================================

JY_DRAFT_DIR = os.path.expandvars(r"%LOCALAPPDATA%\JianyingPro\User Data\Projects\com.lveditor.draft")
JY_EXE_PATH = r"D:\软件\剪映\JianyingPro\JianyingPro.exe"

def cleanup_locked_files():
    """清理剪映草稿目录中的.locked文件"""
    if not os.path.exists(JY_DRAFT_DIR):
        return 0
    count = 0
    for name in os.listdir(JY_DRAFT_DIR):
        if name.endswith(".locked"):
            try:
                locked_path = os.path.join(JY_DRAFT_DIR, name)
                os.remove(locked_path)
                count += 1
            except Exception:
                pass
    return count

def is_jianying_running():
    """检查剪映进程是否在运行"""
    import subprocess
    try:
        result = subprocess.run(
            ["tasklist", "/FI", "IMAGENAME eq JianyingPro.exe", "/FO", "CSV", "/NH"],
            capture_output=True, text=True, timeout=10
        )
        return "JianyingPro.exe" in result.stdout
    except Exception:
        return False

def launch_jianying():
    """启动剪映进程"""
    import subprocess
    if not os.path.exists(JY_EXE_PATH):
        print(f"  ⚠ 剪映路径不存在: {JY_EXE_PATH}", flush=True)
        return False
    try:
        subprocess.Popen([JY_EXE_PATH], creationflags=0x00000008)  # DETACHED_PROCESS
        return True
    except Exception as e:
        print(f"  ⚠ 启动剪映失败: {e}", flush=True)
        return False

def wait_for_jianying(timeout=60):
    """等待剪映窗口就绪（进程启动 + 初始化等待）"""
    print(f"  ⏳ 等待剪映启动...", end="", flush=True)
    for i in range(timeout):
        time.sleep(1)
        if i % 5 == 4:
            print(f"{i+1}s", end="", flush=True)
        else:
            print(".", end="", flush=True)
        if is_jianying_running():
            # 进程已启动，额外等待初始化（剪映UI加载需要时间）
            time.sleep(20)
            print(f" ✓", flush=True)
            return True
    print(f" 超时", flush=True)
    return False

def ensure_jianying_running():
    """确保剪映在运行，没运行则启动并等待"""
    if is_jianying_running():
        return True
    print(f"\n  ⚠ 剪映未运行，正在启动...", flush=True)
    if launch_jianying():
        return wait_for_jianying(timeout=60)
    return False


# ============================================================
# 主入口
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Pipeline标准化6步流水线 — 批量生成短视频")
    parser.add_argument('-n', '--count', type=int, default=5, help='生成视频数量（默认5）')
    parser.add_argument('--force-scan', action='store_true', help='强制重新扫描全部素材')
    parser.add_argument('--material-dir', default=MATERIAL_DIR)
    parser.add_argument('--bgm-dir', default=BGM_DIR)
    parser.add_argument('--output-dir', default=OUTPUT_DIR)
    parser.add_argument('--cache-file', default=CACHE_FILE)
    parser.add_argument('--step', choices=['preprocess','score','bgm','plan','render','qc'],
                        help='单步调试模式')
    parser.add_argument('--step-only', action='store_true', help='与--step联用，仅运行指定步')
    parser.add_argument('--optimize', action='store_true', default=True,
                        help='启用自我优化分析（默认开启）')
    parser.add_argument('--no-optimize', dest='optimize', action='store_false',
                        help='禁用自我优化分析')
    parser.add_argument('--history-file', default=None,
                        help='优化历史文件路径（默认 .optimizer_history.json）')
    parser.add_argument('--start-num', type=int, default=0,
                        help='强制起始编号（0=自动扫描，非0=从指定编号开始）')
    args = parser.parse_args()

    BATCH_COUNT = args.count
    selected_templates = random.choices(ALL_TEMPLATES, k=BATCH_COUNT)

    print("=" * 60)
    print("Pipeline标准化引擎 v2.0 — 6步流水线 + LLM语法引擎 + 自我优化")
    print("4K竖屏 | 60fps | YOLOv8语义检测 | 8维评分 | 自适应节拍 | EditGrammar | Optimizer")
    print("=" * 60)
    print(f"本次随机选中模板: {[t['name'] for t in selected_templates]}")
    print(f"素材目录: {args.material_dir}")
    print(f"BGM目录: {args.bgm_dir}")
    print(f"输出目录: {args.output_dir}")
    print(f"自我优化: {'开启' if args.optimize else '关闭'}")
    print()

    # 单步调试模式
    if args.step and args.step_only:
        _debug_single_step(args, selected_templates[0])
        return

    # 创建EditGrammar引擎
    try:
        from edit_grammar import EditGrammarEngine, FOOD_FILTERS, GLOBAL_EFFECTS
        grammar_engine = EditGrammarEngine()
        print(f"语法引擎: 规则引擎已激活（{len(FOOD_FILTERS)}滤镜/{len(GLOBAL_EFFECTS)}特效）")
    except Exception:
        grammar_engine = None
        print(f"语法引擎: 规则引擎模式（LLM未配置）")

    # 创建Pipeline
    pipeline = VideoPipeline(args.material_dir, args.bgm_dir, args.output_dir, args.cache_file, grammar_engine=grammar_engine)

    start_num = args.start_num if args.start_num > 0 else get_next_number(args.output_dir)
    print(f"起始编号: {start_num}（{'强制指定' if args.start_num > 0 else f'已使用1~{start_num-1}'}）\n")

    # ===== 剪映 5.9 破解版内存泄漏防护（每 80 条自动重启）=====
    JIANYING_PATH = r"D:\软件\M008-剪映 5.9 破解版（window 版）\剪映 5.9 破解版（window 版）\JianyingPro_5.9（windows 版本）\JianyingPro_5.9（windows 版本）\JianyingPro.exe"
    
    def restart_jianying_every_80(batch_idx, batch_size=80):
        if batch_idx % batch_size != 0: return
        import psutil, time, subprocess
        print(f"[健康检查] 已生成{batch_idx}条，重启剪映...");
        for p in psutil.process_iter(["pid","name"]):
            if "JianyingPro" in p.info["name"]: p.terminate()
        time.sleep(3); subprocess.Popen([JIANYING_PATH]); time.sleep(8)
        print(f"✅ 剪映已重启 (第{batch_idx//batch_size}次)\\n")
    
    for idx, template in enumerate(selected_templates, 1):
        seq_num = start_num + idx - 1
        restart_jianying_every_80(idx)
        print(f"\n{'='*60}")
        print(f"视频 {idx}/{BATCH_COUNT}: 编号「{seq_num}」模板「{template['name']}」{template['duration']}秒")
        print('='*60)

        # ---- 进程健康管理 ----
        # 确保剪映在运行（首次启动或崩溃后自动恢复）
        if not ensure_jianying_running():
            print("  ❌ 剪映无法启动，跳过本条")
            continue

        # 每条开始前清理.locked文件
        cleaned = cleanup_locked_files()
        if cleaned > 0:
            print(f"  🧹 预清理 {cleaned} 个.locked文件")

        try:
            result = pipeline.run(template, seq_num)
            if result:
                qc = result["qc"]
                render = result["render"]
                if render.get("export_ok"):
                    print(f"  导出: ✅ ({render.get('file_size_mb', '?')}MB)")
                else:
                    print(f"  导出: ❌ {render.get('export_error', '未知错误')}")
                print(f"  质检: {'✅通过' if qc['passed'] else '❌' + ','.join(qc['issues'])}")
            else:
                print("  ❌ 流水线执行失败")
        except Exception as e:
            print(f"  ❌ 异常退出: {e}")
            import traceback
            traceback.print_exc()
            # 异常时也清理锁文件，防止下一条卡死
            cleanup_locked_files()
            continue

    # 报告
    pipeline.print_report()
    
    # 自我优化分析
    if args.optimize:
        try:
            from self_optimizer import Optimizer
            opt = Optimizer(history_file=args.history_file)
            opt.collect_from_pipeline_log(pipeline.log)
            
            report = opt.analyze()
            print("\n" + report)
            
            suggestions = opt.suggest()
            if suggestions:
                print("\n📈 调优建议:")
                for s in suggestions:
                    icon = {"high": "🔴", "medium": "🟡", "low": "🟢"}.get(s.get("confidence", "low"), "⚪")
                    print(f"  {icon} [{s['category']}] {s['recommendation']}")
            
            best = opt.get_best_config()
            if best:
                print(f"\n⭐ 历史最佳配置: 滤镜={best.get('recommended_filter', 'N/A')} "
                      f"强度={best.get('recommended_intensity', 'N/A')} "
                      f"特效={best.get('recommended_effect', 'N/A')}")
        except ImportError:
            print("\n[Optimizer] self_optimizer模块不可用，跳过优化分析")


def _debug_single_step(args, template):
    """单步调试：只运行指定步骤"""
    step = args.step
    duration = template["duration"]
    print(f"🐛 单步调试模式: Step={step}, 模板={template['name']}({duration}s)\n")

    if step == "preprocess":
        result = step_preprocess(args.material_dir)
        print(f"  素材总数: {result['total']}, 有效: {result['valid']}, 无效: {result['invalid']}")
        print(f"  前5个: {[os.path.basename(v['path'])[:30] for v in result['videos'][:5]]}")

    elif step == "score":
        pre = step_preprocess(args.material_dir)
        result = step_score(pre["videos"], {})
        print(f"  评分完成: {result['total']}个, 新分析: {result['new_analyzed']}")
        for r in result["results"][:5]:
            print(f"  score={r['score']:.1f} type={r['shot_type']} {os.path.basename(r['path'])[:30]}")
            print(f"    dims={r.get('dim_scores', {})}")

    elif step == "bgm":
        result = step_bgm_match(args.bgm_dir, duration)
        print(f"  BGM: {result.get('bgm_name')}")
        bp = result.get('beat_plan', {})
        print(f"  节拍: {bp.get('method')}, BPM={bp.get('bpm')}, {bp.get('shot_count')}镜头")
        print(f"  各镜头: {bp.get('clip_durations')}")

    elif step == "plan":
        pre = step_preprocess(args.material_dir)
        scored = step_score(pre["videos"], {})
        bgm = step_bgm_match(args.bgm_dir, duration)
        result = step_plan(scored["results"], bgm["beat_plan"])
        print(f"  选中: {result['shots']}个镜头, 总时长: {result['total_duration']}s")
        for p in result["plan"]:
            print(f"  {os.path.basename(p['video_path'])[:30]} score={p['score']} dur={p['clip_duration']}s")

    elif step == "render":
        pre = step_preprocess(args.material_dir)
        scored = step_score(pre["videos"], {})
        bgm = step_bgm_match(args.bgm_dir, duration)
        plan = step_plan(scored["results"], bgm["beat_plan"])
        result = step_render(plan["plan"], bgm, template["name"], 999, args.output_dir, duration)
        print(f"  草稿: {result['draft_name']}, 片段: {result['clips_added']}/{result['clips_total']}")
        print(f"  转场: {result.get('transitions_added', 0)}, 关键帧: {result.get('keyframes_added', 0)}")
        print(f"  入场动画: {result.get('intros_added', 0)}, 文字: {result.get('texts_added', 0)}")
        print(f"  滤镜: {'✅' if result['filter_ok'] else '❌'}, 特效: {'✅' if result['effect_ok'] else '❌'}")
        print(f"  导出: {'✅' if result['export_ok'] else '❌'}")

    elif step == "qc":
        pre = step_preprocess(args.material_dir)
        scored = step_score(pre["videos"], {})
        bgm = step_bgm_match(args.bgm_dir, duration)
        plan = step_plan(scored["results"], bgm["beat_plan"])
        render = step_render(plan["plan"], bgm, template["name"], 999, args.output_dir, duration)
        result = step_qc(render)
        print(f"  质检: {'✅通过' if result['passed'] else '❌'}")
        print(f"  转场:{render.get('transitions_added',0)} 关键帧:{render.get('keyframes_added',0)} "
              f"入场动画:{render.get('intros_added',0)} 文字:{render.get('texts_added',0)}")
        if result["issues"]:
            for i in result["issues"]:
                print(f"    - {i}")


if __name__ == "__main__":
    main()

