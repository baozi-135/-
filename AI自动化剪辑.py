"""
AI自动化剪辑 v7.0 — YOLOv8语义检测版
- YOLOv8目标检测替代硬编码颜色分析（bowl/spoon/hand/food）
- OpenCV替代ffmpeg/ffprobe（零外部依赖）
- 自适应节拍混步：2拍/3拍动态交替
- 单镜头0.7~1.1秒，匹配模板节奏
- 从全部12个模板随机选N个
- 素材去重 + 全量扫描缓存
"""
import os, sys, glob, random, time, json, argparse
from datetime import datetime

# ===== 命令行参数 =====
parser = argparse.ArgumentParser()
parser.add_argument('-n', '--count', type=int, default=5, help='生成视频数量（默认5）')
parser.add_argument('--force-scan', action='store_true', help='强制重新扫描全部素材')
parser.add_argument('--start-index', type=int, default=None, help='起始编号（默认自动检测输出目录空位）')
args = parser.parse_args()
BATCH_COUNT = args.count

try:
    import numpy as np
except ImportError:
    np = None

# 新增：AI分析模块（YOLOv8语义检测 + OpenCV）
CWD = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, CWD)
from ai_analyzer import analyze_shot, get_video_info

JY_SKILL_ROOT = r"D:\.workbuddy\skills\jianying-editor-skill"
sys.path.insert(0, os.path.join(JY_SKILL_ROOT, "scripts"))
from utils.env_setup import setup_env
setup_env()

from jy_wrapper import JyProject
import pyJianYingDraft as draft
from pyJianYingDraft import FilterType, TrackType, JianyingController
from pyJianYingDraft.metadata.video_scene_effect import VideoSceneEffectType
from pyJianYingDraft.time_util import Timerange

try:
    import librosa
    HAS_LIBROSA = True
except ImportError:
    HAS_LIBROSA = False
    print("⚠️ librosa未安装")

# ===== 路径配置 =====
MATERIAL_DIR = r"E:/111"
BGM_DIR      = r"E:/音频"
OUTPUT_DIR   = r"D:/混剪成品/auto"
LEARN_DIR    = r"D:\.workbuddy\skills\ai-video-auto-clip\learning"
CACHE_FILE   = os.path.join(LEARN_DIR, "material_cache.json")

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

TEMPLATES = random.choices(ALL_TEMPLATES, k=BATCH_COUNT)
print(f"本次随机选中: {[t['name'] for t in TEMPLATES]}")


# ===== 工具函数 =====

def get_video_duration(vp):
    """获取视频时长（用OpenCV，替换ffprobe）"""
    return get_video_info(vp).get("duration", 0)

# analyze_shot 直接从 ai_analyzer 导入，无需在此重复定义

def load_json(path, default=None):
    if default is None:
        default = {}
    try:
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
    except:
        pass
    return default

def save_json(path, data):
    try:
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return True
    except:
        return False


# ===== MaterialCache（v7.0 YOLO增强版）=====

class MaterialCache:
    """素材分析缓存 - 支持YOLOv8语义检测字段"""
    def __init__(self):
        self.data = load_json(CACHE_FILE, {})

    def get(self, vp):
        key = os.path.abspath(vp).lower()
        return self.data.get(key)

    def set(self, vp, analysis, duration):
        key = os.path.abspath(vp).lower()
        self.data[key] = {
            "path": vp,
            "motion_score": analysis.get("score", 0),
            "motion": analysis.get("motion", 0),
            "appetite": analysis.get("appetite", 0),
            "peak_score": analysis.get("peak_score", 0),
            "peak_timestamp": analysis.get("peak_timestamp", 0),
            "shot_type": analysis.get("shot_type", "unknown"),
            "duration": duration,
            # v7.0 YOLO新字段
            "yolo_score": analysis.get("yolo_score", 0),
            "food_presence": analysis.get("food_presence", 0),
            "hand_presence": analysis.get("hand_presence", 0),
            "spoon_presence": analysis.get("spoon_presence", 0),
            "bowl_presence": analysis.get("bowl_presence", 0),
            "eating_likelihood": analysis.get("eating_likelihood", 0),
            # v7.0 多维度评分
            "dim_scores": analysis.get("dim_scores", {}),
            "multi_dim_total": analysis.get("dim_scores", {}).get("total", 0),
            "cached_at": datetime.now().isoformat()
        }

    def save(self):
        save_json(CACHE_FILE, self.data)

    def scan_and_cache(self, videos, force=False):
        """预扫描未缓存素材"""
        if force:
            uncached = videos
        else:
            cached = set(os.path.abspath(v).lower() for v in self.data.keys())
            uncached = [v for v in videos if os.path.abspath(v).lower() not in cached]

        total = len(uncached)
        if total == 0:
            print(f"  ✅ 所有素材已缓存(共{len(self.data)}个)")
            return

        print(f"  📊 开始扫描 {total} 个新素材（首次扫描，之后复用）...")
        for i, v in enumerate(uncached):
            if i % 20 == 0:
                print(f"     进度: {i}/{total}...")
            analysis = analyze_shot(v)
            self.set(v, analysis, get_video_duration(v))
        self.save()
        print(f"  ✅ 缓存完成！共 {len(self.data)} 个素材已分析")


# ===== 素材收集（跳过扫描，只收集路径）=====

def collect_videos():
    """收集素材路径，不分析（复用缓存）"""
    videos = []
    for ext in ['*.mov', '*.mp4', '*.MOV', '*.MP4']:
        videos += glob.glob(os.path.join(MATERIAL_DIR, "**", ext), recursive=True)
    videos = list({v.lower(): v for v in videos if os.path.exists(v)}.values())
    return videos


# ===== 自适应节拍混步（v6.4核心逻辑）=====

def calc_shots_by_adaptive_beat(bgm_path, duration):
    """
    v6.4 自适应节拍混步：
    - 快节奏段落 → 2拍一镜（镜头更短更动感）
    - 慢节奏段落 → 3拍一镜（镜头更长更从容）
    - 单镜头限制0.7~1.1秒
    """
    beat_times = []
    shot_count = 6

    if bgm_path and os.path.exists(bgm_path) and HAS_LIBROSA:
        try:
            y, sr = librosa.load(bgm_path, sr=None, duration=duration)
            if len(y) == 0:
                raise ValueError("音频信号为空")

            tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
            all_beats = [float(b) for b in librosa.frames_to_time(beat_frames, sr=sr) if 0 < b <= duration]
            try:
                bpm = float(np.asarray(tempo).flat[0]) if tempo is not None else 100
            except Exception:
                bpm = 100

            if all_beats:
                # 估算每段能量
                onset_env = librosa.onset.onset_strength(y=y, sr=sr)

                # 把BGM分成5段，评估每段节奏
                seg_len = max(1, len(all_beats) // 5)
                seg_steps = []
                for i in range(5):
                    start = i * seg_len
                    end = start + seg_len if i < 4 else len(all_beats)
                    seg_beats = all_beats[start:end]
                    if len(seg_beats) < 2:
                        seg_steps.append(2)
                        continue
                    intervals = [seg_beats[j+1] - seg_beats[j] for j in range(len(seg_beats) - 1)]
                    if intervals:
                        var = np.std(intervals) / (np.mean(intervals) + 0.001)
                        # 变化大=节奏感强 -> 2拍；变化小=平稳 -> 3拍
                        seg_steps.append(2 if var > 0.15 else 3)
                    else:
                        seg_steps.append(2)

                # 按段采样
                sampled = []
                for i in range(5):
                    start = i * seg_len
                    end = start + seg_len if i < 4 else len(all_beats)
                    step = seg_steps[i]
                    seg_sampled = [all_beats[j] for j in range(start, min(end, len(all_beats)), step)]
                    sampled.extend(seg_sampled)

                sampled = sorted(set(sampled))
                beat_times = [0.0] + sampled
                if beat_times[-1] < duration - 0.1:
                    beat_times.append(duration)
                else:
                    beat_times[-1] = duration
                shot_count = len(beat_times) - 1
                step_str = "/".join(str(s) for s in seg_steps)
                print(f"    BPM={bpm:.0f} | 镜头{shot_count}个 | 混步:{step_str}")
            else:
                raise ValueError("未检测到节拍")
        except Exception as e:
            print(f"    ⚠️ 节拍分析失败: {e}")
            beat_times = []

    # 保底：均分方案
    if len(beat_times) < 2:
        seg_count = max(8, int(duration / 0.8))
        beat_times = list(np.linspace(0, duration, seg_count + 1))
        shot_count = len(beat_times) - 1
        print(f"    均分{shot_count}镜头，每0.8s一镜")

    # 计算每段时长
    clip_durations = []
    for i in range(len(beat_times) - 1):
        d = beat_times[i+1] - beat_times[i]
        clip_durations.append(d)

    # 归一化到目标时长
    total = sum(clip_durations)
    if total > 0 and abs(total - duration) > 0.1:
        clip_durations = [d * duration / total for d in clip_durations]

    # v6.4核心：限制单镜头0.7~1.1秒
    clip_durations = [max(0.7, min(d, 1.1)) for d in clip_durations]

    # 补齐时长
    actual_total = sum(clip_durations)
    if actual_total < duration * 0.9 and len(clip_durations) > 0:
        deficit = duration - actual_total
        fill_count = max(1, round(deficit / 0.9))
        fill_dur = deficit / fill_count
        fill_dur = max(0.7, min(fill_dur, 1.1))
        clip_durations.append(fill_dur)
        new_total = sum(clip_durations)
        clip_durations = [d * duration / new_total for d in clip_durations]
        clip_durations = [max(0.7, min(d, 1.1)) for d in clip_durations]
        while sum(clip_durations) < duration * 0.9:
            extra = duration - sum(clip_durations)
            clip_durations.append(max(0.7, min(extra, 1.1)))

    # 确保至少8个镜头
    while len(clip_durations) < 8:
        idx = clip_durations.index(max(clip_durations))
        d = clip_durations.pop(idx)
        half = d / 2
        clip_durations.insert(idx, half)
        clip_durations.insert(idx + 1, half)
        clip_durations = [max(0.7, min(d, 1.1)) for d in clip_durations]

    # 最终归一化
    total = sum(clip_durations)
    if total > 0 and abs(total - duration) > 0.1:
        clip_durations = [d * duration / total for d in clip_durations]

    return clip_durations


# ===== 主流程 =====

print("=" * 60)
print("批量剪辑 v7.0 — YOLOv8语义检测版")
print("4K竖屏 | 60fps | YOLOv8目标检测 | 食光II滤镜 | 电影感画幅 | 自适应节拍混步")
print("=" * 60)

# 收集素材
print("\n[1/3] 收集素材...")
videos = collect_videos()
print(f"  合计: {len(videos)} 个视频")
if len(videos) < 6:
    print("❌ 素材不足6个，退出")
    sys.exit(1)

# 加载缓存并扫描未缓存的素材
cache = MaterialCache()
print(f"  已有缓存: {len(cache.data)} 个")
cache.scan_and_cache(videos, force=args.force_scan)  # 首次全量扫描，之后跳过

# 收集BGM（递归扫描子目录，支持 E:/音频/周杰伦 等子文件夹）
bgms = [
    os.path.join(root, f)
    for root, _, files in os.walk(BGM_DIR)
    for f in files
    if f.lower().endswith('.mp3')
]
print(f"  可用BGM: {len(bgms)} 个")

# 按多维度综合分排序素材（v7.0）
def get_dim_score(cache_entry):
    """从缓存提取多维度综合分，兼容旧缓存"""
    if not cache_entry:
        return 0
    dims = cache_entry.get("dim_scores", {})
    if dims and dims.get("total", 0) > 0:
        return dims["total"]
    # 兼容旧缓存：用motion_score
    return cache_entry.get("motion_score", 0)

cached_videos = [(v, cache.get(v)) for v in videos]
scored = [(v, get_dim_score(c), get_video_duration(v)) for v, c in cached_videos]
scored.sort(key=lambda x: x[1], reverse=True)

# 全局已用素材池（去重用）
used_videos = set()

# ===== 找起始编号（扫描输出目录已有数字文件名）=====
def get_next_number():
    if not os.path.exists(OUTPUT_DIR):
        return 1
    nums = []
    for f in os.listdir(OUTPUT_DIR):
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
    # 找最小空缺，没空缺就从max+1继续
    for n in range(1, max(used) + 2):
        if n not in used:
            return n

start_num = args.start_index if args.start_index is not None else get_next_number()
print(f"  起始编号: {start_num}")

# 生成视频
print(f"\n[2/3] 开始生成{BATCH_COUNT}条视频...")

for idx, template in enumerate(TEMPLATES, 1):
    template_name = template["name"]
    duration      = template["duration"]
    seq_num       = start_num + idx - 1  # 编号1, 2, 3...

    print(f"\n{'='*60}")
    print(f"视频 {idx}/{BATCH_COUNT}: 编号「{seq_num}」模板「{template_name}」{duration}秒")
    print('='*60)

    # 选BGM（bgms已是绝对路径）
    bgm_path = random.choice(bgms)
    bgm_name = os.path.basename(bgm_path)
    print(f"\nBGM: {bgm_name}")

    # 自适应节拍计算镜头
    print("分析节拍...")
    clip_durations = calc_shots_by_adaptive_beat(bgm_path, duration)
    shots = len(clip_durations)
    print(f"  镜头数: {shots} 个")
    print(f"  各镜头: {[f'{d:.2f}s' for d in clip_durations]}")

    # ===== 选素材（全局去重 + 时长够用 + 质量过滤）=====
    min_needed = max(clip_durations) + 0.5  # 最短素材也要能截最长的镜头

    # 质量过滤：排除垃圾镜头（分数过低 or shot_type为wide/unknown）
    def is_good_shot(v, s):
        c = cache.get(v)
        if not c:
            return False  # 没缓存的跳过
        st = c.get("shot_type", "unknown")
        # wide和unknown基本都是空镜/垃圾镜头
        if st in ("wide", "unknown"):
            return False
        # 分数太低的也排除（低于15分大概率是暗场/无食物镜头）
        if s < 15:
            return False
        return True

    # 先挑没用过且时长够且质量好的
    available = [(v, s, vd) for v, s, vd in scored
                 if v not in used_videos and vd >= min_needed and is_good_shot(v, s)]

    if len(available) < shots:
        print(f"  ⚠ 优质素材不足({len(available)}<{shots})，放宽复用（允许已用素材，但仍保留质量门槛）")
        # 补充：放宽"未用过"限制，但仍保留质量过滤
        extra = [(v, s, vd) for v, s, vd in scored
                 if v not in [x[0] for x in available] and vd >= min_needed and is_good_shot(v, s)]
        available.extend(extra)
        # 最后兜底：如果优质素材真的不够，才接受medium/low
        if len(available) < shots:
            print(f"  ⚠ 优质素材仍不足，最后兜底接受所有时长足够的素材")
            fallback = [(v, s, vd) for v, s, vd in scored
                        if v not in [x[0] for x in available] and vd >= min_needed]
            available.extend(fallback)

    if len(available) < shots:
        print(f"  ❌ 素材严重不足({len(available)}<{shots})，跳过")
        continue

    # 从高分候选区随机选，确保不重复
    selected = []
    pick_pool = available[:min(shots * 5, len(available))]
    random.shuffle(pick_pool)
    for pick in pick_pool:
        if pick[0] not in [s[0] for s in selected]:
            selected.append(pick)
        if len(selected) >= shots:
            break

    # 标记为已用（全局去重）
    for v, _, _ in selected:
        used_videos.add(v)

    print(f"  选中素材: {len(selected)} 个（已用素材池: {len(used_videos)}）")

    # 创建草稿（数字命名）
    draft_name = f"{seq_num}"
    print(f"\n创建草稿「{draft_name}」...")

    project = JyProject(draft_name, width=2160, height=3840, overwrite=True)
    print("  ✅ 草稿创建成功")

    project.script.add_track(TrackType.video,  "VideoTrack")
    project.script.add_track(TrackType.audio,  "BGM")
    project.script.add_track(TrackType.effect, "EffectTrack")
    project.script.add_track(TrackType.filter, "FilterTrack")
    print("  ✅ 轨道创建完成")

    # 添加视频片段（轨道严格对齐）
    print(f"\n添加 {shots} 个视频片段...")
    ok = 0
    target_start = 0.0

    for i, (vp, _, vid_dur) in enumerate(selected):
        dur = clip_durations[i]

        # 计算截取起始点（素材必须够长，否则跳过换下一个）
        if vid_dur < dur + 0.5:
            print(f"  ⚠ [{i+1:2d}] {os.path.basename(vp)[:30]} 素材太短({vid_dur:.1f}s)，跳过")
            target_start += dur
            continue
        # 修复：避免上限小于下限导致 random.uniform 报错
        max_src_start = max(0.0, vid_dur - dur - 0.3)
        src_start = random.uniform(0.0, max_src_start)

        seg = None
        try:
            seg = project.add_clip(
                media_path=vp, source_start=src_start,
                duration=dur, target_start=target_start,
                track_name="VideoTrack"
            )
        except Exception as e:
            print(f"  ❌ [{i+1:2d}] {e}")

        if seg:
            ok += 1
            print(f"  ✅ [{i+1:2d}] {os.path.basename(vp)[:30]} ({dur:.2f}s)")
        else:
            print(f"  ⚠ [{i+1:2d}] 添加失败，跳过")

        target_start += dur  # 无论片段是否成功添加，时间轴都必须严格推进

    print(f"  成功 {ok}/{shots}")

    # BGM（截断到目标时长）
    print(f"\n添加BGM: {bgm_name}...")
    try:
        bgm_seg = project.add_audio_safe(bgm_path, start_time=0.0, duration=duration, track_name="BGM")
        print(f"  {'✅ BGM添加成功' if bgm_seg else '❌ BGM失败'}")
    except Exception as e:
        print(f"  ❌ BGM失败: {e}")

    # 特效：电影感画幅
    print("\n添加「电影感画幅」特效...")
    try:
        tr = Timerange(0, int(duration * 1_000_000))
        project.script.add_effect(VideoSceneEffectType.电影感画幅, tr, track_name="EffectTrack")
        print("  ✅ 电影感画幅特效已添加")
    except Exception as e:
        print(f"  ❌ 特效失败: {e}")

    # 滤镜：食光II 40%
    print("\n添加「食光II」滤镜 (40%)...")
    try:
        tr = Timerange(0, int(duration * 1_000_000))
        project.script.add_filter(FilterType.食光II, tr, track_name="FilterTrack", intensity=40.0)
        print("  ✅ 食光II 40% 已添加")
    except Exception as e:
        print(f"  ❌ 滤镜失败: {e}")

    # 保存草稿
    print(f"\n保存草稿「{draft_name}」...")
    result = project.save()
    print(f"  {result}")

    # 导出（数字编号命名）
    print(f"\n导出视频（4K/60fps）...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_file = os.path.join(OUTPUT_DIR, f"{seq_num}.mp4")

    try:
        time.sleep(2)
        ctrl = JianyingController()
        ctrl.export_draft(
            draft_name=draft_name,
            output_path=output_file,
            resolution=draft.ExportResolution.RES_4K,
            framerate=draft.ExportFramerate.FR_60
        )
        print(f"  ✅ 导出完成: {os.path.basename(output_file)}")
    except Exception as e:
        print(f"  ⚠️ 导出失败: {e}")
        print(f"     请手动在剪映中导出该草稿: {draft_name}")

print(f"\n{'='*60}")
print(f"✅ 批量视频生成完成!")
print(f"📁 输出目录: {OUTPUT_DIR}")
print('='*60)
