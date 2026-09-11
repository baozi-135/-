"""
LLM剪辑语法引擎 v1.0 — Edit Grammar Engine
============================================
将LLM的结构化编辑决策翻译为剪映API调用序列。
替代 pipeline.py 中 step_render 的硬编码参数。

架构层次:
  1. Grammar Schema   — 定义LLM可输出的编辑操作类型和约束
  2. Context Builder  — 为LLM构建完整的上下文prompt
  3. Grammar Executor — 将LLM输出的JSON翻译为Jianying API调用
  4. Rule Fallback    — LLM不可用时的智能规则引擎回退

使用方式:
  # 作为库使用
  from edit_grammar import EditGrammar, build_edit_context, execute_grammar
  grammar = build_edit_context(template, shot_plan, bgm_info)
  # 传给LLM... 得到grammar_json
  execute_grammar(project, grammar_json, shot_plan, duration)

  # 独立测试
  python edit_grammar.py --test
"""

import json
import random
import os
from typing import Dict, List, Optional, Any, Tuple

# ============================================================
# 1. Grammar Schema — LLM可用的编辑操作词汇表
# ============================================================

# 美食/甜品类精选滤镜（从82个中筛选21个相关滤镜）
FOOD_FILTERS = {
    "食光II":     {"mood": "美食",   "intensity_default": 40, "desc": "增强食物色泽，暖色调"},
    "食欲":       {"mood": "美食",   "intensity_default": 45, "desc": "提升食欲感，饱和度高"},
    "美食":       {"mood": "美食",   "intensity_default": 50, "desc": "通用美食滤镜，色彩鲜明"},
    "深夜食堂":   {"mood": "美食",   "intensity_default": 35, "desc": "暗调美食风，氛围感强"},
    "冬日烧烤":   {"mood": "美食",   "intensity_default": 45, "desc": "暖光烧烤风格"},
    "亮夏":       {"mood": "夏日",   "intensity_default": 50, "desc": "明亮夏日，阳光感"},
    "入夏":       {"mood": "夏日",   "intensity_default": 40, "desc": "清新入夏，绿色调"},
    "夏日小美好": {"mood": "夏日",   "intensity_default": 45, "desc": "温馨夏日记忆感"},
    "元气新年":   {"mood": "活力",   "intensity_default": 50, "desc": "元气满满，色彩跳跃"},
    "仲夏绿光":   {"mood": "清新",   "intensity_default": 40, "desc": "绿色清新自然风"},
    "净白肤":     {"mood": "人像",   "intensity_default": 50, "desc": "净白肤色，适合人物出镜"},
    "冰肌":       {"mood": "人像",   "intensity_default": 45, "desc": "冰感冷白皮"},
    "冷白":       {"mood": "冷调",   "intensity_default": 40, "desc": "冷白高级感"},
    "电影感II":   {"mood": "电影",   "intensity_default": 35, "desc": "电影调色，质感强"},
    "复古胶片":   {"mood": "复古",   "intensity_default": 40, "desc": "复古胶片质感"},
    "港风":       {"mood": "复古",   "intensity_default": 45, "desc": "港风复古色调"},
    "轻复古":     {"mood": "复古",   "intensity_default": 35, "desc": "轻度复古，不压暗"},
    "Lofi_II":    {"mood": "复古",   "intensity_default": 40, "desc": "Lofi低保真风格"},
    "原木":       {"mood": "自然",   "intensity_default": 35, "desc": "原木色调，温暖自然"},
    "暖食":       {"mood": "美食",   "intensity_default": 45, "desc": "暖色调美食"},
    "春日序":     {"mood": "清新",   "intensity_default": 40, "desc": "春天清新感"},
}

# 全局画面特效（精选12个常用）
GLOBAL_EFFECTS = {
    "电影感画幅": {"desc": "上下黑边，电影感",           "params": {}},
    "CCD闪光":    {"desc": "CCD相机闪光效果",          "params": {"effects_adjust_speed": 0.5, "effects_adjust_luminance": 0.65}},
    "90s画质":    {"desc": "90年代复古画质",            "params": {"effects_adjust_filter": 1.0, "effects_adjust_sharpen": 0.63}},
    "70s":        {"desc": "70年代复古风",              "params": {"effects_adjust_speed": 0.33}},
    "JVC":        {"desc": "JVC摄像机风格",             "params": {"effects_adjust_color": 0.5, "effects_adjust_intensity": 0.35}},
    "VCR":        {"desc": "VCR录像带风格",             "params": {"effects_adjust_speed": 0.33, "effects_adjust_sharpen": 0.3}},
    "MV封面":     {"desc": "MV封面虚化边框",            "params": {"effects_adjust_blur": 1.0}},
    "RGB描边":    {"desc": "RGB色彩分离描边",           "params": {"effects_adjust_speed": 0.67}},
    "DV录制框":   {"desc": "DV录像框叠加",              "params": {}},
    "雪花":       {"desc": "雪花飘落叠加",              "params": {}},
    "星星闪烁":   {"desc": "星星闪烁特效",              "params": {}},
    "光斑":       {"desc": "柔和光斑飘动",              "params": {}},
}

# 转场（精选15个常用）
TRANSITIONS = {
    "叠化":       {"default_dur": 0.15, "desc": "柔和渐隐渐显，最常用"},
    "叠加":       {"default_dur": 0.15, "desc": "叠加混合，梦幻感"},
    "模糊":       {"default_dur": 0.15, "desc": "模糊过渡"},
    "黑场":       {"default_dur": 0.15, "desc": "黑场过渡，情绪切换", "enum_name": "闪黑"},
    "白场":       {"default_dur": 0.15, "desc": "白场过渡，闪白效果", "enum_name": "闪白"},
    "闪白":       {"default_dur": 0.15, "desc": "闪白转场，快节奏"},
    "向右滑动":   {"default_dur": 0.15, "desc": "向右滑动切入"},
    "向左滑动":   {"default_dur": 0.15, "desc": "向左滑动切入"},
    "向上滑动":   {"default_dur": 0.15, "desc": "向上滑动切入"},
    "向下滑动":   {"default_dur": 0.15, "desc": "向下滑动切入"},
    "放大":       {"default_dur": 0.15, "desc": "放大转场"},
    "缩小":       {"default_dur": 0.15, "desc": "缩小转场"},
    "分割":       {"default_dur": 0.15, "desc": "分割转场，动感"},
    "冲鸭":       {"default_dur": 0.15, "desc": "可爱冲鸭转场"},
}

# 入场动画（精选12个常用）
INTRO_ANIMATIONS = {
    "渐显":       {"duration": 0.5, "desc": "柔和渐显"},
    "动感放大":   {"duration": 0.5, "desc": "动感放大入场"},
    "轻微抖动":   {"duration": 0.5, "desc": "轻微抖动，活力感"},
    "旋涡旋转":   {"duration": 0.5, "desc": "旋涡旋转入场"},
    "向右甩入":   {"duration": 0.5, "desc": "向右甩入"},
    "向左甩入":   {"duration": 0.5, "desc": "向左甩入"},
    "折叠开幕":   {"duration": 1.5, "desc": "折叠开幕，仪式感"},
    "旋转开幕":   {"duration": 1.0, "desc": "旋转开幕"},
    "跳转开幕":   {"duration": 0.73, "desc": "跳转开幕，节奏感强"},
    "翻入":       {"duration": 1.17, "desc": "翻页入"},
    "斜切":       {"duration": 0.7, "desc": "斜切入场"},
    "Kira游动":   {"duration": 2.27, "desc": "Kira闪光游动，炫酷"},
}

# 关键帧运动类型
KEYFRAME_MOTIONS = {
    "slow_zoom_in":   {"desc": "缓慢放大（Ken Burns效果）", "start_scale": 1.0, "end_scale": 1.08},
    "slow_zoom_out":  {"desc": "缓慢缩小",                 "start_scale": 1.08, "end_scale": 1.0},
    "pan_left":       {"desc": "向左平移",                 "start_x": 0.05, "end_x": -0.05},
    "pan_right":      {"desc": "向右平移",                 "start_x": -0.05, "end_x": 0.05},
    "tilt_up":        {"desc": "向上微移",                 "start_y": 0.05, "end_y": -0.05},
    "wobble":         {"desc": "轻微抖动",                 "start_scale": 1.0, "end_scale": 1.02, "start_x": -0.02, "end_x": 0.02},
}

# ============================================================
# 编辑语法JSON Schema（供LLM参考）
# ============================================================

GRAMMAR_SCHEMA = """
{
  "version": "1.0",
  "global": {
    "filter": {"name": "食光II", "intensity": 40},        // 全局滤镜，从FOOD_FILTERS中选
    "effect": {"name": null},                              // 全局画面特效，从GLOBAL_EFFECTS中选，null=不加
    "bgm_volume": 0.8,                                     // BGM音量 0.0~1.0
    "color_temperature": null                              // 色温倾向: "warm"|"cool"|null
  },
  "per_clip": [                                            // 每个镜头的编辑决策，数量=镜头数
    {
      "clip_index": 0,                                     // 镜头序号（从0开始）
      "filter_override": null,                             // 覆盖全局滤镜，null=用全局
      "keyframe_motion": "slow_zoom_in",                   // 关键帧运动，null=静止
      "intro_animation": null,                             // 入场动画，null=无动画
      "speed_ramp": 1.0                                    // 变速倍率，1.0=原速
    }
  ],
  "transitions": [                                         // 镜头间转场，数量=镜头数-1
    {
      "after_clip_index": 0,                               // 在第N个镜头后添加转场
      "type": "叠化",                                      // 转场类型
      "duration": 0.5                                      // 转场时长(秒)
    }
  ],
  "text_overlays": [                                       // 花字/文字叠加，可选
    {
      "time": 0.5,                                          // 出现时间(秒)
      "duration": 2.0,                                      // 持续时长(秒)
      "content": "🍧夏日限定",                              // 文字内容
      "style": "subtitle"                                   // subtitle|title|sticker
    }
  ]
}
"""


# ============================================================
# 2. Context Builder — 构建LLM Prompt
# ============================================================

def build_edit_context(
    template: Dict,
    shot_plan: List[Dict],
    bgm_info: Dict,
    include_schema: bool = True
) -> str:
    """构建LLM剪辑决策所需的完整上下文文本
    
    返回一个可直接喂给LLM的prompt字符串。
    """
    lines = []
    lines.append("=" * 60)
    lines.append("🎬 剪辑上下文 — 请为以下素材生成编辑决策JSON")
    lines.append("=" * 60)
    
    # 模板信息
    lines.append(f"\n## 参考模板")
    lines.append(f"- 名称: {template.get('name', '未知')}")
    lines.append(f"- 总时长: {template.get('duration', 0):.1f}秒")
    
    # BGM信息
    lines.append(f"\n## BGM")
    lines.append(f"- 曲名: {bgm_info.get('bgm_name', '未知')}")
    bp = bgm_info.get('beat_plan', {})
    lines.append(f"- 节拍方案: {bp.get('method', 'unknown')}")
    lines.append(f"- BPM: {bp.get('bpm', 0)}")
    lines.append(f"- 镜头数: {bp.get('shot_count', 0)}")
    lines.append(f"- 各镜头时长: {bp.get('clip_durations', [])}")
    
    # 镜头方案
    lines.append(f"\n## 镜头方案 ({len(shot_plan)}个)")
    for i, shot in enumerate(shot_plan):
        fname = os.path.basename(shot.get('video_path', ''))[:40]
        lines.append(f"  镜头{i}: {fname}")
        lines.append(f"    score={shot.get('score', 0):.1f} type={shot.get('shot_type', 'unknown')}")
        lines.append(f"    duration={shot.get('clip_duration', 0):.2f}s src_start={shot.get('source_start', 0):.1f}s")
        dims = shot.get('dim_scores', {})
        if dims:
            top3 = sorted([(k, v) for k, v in dims.items() if k != 'total'], key=lambda x: -x[1])[:3]
            lines.append(f"    dims: {[(k, f'{v:.1f}') for k, v in top3]}")
    
    # 可选操作词汇表
    if include_schema:
        lines.append(f"\n## 可用滤镜 (美食/甜品场景精选)")
        for name, info in FOOD_FILTERS.items():
            lines.append(f"  - {name} [{info['mood']}]: {info['desc']}")
        
        lines.append(f"\n## 可用全局特效")
        for name, info in GLOBAL_EFFECTS.items():
            lines.append(f"  - {name}: {info['desc']}")
        
        lines.append(f"\n## 可用转场")
        for name, info in TRANSITIONS.items():
            lines.append(f"  - {name} ({info['default_dur']}s): {info['desc']}")
        
        lines.append(f"\n## 可用入场动画")
        for name, info in INTRO_ANIMATIONS.items():
            lines.append(f"  - {name} ({info['duration']}s): {info['desc']}")
        
        lines.append(f"\n## 可用关键帧运动")
        for name, info in KEYFRAME_MOTIONS.items():
            lines.append(f"  - {name}: {info['desc']}")
    
    # 输出格式
    lines.append(f"\n## 输出格式")
    lines.append("请输出JSON，格式如下:")
    lines.append(GRAMMAR_SCHEMA)
    lines.append("\n输出规则:")
    lines.append("1. per_clip数组长度必须等于镜头数")
    lines.append("2. transitions数组长度必须等于(镜头数-1)")
    lines.append("3. 根据shot_type智能选择: eating_action镜头用食欲/食光滤镜+slow_zoom_in; closeup用净白肤; wide镜头可加动感放大入场")
    lines.append("4. 甜品/夏日主题: 优先暖色调滤镜(食光II/食欲/亮夏/入夏)")
    lines.append("5. 转场默认叠化0.5s, 快节奏片段用闪光/分割")
    lines.append("6. text_overlays根据模板名生成对应的标题文字（如模板名含'昆明'→'春城味道', '新化'→'湘味糖水'）")
    lines.append("7. 只输出JSON，不要额外说明")
    
    return "\n".join(lines)


def build_grammar_prompt(context_text: str) -> str:
    """将上下文包装为完整prompt，可直接发给LLM API"""
    return f"""{context_text}

请基于以上上下文，输出编辑决策JSON。只输出JSON，不要任何解释。"""


# ============================================================
# 3. Grammar Executor — 执行LLM输出的编辑决策
# ============================================================

def execute_grammar(
    project,           # JyProject 实例
    grammar: Dict,
    shot_plan: List[Dict],
    bgm_info: Dict,
    duration: float
) -> Dict[str, Any]:
    """将编辑语法JSON翻译为剪映API调用序列
    
    Args:
        project: JyProject实例（已创建track）
        grammar: LLM输出的编辑决策JSON
        shot_plan: 镜头方案列表
        bgm_info: BGM信息
        duration: 视频总时长
    
    Returns:
        {"filter_ok": bool, "effect_ok": bool, "transitions_added": int, ...}
    """
    import pyJianYingDraft as draft
    from pyJianYingDraft import FilterType, TrackType, KeyframeProperty as KP
    from pyJianYingDraft import TextStyle, TextBorder, TextBackground, ClipSettings
    from pyJianYingDraft.metadata.video_scene_effect import VideoSceneEffectType
    from pyJianYingDraft.time_util import Timerange
    
    stats = {
        "filter_ok": False, "effect_ok": False,
        "transitions_added": 0, "keyframes_added": 0,
        "intros_added": 0, "texts_added": 0,
        "errors": []
    }
    
    # ---- 关键修复：materials 后注册 ----
    # JyWrapper架构缺陷：add_segment()只在添加时扫描一次segment并注册素材，
    # 之后调用 seg.add_transition/add_animation/add_filter 只修改内存，
    # 不会同步到 project.script.materials 注册表 → 保存时丢失。
    # 解决：每次修改segment后手动注册到materials。
    def _register_material(seg):
        """将segment上的素材注册到project.script.materials，防止序列化丢失"""
        try:
            mats = project.script.materials
            # 转场
            if hasattr(seg, 'transition') and seg.transition is not None:
                if seg.transition not in mats.transitions:
                    mats.transitions.append(seg.transition)
            # 动画
            anim = getattr(seg, 'animations_instance', None)
            if anim is not None and anim not in mats.animations:
                mats.animations.append(anim)
            # 滤镜（逐镜头滤镜）
            for f in getattr(seg, 'filters', []):
                if f not in mats.filters:
                    mats.filters.append(f)
            # 特效（逐镜头特效）
            for e in getattr(seg, 'effects', []):
                if e not in mats.video_effects:
                    mats.video_effects.append(e)
        except Exception:
            pass  # materials注册失败不阻塞主流程
    
    global_cfg = grammar.get("global", {})
    
    # ---- 全局滤镜 ----
    filter_name = global_cfg.get("filter", {}).get("name") if isinstance(global_cfg.get("filter"), dict) else global_cfg.get("filter")
    filter_intensity = global_cfg.get("filter", {}).get("intensity", 40) if isinstance(global_cfg.get("filter"), dict) else 40
    
    if filter_name:
        try:
            filter_enum = _resolve_enum(FilterType, filter_name)
            if filter_enum:
                tr = Timerange(0, int(duration * 1_000_000))
                project.script.add_filter(filter_enum, tr, track_name="FilterTrack", intensity=float(filter_intensity))
                stats["filter_ok"] = True
            else:
                stats["errors"].append(f"滤镜 '{filter_name}' 未在枚举中找到")
        except Exception as e:
            stats["errors"].append(f"滤镜添加失败: {e}")
    
    # ---- 全局特效 ----
    effect_name = global_cfg.get("effect", {}).get("name") if isinstance(global_cfg.get("effect"), dict) else global_cfg.get("effect")
    if effect_name:
        try:
            effect_enum = _resolve_enum(VideoSceneEffectType, effect_name)
            if effect_enum:
                tr = Timerange(0, int(duration * 1_000_000))
                project.script.add_effect(effect_enum, tr, track_name="EffectTrack")
                stats["effect_ok"] = True
        except Exception as e:
            stats["errors"].append(f"特效添加失败: {e}")
    
    # ---- 逐镜头处理 ----
    per_clip = grammar.get("per_clip", [])
    transitions = grammar.get("transitions", [])
    
    video_segments = []  # 记录每个镜头的segment引用
    
    # 先收集已添加的视频segment（从project中获取）
    # 注意：project.script.tracks 是字典，key为track_name
    try:
        video_track = project.script.tracks.get("VideoTrack")
        if video_track and hasattr(video_track, "segments"):
            video_segments = list(video_track.segments)
    except Exception as e:
        stats["errors"].append(f"获取VideoTrack失败: {e}")
    
    # 逐镜头设置关键帧和入场动画
    for clip_cfg in per_clip:
        idx = clip_cfg.get("clip_index", 0)
        
        # 入场动画
        intro = clip_cfg.get("intro_animation")
        if intro and idx < len(video_segments):
            try:
                seg = video_segments[idx]
                from pyJianYingDraft.metadata.video_intro import IntroType
                intro_enum = _resolve_enum(IntroType, intro)
                if intro_enum:
                    seg.add_animation(intro_enum)
                    stats["intros_added"] += 1
                    _register_material(seg)
            except Exception as e:
                stats["errors"].append(f"镜头{idx}入场动画失败: {e}")
        
        # 关键帧运动
        motion = clip_cfg.get("keyframe_motion")
        if motion and motion in KEYFRAME_MOTIONS and idx < len(video_segments):
            try:
                seg = video_segments[idx]
                mcfg = KEYFRAME_MOTIONS[motion]
                clip_dur = shot_plan[idx].get("clip_duration", 1.0) if idx < len(shot_plan) else 1.0
                dur_us = int(clip_dur * 1_000_000)
                
                if "start_scale" in mcfg and "end_scale" in mcfg:
                    seg.add_keyframe(KP.uniform_scale, 0, mcfg["start_scale"])
                    seg.add_keyframe(KP.uniform_scale, dur_us, mcfg["end_scale"])
                if "start_x" in mcfg and "end_x" in mcfg:
                    seg.add_keyframe(KP.position_x, 0, mcfg["start_x"])
                    seg.add_keyframe(KP.position_x, dur_us, mcfg["end_x"])
                if "start_y" in mcfg and "end_y" in mcfg:
                    seg.add_keyframe(KP.position_y, 0, mcfg["start_y"])
                    seg.add_keyframe(KP.position_y, dur_us, mcfg["end_y"])
                
                stats["keyframes_added"] += 1
            except Exception as e:
                stats["errors"].append(f"镜头{idx}关键帧失败: {e}")
        
        # 逐镜头滤镜覆盖
        filter_override = clip_cfg.get("filter_override")
        if filter_override and idx < len(video_segments):
            try:
                seg = video_segments[idx]
                override_enum = _resolve_enum(FilterType, filter_override)
                if override_enum:
                    seg.add_filter(override_enum, intensity=45)
                    _register_material(seg)
            except Exception as e:
                stats["errors"].append(f"镜头{idx}滤镜覆盖失败: {e}")
    
    # ---- 转场（直接调用VideoSegment.add_transition，绕过有bug的add_transition_simple）----
    from pyJianYingDraft.metadata import TransitionType
    for trans in transitions:
        after_idx = trans.get("after_clip_index", 0)
        trans_name = trans.get("type", "叠化")
        trans_dur = trans.get("duration", 0.5)
        
        if after_idx < len(video_segments):
            try:
                seg = video_segments[after_idx]
                trans_enum = _resolve_enum(TransitionType, trans_name)
                if trans_enum:
                    seg.add_transition(trans_enum, duration=int(trans_dur * 1_000_000))
                    stats["transitions_added"] += 1
                    _register_material(seg)
                else:
                    stats["errors"].append(f"转场 '{trans_name}' 枚举未找到")
            except Exception as e:
                stats["errors"].append(f"转场(after {after_idx})失败: {e}")
    
    # ---- 文字叠加 ----
    for text_cfg in grammar.get("text_overlays", []):
        try:
            content = text_cfg.get("content", "")
            t = text_cfg.get("time", 0)
            d = text_cfg.get("duration", 2.0)
            text_style = text_cfg.get("style", "subtitle")
            
            if not content:
                continue
            
            if text_style == "title":
                # 标题: 大号金色粗体 + 黑色描边 + 半透明深色背景
                project.add_text_simple(
                    content,
                    start_time=f"{t}s", duration=f"{d}s",
                    track_name="TitleTrack",
                    style=TextStyle(size=16.0, bold=True, color=(1.0, 0.82, 0.15), alpha=1.0),
                    border=TextBorder(color=(0.0, 0.0, 0.0), alpha=1.0, width=50.0),
                    background=TextBackground(style=2, alpha=0.55, color="#1A1A1A", round_radius=10.0),
                    clip_settings=ClipSettings(transform_y=-0.65),
                    anim_in="渐显", anim_out="淡出"
                )
            else:
                # 字幕: 暖色奶油白粗体 + 深色描边 + 暗底
                project.add_text_simple(
                    content,
                    start_time=f"{t}s", duration=f"{d}s",
                    track_name="Subtitles",
                    style=TextStyle(size=12.0, bold=True, color=(1.0, 0.92, 0.72), alpha=1.0),
                    border=TextBorder(color=(0.08, 0.06, 0.02), alpha=0.95, width=38.0),
                    background=TextBackground(style=2, alpha=0.45, color="#0D0D0D", round_radius=6.0),
                    clip_settings=ClipSettings(transform_y=-0.78)
                )
            stats["texts_added"] += 1
        except Exception as e:
            stats["errors"].append(f"文字叠加失败: {e}")
    
    return stats


def _resolve_enum(enum_class, name: str):
    """解析枚举值，支持中文名和模糊匹配"""
    # 精确匹配
    for member in enum_class:
        if hasattr(member, 'name'):
            if member.name == name:
                return member
        if hasattr(member, 'value'):
            meta = member.value
            if isinstance(meta, str) and meta == name:
                return member
            if hasattr(meta, 'name') and (meta.name == name or getattr(meta, 'display_name', '') == name):
                return member
    
    # 模糊匹配: 枚举名包含name 或 name包含枚举名
    for member in enum_class:
        mn = member.name if hasattr(member, 'name') else ''
        if name in mn or mn in name:
            return member
        if hasattr(member, 'value') and hasattr(member.value, 'name'):
            vn = member.value.name
            if name in vn or vn in name:
                return member
    
    return None


# ============================================================
# 4. Rule Fallback — LLM不可用时的智能规则引擎
# ============================================================

def generate_rule_grammar(
    template: Dict,
    shot_plan: List[Dict],
    bgm_info: Dict,
    duration: float
) -> Dict:
    """规则引擎回退：根据模板名和镜头特征自动生成编辑决策
    
    完全不依赖LLM，基于规则和随机化。
    """
    template_name = template.get("name", "")
    bp = bgm_info.get("beat_plan", {})
    n_shots = len(shot_plan)
    
    # --- 根据模板名推断风格 ---
    is_summer = any(kw in template_name for kw in ["夏日", "夏天", "summer", "昆明", "新化"])
    is_vibe = any(kw in template_name for kw in ["氛围感", "vibe", "晚风", "海阔天空"])
    is_energetic = any(kw in template_name for kw in ["十面埋伏", "列车", "爷爷泡的茶"])
    bgm_style = "adaptive" if bp.get("method") == "adaptive" else "uniform"
    
    # --- 全局滤镜（食光II：100%通过率5次，美食专用）---
    chosen_filter = "食光II"
    filter_intensity = 55

    # --- 全局特效（始终加电影感画幅）---
    chosen_effect = "电影感画幅"
    
    # --- 逐镜头决策 ---
    per_clip = []
    shot_types_seen = []
    for i, shot in enumerate(shot_plan):
        st = shot.get("shot_type", "unknown")
        score = shot.get("score", 50)
        
        # 关键帧运动（更多镜头添加slow_zoom_in提升动态感）
        if st == "eating_action":
            motion = "slow_zoom_in"  # eating_action全部加
        elif st == "closeup":
            motion = random.choice(["slow_zoom_in", "slow_zoom_in", "slow_zoom_out", None])
        elif st == "wide":
            motion = random.choice(["pan_right", "pan_left", None, None])
        else:
            motion = "slow_zoom_in"  # 所有镜头都加关键帧运动，提升动态感
        
        # 入场动画（仅前2个镜头，符合优化器建议"前2-3个加渐显/动感放大"）
        intro = None
        if i == 0:
            intro = random.choice(["渐显", "动感放大", "渐显"])  # 第1个镜头必加
        elif i == 1:
            intro = random.choice(["动感放大", "轻微抖动", "渐显"])  # 第2个镜头必加
        # v16: 第3个镜头不再加入场动画，避免泛滥
        
        # 逐镜头滤镜覆盖（只在分数差异大时）
        filter_override = None
        
        per_clip.append({
            "clip_index": i,
            "filter_override": filter_override,
            "keyframe_motion": motion,
            "intro_animation": intro,
            "speed_ramp": 1.0
        })
        shot_types_seen.append(st)
    
    # --- 转场（v16：仅前2-3个镜头之间加，避免泛滥）---
    transitions = []
    max_trans = min(2, n_shots - 1)  # 前2个转场（0→1、1→2）
    for i in range(max_trans):
        if is_energetic or bgm_style == "adaptive":
            trans_type = random.choice(["叠化", "叠化", "叠化", "闪白", "分割", "模糊"])
        else:
            trans_type = random.choice(["叠化", "叠化", "叠化", "模糊", "叠加"])
        
        trans_dur = random.choice([0.1, 0.12, 0.15, 0.15])
        if trans_type in TRANSITIONS:
            trans_dur = TRANSITIONS[trans_type]["default_dur"]
        
        transitions.append({
            "after_clip_index": i,
            "type": trans_type,
            "duration": trans_dur
        })
    
    # --- 文字叠加 ---
    text_overlays = _generate_text_overlay(template_name, duration)
    
    return {
        "version": "1.0",
        "generated_by": "rule_engine",
        "global": {
            "filter": {"name": chosen_filter, "intensity": filter_intensity},
            "effect": {"name": chosen_effect} if chosen_effect else {"name": None},
            "bgm_volume": 0.8
        },
        "per_clip": per_clip,
        "transitions": transitions,
        "text_overlays": text_overlays
    }


def _generate_text_overlay(template_name: str, duration: float) -> List[Dict]:
    """字幕已禁用，等用户提供模板后再启用"""
    return []


# ============================================================
# 便捷接口 — 供pipeline调用
# ============================================================

class EditGrammarEngine:
    """剪辑语法引擎封装，统一LLM/规则两种模式"""
    
    def __init__(self, llm_api_fn=None):
        """
        Args:
            llm_api_fn: 可选的LLM调用函数，签名: fn(prompt: str) -> str (JSON)
                        为None时自动使用规则引擎回退
        """
        self.llm_api_fn = llm_api_fn
        self.use_llm = llm_api_fn is not None
        self.history = []  # 记录历史决策，供优化层使用
    
    def generate_grammar(self, template: Dict, shot_plan: List[Dict], bgm_info: Dict, duration: float) -> Dict:
        """生成编辑语法JSON（LLM优先，回退规则）"""
        if self.use_llm:
            try:
                context = build_edit_context(template, shot_plan, bgm_info)
                prompt = build_grammar_prompt(context)
                llm_response = self.llm_api_fn(prompt)
                grammar = json.loads(llm_response) if isinstance(llm_response, str) else llm_response
                grammar["generated_by"] = "llm"
                self.history.append({"template": template.get("name"), "mode": "llm", "grammar": grammar})
                return grammar
            except Exception as e:
                print(f"[EditGrammar] LLM调用失败，回退规则引擎: {e}")
        
        grammar = generate_rule_grammar(template, shot_plan, bgm_info, duration)
        self.history.append({"template": template.get("name"), "mode": "rule", "grammar": grammar})
        return grammar
    
    def execute(self, project, grammar: Dict, shot_plan: List[Dict], bgm_info: Dict, duration: float) -> Dict:
        """执行编辑语法到剪映项目"""
        return execute_grammar(project, grammar, shot_plan, bgm_info, duration)
    
    def get_history(self) -> List[Dict]:
        return self.history


# ============================================================
# 测试入口
# ============================================================

if __name__ == "__main__":
    import sys
    
    if "--test" in sys.argv:
        print("=" * 50)
        print("EditGrammar 引擎自测")
        print("=" * 50)
        
        # 模拟数据
        template = {"name": "0430 新化氛围感", "duration": 11.82}
        shot_plan = [
            {"video_path": "E:/111/1/test.mov", "score": 78.5, "shot_type": "eating_action",
             "clip_duration": 0.9, "source_start": 2.0,
             "dim_scores": {"appetite": 85, "color": 70, "action": 92}},
            {"video_path": "E:/111/2/test2.mov", "score": 65.2, "shot_type": "closeup",
             "clip_duration": 0.8, "source_start": 1.5,
             "dim_scores": {"appetite": 60, "color": 80, "action": 45}},
            {"video_path": "E:/111/3/test3.mov", "score": 55.0, "shot_type": "medium",
             "clip_duration": 1.0, "source_start": 3.0,
             "dim_scores": {"appetite": 50, "color": 55, "action": 30}},
        ]
        bgm_info = {
            "bgm_name": "夏日限定.wav",
            "beat_plan": {"method": "adaptive", "bpm": 120, "shot_count": 3,
                         "clip_durations": [0.9, 0.8, 1.0]}
        }
        
        # 测试规则引擎
        print("\n--- 规则引擎模式 ---")
        grammar = generate_rule_grammar(template, shot_plan, bgm_info, 11.82)
        print(json.dumps(grammar, ensure_ascii=False, indent=2))
        
        # 测试Context Builder
        print("\n--- LLM上下文构建 ---")
        ctx = build_edit_context(template, shot_plan, bgm_info)
        print(f"上下文长度: {len(ctx)}字符")
        print(ctx[:500] + "...")
        
        print("\n✅ 自测通过")
    else:
        print("EditGrammar引擎已加载。使用 --test 运行自测。")
