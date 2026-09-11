"""
自我优化层 v1.0 — Self-Optimizer
=================================
剪辑完成后自动分析质量趋势，提炼最佳实践，持续改进参数。

核心能力:
  1. 质量指标采集  — 从Pipeline日志提取可量化指标
  2. 趋势分析      — 关联编辑决策与输出质量
  3. 参数调优建议  — 基于历史数据推荐最优配置
  4. 经验持久化    — JSON历史记录，跨会话累积

使用方式:
  from self_optimizer import Optimizer
  opt = Optimizer()
  
  # 收集本轮数据
  opt.collect_batch(pipeline.log, grammar_engine.history)
  
  # 生成分析报告
  report = opt.analyze()
  print(report)
  
  # 获取调优建议
  suggestions = opt.suggest()
  for s in suggestions:
      print(f"  {s['category']}: {s['recommendation']}")
"""

import json
import os
import time
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from collections import Counter, defaultdict


# ============================================================
# 数据结构
# ============================================================

class RunRecord:
    """单次运行的记录"""
    def __init__(self, seq: int, template: str, duration: float):
        self.seq = seq
        self.template = template
        self.duration = duration
        self.timestamp = datetime.now().isoformat()
        
        # 素材指标
        self.total_materials = 0
        self.analyzed_materials = 0
        self.avg_shot_score = 0.0
        self.shot_types = {}
        
        # BGM指标
        self.bgm_name = ""
        self.bgm_bpm = 0
        self.beat_method = ""
        self.shot_count = 0
        
        # 渲染指标
        self.clips_added = 0
        self.clips_total = 0
        self.clip_ratio = 0.0
        self.transitions_added = 0
        self.keyframes_added = 0
        self.intros_added = 0
        self.texts_added = 0
        
        # 编辑决策
        self.filter_name = ""
        self.filter_intensity = 0
        self.effect_name = ""
        self.grammar_mode = "rule"
        
        # 质量
        self.qc_passed = False
        self.qc_issues = []
        self.file_size_mb = 0.0
        self.export_ok = False
        self.bgm_ok = False
        self.filter_ok = False
        self.effect_ok = False
        
        # 耗时
        self.total_time = 0.0
        self.step_times = {}
    
    def to_dict(self) -> Dict:
        return {
            "seq": self.seq, "template": self.template, "duration": self.duration,
            "timestamp": self.timestamp,
            "materials": {"total": self.total_materials, "analyzed": self.analyzed_materials,
                         "avg_score": round(self.avg_shot_score, 1), "shot_types": self.shot_types},
            "bgm": {"name": self.bgm_name, "bpm": self.bgm_bpm,
                   "method": self.beat_method, "shot_count": self.shot_count},
            "render": {"clips": f"{self.clips_added}/{self.clips_total}",
                      "ratio": round(self.clip_ratio, 2),
                      "transitions": self.transitions_added,
                      "keyframes": self.keyframes_added,
                      "intros": self.intros_added, "texts": self.texts_added},
            "editing": {"filter": self.filter_name, "filter_intensity": self.filter_intensity,
                       "effect": self.effect_name, "grammar_mode": self.grammar_mode},
            "quality": {"qc_passed": self.qc_passed, "issues": self.qc_issues,
                       "file_size_mb": self.file_size_mb, "export_ok": self.export_ok,
                       "bgm_ok": self.bgm_ok, "filter_ok": self.filter_ok, "effect_ok": self.effect_ok},
            "timing": {"total": round(self.total_time, 1), "steps": self.step_times}
        }
    
    def quality_score(self) -> float:
        """综合质量分 0~100"""
        score = 0.0
        if self.qc_passed: score += 30
        if self.export_ok: score += 25
        if self.bgm_ok: score += 10
        if self.filter_ok: score += 10
        if self.effect_ok: score += 10
        score += min(self.clip_ratio * 15, 15)
        if self.transitions_added > 0: score += 5
        if self.keyframes_added > 0: score += 3
        if self.intros_added > 0: score += 2
        return min(score, 100)


# ============================================================
# 核心优化器
# ============================================================

class Optimizer:
    """自我优化器：采集、分析、建议"""
    
    def __init__(self, history_file: str = None):
        self.records: List[RunRecord] = []
        self.history_file = history_file or os.path.join(
            os.path.dirname(os.path.abspath(__file__)), ".optimizer_history.json"
        )
        self._load_history()
    
    def _load_history(self):
        try:
            if os.path.exists(self.history_file):
                with open(self.history_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    for item in data:
                        r = RunRecord(item["seq"], item["template"], item["duration"])
                        r.timestamp = item.get("timestamp", "")
                        m = item.get("materials", {})
                        r.total_materials = m.get("total", 0)
                        r.analyzed_materials = m.get("analyzed", 0)
                        r.avg_shot_score = m.get("avg_score", 0)
                        r.shot_types = m.get("shot_types", {})
                        b = item.get("bgm", {})
                        r.bgm_name = b.get("name", "")
                        r.bgm_bpm = b.get("bpm", 0)
                        r.beat_method = b.get("method", "")
                        r.shot_count = b.get("shot_count", 0)
                        rd = item.get("render", {})
                        parts = rd.get("clips", "0/1").split("/")
                        r.clips_added = int(parts[0]) if len(parts) > 0 else 0
                        r.clips_total = int(parts[1]) if len(parts) > 1 else 1
                        r.clip_ratio = rd.get("ratio", 0)
                        r.transitions_added = rd.get("transitions", 0)
                        r.keyframes_added = rd.get("keyframes", 0)
                        r.intros_added = rd.get("intros", 0)
                        r.texts_added = rd.get("texts", 0)
                        e = item.get("editing", {})
                        r.filter_name = e.get("filter", "")
                        r.filter_intensity = e.get("filter_intensity", 0)
                        r.effect_name = e.get("effect", "")
                        r.grammar_mode = e.get("grammar_mode", "rule")
                        q = item.get("quality", {})
                        r.qc_passed = q.get("qc_passed", False)
                        r.qc_issues = q.get("issues", [])
                        r.file_size_mb = q.get("file_size_mb", 0)
                        r.export_ok = q.get("export_ok", False)
                        r.bgm_ok = q.get("bgm_ok", False)
                        r.filter_ok = q.get("filter_ok", False)
                        r.effect_ok = q.get("effect_ok", False)
                        t = item.get("timing", {})
                        r.total_time = t.get("total", 0)
                        r.step_times = t.get("steps", {})
                        self.records.append(r)
        except Exception:
            pass
    
    def _save_history(self):
        try:
            os.makedirs(os.path.dirname(self.history_file), exist_ok=True)
            with open(self.history_file, 'w', encoding='utf-8') as f:
                json.dump([r.to_dict() for r in self.records], f, ensure_ascii=False, indent=2)
        except Exception:
            pass
    
    def collect_from_pipeline_log(self, pipeline_log: List[Dict], grammar_history: List[Dict] = None):
        """从Pipeline的log和Grammar的history中提取指标
        
        Args:
            pipeline_log: VideoPipeline.log 列表
            grammar_history: EditGrammarEngine.history 列表
        """
        grammar_map = {}
        if grammar_history:
            for gh in grammar_history:
                grammar_map[gh.get("template", "")] = gh
        
        for entry in pipeline_log:
            seq = entry.get("seq", 0)
            template_name = entry.get("template", "未知")
            duration = 0
            
            # 查找对应模板的时长
            from pipeline import ALL_TEMPLATES
            for t in ALL_TEMPLATES:
                if t["name"] == template_name:
                    duration = t["duration"]
                    break
            
            r = RunRecord(seq, template_name, duration)
            
            steps = entry.get("steps", {})
            r.total_time = entry.get("total_time", 0)
            
            # Step 1: 素材
            pre = steps.get("preprocess", {})
            r.total_materials = pre.get("total", pre.get("valid", 0))  # 兼容新旧格式
            
            # Step 2: 评分 (从plan推断)
            score_data = steps.get("score", {})
            # 从plan中取平均值
            plan_data = steps.get("plan", {})
            
            # Step 3: BGM
            bgm_data = steps.get("bgm", {})
            r.bgm_name = bgm_data.get("bgm", "")
            r.beat_method = bgm_data.get("beat_method", "")
            r.shot_count = bgm_data.get("shots", 0)
            
            # Step 5: 渲染
            render_data = steps.get("render", {})
            clips_str = render_data.get("clips", "0/1")
            parts = clips_str.split("/")
            r.clips_added = int(parts[0]) if len(parts) > 0 else 0
            r.clips_total = int(parts[1]) if len(parts) > 1 else 1
            r.clip_ratio = r.clips_added / max(r.clips_total, 1)
            r.transitions_added = render_data.get("transitions", 0)
            r.keyframes_added = render_data.get("keyframes", 0)
            r.intros_added = render_data.get("intros", 0)
            r.texts_added = render_data.get("texts", 0)
            
            # 编辑决策（从grammar history获取）
            gh = grammar_map.get(template_name, {})
            grammar = gh.get("grammar", {})
            r.grammar_mode = gh.get("mode", "rule")
            g = grammar.get("global", {})
            r.filter_name = g.get("filter", {}).get("name", "") if isinstance(g.get("filter"), dict) else ""
            r.filter_intensity = g.get("filter", {}).get("intensity", 0) if isinstance(g.get("filter"), dict) else 0
            r.effect_name = g.get("effect", {}).get("name", "") if isinstance(g.get("effect"), dict) else ""
            
            # Step 6: 质检
            qc_data = steps.get("qc", {})
            r.qc_passed = qc_data.get("passed", False)
            r.qc_issues = qc_data.get("issues", [])
            r.export_ok = render_data.get("export", False)
            
            # 耗时
            r.step_times = {k: v.get("time", 0) for k, v in steps.items()}
            
            # 质量指标
            r.filter_ok = not any("滤镜" in i for i in r.qc_issues)
            r.effect_ok = not any("特效" in i for i in r.qc_issues)
            r.bgm_ok = not any("BGM" in i for i in r.qc_issues)
            
            self.records.append(r)
        
        self._save_history()
    
    def analyze(self) -> str:
        """生成分析报告"""
        if not self.records:
            return "暂无数据，请先运行一批生成任务。"
        
        lines = []
        lines.append("=" * 60)
        lines.append("📊 自我优化分析报告")
        lines.append("=" * 60)
        
        total = len(self.records)
        passed = sum(1 for r in self.records if r.qc_passed)
        export_ok = sum(1 for r in self.records if r.export_ok)
        
        lines.append(f"\n## 总体统计")
        lines.append(f"  总批次数: {total}")
        lines.append(f"  质检通过率: {passed}/{total} ({passed*100//total}%)")
        lines.append(f"  导出成功率: {export_ok}/{total} ({export_ok*100//total}%)")
        
        # 剪辑效率
        avg_ratio = sum(r.clip_ratio for r in self.records) / max(total, 1)
        avg_trans = sum(r.transitions_added for r in self.records) / max(total, 1)
        avg_kfs = sum(r.keyframes_added for r in self.records) / max(total, 1)
        lines.append(f"\n## 剪辑效率")
        lines.append(f"  平均片段添加率: {avg_ratio*100:.0f}%")
        lines.append(f"  平均转场数: {avg_trans:.1f}")
        lines.append(f"  平均关键帧运动: {avg_kfs:.1f}")
        
        # 滤镜效果对比
        filter_stats = defaultdict(lambda: {"count": 0, "passed": 0, "scores": []})
        for r in self.records:
            if r.filter_name:
                fs = filter_stats[r.filter_name]
                fs["count"] += 1
                if r.qc_passed:
                    fs["passed"] += 1
                fs["scores"].append(r.quality_score())
        
        if filter_stats:
            lines.append(f"\n## 滤镜效果对比")
            lines.append(f"  {'滤镜':<12s} {'使用次数':>6s} {'通过率':>8s} {'平均质量分':>10s}")
            lines.append(f"  {'-'*40}")
            for fname, fs in sorted(filter_stats.items(), key=lambda x: -x[1]["passed"]/max(x[1]["count"],1)):
                pass_rate = fs["passed"] / max(fs["count"], 1) * 100
                avg_score = sum(fs["scores"]) / max(len(fs["scores"]), 1)
                lines.append(f"  {fname:<12s} {fs['count']:>6d} {pass_rate:>7.0f}% {avg_score:>9.1f}")
        
        # 编辑模式对比
        mode_stats = defaultdict(lambda: {"count": 0, "passed": 0})
        for r in self.records:
            ms = mode_stats[r.grammar_mode]
            ms["count"] += 1
            if r.qc_passed:
                ms["passed"] += 1
        
        lines.append(f"\n## 编辑模式对比")
        for mode, ms in mode_stats.items():
            mode_name = {"rule": "规则引擎", "llm": "LLM引擎"}.get(mode, mode)
            pass_rate = ms["passed"] / max(ms["count"], 1) * 100
            lines.append(f"  {mode_name}: {pass_rate:.0f}%通过 ({ms['passed']}/{ms['count']})")
        
        # 耗时分析
        avg_total = sum(r.total_time for r in self.records) / max(total, 1)
        lines.append(f"\n## 耗时分析")
        lines.append(f"  平均总耗时: {avg_total:.1f}s/条")
        
        step_names = {"preprocess": "素材收集", "score": "素材评分", "bgm": "BGM匹配",
                      "plan": "镜头方案", "render": "渲染导出", "qc": "质检"}
        for sk, slabel in step_names.items():
            times = [r.step_times.get(sk, 0) for r in self.records if r.step_times.get(sk, 0) > 0]
            if times:
                avg = sum(times) / len(times)
                lines.append(f"  {slabel}: 平均{avg:.1f}s")
        
        # 问题汇总
        all_issues = Counter()
        for r in self.records:
            for issue in r.qc_issues:
                all_issues[issue] += 1
        
        if all_issues:
            lines.append(f"\n## 常见问题")
            for issue, count in all_issues.most_common(10):
                lines.append(f"  [{count}次] {issue}")
        
        lines.append(f"\n{'='*60}")
        return "\n".join(lines)
    
    def suggest(self) -> List[Dict]:
        """基于历史数据生成参数调优建议"""
        if len(self.records) < 3:
            return [{"category": "数据不足", "recommendation": "至少需要3条记录才能生成有意义的建议", "confidence": "low"}]
        
        suggestions = []
        
        # 1. 滤镜推荐
        filter_stats = defaultdict(lambda: {"count": 0, "passed": 0, "avg_quality": 0})
        for r in self.records:
            if r.filter_name:
                fs = filter_stats[r.filter_name]
                fs["count"] += 1
                if r.qc_passed:
                    fs["passed"] += 1
        
        if filter_stats:
            best_filter = max(filter_stats.items(), 
                            key=lambda x: x[1]["passed"] / max(x[1]["count"], 1))
            pass_rate = best_filter[1]["passed"] / max(best_filter[1]["count"], 1)
            if pass_rate >= 0.8 and best_filter[1]["count"] >= 2:
                suggestions.append({
                    "category": "滤镜选择",
                    "recommendation": f"推荐优先使用「{best_filter[0]}」滤镜（{pass_rate*100:.0f}%通过率，{best_filter[1]['count']}次使用）",
                    "confidence": "high"
                })
        
        # 2. 转场密度建议
        avg_trans = sum(r.transitions_added for r in self.records) / len(self.records)
        if avg_trans < 1 and len(self.records) > 0:
            suggestions.append({
                "category": "转场密度",
                "recommendation": f"当前平均{avg_trans:.1f}个转场/视频，建议增加到镜头数-1以提升流畅度",
                "confidence": "medium"
            })
        
        # 3. 关键帧运动建议
        avg_kfs = sum(r.keyframes_added for r in self.records) / len(self.records)
        last_shot_count = self.records[-1].shot_count if self.records else 0
        if avg_kfs < last_shot_count * 0.3:
            suggestions.append({
                "category": "关键帧运动",
                "recommendation": "建议为更多镜头添加关键帧运动（slow_zoom_in），提升画面动态感",
                "confidence": "medium"
            })
        
        # 4. 入场动画建议
        avg_intros = sum(r.intros_added for r in self.records) / len(self.records)
        if avg_intros < 2 and len(self.records) > 0:
            suggestions.append({
                "category": "入场动画",
                "recommendation": "建议为前2-3个镜头添加入场动画（渐显/动感放大），提升开篇吸引力",
                "confidence": "low"
            })
        
        # 5. 片段添加率优化
        avg_ratio = sum(r.clip_ratio for r in self.records) / len(self.records)
        if avg_ratio < 0.85:
            suggestions.append({
                "category": "片段添加",
                "recommendation": f"片段添加率偏低({avg_ratio*100:.0f}%)，建议检查素材格式兼容性",
                "confidence": "high"
            })
        
        # 6. 特效使用
        effect_used = sum(1 for r in self.records if r.effect_name)
        if effect_used < len(self.records) * 0.5:
            suggestions.append({
                "category": "画面特效",
                "recommendation": "建议尝试添加「电影感画幅」特效提升电影质感",
                "confidence": "low"
            })
        
        return suggestions
    
    def get_best_config(self) -> Dict:
        """提取历史最佳配置"""
        if not self.records:
            return {}
        
        # 评分最高的记录
        best = max(self.records, key=lambda r: r.quality_score())
        
        # 最常用的成功配置
        filter_counter = Counter()
        for r in self.records:
            if r.qc_passed and r.filter_name:
                filter_counter[r.filter_name] += 1
        
        return {
            "best_quality_score": best.quality_score(),
            "best_template": best.template,
            "recommended_filter": filter_counter.most_common(1)[0][0] if filter_counter else "食光II",
            "recommended_intensity": best.filter_intensity or 45,
            "recommended_effect": best.effect_name or "电影感画幅",
            "avg_transitions_per_video": round(sum(r.transitions_added for r in self.records) / len(self.records), 1),
            "avg_keyframes_per_video": round(sum(r.keyframes_added for r in self.records) / len(self.records), 1),
        }


# ============================================================
# 测试入口
# ============================================================

if __name__ == "__main__":
    import sys
    
    if "--test" in sys.argv:
        print("=" * 50)
        print("Optimizer 自测")
        print("=" * 50)
        
        opt = Optimizer()
        
        # 模拟数据
        mock_log = [
            {
                "seq": 1, "template": "0430 新化氛围感", "total_time": 45.2,
                "steps": {
                    "preprocess": {"total": 1636, "valid": 1636, "time": 0.02},
                    "score": {"total": 1636, "new_analyzed": 5, "time": 18.5},
                    "bgm": {"bgm": "夏日限定.wav", "beat_method": "adaptive", "shots": 10, "time": 2.1},
                    "plan": {"shots": 10, "time": 0.3},
                    "render": {"clips": "10/10", "transitions": 9, "keyframes": 3, "intros": 2, "texts": 2,
                              "export": True, "time": 25.0},
                    "qc": {"passed": True, "issues": [], "time": 0.1}
                }
            },
            {
                "seq": 2, "template": "9月25日海阔天空", "total_time": 42.8,
                "steps": {
                    "preprocess": {"total": 1636, "valid": 1636, "time": 0.02},
                    "score": {"total": 1636, "new_analyzed": 0, "time": 10.5},
                    "bgm": {"bgm": "海阔天空.mp3", "beat_method": "uniform", "shots": 12, "time": 1.5},
                    "plan": {"shots": 12, "time": 0.3},
                    "render": {"clips": "11/12", "transitions": 10, "keyframes": 1, "intros": 1, "texts": 2,
                              "export": True, "time": 30.0},
                    "qc": {"passed": True, "issues": [], "time": 0.1}
                }
            },
            {
                "seq": 3, "template": "0508昆明", "total_time": 40.1,
                "steps": {
                    "preprocess": {"total": 1636, "valid": 1636, "time": 0.02},
                    "score": {"total": 1636, "new_analyzed": 0, "time": 9.0},
                    "bgm": {"bgm": "昆明.mp3", "beat_method": "adaptive", "shots": 11, "time": 2.0},
                    "plan": {"shots": 11, "time": 0.3},
                    "render": {"clips": "11/11", "transitions": 10, "keyframes": 4, "intros": 2, "texts": 2,
                              "export": True, "time": 28.0},
                    "qc": {"passed": True, "issues": [], "time": 0.1}
                }
            }
        ]
        
        mock_grammar = [
            {"template": "0430 新化氛围感", "mode": "rule",
             "grammar": {"global": {"filter": {"name": "亮夏", "intensity": 55},
                                   "effect": {"name": "电影感画幅"}}}},
            {"template": "9月25日海阔天空", "mode": "rule",
             "grammar": {"global": {"filter": {"name": "食光II", "intensity": 40},
                                   "effect": {"name": None}}}},
            {"template": "0508昆明", "mode": "rule",
             "grammar": {"global": {"filter": {"name": "食光II", "intensity": 45},
                                   "effect": {"name": "CCD闪光"}}}},
        ]
        
        opt.collect_from_pipeline_log(mock_log, mock_grammar)
        print(f"收集完成: {len(opt.records)}条记录\n")
        
        report = opt.analyze()
        print(report)
        
        print("\n--- 调优建议 ---")
        for s in opt.suggest():
            print(f"  [{s['confidence']}] {s['category']}: {s['recommendation']}")
        
        print("\n--- 最佳配置 ---")
        best = opt.get_best_config()
        for k, v in best.items():
            print(f"  {k}: {v}")
        
        print("\n✅ 自测通过")
    else:
        print("Optimizer已加载。使用 --test 运行自测。")
