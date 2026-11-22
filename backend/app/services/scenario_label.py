"""业务场景标注领域服务：数据分析场景的业务标注、规则表达与知识管理。"""
from __future__ import annotations

from typing import Any

from app.services.base import BaseModuleService
from app.store import store


class ScenarioLabelService(BaseModuleService):
    MODULE = "scenario_label"
    ENTITY = "场景标注"
    OPERATION_TERM = "关联"
    LIST_FIELDS = ["标注编号", "场景名称", "关联特征", "业务规则", "标注时间", "标注人", "标注状态", "场景版本"]
    REQUIRED_FIELDS = ["标注编号", "场景名称", "关联特征"]
    STATUS_ORDER = ["草稿", "待校验", "已生效", "已废弃"]
    ACTION_RULES = {"提交校验": "待校验", "标记生效": "已生效", "标记废弃": "已废弃"}
    NEGATIVE_ACTIONS = []

    # ── 链路4: 洞察投影 → 场景标注（来源追溯） ──
    def describe_label_source(self, label_id: int) -> dict[str, Any]:
        """查询场景标注的来源明细，区分来自模型采集还是预测任务。"""
        label = store.find(self.MODULE, label_id)
        if label is None:
            return {"ok": False}
        source_type = label.get("来源类型")
        return {
            "ok": True,
            "label_id": label_id,
            "场景名称": label.get("场景名称"),
            "来源类型": source_type,
            "来源编号": label.get("来源编号"),
            "来源名称": label.get("来源名称", label.get("场景名称")),
            "结果ID": label.get("结果ID"),
            "关联特征": label.get("关联特征", ""),
            "业务规则": label.get("业务规则", ""),
            "由洞察投影生成": source_type is not None,
        }

    # ── 链路6: 聚类配置 → 分群 → 场景标注 ──
    def validate_label_rules(self, label_id: int) -> dict[str, Any]:
        """校验场景标注的业务规则合法性，检查关联特征是否存在且可查询。"""
        label = store.find(self.MODULE, label_id)
        if label is None:
            return {"ok": False}
        feats = label.get("关联特征", "")
        define_rows = store.rows("feature_define")
        matched = [r for r in define_rows if r.get("特征名称", "") in feats.split(",")]
        if len(matched) == 0:
            label["status"] = "待校验"
            label["abnormal"] = True
            return {"ok": False, "reason": "关联特征均不存在于特征定义中"}
        label["status"] = "已生效"
        label["abnormal"] = False
        return {"ok": True, "label_id": label_id, "matched_features": len(matched)}

    def link_to_cluster_segments(self, label_id: int) -> dict[str, Any]:
        """将场景标注与聚类分群关联，形成聚类→分群→标注的完整链路。"""
        label = store.find(self.MODULE, label_id)
        if label is None:
            return {"ok": False}
        cluster_rows = store.rows("cluster_config")
        related = []
        for c in cluster_rows:
            if c.get("特征集") and label.get("关联特征") and any(
                f.strip() in str(c.get("特征集", "")) for f in str(label.get("关联特征", "")).split(",")
            ):
                related.append({"cluster_id": c.get("id"), "algorithm": c.get("算法类型")})
        return {"ok": True, "label_id": label_id, "related_clusters": related, "count": len(related)}

    def export_label_knowledge_base(self) -> dict[str, Any]:
        """导出全量场景标注知识库，按业务主题聚合后供给外部系统查询。"""
        rows = store.rows(self.MODULE)
        by_status = {}
        for r in rows:
            st = r.get("status", "unknown")
            if st not in by_status:
                by_status[st] = []
            by_status[st].append({
                "id": r.get("id"),
                "scene": r.get("场景名称"),
                "features": r.get("关联特征"),
                "rules": r.get("业务规则"),
                "annotator": r.get("标注人"),
            })
        return {"ok": True, "total_labels": len(rows), "by_status": by_status}

    def standardize_label_format(self, label_id: int) -> dict[str, Any]:
        """标准化场景标注的格式，确保规则表达符合统一的标注规范。"""
        label = store.find(self.MODULE, label_id)
        if label is None:
            return {"ok": False}
        raw_rules = label.get("业务规则", "")
        standardized = raw_rules.replace(" ", "_").replace("，", ",").lower()
        label["业务规则"] = standardized
        return {"ok": True, "label_id": label_id, "raw": raw_rules, "standardized": standardized}