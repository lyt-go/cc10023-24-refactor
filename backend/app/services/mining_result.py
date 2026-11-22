"""挖掘结果管理领域服务：分析结果、洞察发现与关键发现的管理。"""
from __future__ import annotations

from typing import Any

from app.services.base import BaseModuleService
from app.services.insight import insight_writer, source_display_name
from app.store import store


class MiningResultService(BaseModuleService):
    MODULE = "mining_result"
    ENTITY = "挖掘结果"
    OPERATION_TERM = "存储"
    LIST_FIELDS = ["结果编号", "分析主题", "算法类型", "洞察摘要", "关键分群", "置信度", "结果文件", "结果版本"]
    REQUIRED_FIELDS = ["结果编号", "分析主题", "算法类型"]
    STATUS_ORDER = ["草稿", "审核中", "已采纳", "已搁置"]
    ACTION_RULES = {"提交审核": "审核中", "确认采纳": "已采纳", "标记搁置": "已搁置"}
    NEGATIVE_ACTIONS = []

    # ── 链路4: 预测 → 挖掘结果 ──
    def export_insights_to_labels(self, result_id: Any) -> dict[str, Any]:
        """按成功回执读取洞察投影结果并导出为业务场景标注（场景规则）。

        ``result_id`` 可以是模型注册/批量预测返回的成功回执，也可以是结果 ID；
        失败回执或尚未提交的结果不会生成标注。已有洞察转标注保持原有规则字段。
        """
        result = self._resolve_receipt(result_id)
        if result is None:
            return {"ok": False, "message": "回执对应的挖掘结果尚未提交，无法转标注"}
        label_rows = store.rows("scenario_label")
        next_id = max((int(r.get("id", 0)) for r in label_rows), default=0) + 1
        resolved_id = result["id"]
        source_type = result.get("来源类型")
        source_id = result.get("来源编号")
        if source_type:
            scene_name = (
                f"挖掘场景_{source_display_name(source_type, source_id)}"
                f"_{result.get('分析主题', 'unknown')}"
            )
        else:
            scene_name = f"挖掘场景_{result.get('分析主题', 'unknown')}"
        result["status"] = "已采纳"
        result["pending"] = False
        label = {
            "id": next_id,
            "status": "草稿",
            "pending": True,
            "abnormal": False,
            "标注编号": f"LABEL-{next_id:04d}",
            "场景名称": scene_name,
            # 规则字段沿用对应结果的分群及摘要，分群保持原有顺序。
            "关联特征": result.get("关键分群", ""),
            "业务规则": result.get("洞察摘要", ""),
            "标注时间": "2026-11-18",
            "标注人": "system_auto",
            "标注状态": "草稿",
            "场景版本": "v1.0.0",
        }
        if source_type:
            label["结果ID"] = resolved_id
            label["来源类型"] = source_type
            label["来源编号"] = str(source_id)
            label["来源名称"] = source_display_name(source_type, source_id)
        label_rows.append(label)
        return {
            "ok": True,
            "result_id": resolved_id,
            "label_id": next_id,
            "source_type": source_type,
            "source_id": source_id,
            "关联特征": label["关联特征"],
            "业务规则": label["业务规则"],
        }

    def _resolve_receipt(self, receipt: Any) -> dict[str, Any] | None:
        """把回执解析成结果表里已经完整提交的那一行。"""
        if isinstance(receipt, dict):
            return insight_writer.find_committed(receipt)
        if isinstance(receipt, int):
            return store.find(self.MODULE, receipt)
        if isinstance(receipt, str) and receipt.strip().isdigit():
            return store.find(self.MODULE, int(receipt.strip()))
        return None

    # ── 挖掘结果写入层对外入口：分配编号 + 完整提交 ──
    def allocate_insight_id(self) -> int:
        """预留一个挖掘结果编号，供需要显式交错的写入方使用。"""
        return insight_writer.allocate_result_id()

    def release_insight_id(self, result_id: int) -> None:
        insight_writer.release_result_id(result_id)

    def commit_insight(self, draft: Any) -> dict[str, Any]:
        """提交一条洞察记录并返回成功回执；仅在完整落表后返回。"""
        return insight_writer.commit(draft)

    def query_insight_projection(self, result_id: Any) -> dict[str, Any]:
        """查询预测洞察投影结果，输出可追溯的来源与规则信息。"""
        result = self._resolve_receipt(result_id)
        if result is None:
            return {"ok": False, "message": "洞察投影结果尚未提交"}
        source_type = result.get("来源类型")
        return {
            "ok": True,
            "result_id": result["id"],
            "结果编号": result.get("结果编号"),
            "分析主题": result.get("分析主题"),
            "洞察类型": result.get("洞察类型", ""),
            "来源类型": source_type,
            "来源编号": result.get("来源编号"),
            "来源名称": (source_display_name(source_type, result.get("来源编号"))
                       if source_type else ""),
            "关键分群": result.get("关键分群", ""),
            "洞察摘要": result.get("洞察摘要", ""),
            "结果文件": result.get("结果文件"),
        }

    def aggregate_cluster_insights(self, result_id: int) -> dict[str, Any]:
        """聚合挖掘结果中关键分群的洞察数据并关联到聚类分析配置。"""
        result = store.find(self.MODULE, result_id)
        if result is None:
            return {"ok": False}
        cluster_rows = store.rows("cluster_config")
        related_clusters = [
            r.get("id") for r in cluster_rows
            if result.get("关键分群", "") and any(g.strip() in str(r.get("特征集", ""))
            for g in result.get("关键分群", "").split(","))
        ]
        return {
            "ok": True,
            "result_id": result_id,
            "clusters": result.get("关键分群", "").split(","),
            "related_cluster_configs": related_clusters,
        }

    def convert_insight_to_feature_source(self, result_id: int) -> dict[str, Any]:
        """将采纳的挖掘洞察转换为新的特征数据源，丰富下游模型的特征输入。"""
        result = store.find(self.MODULE, result_id)
        if result is None:
            return {"ok": False}
        source_rows = store.rows("feature_source")
        next_id = max((int(r.get("id", 0)) for r in source_rows), default=0) + 1
        source_rows.append({
            "id": next_id,
            "status": "草稿",
            "pending": True,
            "abnormal": False,
            "源编号": f"SRC-R{result_id:04d}",
            "源名称": f"洞察衍生特征_{result.get('分析主题', '')}",
            "数据表名": f"insight_derived_{result_id}",
            "连接配置": "hive_metastore",
            "字段清单": "insight_score,cluster_label,confidence",
            "采样比例": "0.5",
            "缓存策略": "memory_1h",
            "源版本": "v1.0.0",
        })
        return {"ok": True, "source_id": next_id, "result_id": result_id}

    def deduplicate_mining_results(self, result_ids: list[int]) -> dict[str, Any]:
        """对挖掘结果集进行去重处理，识别相似洞察并合并重复发现。"""
        topics = {}
        duplicates = []
        for rid in result_ids:
            r = store.find(self.MODULE, rid)
            if r is None:
                continue
            topic = r.get("分析主题", "")
            if topic in topics:
                duplicates.append({"id": rid, "similar_to": topics[topic]})
                r["status"] = "已搁置"
            else:
                topics[topic] = rid
        return {"ok": True, "unique_count": len(topics), "duplicate_count": len(duplicates),
                "duplicates": duplicates}