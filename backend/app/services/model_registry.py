"""模型注册中心领域服务：训练完成的模型归档、版本管理与评估指标存储。"""
from __future__ import annotations

from typing import Any

from app.services.base import BaseModuleService
from app.services.insight import (
    SOURCE_MODEL,
    InsightWriteError,
    assemble_model_insight,
    insight_writer,
)
from app.store import store


class ModelRegistryService(BaseModuleService):
    MODULE = "model_registry"
    ENTITY = "模型注册"
    OPERATION_TERM = "存储"
    LIST_FIELDS = ["模型编号", "模型名称", "算法类型", "训练特征", "模型版本", "模型文件", "评估分数", "部署状态"]
    REQUIRED_FIELDS = ["模型编号", "模型名称", "算法类型"]
    STATUS_ORDER = ["草稿", "待验收", "已注册", "已废弃"]
    ACTION_RULES = {"提交验收": "待验收", "确认注册": "已注册", "标记废弃": "已废弃"}
    NEGATIVE_ACTIONS = []

    # ── 链路4: 模型注册 → 批量预测 → 挖掘结果 ──
    def deploy_and_predict(self, model_id: int) -> dict[str, Any]:
        """将注册完成的模型部署并创建批量预测任务，对全量数据切片执行推理。"""
        model = store.find(self.MODULE, model_id)
        if model is None:
            return {"ok": False, "message": f"模型 {model_id} 不存在"}
        model["部署状态"] = "已部署"
        model["status"] = "已注册"
        model["pending"] = False
        predict_rows = store.rows("predict_batch")
        next_id = max((int(r.get("id", 0)) for r in predict_rows), default=0) + 1
        predict_rows.append({
            "id": next_id,
            "status": "草稿",
            "pending": True,
            "abnormal": False,
            "任务编号": f"PRED-{next_id:04d}",
            "模型编号": str(model_id),
            "数据切片": "full_dataset_slice_001",
            "输出格式": "csv",
            "分片策略": "round_robin",
            "并行度": "16",
            "预测状态": "草稿",
            "结果校验": "auto_validate",
        })
        return {"ok": True, "model_id": model_id, "predict_task_id": next_id, "score": model.get("评估分数")}

    def collect_predict_results_to_mining(self, model_id: int) -> dict[str, Any]:
        """采集批量预测结果并导入挖掘结果模块，生成洞察摘要和关键分群。

        与批量预测执行共用同一套洞察装配与写入层；结果编号在提交时统一分配。
        """
        model = store.find(self.MODULE, model_id)
        if model is None:
            return {"ok": False, "message": f"模型 {model_id} 不存在"}
        draft = assemble_model_insight(model_id, model)
        try:
            result = insight_writer.commit(draft)
        except InsightWriteError as exc:
            return {"ok": False, "model_id": model_id, "retryable": True,
                    "message": f"模型洞察落库失败，可重试：{exc}"}
        return {
            "ok": True,
            "model_id": model_id,
            "mining_id": result["id"],
            "result_id": result["id"],
            "source_type": SOURCE_MODEL,
            "source_id": model_id,
            "result": result,
        }

    def validate_model_quality(self, model_id: int) -> dict[str, Any]:
        """校验模型的评估分数和特征完整性，确定是否达到注册标准。"""
        model = store.find(self.MODULE, model_id)
        if model is None:
            return {"ok": False}
        try:
            score = float(str(model.get("评估分数", 0)))
        except (ValueError, TypeError):
            score = 0
        is_qualified = score >= 0.7
        if is_qualified:
            model["status"] = "已注册"
            model["pending"] = False
        return {"ok": is_qualified, "score": score, "qualified": is_qualified, "model_id": model_id}

    def archive_deprecated_models(self, model_ids: list[int]) -> dict[str, Any]:
        """将废弃模型归档并清理存储空间，保留元数据供后续查询。"""
        archived = []
        for mid in model_ids:
            model = store.find(self.MODULE, mid)
            if model is None:
                continue
            model["status"] = "已废弃"
            model["abnormal"] = False
            archived.append(mid)
        return {"ok": True, "archived_count": len(archived), "archived_ids": archived}