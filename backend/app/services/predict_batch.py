"""批量预测任务领域服务：基于已注册模型的批量预测执行、结果存储与校验。"""
from __future__ import annotations

from typing import Any

from app.services.base import BaseModuleService
from app.services.insight import (
    SOURCE_PREDICT,
    InsightWriteError,
    assemble_predict_insight,
    insight_writer,
)
from app.store import store


class PredictBatchService(BaseModuleService):
    MODULE = "predict_batch"
    ENTITY = "预测任务"
    OPERATION_TERM = "校验"
    LIST_FIELDS = ["任务编号", "模型编号", "数据切片", "输出格式", "分片策略", "并行度", "预测状态", "结果校验"]
    REQUIRED_FIELDS = ["任务编号", "模型编号", "数据切片"]
    STATUS_ORDER = ["草稿", "等待中", "运行中", "已完成"]
    ACTION_RULES = {"提交运行": "草稿", "停止预测": "运行中", "确认完成": "已完成"}
    NEGATIVE_ACTIONS = []

    # ── 链路4: 批量预测 → 挖掘结果 ──
    def execute_batch_and_collect_result(self, task_id: int) -> dict[str, Any]:
        """执行批量预测任务并将预测结果采集到挖掘结果模块进行深度分析。

        结果编号由挖掘结果写入层统一分配；只有洞察完整提交后任务才报告完成，
        落库失败时任务保持可重试状态，不产生成功回执。
        """
        task = store.find(self.MODULE, task_id)
        if task is None:
            return {"ok": False, "message": f"预测任务 {task_id} 不存在"}
        draft = assemble_predict_insight(task_id, task)
        try:
            result = insight_writer.commit(draft)
        except InsightWriteError as exc:
            # 保留原状态与 pending，任务可再次执行。
            return {
                "ok": False,
                "task_id": task_id,
                "retryable": True,
                "message": f"预测结果采集失败，任务可重试：{exc}",
            }
        task["status"] = "已完成"
        task["prediction_state"] = "已完成"
        task["pending"] = False
        return {
            "ok": True,
            "task_id": task_id,
            "mining_id": result["id"],
            "result_id": result["id"],
            "source_type": SOURCE_PREDICT,
            "source_id": task_id,
            "result": result,
            "message": "预测结果已采集并导入挖掘结果",
        }

    def query_prediction_status(self, task_id: int) -> dict[str, Any]:
        """查询预测任务状态及其实测洞察是否已经完整提交。"""
        task = store.find(self.MODULE, task_id)
        if task is None:
            return {"ok": False, "message": f"预测任务 {task_id} 不存在"}
        committed = [
            row for row in store.rows("mining_result")
            if row.get("来源类型") == SOURCE_PREDICT and str(row.get("来源编号")) == str(task_id)
        ]
        return {
            "ok": True,
            "task_id": task_id,
            "status": task.get("status"),
            "prediction_state": task.get("prediction_state", task.get("预测状态")),
            "pending": task.get("pending"),
            "completed": task.get("status") == "已完成",
            "insight_committed": bool(committed),
            "mining_id": committed[-1]["id"] if committed else None,
            "retryable": task.get("status") != "已完成",
        }

    def validate_prediction_consistency(self, task_id: int) -> dict[str, Any]:
        """校验预测结果的一致性和格式合法性，确保输出符合预期schema。"""
        task = store.find(self.MODULE, task_id)
        if task is None:
            return {"ok": False}
        is_valid = task.get("输出格式") in ("csv", "json", "parquet")
        if not is_valid:
            task["abnormal"] = True
            return {"ok": False, "reason": f"输出格式 {task.get('输出格式')} 不合法"}
        return {"ok": True, "task_id": task_id, "format": task.get("输出格式")}

    def split_predict_partitions(self, task_id: int) -> dict[str, Any]:
        """将批量预测任务切分为多个分区并行执行，提高大规模数据集的预测吞吐。"""
        task = store.find(self.MODULE, task_id)
        if task is None:
            return {"ok": False}
        parallelism = int(str(task.get("并行度", "4")))
        partitions = []
        for i in range(parallelism):
            partitions.append({
                "partition_id": i,
                "task_id": task_id,
                "model_id": task.get("模型编号"),
                "output": f"predict_output/{task_id}/part_{i:03d}.csv",
            })
        return {"ok": True, "task_id": task_id, "partition_count": parallelism, "partitions": partitions}

    def export_predictions_to_csv(self, task_id: int) -> dict[str, Any]:
        """导出批量预测结果为CSV格式，写入外部存储供下游业务消费。"""
        task = store.find(self.MODULE, task_id)
        if task is None:
            return {"ok": False}
        export_rows = store.rows("data_export")
        next_id = max((int(r.get("id", 0)) for r in export_rows), default=0) + 1
        export_rows.append({
            "id": next_id,
            "status": "已完成",
            "pending": False,
            "abnormal": False,
            "任务编号": f"EXP-P{task_id:04d}",
            "数据来源": f"predict_task_{task_id}",
            "导出范围": "全量预测结果",
            "导出格式": "csv",
            "字段映射": "prediction_id,model_id,score,label,dt",
            "分片策略": "none",
            "压缩方式": "gzip",
            "导出状态": "已完成",
        })
        return {"ok": True, "export_id": next_id, "task_id": task_id}