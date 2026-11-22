"""洞察装配与落表：模型注册采集与批量预测执行共用的预测洞察投影写入层。

两种来源（模型注册、批量预测任务）只负责装配自己的洞察草稿，结果编号分配、
提交与成功回执统一由 :class:`MiningInsightWriter` 完成。写入层在提交时才在锁内
占号，并且只有整条记录装配成功后才在锁内一次性落表，因此即使两个来源的原始
数字编号相同、甚至在「已分配、未提交」的窗口里交错执行，挖掘结果表中的
``id`` 与结果编号仍然唯一。
"""
from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any, Callable

from app.store import store

MINING_MODULE = "mining_result"

# 来源类型沿用模块 key，来源编号保留原始业务编号，便于跨模块追溯。
SOURCE_MODEL = "model_registry"
SOURCE_PREDICT = "predict_batch"

# 同一张挖掘结果表里标识「这是预测洞察投影结果」。
INSIGHT_KIND = "预测洞察投影"

DEFAULT_VERSION = "v1.0.0"


class InsightWriteError(RuntimeError):
    """洞察落库失败：本次没有产生可用的成功回执。"""


@dataclass
class InsightDraft:
    """尚未分配结果编号的洞察草稿。

    ``code_template`` 只包含结果编号的格式，真正的结果序号由写入层在提交时
    统一分配，装配阶段不读取、也不占用挖掘结果表的编号。
    """

    source_type: str
    source_id: int
    topic: str
    algorithm: str
    summary: str
    clusters: str
    confidence: str
    file_path: str
    code_template: str
    version: str = DEFAULT_VERSION
    attributes: dict[str, Any] = field(default_factory=dict)

    @property
    def source_key(self) -> str:
        return f"{self.source_type}:{self.source_id}"

    def code_for(self, result_id: int) -> str:
        return self.code_template.format(result_id=result_id)

    def build_row(self, result_id: int) -> dict[str, Any]:
        """装配一条完整的结果记录；任何字段缺失都由提交前的这一步暴露。"""
        code = self.code_for(result_id)
        if not code or not self.topic or not self.summary:
            raise InsightWriteError(f"洞察 {self.source_key} 装配不完整，无法提交")
        row: dict[str, Any] = {
            "id": result_id,
            "status": "草稿",
            "pending": True,
            "abnormal": False,
            "结果编号": code,
            "分析主题": self.topic,
            "算法类型": self.algorithm,
            "洞察摘要": self.summary,
            "关键分群": self.clusters,
            "置信度": self.confidence,
            "结果文件": self.file_path,
            "结果版本": self.version,
            # 预测洞察投影的追溯信息。
            "洞察类型": INSIGHT_KIND,
            "来源类型": self.source_type,
            "来源编号": str(self.source_id),
            "来源ID": self.source_id,
            "来源标识": self.source_key,
            "来源名称": source_display_name(self.source_type, self.source_id),
        }
        row.update(self.attributes)
        return row


def source_display_name(source_type: str, source_id: Any) -> str:
    """把来源类型/编号转成便于场景详情展示的名称。"""
    if source_type == SOURCE_MODEL:
        return f"模型{source_id}"
    if source_type == SOURCE_PREDICT:
        return f"预测任务{source_id}"
    if source_type == MINING_MODULE:
        return f"挖掘结果{source_id}"
    return f"{source_type}{source_id}"


def assemble_model_insight(model_id: int, model: dict[str, Any]) -> InsightDraft:
    """按模型注册采集的原有规则装配洞察草稿。"""
    return InsightDraft(
        source_type=SOURCE_MODEL,
        source_id=model_id,
        topic=f"模型{model_id}预测结果挖掘分析",
        algorithm=model.get("算法类型", "unknown"),
        summary=f"基于模型{model_id}的批量预测发现了3个高价值用户分群，置信度0.85",
        clusters="高价值_群1,活跃_群2,流失风险_群3",
        confidence="0.85",
        file_path=f"mining/model_{model_id}_results.json",
        code_template="RES-{result_id:04d}",
        attributes={
            "模型编号": str(model_id),
            "模型名称": model.get("模型名称", ""),
            "模型版本": model.get("模型版本", ""),
            "评估分数": model.get("评估分数", ""),
        },
    )


def assemble_predict_insight(task_id: int, task: dict[str, Any]) -> InsightDraft:
    """按批量预测执行的原有规则装配洞察草稿。"""
    return InsightDraft(
        source_type=SOURCE_PREDICT,
        source_id=task_id,
        topic=f"批量预测任务{task_id}挖掘分析",
        algorithm="ensemble_voting",
        summary="从批量预测结果中识别出用户行为模式并写入挖掘结果；发现2个关键行为路径",
        clusters="高转化_群A,待激活_群B",
        confidence="0.82",
        file_path=f"mining/predict_{task_id}_result.json",
        code_template=f"RES-P{task_id:04d}-{{result_id:04d}}",
        attributes={
            "预测任务编号": str(task_id),
            "模型编号": str(task.get("模型编号", "unknown")),
            "数据切片": task.get("数据切片", ""),
            "输出格式": task.get("输出格式", ""),
        },
    )


class MiningInsightWriter:
    """挖掘结果写入层：负责编号分配、完整提交与成功回执。

    提交过程：
    1. 在写锁内基于「已落库编号 + 已占号」分配结果编号；
    2. 锁外执行提交前钩子并装配完整记录（允许构造两个写入的交错）；
    3. 装配失败则释放占号、不落任何半成品；
    4. 装配成功后在锁内一次性 append 并返回成功回执（结果记录本身）。
    """

    def __init__(self) -> None:
        # 可重入：提交前钩子里可以再次进入写入层，完成交错提交。
        self._lock = threading.RLock()
        self._reserved: set[int] = set()
        self._before_commit: list[Callable[[InsightDraft, int], None]] = []
        self._faults: list[dict[str, Any]] = []
        self._fail_next_write = False
        self.commits: list[dict[str, Any]] = []

    # ── 故障注入：让「分配成功、提交失败」成为可验证的场景 ──
    def inject_failure(self, source_type: str | None = None, times: int = 1) -> None:
        """让接下来 ``times`` 次匹配来源的提交在占号之后失败。"""
        if times <= 0:
            return
        self._faults.append({"source_type": source_type, "remaining": times})

    def clear_failures(self) -> None:
        self._faults.clear()
        self._fail_next_write = False

    @property
    def fail_next_write(self) -> bool:
        return self._fail_next_write

    @fail_next_write.setter
    def fail_next_write(self, value: bool) -> None:
        self._fail_next_write = bool(value)

    # ── 交错钩子：编号已经分配、记录尚未提交时触发 ──
    def set_before_commit_hook(
        self, hook: Callable[[InsightDraft, int], None] | None
    ) -> None:
        self._before_commit = [hook] if hook is not None else []

    def add_before_commit_hook(self, hook: Callable[[InsightDraft, int], None]) -> None:
        self._before_commit.append(hook)

    def remove_before_commit_hook(self, hook: Callable[[InsightDraft, int], None]) -> None:
        if hook in self._before_commit:
            self._before_commit.remove(hook)

    # ── 写入层核心 ──
    def allocate_result_id(self) -> int:
        """仅分配编号（预留），供需要显式构造交错的调用方使用。"""
        with self._lock:
            return self._allocate_locked()

    def release_result_id(self, result_id: int) -> None:
        with self._lock:
            self._reserved.discard(result_id)

    def commit(self, draft: InsightDraft) -> dict[str, Any]:
        """分配编号并完整提交一条洞察记录，返回成功回执。

        编号预留在锁内完成，装配与提交前钩子不持锁——这使另一个写入方可以在
        本次「已分配、未提交」的窗口里完成自己的分配与提交；编号因此不会重复。
        """
        result_id = self.allocate_result_id()
        try:
            for hook in tuple(self._before_commit):
                hook(draft, result_id)
            self._raise_injected_failure(draft)
            row = draft.build_row(result_id)
        except Exception as exc:
            # 失败时释放占号并保证结果表没有本次的半成品；预测任务因此仍可重试。
            # 只清理本次编号的记录，保留交错钩子里其他写入的成果。
            with self._lock:
                mining_rows = store.rows(MINING_MODULE)
                mining_rows[:] = [
                    r for r in mining_rows if int(r.get("id", 0)) != result_id
                ]
                self._reserved.discard(result_id)
            if isinstance(exc, InsightWriteError):
                raise
            raise InsightWriteError(str(exc)) from exc

        # 完整装配后一次性落表：落表与释放占号在同一把锁内完成。
        with self._lock:
            store.rows(MINING_MODULE).append(row)
            self._reserved.discard(result_id)
            self.commits.append({
                "result_id": result_id,
                "来源类型": draft.source_type,
                "来源编号": str(draft.source_id),
                "结果编号": row["结果编号"],
            })
        return row

    # 别名：不同调用方对「落表」的称呼不同，统一指向同一次完整提交。
    write = commit

    def _allocate_locked(self) -> int:
        used = {int(row.get("id", 0)) for row in store.rows(MINING_MODULE)}
        candidate = max(used | self._reserved, default=0) + 1
        self._reserved.add(candidate)
        return candidate

    def _raise_injected_failure(self, draft: InsightDraft) -> None:
        if self._fail_next_write:
            self._fail_next_write = False
            raise InsightWriteError(f"洞察落库失败：{draft.source_key}")
        for fault in list(self._faults):
            if fault["source_type"] not in (None, draft.source_type):
                continue
            fault["remaining"] -= 1
            if fault["remaining"] <= 0 and fault in self._faults:
                self._faults.remove(fault)
            raise InsightWriteError(f"洞察落库失败：{draft.source_key}")

    # ── 成功回执 → 已提交的投影结果 ──
    def find_committed(self, receipt: Any) -> dict[str, Any] | None:
        """按成功回执读取已提交的洞察投影结果；失败回执或未提交时返回 None。"""
        if isinstance(receipt, dict):
            if receipt.get("ok") is False:
                return None
            result_id = receipt.get("result_id", receipt.get("mining_id"))
            if result_id is None and "id" in receipt and (
                "结果编号" in receipt or "洞察类型" in receipt
            ):
                # 直接传入已落表的结果记录时，按记录 ID 校验它确实已经提交。
                result_id = receipt.get("id")
                return store.find(MINING_MODULE, int(result_id)) if result_id else None
            if result_id is None:
                return None
            row = store.find(MINING_MODULE, int(result_id))
            if row is None:
                return None
            source_type = receipt.get("source_type")
            source_id = receipt.get("source_id")
            if source_type is None:
                if "model_id" in receipt and "task_id" not in receipt:
                    source_type = SOURCE_MODEL
                    source_id = receipt.get("model_id")
                elif "task_id" in receipt:
                    source_type = SOURCE_PREDICT
                    source_id = receipt.get("task_id")
            if source_type is not None and row.get("来源类型") != source_type:
                return None
            if source_id is not None and str(row.get("来源编号")) != str(source_id):
                return None
            return row
        if isinstance(receipt, bool):
            return None
        if isinstance(receipt, int):
            return store.find(MINING_MODULE, receipt)
        if isinstance(receipt, str):
            code = receipt.strip()
            if code.isdigit():
                return store.find(MINING_MODULE, int(code))
            for row in store.rows(MINING_MODULE):
                if str(row.get("结果编号", "")) == code:
                    return row
        return None


# 两种来源共享同一个写入层单例，编号预留与提交因此对所有调用方可见。
insight_writer = MiningInsightWriter()
