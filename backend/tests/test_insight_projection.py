"""洞察投影写入链路测试：交错提交、失败重试、回执转标注与来源追溯。"""
from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.services import service_for
from app.services.insight import (
    SOURCE_MODEL,
    SOURCE_PREDICT,
    InsightWriteError,
    assemble_model_insight,
    insight_writer,
)
from app.store import store


def assert_ok(result, msg=""):
    assert result.get("ok"), f"{msg}: {result}"
    return result


def test_interleaved_writes_keep_ids_unique_and_fail_first_predict_write():
    """模型采集与预测写入交错，预测首次失败，模型先提交；重试后各自可追溯。"""
    model_svc = service_for("model_registry")
    predict_svc = service_for("predict_batch")
    mining_svc = service_for("mining_result")

    model_id = 1
    task_id = 1
    assert store.find("model_registry", model_id) is not None
    assert store.find("predict_batch", task_id) is not None

    # 预测首次写入在占号后失败。
    insight_writer.clear_failures()
    insight_writer.inject_failure(SOURCE_PREDICT, times=1)

    model_receipt: dict = {}
    interleaved = {"done": False}

    def hook(draft, reserved_id):
        # 预测已经分配编号、尚未提交时，模型采集完成自己的分配与提交。
        if draft.source_type == SOURCE_PREDICT and not interleaved["done"]:
            interleaved["done"] = True
            model_receipt.update(model_svc.collect_predict_results_to_mining(model_id))

    insight_writer.set_before_commit_hook(hook)
    try:
        failed = predict_svc.execute_batch_and_collect_result(task_id)
    finally:
        insight_writer.set_before_commit_hook(None)
        insight_writer.clear_failures()

    # 预测首次写入失败：没有新的洞察回执，任务也没有完成。
    assert not failed.get("ok"), failed
    assert failed.get("retryable") is True
    assert "mining_id" not in failed
    assert_ok(model_receipt, "模型洞察应已提交")

    model_result_id = model_receipt["mining_id"]
    model_row = store.find("mining_result", model_result_id)
    assert model_row is not None
    assert model_row["来源类型"] == SOURCE_MODEL
    assert model_row["来源编号"] == str(model_id)

    # 模型洞察已提交、预测洞察未提交时查询预测状态。
    status = predict_svc.query_prediction_status(task_id)
    assert status["completed"] is False
    assert status["insight_committed"] is False
    assert status["retryable"] is True
    assert status["pending"] is True

    # 模型洞察可正常转标注。
    model_label = assert_ok(
        mining_svc.export_insights_to_labels(model_receipt), "模型回执转标注")
    assert model_label["result_id"] == model_result_id

    # 预测重试成功后再转它自己的回执。
    ok = assert_ok(predict_svc.execute_batch_and_collect_result(task_id), "预测重试")
    predict_result_id = ok["mining_id"]
    assert predict_result_id != model_result_id, "两种来源的结果 ID 必须唯一"
    assert store.find("mining_result", model_result_id) is not None
    assert ok["retryable"] if "retryable" in ok else True

    status2 = predict_svc.query_prediction_status(task_id)
    assert status2["completed"] is True
    assert status2["insight_committed"] is True
    assert status2["mining_id"] == predict_result_id

    predict_label = assert_ok(
        mining_svc.export_insights_to_labels(ok), "预测回执转标注")
    assert predict_label["result_id"] == predict_result_id

    model_row = store.find("mining_result", model_result_id)
    predict_row = store.find("mining_result", predict_result_id)

    # 规则字段分别采用各自的洞察摘要与原有分群顺序。
    assert model_label["业务规则"] == model_row["洞察摘要"]
    assert model_label["关联特征"] == model_row["关键分群"]
    assert predict_label["业务规则"] == predict_row["洞察摘要"]
    assert predict_label["关联特征"] == predict_row["关键分群"]
    assert model_row["关键分群"] == "高价值_群1,活跃_群2,流失风险_群3"
    assert predict_row["关键分群"] == "高转化_群A,待激活_群B"

    model_label_row = store.find("scenario_label", model_label["label_id"])
    predict_label_row = store.find("scenario_label", predict_label["label_id"])
    assert "模型1" in model_label_row["场景名称"]
    assert "预测任务1" in predict_label_row["场景名称"]
    assert model_label_row["来源类型"] == SOURCE_MODEL
    assert predict_label_row["来源类型"] == SOURCE_PREDICT
    assert model_label_row["结果ID"] == model_result_id
    assert predict_label_row["结果ID"] == predict_result_id

    # 场景详情应能区分来源是模型 1 还是预测任务 1。
    label_svc = service_for("scenario_label")
    model_detail = assert_ok(
        label_svc.describe_label_source(model_label["label_id"]), "模型来源详情")
    predict_detail = assert_ok(
        label_svc.describe_label_source(predict_label["label_id"]), "预测来源详情")
    assert model_detail["来源类型"] == SOURCE_MODEL
    assert predict_detail["来源类型"] == SOURCE_PREDICT
    assert model_detail["来源名称"] == "模型1"
    assert predict_detail["来源名称"] == "预测任务1"
    assert model_detail["结果ID"] == model_result_id
    assert predict_detail["结果ID"] == predict_result_id

    # 两个来源的原始数字编号相同，字号编号不同。
    assert model_row["来源ID"] == 1
    assert predict_row["来源ID"] == 1
    assert model_row["结果编号"] != predict_row["结果编号"]


def test_concurrent_model_and_predict_writes_do_not_collide():
    """两个写入线程真正并发时，结果编号不冲突且两条记录都完整提交。"""
    import threading

    model_svc = service_for("model_registry")
    predict_svc = service_for("predict_batch")

    model_id = 3
    predict_task = service_for("predict_batch")
    rows, _ = predict_task.list_entries()
    task_id = next(row["id"] for row in rows if row["status"] != "已完成")

    results: dict[str, dict] = {}
    barrier = threading.Barrier(2)

    def run_model():
        barrier.wait()
        results["model"] = model_svc.collect_predict_results_to_mining(model_id)

    def run_predict():
        barrier.wait()
        results["predict"] = predict_svc.execute_batch_and_collect_result(task_id)

    threads = [threading.Thread(target=run_model), threading.Thread(target=run_predict)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    model_receipt = assert_ok(results["model"], "并发模型采集")
    predict_receipt = assert_ok(results["predict"], "并发预测采集")
    model_row = store.find("mining_result", model_receipt["mining_id"])
    predict_row = store.find("mining_result", predict_receipt["mining_id"])
    assert model_row is not None and predict_row is not None
    assert model_row["id"] != predict_row["id"]
    ids = [row["id"] for row in store.rows("mining_result")]
    assert len(ids) == len(set(ids)), "挖掘结果 ID 必须唯一"
    codes = [row["结果编号"] for row in store.rows("mining_result")]
    assert len(codes) == len(set(codes)), "结果编号必须唯一"


def test_original_entrypoints_keep_return_fields_and_file_paths():
    """原有入口的洞察返回字段与记录中的结果文件路径保持兼容。"""
    model_svc = service_for("model_registry")
    predict_svc = service_for("predict_batch")
    mining_svc = service_for("mining_result")

    model_receipt = assert_ok(
        model_svc.collect_predict_results_to_mining(2), "模型采集入口")
    assert {"ok", "model_id", "mining_id"} <= set(model_receipt)
    model_row = store.find("mining_result", model_receipt["mining_id"])
    assert model_row["结果文件"] == "mining/model_2_results.json"
    projection = assert_ok(
        mining_svc.query_insight_projection(model_receipt), "模型洞察投影")
    assert projection["结果文件"] == "mining/model_2_results.json"
    assert projection["来源类型"] == SOURCE_MODEL

    ok, _ = service_for("predict_batch").list_entries()
    task = next(row for row in ok if row["status"] != "已完成")
    task_id = task["id"]
    predict_receipt = assert_ok(
        predict_svc.execute_batch_and_collect_result(task_id), "预测执行入口")
    assert {"ok", "task_id", "mining_id", "message"} <= set(predict_receipt)
    predict_row = store.find("mining_result", predict_receipt["mining_id"])
    assert predict_row["结果文件"] == f"mining/predict_{task_id}_result.json"
    projection = assert_ok(
        mining_svc.query_insight_projection(predict_receipt), "预测洞察投影")
    assert projection["结果文件"] == f"mining/predict_{task_id}_result.json"
    assert projection["来源类型"] == SOURCE_PREDICT


def test_existing_insight_conversion_keeps_original_fields():
    """已有洞察（无投影来源）转标注仍沿用原有规则字段。"""
    mining_svc = service_for("mining_result")
    label = assert_ok(mining_svc.export_insights_to_labels(1), "已有洞察转标注")
    row = store.find("mining_result", 1)
    label_row = store.find("scenario_label", label["label_id"])
    assert label_row["业务规则"] == row["洞察摘要"]
    assert label_row["关联特征"] == row["关键分群"]
    assert label_row["场景名称"] == f"挖掘场景_{row['分析主题']}"
    assert "来源类型" not in label_row


if __name__ == "__main__":
    tests = [
        test_interleaved_writes_keep_ids_unique_and_fail_first_predict_write,
        test_concurrent_model_and_predict_writes_do_not_collide,
        test_original_entrypoints_keep_return_fields_and_file_paths,
        test_existing_insight_conversion_keeps_original_fields,
    ]
    failed = 0
    for test in tests:
        try:
            test()
            print(f"[PASS] {test.__name__}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"[FAIL] {test.__name__}: {exc!r}")
    print(f"\n总测试: {len(tests)}, 通过: {len(tests) - failed}, 失败: {failed}")
    sys.exit(1 if failed else 0)
