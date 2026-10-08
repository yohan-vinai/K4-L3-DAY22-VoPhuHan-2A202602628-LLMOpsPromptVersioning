"""
Bước 3 — RAGAS Evaluation
===========================
NHIỆM VỤ:
  1. Chạy 50 QA pairs qua CẢ 2 prompt version, lưu answers + contexts
  2. Tạo EvaluationDataset với các SingleTurnSample object
  3. Đánh giá với 4 RAGAS metrics: faithfulness, answer_relevancy,
     context_recall, context_precision
  4. In bảng so sánh V1 vs V2
  5. Lưu kết quả vào data/ragas_report.json

DELIVERABLE: faithfulness ≥ 0.8 cho ít nhất 1 prompt version
             + file data/ragas_report.json được tạo ra

⏰ LƯU Ý: Bước này mất ~15-30 phút. Hãy bắt đầu sớm!
"""
import sys
import json
import os
import time
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import config  # ⚠️ phải import trước LangChain

import numpy as np
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from ragas import evaluate, EvaluationDataset, SingleTurnSample
from ragas.metrics import faithfulness, answer_relevancy, context_recall, context_precision
from ragas.run_config import RunConfig

# Gemini 3.5 Flash-Lite accepts one candidate per request; RAGAS defaults to 3.
answer_relevancy.strictness = 1
EVALUATOR_MODELS = {
    "faithfulness": config.GEMINI_EVALUATOR_MODEL,
    "answer_relevancy": config.GEMINI_EVALUATOR_ANSWER_RELEVANCY_MODEL,
    "context_recall": config.GEMINI_EVALUATOR_CONTEXT_RECALL_MODEL,
    "context_precision": config.GEMINI_EVALUATOR_CONTEXT_PRECISION_MODEL,
}

from utils.llm_factory import get_llm, get_embeddings
from utils.data_loader import load_knowledge_base, split_text, build_vectorstore
from qa_pairs import QA_PAIRS


# ── 1. Prompt Templates (copy từ Bước 2) ──────────────────────────────────
# TODO: Copy SYSTEM_V1 và SYSTEM_V2 mà bạn đã viết ở file 02_prompt_hub_ab_routing.py
SYSTEM_V1 = (
    "Bạn là trợ lý AI thân thiện. Trả lời ngắn gọn trong 2-4 câu và chỉ dựa trên context. "
    "Nếu context không có câu trả lời, hãy nói rõ là bạn không biết.\n\n"
    "Context:\n{context}"
)
PROMPT_V1 = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_V1),
    ("human",  "{question}"),
])

SYSTEM_V2 = (
    "Bạn là chuyên gia phân tích thông tin. Đọc kỹ context, xác định các dữ kiện liên quan, "
    "rồi trình bày câu trả lời rõ ràng, có tổ chức trong 3-5 câu. Không suy đoán ngoài context.\n\n"
    "Context:\n{context}"
)
PROMPT_V2 = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_V2),
    ("human",  "{question}"),
])

PROMPTS = {"v1": PROMPT_V1, "v2": PROMPT_V2}


# ── 2. Setup Vectorstore ───────────────────────────────────────────────────
def setup_vectorstore():
    """Tái sử dụng — tạo FAISS vectorstore từ knowledge base."""
    embeddings  = get_embeddings()
    text        = load_knowledge_base()
    chunks      = split_text(text)
    return build_vectorstore(chunks, embeddings)


# ── 3. Chạy RAG và thu thập kết quả ───────────────────────────────────────
def run_rag(retriever, llm, prompt, question: str) -> dict:
    """
    Chạy RAG chain cho 1 câu hỏi.

    ⚠️ QUAN TRỌNG: trả về contexts là LIST of strings, KHÔNG phải string đã ghép!
    RAGAS cần từng đoạn riêng để tính context_recall và context_precision.

    Trả về: {"answer": str, "contexts": list[str]}
    """
    for attempt in range(3):
        try:
            docs = retriever.invoke(question)
            contexts = [doc.page_content for doc in docs]
            ctx_str = "\n\n".join(contexts)
            answer = (prompt | llm | StrOutputParser()).invoke({
                "context": ctx_str,
                "question": question,
            })
            return {"answer": answer, "contexts": contexts}
        except Exception:
            if attempt == 2:
                raise
            print("  ⚠️ Request Gemini tạm lỗi; đợi 15 giây rồi thử lại ...")
            time.sleep(15)


def collect_rag_outputs(vectorstore, prompt_version: str) -> list:
    """
    Chạy tất cả 50 QA pairs qua prompt version được chỉ định.
    Trả về: list of dict với keys: question, reference, answer, contexts
    """
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    llm       = get_llm()
    prompt    = PROMPTS[prompt_version]

    checkpoint_path = Path(__file__).parent.parent / "data" / f".ragas_checkpoint_{prompt_version}.json"
    results = json.loads(checkpoint_path.read_text(encoding="utf-8")) if checkpoint_path.exists() else []
    print(f"\n🚀 Đang chạy 50 câu hỏi với prompt {prompt_version} ...")

    for i, qa in enumerate(QA_PAIRS[len(results):], len(results) + 1):
        # TODO: Gọi run_rag() cho câu hỏi hiện tại
        out = run_rag(retriever, llm, prompt, qa["question"])

        # TODO: Append vào results dict với 4 keys
        results.append({
            "question":  qa["question"],
            "reference": qa["reference"],
            "answer":    out["answer"],
            "contexts":  out["contexts"],
        })
        print(f"  [{i:02d}/50] {qa['question'][:60]}")
        checkpoint_path.write_text(json.dumps(results, ensure_ascii=False), encoding="utf-8")

    return results


# ── 4. Tạo RAGAS EvaluationDataset ────────────────────────────────────────
def build_ragas_dataset(rag_results: list) -> EvaluationDataset:
    """
    Chuyển đổi kết quả RAG thành RAGAS EvaluationDataset.

    Mỗi SingleTurnSample cần 4 trường:
      user_input         → câu hỏi
      response           → câu trả lời đã tạo
      retrieved_contexts → list[str] các đoạn đã retrieve
      reference          → đáp án chuẩn (ground truth)
    """
    # TODO: Tạo list các SingleTurnSample từ rag_results
    samples = [
        SingleTurnSample(
            user_input=r["question"],
            response=r["answer"],
            retrieved_contexts=r["contexts"],
            reference=r["reference"],
        )
        for r in rag_results
    ]

    # TODO: Wrap thành EvaluationDataset và trả về
    return EvaluationDataset(samples=samples)


# ── 5. Chạy RAGAS Evaluation ──────────────────────────────────────────────
def run_ragas_eval(rag_results: list, version: str) -> tuple[dict, dict]:
    """
    Đánh giá kết quả RAG với 4 RAGAS metrics.
    Trả về: dict {metric_name: mean_score}

    Lưu ý: evaluate() thực hiện rất nhiều lần gọi LLM → mất 5-10 phút / version.
    """
    print(f"\n📐 Đang đánh giá RAGAS cho prompt {version} ... (thời gian phụ thuộc quota/model)")

    # Evaluation is already covered by the 100 application traces from steps 1-2.
    os.environ["LANGCHAIN_TRACING_V2"] = "false"
    emb_eval = get_embeddings()

    metric_objects = {
        "faithfulness": faithfulness,
        "answer_relevancy": answer_relevancy,
        "context_recall": context_recall,
        "context_precision": context_precision,
    }
    scores = {}
    valid_counts = {}
    for key, metric in metric_objects.items():
        score_checkpoint = (
            Path(__file__).parent.parent / "data" / f".ragas_scores_{version}_{key}.json"
        )
        if score_checkpoint.exists():
            saved = json.loads(score_checkpoint.read_text(encoding="utf-8"))
            if (
                saved.get("model") == EVALUATOR_MODELS[key]
                and saved.get("sample_count") == len(rag_results)
            ):
                values = saved.get("scores", [])
            else:
                values = []
        else:
            values = []

        batch_size = 5
        if len(values) < len(rag_results):
            values.extend([None] * (len(rag_results) - len(values)))
        for start in range(0, len(rag_results), batch_size):
            end = min(start + batch_size, len(rag_results))
            if all(value is not None for value in values[start:end]):
                continue

            batch_results = rag_results[start : start + batch_size]
            batch_dataset = build_ragas_dataset(batch_results)
            llm_eval = get_llm(
                temperature=0,
                model_override=EVALUATOR_MODELS[key],
                response_mime_type="application/json",
            )
            result = evaluate(
                batch_dataset,
                metrics=[metric],
                llm=llm_eval,
                embeddings=emb_eval,
                run_config=RunConfig(max_workers=1),
            )
            raw = result[key]
            values[start:end] = [
                float(value) if value is not None and not np.isnan(value) else None
                for value in raw
            ]
            score_checkpoint.write_text(
                json.dumps({
                    "model": EVALUATOR_MODELS[key],
                    "sample_count": len(rag_results),
                    "scores": values,
                }, indent=2),
                encoding="utf-8",
            )

        valid = [value for value in values if value is not None]
        scores[key] = float(np.mean(valid)) if valid else None
        valid_counts[key] = len(valid)

    # In kết quả
    print(f"\n📊 Kết quả RAGAS — Prompt {version.upper()}:")
    for k, v in scores.items():
        star = " ⭐" if k == "faithfulness" and v is not None and v >= 0.8 else ""
        rendered = f"{v:.4f}" if v is not None else "N/A"
        print(f"  {k:30s}: {rendered} ({valid_counts[k]}/{len(rag_results)} cases){star}")

    return scores, valid_counts


# ── 6. Main ────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  Bước 3: RAGAS Evaluation")
    print("=" * 60)

    if not config.validate():
        sys.exit(1)

    checkpoint_root = Path(__file__).parent.parent / "data"
    v1_checkpoint = checkpoint_root / ".ragas_checkpoint_v1.json"
    v2_checkpoint = checkpoint_root / ".ragas_checkpoint_v2.json"
    if (
        v1_checkpoint.exists()
        and v2_checkpoint.exists()
        and len(json.loads(v1_checkpoint.read_text(encoding="utf-8"))) == len(QA_PAIRS)
        and len(json.loads(v2_checkpoint.read_text(encoding="utf-8"))) == len(QA_PAIRS)
    ):
        v1_results = json.loads(v1_checkpoint.read_text(encoding="utf-8"))
        v2_results = json.loads(v2_checkpoint.read_text(encoding="utf-8"))
        print("✅ Dùng lại 50 QA outputs đã lưu cho mỗi prompt version.")
    else:
        vectorstore = setup_vectorstore()
        v1_results = collect_rag_outputs(vectorstore, "v1")
        v2_results = collect_rag_outputs(vectorstore, "v2")

    # Chạy RAGAS evaluation
    v1_scores, v1_valid = run_ragas_eval(v1_results, "v1")
    v2_scores, v2_valid = run_ragas_eval(v2_results, "v2")

    # In bảng so sánh
    print("\n" + "=" * 65)
    print(f"  {'Metric':30s}  {'V1':>8}  {'V2':>8}  Winner")
    print("=" * 65)
    for metric in ["faithfulness", "answer_relevancy", "context_recall", "context_precision"]:
        s1, s2  = v1_scores[metric], v2_scores[metric]
        winner = "N/A" if s1 is None or s2 is None else ("← V1" if s1 > s2 else "← V2")
        fmt = lambda score: f"{score:>8.4f}" if score is not None else f"{'N/A':>8s}"
        print(f"  {metric:30s}  {fmt(s1)}  {fmt(s2)}  {winner}")

    # Kiểm tra mục tiêu
    faithfulness_scores = [score for score in (v1_scores["faithfulness"], v2_scores["faithfulness"]) if score is not None]
    best_faith = max(faithfulness_scores) if faithfulness_scores else None
    if best_faith is not None and best_faith >= 0.8:
        print(f"\n✅ Đạt mục tiêu: faithfulness = {best_faith:.4f} ≥ 0.8")
    else:
        rendered = f"{best_faith:.4f}" if best_faith is not None else "N/A"
        print(f"\n⚠️  Chưa đạt mục tiêu ({rendered} < 0.8).")
        print("   Gợi ý: giảm chunk_size, tăng k, hoặc điều chỉnh prompt.")

    # TODO: Lưu báo cáo vào data/ragas_report.json
    report = {
        "prompt_v1_scores": v1_scores,
        "prompt_v2_scores": v2_scores,
        "valid_case_counts": {"v1": v1_valid, "v2": v2_valid},
        "evaluator_models": EVALUATOR_MODELS,
        "target_met": best_faith is not None and best_faith >= 0.8,
    }
    report_path = Path(__file__).parent.parent / "data" / "ragas_report.json"
    # TODO: Ghi report vào file bằng json.dumps hoặc json.dump
    # Gợi ý: report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"💾 Đã lưu báo cáo vào {report_path}")


if __name__ == "__main__":
    main()
