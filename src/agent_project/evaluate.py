"""Evaluate retrieval and answers separately on data/eval/questions.json."""

import argparse
import json
from pathlib import Path
from typing import Any

from agent_project.llm import OpenRouterClient
from agent_project.qa_graph import answer_question
from agent_project.retrieval import TOP_K, DocumentIndex
from agent_project.tools import DATA_DIR

QUESTIONS_PATH = DATA_DIR / "eval" / "questions.json"
EVIDENCE_DIR = DATA_DIR.parent / "evidence"


def load_questions(path: Path = QUESTIONS_PATH) -> list[dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8"))


def evaluate_retrieval(
    index: DocumentIndex, questions: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Rank of the first expected document chunk, using the question as the query."""
    rows = []
    for q in questions:
        ranking = index.search(q["question"], k=len(index.chunks))
        doc_sources = [s for s in q["sources"] if not s.startswith("tool:")]
        rank = next(
            (i + 1 for i, r in enumerate(ranking) if r["id"] in doc_sources), None
        )
        top_k = [r["id"] for r in ranking[:TOP_K]]
        rows.append(
            {
                "id": q["id"],
                "expected": q["expected"],
                "top_k": top_k,
                "top_score": ranking[0]["score"],
                "rank": rank if doc_sources else None,
                "all_expected_in_top_k": all(s in top_k for s in doc_sources)
                if doc_sources
                else None,
            }
        )
    return rows


def retrieval_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    ranked = [r for r in rows if r["expected"] == "answered" and r["rank"] is not None]
    unanswerable = [r["top_score"] for r in rows if r["expected"] == "abstained"]
    answerable = [r["top_score"] for r in rows if r["expected"] == "answered"]
    return {
        "questions": len(ranked),
        "hit_at_1": sum(r["rank"] == 1 for r in ranked) / len(ranked),
        f"hit_at_{TOP_K}": sum(r["rank"] <= TOP_K for r in ranked) / len(ranked),
        "mrr": round(sum(1 / r["rank"] for r in ranked) / len(ranked), 3),
        f"all_expected_in_top_{TOP_K}": sum(r["all_expected_in_top_k"] for r in ranked)
        / len(ranked),
        "min_top_score_answerable": min(answerable),
        "max_top_score_unanswerable": max(unanswerable),
    }


def evaluate_answers(
    index: DocumentIndex, questions: list[dict[str, Any]], trace_dir: Path
) -> list[dict[str, Any]]:
    client = OpenRouterClient()
    trace_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for q in questions:
        trace_path = trace_dir / f"{q['id']}.jsonl"
        trace_path.unlink(missing_ok=True)
        state = answer_question(client, q["question"], index.search, trace_path)
        events = [
            json.loads(line)
            for line in trace_path.read_text(encoding="utf-8").splitlines()
        ]
        citations = state.get("citations", [])
        rows.append(
            {
                "id": q["id"],
                "question": q["question"],
                "expected": q["expected"],
                "status": state["status"],
                "outcome_correct": state["status"] == q["expected"],
                "citations": citations,
                "cites_expected_source": any(
                    c.startswith(source) for c in citations for source in q["sources"]
                ),
                "rejected_citations": sum(
                    1
                    for e in events
                    if e["event"] == "verify" and e["payload"]["problem"]
                ),
                "searches": len(state.get("queries", [])),
                "tool_calls": state.get("tool_calls", 0),
                "decisions": state.get("decisions", 0),
                "cost": round(state.get("cost", 0.0), 6),
                "output": state.get("answer")
                or state.get("missing")
                or state.get("clarification"),
            }
        )
        print(f"{q['id']}: {state['status']} (expected {q['expected']})")
    return rows


def answer_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    answered = [r for r in rows if r["expected"] == "answered"]
    return {
        "questions": len(rows),
        "outcome_accuracy": sum(r["outcome_correct"] for r in rows) / len(rows),
        "answered_correctly": sum(r["status"] == "answered" for r in answered),
        "answerable": len(answered),
        "abstained_correctly": sum(
            r["status"] == "abstained" for r in rows if r["expected"] == "abstained"
        ),
        "unanswerable": sum(r["expected"] == "abstained" for r in rows),
        "cites_expected_source": sum(
            r["cites_expected_source"] for r in answered if r["status"] == "answered"
        ),
        "rejected_citation_attempts": sum(r["rejected_citations"] for r in rows),
        "total_cost_usd": round(sum(r["cost"] for r in rows), 6),
    }


def _dash(value: Any) -> Any:
    return "-" if value is None else value


def write_report(
    retrieval: list[dict], r_metrics: dict, answers: list[dict] | None, a_metrics
) -> str:
    lines = [
        "# Evaluation Report",
        "",
        "Generated by `uv run outfit-eval`.",
        "",
        "## Retrieval quality (no model calls)",
        "",
        "| Metric | Value |",
        "| --- | --- |",
        *[f"| {k} | {v} |" for k, v in r_metrics.items()],
        "",
        "| Question | Expected | First expected rank | All expected in top k "
        "| Top score | Top results |",
        "| --- | --- | --- | --- | --- | --- |",
        *[
            f"| {r['id']} | {r['expected']} | {r['rank'] or '-'} | "
            f"{_dash(r['all_expected_in_top_k'])} | {r['top_score']} | "
            f"{', '.join(r['top_k'])} |"
            for r in retrieval
        ],
    ]
    if answers is not None:
        lines += [
            "",
            "## Answer quality (live model)",
            "",
            "| Metric | Value |",
            "| --- | --- |",
            *[f"| {k} | {v} |" for k, v in a_metrics.items()],
            "",
            "| Question | Expected | Got | Citations | Searches | Tools | Cost |",
            "| --- | --- | --- | --- | --- | --- | --- |",
            *[
                f"| {r['id']} | {r['expected']} | {r['status']} | "
                f"{', '.join(r['citations']) or '-'} | {r['searches']} | "
                f"{r['tool_calls']} | {r['cost']} |"
                for r in answers
            ],
            "",
            "## Outputs",
            "",
            *[
                f"- **{r['id']}** {r['question']}  \n  {' '.join(r['output'].split())}"
                for r in answers
            ],
        ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate retrieval and answers")
    parser.add_argument(
        "--retrieval-only", action="store_true", help="Skip the live model run."
    )
    args = parser.parse_args()

    index = DocumentIndex.load()
    questions = load_questions()
    retrieval = evaluate_retrieval(index, questions)
    r_metrics = retrieval_metrics(retrieval)
    answers = a_metrics = None
    if not args.retrieval_only:
        answers = evaluate_answers(index, questions, EVIDENCE_DIR / "eval-traces")
        a_metrics = answer_metrics(answers)

    EVIDENCE_DIR.mkdir(exist_ok=True)
    results = {
        "retrieval": {"metrics": r_metrics, "rows": retrieval},
        "answers": {"metrics": a_metrics, "rows": answers},
    }
    (EVIDENCE_DIR / "eval-results.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    report = write_report(retrieval, r_metrics, answers, a_metrics)
    (EVIDENCE_DIR / "eval-report.md").write_text(report, encoding="utf-8")
    print(json.dumps({"retrieval": r_metrics, "answers": a_metrics}, indent=2))


if __name__ == "__main__":
    main()
