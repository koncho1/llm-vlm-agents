"""Interactive entry point for evidence-grounded questions about clothes."""

import sys
from datetime import UTC, datetime

from agent_project.cli import TRACES_DIR
from agent_project.llm import OpenRouterClient
from agent_project.qa_graph import QAState, answer_question
from agent_project.retrieval import DocumentIndex


def print_result(state: QAState) -> None:
    if state["status"] == "answered":
        print("\n" + state["answer"])
        evidence = state["evidence"]
        print("\nSources:")
        for item_id in state["citations"]:
            print(f"  [{item_id}] {evidence[item_id]['heading']}")
    elif state["status"] == "abstained":
        print(f"\nI can't answer that from the documents. Missing: {state['missing']}")


def main() -> None:
    question = " ".join(sys.argv[1:]).strip() or input("Ask about clothes: ").strip()
    if not question:
        print("No question provided.")
        return

    index = DocumentIndex.load()
    client = OpenRouterClient()
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    trace_path = TRACES_DIR / f"ask-{stamp}.jsonl"

    state = answer_question(client, question, index.search, trace_path)
    if state["status"] == "needs_clarification":
        reply = input(f"\n{state['clarification']}\n> ").strip()
        if reply:
            question = f"{question}\nClarification from the user: {reply}"
            state = answer_question(client, question, index.search, trace_path)
    if state["status"] == "needs_clarification":
        print(f"\nStill unclear: {state['clarification']}")
    print_result(state)
    print(f"\nTrace saved to {trace_path}")


if __name__ == "__main__":
    main()
