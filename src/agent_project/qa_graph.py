"""LangGraph workflow: retrieve, then let the evidence decide the next step.

The model chooses one action per turn (answer, search again, use a wardrobe or
weather tool, ask the user, or abstain). Code decides which actions are offered,
enforces the budgets, and verifies every citation before an answer is accepted.
"""

import json
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field, ValidationError

from agent_project.llm import ModelClient
from agent_project.policy import execute_tool
from agent_project.tools import TOOLS
from agent_project.tracing import append_trace

MAX_SEARCHES = 2
MAX_TOOL_CALLS = 2
MAX_ANSWER_ATTEMPTS = 2
MAX_DECISIONS = 6

Search = Callable[[str], list[dict[str, Any]]]

SYSTEM_PROMPT = """You answer questions about clothing, outfits, and garment \
care for an outfit planner. Use only the evidence you are given, which comes \
from the project's documents and tools, never general knowledge.

Decide the next step by calling exactly one tool:
- answer: the evidence fully answers the question. Cite the evidence ids each \
sentence relies on in square brackets, e.g. [shoe-care#1]. Only cite ids that \
appear in the evidence list. Tool results are evidence too: cite them by their \
evidence id, e.g. [tool:search_wardrobe#1], never by a wardrobe item id like w11.
- search_again: the evidence is related but incomplete, and a differently \
worded query could find the rest. The evidence holds only the top few search \
results, not the whole collection: if one part of the question is not covered, \
search again for that part on its own before answering or abstaining.
- search_wardrobe or get_weather_profile: the question is about the user's own \
clothes or a weather condition, and their result is not in the evidence yet.
- ask_user: the question is too vague to know what to look for, e.g. it names \
no occasion or item.
- abstain: the evidence does not cover the question, e.g. prices, shops, live \
weather forecasts, or garments the documents never mention. Say exactly what \
is missing. Never use answer to say that the evidence is missing.
"""


class AnswerArgs(BaseModel):
    text: str = Field(min_length=1, description="Answer with inline [id] citations.")
    citations: list[str] = Field(
        min_length=1, description="Every evidence id cited in the answer."
    )


class SearchAgainArgs(BaseModel):
    query: str = Field(min_length=3, description="A differently worded search.")


class AskUserArgs(BaseModel):
    question: str = Field(min_length=3, description="One clarifying question.")


class AbstainArgs(BaseModel):
    missing: str = Field(min_length=3, description="What evidence is missing.")


DECISIONS: dict[str, tuple[str, type[BaseModel], str]] = {
    "answer": ("answer", AnswerArgs, "Answer from the evidence with citations."),
    "search_again": (
        "search",
        SearchAgainArgs,
        "Search the documents again with a better query.",
    ),
    "ask_user": ("ask_user", AskUserArgs, "Ask the user to clarify the question."),
    "abstain": ("abstain", AbstainArgs, "Say the documents lack the evidence."),
}
READ_ONLY_TOOLS = [
    t
    for t in TOOLS
    if t["function"]["name"] in {"search_wardrobe", "get_weather_profile"}
]
_CITATION_ID = re.compile(r"^(?:tool:)?[a-z0-9_-]+#\d+$")


class QAState(TypedDict, total=False):
    question: str
    queries: list[str]
    evidence: dict[str, dict[str, Any]]
    tool_calls: int
    decisions: int
    answer_attempts: int
    decision: dict[str, Any]
    feedback: str | None
    status: str
    answer: str
    citations: list[str]
    missing: str
    clarification: str
    cost: float


def _decision_tool(name: str) -> dict[str, Any]:
    _, model, description = DECISIONS[name]
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": model.model_json_schema(),
        },
    }


def offered_tools(state: QAState) -> list[dict[str, Any]]:
    """Code, not the model, decides which next steps are still allowed."""
    names = ["answer", "ask_user", "abstain"]
    if len(state.get("queries", [])) < MAX_SEARCHES:
        names.insert(1, "search_again")
    tools = [_decision_tool(name) for name in names]
    if state.get("tool_calls", 0) < MAX_TOOL_CALLS:
        tools.extend(READ_ONLY_TOOLS)
    return tools


def cited_ids(text: str) -> list[str]:
    ids = []
    for group in re.findall(r"\[([^\[\]]+)\]", text):
        for part in re.split(r"[,;]", group):
            if _CITATION_ID.match(part.strip()):
                ids.append(part.strip())
    return ids


def verify_citations(
    text: str, citations: list[str], evidence: dict[str, Any]
) -> str | None:
    """Return a problem description, or None when every citation was retrieved."""
    valid = f"Valid evidence ids: {sorted(evidence)}."
    cited = set(cited_ids(text)) | set(citations)
    if not cited:
        return f"The answer cites no evidence. {valid}"
    unknown = sorted(cited - evidence.keys())
    if unknown:
        return f"These ids were never retrieved: {unknown}. {valid}"
    return None


def _evidence_text(state: QAState) -> str:
    lines = []
    for item in state.get("evidence", {}).values():
        score = f", score {item['score']}" if "score" in item else ""
        lines.append(f"[{item['id']}] {item['heading']}{score}\n{item['text']}")
    return "\n\n".join(lines) or "(no evidence)"


def _messages(state: QAState) -> list[dict[str, Any]]:
    content = (
        f"Question: {state['question']}\n\n"
        f"Searches so far: {state.get('queries', [])}\n\n"
        f"Evidence:\n{_evidence_text(state)}"
    )
    if state.get("feedback"):
        content += f"\n\nNote from the application: {state['feedback']}"
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": content},
    ]


def _parse_decision(result: dict[str, Any], allowed: set[str]) -> dict[str, Any]:
    calls = result.get("tool_calls") or []
    if not calls:
        return {"action": "invalid", "error": "Reply by calling one of the tools."}
    name, raw = calls[0]["name"], calls[0]["arguments"]
    if name not in allowed:
        return {"action": "invalid", "error": f"'{name}' is not available now."}
    if name not in DECISIONS:
        return {"action": "use_tool", "args": {"name": name, "arguments": raw}}
    action, model, _ = DECISIONS[name]
    try:
        args = model.model_validate(json.loads(raw or "{}")).model_dump()
    except (json.JSONDecodeError, ValidationError) as exc:
        return {"action": "invalid", "error": f"Invalid arguments for '{name}': {exc}"}
    return {"action": action, "args": args}


def build_graph(client: ModelClient, search: Search, trace_path: Path | None = None):
    def trace(event: str, payload: dict[str, Any]) -> None:
        if trace_path:
            append_trace(trace_path, event, payload)

    def retrieve(state: QAState) -> QAState:
        decision = state.get("decision") or {}
        args = decision.get("args", {})
        query = args.get("query") or args.get("missing") or state["question"]
        feedback = None
        if decision.get("action") == "abstain":
            feedback = (
                f"Before accepting your abstention, the application searched for "
                f"what you said is missing ({query!r}). Answer if the new evidence "
                "covers it; otherwise abstain again."
            )
        results = search(query)
        evidence = {**state.get("evidence", {}), **{r["id"]: r for r in results}}
        trace(
            "retrieve",
            {"query": query, "results": [[r["id"], r["score"]] for r in results]},
        )
        return {
            "queries": [*state.get("queries", []), query],
            "evidence": evidence,
            "feedback": feedback,
        }

    def decide(state: QAState) -> QAState:
        tools = offered_tools(state)
        allowed = {t["function"]["name"] for t in tools}
        result = client.complete(_messages(state), tools=tools, tool_choice="required")
        decision = _parse_decision(result, allowed)
        cost = state.get("cost", 0.0) + ((result.get("usage") or {}).get("cost") or 0)
        trace(
            "decide",
            {"offered": sorted(allowed), "decision": decision, "model": result},
        )
        return {
            "decision": decision,
            "decisions": state.get("decisions", 0) + 1,
            "feedback": decision.get("error"),
            "cost": cost,
        }

    def run_tool(state: QAState) -> QAState:
        args = state["decision"]["args"]
        result = execute_tool(args["name"], args["arguments"])
        number = state.get("tool_calls", 0) + 1
        item_id = f"tool:{args['name']}#{number}"
        item = {
            "id": item_id,
            "source": f"tool:{args['name']}",
            "heading": f"{args['name']}({args['arguments']})",
            "text": json.dumps(result)[:2000],
        }
        trace("tool", {"id": item_id, "result": result})
        return {
            "evidence": {**state.get("evidence", {}), item_id: item},
            "tool_calls": number,
        }

    def verify(state: QAState) -> QAState:
        args = state["decision"]["args"]
        problem = verify_citations(args["text"], args["citations"], state["evidence"])
        trace("verify", {"citations": cited_ids(args["text"]), "problem": problem})
        if problem:
            return {
                "feedback": problem,
                "answer_attempts": state.get("answer_attempts", 0) + 1,
            }
        return {
            "status": "answered",
            "answer": args["text"],
            "citations": sorted(set(cited_ids(args["text"])) | set(args["citations"])),
        }

    def ask_user(state: QAState) -> QAState:
        return {
            "status": "needs_clarification",
            "clarification": state["decision"]["args"]["question"],
        }

    def abstain(state: QAState) -> QAState:
        decision = state.get("decision") or {}
        if decision.get("action") == "abstain":
            missing = decision["args"]["missing"]
        else:
            missing = (
                "No verified answer within the step budget. "
                f"Last problem: {state.get('feedback') or 'none'}"
            )
        return {"status": "abstained", "missing": missing}

    def after_decide(state: QAState) -> str:
        action = state["decision"]["action"]
        if action == "abstain" and len(state["queries"]) < MAX_SEARCHES:
            # One top-k search shows only part of the collection, so it cannot
            # prove the evidence is missing: search for the missing part first.
            return "retrieve"
        if action in ("answer", "ask_user", "abstain"):
            return action
        if state["decisions"] >= MAX_DECISIONS:
            return "abstain"
        return {"search": "retrieve", "use_tool": "run_tool"}.get(action, "decide")

    def after_verify(state: QAState) -> str:
        if state.get("status") == "answered":
            return END
        if (
            state["answer_attempts"] < MAX_ANSWER_ATTEMPTS
            and state["decisions"] < MAX_DECISIONS
        ):
            return "decide"
        return "abstain"

    graph = StateGraph(QAState)
    graph.add_node("retrieve", retrieve)
    graph.add_node("decide", decide)
    graph.add_node("run_tool", run_tool)
    graph.add_node("verify", verify)
    graph.add_node("ask_user", ask_user)
    graph.add_node("abstain", abstain)
    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "decide")
    graph.add_conditional_edges(
        "decide",
        after_decide,
        {
            "answer": "verify",
            "retrieve": "retrieve",
            "run_tool": "run_tool",
            "ask_user": "ask_user",
            "abstain": "abstain",
            "decide": "decide",
        },
    )
    graph.add_edge("run_tool", "decide")
    graph.add_conditional_edges(
        "verify", after_verify, {END: END, "decide": "decide", "abstain": "abstain"}
    )
    graph.add_edge("ask_user", END)
    graph.add_edge("abstain", END)
    return graph.compile()


def answer_question(
    client: ModelClient,
    question: str,
    search: Search,
    trace_path: Path | None = None,
) -> QAState:
    graph = build_graph(client, search, trace_path)
    state = graph.invoke({"question": question}, config={"recursion_limit": 40})
    if trace_path:
        append_trace(
            trace_path,
            "run_complete",
            {
                key: state.get(key)
                for key in (
                    "status",
                    "answer",
                    "citations",
                    "missing",
                    "clarification",
                    "queries",
                    "decisions",
                    "cost",
                )
            },
        )
    return state
