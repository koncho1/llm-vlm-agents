import json

from agent_project import MockClient
from agent_project.qa_graph import (
    MAX_SEARCHES,
    answer_question,
    cited_ids,
    offered_tools,
    verify_citations,
)

WOOL = {"id": "fabric-care#1", "heading": "Wool", "text": "Hand wash.", "score": 0.9}
PILLING = {"id": "fabric-care#2", "heading": "Pilling", "text": "Shave.", "score": 0.6}


def fake_search(query: str) -> list[dict]:
    """Deterministic retrieval: only the word 'pilling' finds the pilling chunk."""
    return [PILLING] if "pilling" in query else [WOOL]


def _call(name: str, **arguments) -> dict:
    call = {"id": "c1", "name": name, "arguments": json.dumps(arguments)}
    return {"text": None, "tool_calls": [call], "usage": {"cost": 0.001}}


def _events(path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_citation_parsing_handles_lists() -> None:
    text = "Wash it cold [fabric-care#1, shoe-care#2]. Not an id [see above]."
    assert cited_ids(text) == ["fabric-care#1", "shoe-care#2"]


def test_verify_rejects_missing_and_unretrieved_citations() -> None:
    evidence = {"fabric-care#1": WOOL}
    assert (
        verify_citations("Hand wash [fabric-care#1].", ["fabric-care#1"], evidence)
        is None
    )
    assert verify_citations("Hand wash.", ["fabric-care#1"], evidence) is None
    assert "cites no evidence" in verify_citations("Hand wash.", [], evidence)
    problem = verify_citations(
        "Use bleach [fabric-care#9].", ["fabric-care#9"], evidence
    )
    assert "fabric-care#9" in problem


def test_item_id_citation_feedback_lists_valid_ids() -> None:
    # Regression for eval q08: the model cited a wardrobe item id, "w11".
    evidence = {"tool:search_wardrobe#1": {}}
    problem = verify_citations("Wear the boots (w11).", ["w11"], evidence)
    assert "tool:search_wardrobe#1" in problem


def test_answer_with_verified_citation(tmp_path) -> None:
    client = MockClient(
        [
            _call(
                "answer",
                text="Hand wash it [fabric-care#1].",
                citations=["fabric-care#1"],
            )
        ]
    )
    state = answer_question(client, "Wash wool?", fake_search, tmp_path / "t.jsonl")
    assert state["status"] == "answered"
    assert state["citations"] == ["fabric-care#1"]


def test_insufficient_first_retrieval_searches_again(tmp_path) -> None:
    client = MockClient(
        [
            _call("search_again", query="sweater pilling fuzzy balls"),
            _call(
                "answer",
                text="Shave the pills [fabric-care#2].",
                citations=["fabric-care#2"],
            ),
        ]
    )
    trace = tmp_path / "t.jsonl"
    state = answer_question(client, "Fuzzy balls on my jumper?", fake_search, trace)
    assert state["status"] == "answered"
    assert state["queries"] == [
        "Fuzzy balls on my jumper?",
        "sweater pilling fuzzy balls",
    ]
    assert [e["event"] for e in _events(trace)][:4] == [
        "retrieve",
        "decide",
        "retrieve",
        "decide",
    ]


def test_search_again_is_withdrawn_after_the_limit() -> None:
    names = lambda state: {t["function"]["name"] for t in offered_tools(state)}  # noqa: E731
    assert "search_again" in names({"queries": ["q"]})
    assert "search_again" not in names({"queries": ["q"] * MAX_SEARCHES})
    assert "search_wardrobe" not in names({"tool_calls": 2})


def test_withdrawn_action_is_refused_then_model_abstains(tmp_path) -> None:
    client = MockClient(
        [
            _call("search_again", query="pilling"),
            _call("search_again", query="pilling again"),  # over the limit
            _call("abstain", missing="Nothing about this fabric."),
        ]
    )
    state = answer_question(client, "Q?", fake_search, tmp_path / "t.jsonl")
    assert state["status"] == "abstained"
    assert len(state["queries"]) == MAX_SEARCHES


def test_unretrieved_citation_is_rejected_then_corrected(tmp_path) -> None:
    client = MockClient(
        [
            _call(
                "answer", text="Bleach it [fabric-care#9].", citations=["fabric-care#9"]
            ),
            _call(
                "answer",
                text="Hand wash it [fabric-care#1].",
                citations=["fabric-care#1"],
            ),
        ]
    )
    trace = tmp_path / "t.jsonl"
    state = answer_question(client, "Wash wool?", fake_search, trace)
    assert state["status"] == "answered"
    verify = [e["payload"] for e in _events(trace) if e["event"] == "verify"]
    assert "fabric-care#9" in verify[0]["problem"]
    assert verify[1]["problem"] is None


def test_repeated_bad_citations_end_in_abstention(tmp_path) -> None:
    bad = _call(
        "answer", text="Bleach it [fabric-care#9].", citations=["fabric-care#9"]
    )
    client = MockClient([bad, bad])
    state = answer_question(client, "Wash wool?", fake_search, tmp_path / "t.jsonl")
    assert state["status"] == "abstained"
    assert "fabric-care#9" in state["missing"]


def test_abstain_names_what_is_missing_after_a_second_search(tmp_path) -> None:
    abstain = _call("abstain", missing="jacket prices")
    client = MockClient([abstain, abstain])
    state = answer_question(client, "Price?", fake_search, tmp_path / "t.jsonl")
    assert state["status"] == "abstained"
    assert state["missing"] == "jacket prices"
    assert state["queries"] == ["Price?", "jacket prices"]


def test_early_abstention_triggers_a_search_for_the_missing_part(tmp_path) -> None:
    # Regression for eval q11: the model abstained after one search although a
    # second search would have found the missing part.
    client = MockClient(
        [
            _call("abstain", missing="anything about pilling"),
            _call(
                "answer", text="Shave it [fabric-care#2].", citations=["fabric-care#2"]
            ),
        ]
    )
    trace = tmp_path / "t.jsonl"
    state = answer_question(client, "Fuzzy jumper?", fake_search, trace)
    assert state["status"] == "answered"
    assert state["queries"] == ["Fuzzy jumper?", "anything about pilling"]
    assert [e["event"] for e in _events(trace)][:4] == [
        "retrieve",
        "decide",
        "retrieve",
        "decide",
    ]


def test_ambiguous_question_asks_the_user(tmp_path) -> None:
    client = MockClient([_call("ask_user", question="Which event is it?")])
    state = answer_question(client, "What to wear?", fake_search, tmp_path / "t.jsonl")
    assert state["status"] == "needs_clarification"
    assert state["clarification"] == "Which event is it?"


def test_wardrobe_tool_result_becomes_citable_evidence(tmp_path) -> None:
    client = MockClient(
        [
            _call("search_wardrobe", category="shoes", weather_tags=["rain"]),
            _call(
                "answer",
                text="Wear the waterproof boots [tool:search_wardrobe#1].",
                citations=["tool:search_wardrobe#1"],
            ),
        ]
    )
    state = answer_question(client, "Rain shoes?", fake_search, tmp_path / "t.jsonl")
    assert state["status"] == "answered"
    assert (
        "Black Waterproof Boots" in state["evidence"]["tool:search_wardrobe#1"]["text"]
    )


def test_plain_text_reply_is_not_accepted_as_an_answer(tmp_path) -> None:
    client = MockClient([{"text": "Just wash it.", "tool_calls": [], "usage": {}}] * 6)
    state = answer_question(client, "Wash wool?", fake_search, tmp_path / "t.jsonl")
    assert state["status"] == "abstained"
    assert state["decisions"] == 6
