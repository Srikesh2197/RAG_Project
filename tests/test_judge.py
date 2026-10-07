from types import SimpleNamespace

from ragbasics.cache import CachedGenerator, DiskCache
from ragbasics.eval.judge import (
    judge_correctness,
    judge_faithfulness,
    parse_verdict,
)
from ragbasics.generation.llm import Generation, OpenAIGenerator
from ragbasics.types import Chunk


class ScriptedLLM:
    model = "scripted"

    def __init__(self, reply: str):
        self.reply = reply
        self.calls: list[tuple[str, str]] = []

    def generate(self, system: str, user: str) -> Generation:
        self.calls.append((system, user))
        return Generation(self.reply, input_tokens=100, output_tokens=20)


def test_parse_verdict_reads_json_and_tolerates_text_around_it():
    assert parse_verdict('{"reasoning": "same name", "verdict": "correct"}', "correct",
                         "incorrect") == (True, "same name")
    wrapped = 'Here you go:\n```json\n{"reasoning": "r", "verdict": "Incorrect"}\n```'
    assert parse_verdict(wrapped, "correct", "incorrect") == (False, "r")


def test_an_unreadable_reply_is_no_verdict_not_a_failure():
    assert parse_verdict("I think it is fine.", "correct", "incorrect") == (None, "")
    assert parse_verdict('{"verdict": "maybe"}', "correct", "incorrect")[0] is None
    assert parse_verdict("{not json}", "correct", "incorrect")[0] is None


def test_correctness_judge_sees_question_reference_and_answer():
    llm = ScriptedLLM('{"reasoning": "matches", "verdict": "correct"}')
    verdict = judge_correctness(llm, "Who?", "Sam Bankman-Fried", "It was SBF.")
    assert verdict.passed is True and verdict.reasoning == "matches"
    system, user = llm.calls[0]
    assert "reference" in system.lower()
    assert user == (
        "Question: Who?\n\nReference answer: Sam Bankman-Fried\n\nCandidate answer: It was SBF."
    )


def test_faithfulness_judge_sees_the_passages_and_not_the_reference():
    llm = ScriptedLLM('{"reasoning": "not stated", "verdict": "unsupported"}')
    context = [Chunk("d:0", "d", "The ferry was late.", 0, 19)]
    verdict = judge_faithfulness(llm, "Why?", "Fog.", context)
    assert verdict.passed is False
    _, user = llm.calls[0]
    assert "The ferry was late." in user and user.endswith("Question: Why?\n\nAnswer: Fog.")


def test_cache_pays_once_per_distinct_request_and_per_sample(tmp_path):
    llm = ScriptedLLM("answer")
    cache = DiskCache(tmp_path / "cache" / "llm.sqlite")
    cached = CachedGenerator(llm, cache)

    assert cached.is_cached("s", "u") is False
    first = cached.generate("s", "u")
    again = cached.generate("s", "u")
    assert (first.cached, again.cached) == (False, True)
    assert again.text == "answer" and again.input_tokens == 100
    assert len(llm.calls) == 1

    cached.generate("s", "another question")
    assert len(llm.calls) == 2
    # A second sample is a deliberate new draw, cached under its own key.
    repeat = CachedGenerator(llm, cache, sample=1)
    assert repeat.generate("s", "u").cached is False
    assert repeat.generate("s", "u").cached is True
    assert len(llm.calls) == 3


def test_cache_survives_reopening_the_file(tmp_path):
    path = tmp_path / "llm.sqlite"
    CachedGenerator(ScriptedLLM("answer"), DiskCache(path)).generate("s", "u")
    fresh = ScriptedLLM("different")
    assert CachedGenerator(fresh, DiskCache(path)).generate("s", "u").text == "answer"
    assert fresh.calls == []


class StubOpenAIChat:
    def __init__(self):
        self.chat = SimpleNamespace(completions=self)
        self.request = None

    def create(self, **request):
        self.request = request
        message = SimpleNamespace(content=' {"verdict": "correct"} ')
        return SimpleNamespace(
            choices=[SimpleNamespace(message=message, finish_reason="stop")],
            usage=SimpleNamespace(prompt_tokens=300, completion_tokens=25),
        )


def test_openai_generator_sends_system_and_user_and_reads_usage():
    client = StubOpenAIChat()
    generation = OpenAIGenerator("gpt-6-luna", client=client).generate("system text", "user text")
    assert generation.text == '{"verdict": "correct"}'
    assert (generation.input_tokens, generation.output_tokens) == (300, 25)
    assert client.request["model"] == "gpt-6-luna"
    assert client.request["messages"] == [
        {"role": "system", "content": "system text"},
        {"role": "user", "content": "user text"},
    ]
    assert "temperature" not in client.request and "reasoning_effort" not in client.request
