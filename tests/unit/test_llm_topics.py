
from reqcluster.llm import topics as T
from reqcluster.llm.gateway import LLMResult


class Gw:
    def __init__(self, tax=None, asg=None, source="llm"):
        self.tax, self.asg, self.source = tax, asg or [], source

    def complete(self, prompt, fallback, provider=None):
        assert prompt.task == "topics_taxonomy" and len(prompt.data["requirements"]) == 3
        return LLMResult(self.tax or fallback(), self.source, "x")

    def map(self, prompts, fallback_for, provider=None):
        return [LLMResult(a, "llm", "x") for a in self.asg]


def test_discover_dedupes_and_falls_back():
    tops = [T.Topic(name="Brakes"), T.Topic(name=" brakes "), T.Topic(name="Display")]
    got = T.discover(Gw(T.Taxonomy(topics=tops)), ["a", "b", "c"])
    assert [t.name for t in got] == ["Brakes", "Display"]
    assert T.discover(Gw(source="fallback"), ["a", "b", "c"]) is None
    assert T.discover(Gw(T.Taxonomy(topics=tops[:2])), ["a", "b", "c"]) is None  # 1 unique


def test_classify_batches_and_invalid_ids():
    A = T.Assignment
    asg = [T.Assignments(assignments=[A(id=1, topic=2), A(id=2, topic=9), A(id=7, topic=1)]),
           T.Assignments(assignments=[A(id=1, topic=1)])]
    tops = [T.Topic(name="a"), T.Topic(name="b")]
    out = T.classify(Gw(asg=asg), ["r"] * 4, tops, batch=3, progress=lambda *_: None)
    assert out.tolist() == [1, -1, -1, 0]
