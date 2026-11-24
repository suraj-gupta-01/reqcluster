from reqcluster.texts import build_texts, strip_boilerplate


W = [f"w{chr(97 + i // 26)}{chr(97 + i % 26)}" for i in range(40)]


def _rows(k, n=40):
    return [f"{W[i]} unit shall handle case {i}. Page {i} shall be traced." if i < k else f"{W[i]} unit shall handle case {i}." for i in range(n)]


def test_strip_repeated_sentence_with_digits():
    out = strip_boilerplate(_rows(5))
    assert out[0] == f"{W[0]} unit shall handle case 0." and out[5] == _rows(5)[5]


def test_keeps_sentence_in_four_rows():
    rows = _rows(4)
    assert strip_boilerplate(rows) == rows


def test_boilerplate_only_row_keeps_original():
    rows = _rows(5)
    rows[0] = "Page 0 shall be traced."
    assert strip_boilerplate(rows)[0] == rows[0]


def test_build_texts_base_uses_stripped():
    assert build_texts(_rows(5), "base")[0][0] == f"{W[0]} unit shall handle case 0."
