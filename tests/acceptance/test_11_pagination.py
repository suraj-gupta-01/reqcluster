"""11. Pagination / sorting / search."""
import pytest

pytestmark = pytest.mark.acceptance


@pytest.fixture
def big(api):
    return api.upload_texts([f"The subsystem number {i:04d} shall report status alpha" for i in range(250)])


def test_invalid_sort_field_422(api, big):
    st, b, _ = api.requirements(big, sort_field="not_a_column")
    assert st == 422, (st, b)


def test_invalid_sort_dir_422(api, big):
    st, b, _ = api.requirements(big, sort_dir="sideways")
    assert st == 422


@pytest.mark.parametrize("q", [{"page": 0}, {"page_size": 0}, {"page_size": 10**6}])
def test_invalid_page_params_422(api, big, q):
    st, b, _ = api.requirements(big, **q)
    assert st == 422


def test_page_without_page_size_uses_default_cap(api, big):
    st, items, total = api.requirements(big, page=1)
    assert st == 200
    assert 0 < len(items) < 250
    assert total == 250


def test_pages_partition_the_collection(api, big):
    seen = []
    for p in range(1, 6):
        st, items, total = api.requirements(big, page=p, page_size=60)
        assert st == 200 and total == 250
        seen += [r["id"] for r in items]
    assert len(seen) == 250 and len(set(seen)) == 250


def test_sort_by_req_id_desc(api, big):
    st, items, _ = api.requirements(big, sort_field="req_id", sort_dir="desc", page=1, page_size=20)
    ids = [r["req_id"] for r in items]
    assert ids == sorted(ids, reverse=True)


@pytest.fixture
def special(api):
    return api.upload_texts([
        "Coverage shall reach 100% of statements in the unit",
        "Coverage shall reach 100x of statements in the unit",
        "The a_b interface shall validate every frame header",
        "The aXb interface shall validate every frame header",
        "The plain interface shall validate every frame header",
    ])


def test_search_percent_is_literal(api, special):
    st, items, total = api.requirements(special, search="100%")
    assert st == 200 and [r["text"] for r in items] == ["Coverage shall reach 100% of statements in the unit"]
    st, items, _ = api.requirements(special, search="%")
    assert len(items) == 1


def test_search_underscore_is_literal(api, special):
    st, items, _ = api.requirements(special, search="a_b")
    assert [r["text"] for r in items] == ["The a_b interface shall validate every frame header"]


def test_search_backslash_does_not_error(api, special):
    st, items, _ = api.requirements(special, search="\\")
    assert st == 200 and items == []
