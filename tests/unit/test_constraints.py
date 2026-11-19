import numpy as np
import pytest

from reqcluster.errors import Invalid
from reqcluster.ml.constraints import apply, from_correction


def _emb(n, dim=8):
    x = np.random.default_rng(0).standard_normal((n, dim)).astype(np.float32)
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def test_no_constraints_is_noop():
    labels = np.array([0, 1, 2])
    new, moved = apply(_emb(3), labels, [], [])
    assert np.array_equal(new, labels) and moved == 0


def test_must_link_merges():
    new, moved = apply(_emb(5), np.array([0, 0, 1, 1, 2]), [(0, 2)], [])
    assert new[0] == new[2] and moved >= 1


def test_must_link_chain_moves_together():
    new, _ = apply(_emb(6), np.array([0, 0, 0, 1, 2, 2]), [(0, 3), (3, 4)], [])
    assert new[0] == new[3] == new[4]


def test_cannot_link_separates_and_keeps_must_links():
    labels = np.array([0, 0, 0, 0, 1, 1])
    new, _ = apply(_emb(6), labels, [(0, 1)], [(1, 2)])
    assert new[1] != new[2]
    assert new[0] == new[1]  # must-link component never split


def test_cannot_link_moves_smaller_component():
    new, _ = apply(_emb(6), np.array([0, 0, 0, 0, 1, 1]), [(1, 2)], [(0, 1)])
    assert new[0] != new[1] and new[1] == new[2]


def test_contradiction_raises_naming_pair():
    with pytest.raises(Invalid, match=r"\(0, 2\)"):
        apply(_emb(4), np.array([0, 0, 1, 1]), [(0, 1), (1, 2)], [(2, 0)])


def test_out_of_range_raises():
    with pytest.raises(Invalid):
        apply(_emb(3), np.array([0, 0, 1]), [(0, 9)], [])


def test_from_correction_pairs_sorted_unique():
    X = _emb(8)
    labels = np.array([1, 1, 1, 1, 0, 0, 0, 0])  # point 0 already moved into 1
    must, cannot = from_correction(labels, 0, 1, 0, X=X)
    assert must and cannot
    for a, b in must + cannot:
        assert a < b
    assert len(set(must)) == len(must) and len(set(cannot)) == len(cannot)
    assert all(0 in p for p in must + cannot)


def test_from_correction_to_noise_only_cannot():
    must, cannot = from_correction(np.array([-1, 0, 0, 0]), 0, -1, 0, X=_emb(4))
    assert must == [] and cannot
