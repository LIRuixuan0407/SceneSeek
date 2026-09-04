from __future__ import annotations

from pathlib import Path

import numpy as np

from sceneseek.retrieval.index import IndexedItem, VectorIndex


def item(name: str) -> IndexedItem:
    return IndexedItem(
        item_id=name,
        media_id=name,
        kind="image",
        path=f"/{name}.jpg",
        media_type="image",
        duration=None,
        width=100,
        height=100,
        start_sec=None,
        end_sec=None,
    )


def test_numpy_index_matches_inner_product_order(tmp_path: Path) -> None:
    index = VectorIndex(tmp_path / "index.npz", backend="numpy")
    vectors = np.array([[1, 0], [0.8, 0.2], [0, 1]], dtype=np.float32)
    index.set_data([item("a"), item("b"), item("c")], vectors)

    results = index.search(np.array([1, 0], dtype=np.float32), 3)

    assert [result.item.item_id for result in results] == ["a", "b", "c"]
    assert results[0].score == 1.0


def test_index_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "index.npz"
    index = VectorIndex(path, backend="numpy")
    index.set_data([item("a")], np.array([[3, 4]], dtype=np.float32))
    index.save()

    loaded = VectorIndex(path, backend="numpy")

    assert loaded.size == 1
    assert loaded.dimension == 2
    assert np.linalg.norm(loaded.vectors[0]) == np.float32(1.0)
