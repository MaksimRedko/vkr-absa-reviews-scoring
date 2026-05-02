from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from src.pipeline.stage_cache import StageCacheManager
from src.pipeline.stages import s1_extraction, s2_encoding, s3_vocab_matching, s4_discovery


def test_stage_cache_manager_roundtrip(tmp_path: Path) -> None:
    source_dir = tmp_path / "run"
    source_dir.mkdir()
    (source_dir / "a.txt").write_text("hello", encoding="utf-8")
    nested = source_dir / "nested"
    nested.mkdir()
    (nested / "b.bin").write_bytes(b"\x00\x01\x02")

    cache = StageCacheManager(root_dir=tmp_path / "cache", enabled=True)
    fingerprint = cache.fingerprint({"stage": "demo", "x": 1})
    cache.store_from_run_dir(
        "demo",
        fingerprint,
        source_dir,
        ["a.txt", "nested/b.bin"],
        inputs={"stage": "demo", "x": 1},
    )

    restored_dir = tmp_path / "restored"
    restored_dir.mkdir()
    meta = cache.restore_to_run_dir("demo", fingerprint, restored_dir)

    assert meta is not None
    assert (restored_dir / "a.txt").read_text(encoding="utf-8") == "hello"
    assert (restored_dir / "nested" / "b.bin").read_bytes() == b"\x00\x01\x02"


def test_restore_helpers_rebuild_review_state_and_vectors() -> None:
    reviews = [
        SimpleNamespace(review_id="r1", candidate_surfaces_by_lemma={}, candidate_lemmas=set(), vocab_aspect_ids=set(), unmatched_phrases=[], discovery_cluster_ids=set()),
        SimpleNamespace(review_id="r2", candidate_surfaces_by_lemma={}, candidate_lemmas=set(), vocab_aspect_ids=set(), unmatched_phrases=[], discovery_cluster_ids=set()),
    ]
    candidates = pd.DataFrame(
        [
            {
                "candidate_id": "c1",
                "review_id": "r1",
                "text": "катушка",
                "text_lemmatized": "катушка",
                "start_offset": 0,
                "end_offset": 7,
            },
            {
                "candidate_id": "c2",
                "review_id": "r1",
                "text": "доставка",
                "text_lemmatized": "доставка",
                "start_offset": 10,
                "end_offset": 18,
            },
            {
                "candidate_id": "c3",
                "review_id": "r2",
                "text": "упаковка",
                "text_lemmatized": "упаковка",
                "start_offset": 0,
                "end_offset": 9,
            },
        ]
    )
    s1_extraction.apply_cached_results(reviews, candidates)

    assert reviews[0].candidate_lemmas == {"катушка", "доставка"}
    assert reviews[1].candidate_surfaces_by_lemma["упаковка"] == ["упаковка"]

    matches = pd.DataFrame(
        [
            {
                "candidate_id": "c1",
                "matched_aspect_id": "coil",
                "match_method": "lexical",
                "match_score": 1.0,
                "matched_lemmas": ["катушка"],
                "cosine_similarity": 0.9,
                "is_unmatched": False,
            },
            {
                "candidate_id": "c2",
                "matched_aspect_id": None,
                "match_method": None,
                "match_score": float("nan"),
                "matched_lemmas": [],
                "cosine_similarity": 0.3,
                "is_unmatched": True,
            },
            {
                "candidate_id": "c3",
                "matched_aspect_id": None,
                "match_method": None,
                "match_score": float("nan"),
                "matched_lemmas": [],
                "cosine_similarity": 0.2,
                "is_unmatched": True,
            },
        ]
    )
    s3_vocab_matching.apply_cached_results(reviews, candidates, matches, {})

    assert reviews[0].vocab_aspect_ids == {"coil"}
    assert reviews[0].unmatched_phrases == ["доставка"]
    assert reviews[1].unmatched_phrases == ["упаковка"]

    candidate_embeddings = np.asarray([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]], dtype=np.float32)
    candidate_index = pd.DataFrame(
        [
            {"candidate_id": "c1", "row_index": 0},
            {"candidate_id": "c2", "row_index": 1},
            {"candidate_id": "c3", "row_index": 2},
        ]
    )
    vocab_embeddings = np.asarray([[0.9, 0.1], [0.1, 0.9]], dtype=np.float32)
    vocab_index = pd.DataFrame(
        [
            {"aspect_id": "coil", "category_id": "physical_goods", "row_index": 0},
            {"aspect_id": "delivery", "category_id": "physical_goods", "row_index": 1},
        ]
    )
    enc = s2_encoding.restore_stage(
        candidates,
        {"physical_goods": []},
        {"models": {"encoder": "stub"}, "discovery": {"encoder_batch_size": 2}},
        candidate_embeddings=candidate_embeddings,
        candidate_index=candidate_index,
        vocab_embeddings=vocab_embeddings,
        vocab_index=vocab_index,
        encoder=None,
        cache={},
    )

    assert np.allclose(enc["candidate_vectors_by_id"]["c1"], np.asarray([1.0, 0.0], dtype=np.float32))
    assert np.allclose(enc["aspect_vectors_by_category"]["physical_goods"]["delivery"], np.asarray([0.1, 0.9], dtype=np.float32))


def test_restore_discovery_by_product_and_apply_cached_results() -> None:
    payloads = {
        101: {
            "nm_id": 101,
            "category_id": "physical_goods",
            "clusters": [
                {
                    "cluster_id": 7,
                    "top_phrases": [["катушка трещит", 3], ["сломалась катушка", 2]],
                    "medoid_phrase": "катушка трещит",
                    "gold_matches": {"катушка": 0.88},
                }
            ],
        }
    }
    centroids = {
        101: np.asarray([[0.2, 0.8]], dtype=np.float32),
    }
    discovery = s4_discovery.restore_discovery_by_product(payloads, centroids)

    assert 101 in discovery
    assert discovery[101].clusters[7].medoid == "катушка трещит"
    assert discovery[101].clusters[7].gold_matches == {"катушка": 0.88}
    assert np.allclose(discovery[101].clusters[7].centroid, np.asarray([0.2, 0.8], dtype=np.float32))

    reviews = [
        SimpleNamespace(review_id="r1", discovery_cluster_ids=set()),
        SimpleNamespace(review_id="r2", discovery_cluster_ids=set()),
    ]
    bindings = pd.DataFrame(
        [
            {"review_id": "r1", "cluster_id": 7},
            {"review_id": "r1", "cluster_id": 9},
            {"review_id": "r2", "cluster_id": 8},
        ]
    )
    s4_discovery.apply_cached_results(reviews, bindings)

    assert reviews[0].discovery_cluster_ids == {7, 9}
    assert reviews[1].discovery_cluster_ids == {8}
