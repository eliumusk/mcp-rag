from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import utils


def test_compute_content_hash_is_deterministic():
    hash_a = utils.compute_content_hash("hello world")
    hash_b = utils.compute_content_hash("hello world")
    hash_c = utils.compute_content_hash("another")
    assert hash_a == hash_b
    assert hash_a != hash_c


@patch("utils.store_embeddings_in_cache")
@patch("utils.create_embeddings_batch")
@patch("utils.fetch_embedding_cache")
def test_get_embeddings_with_cache_hits_and_misses(fetch_mock, create_mock, store_mock):
    now_iso = datetime.now(timezone.utc).isoformat()
    fetch_mock.return_value = {
        "hash_cached": {
            "embedding": [0.5, 0.0],
            "refreshed_at": now_iso,
            "needs_refresh": False,
        }
    }
    create_mock.return_value = [[1.0, 1.0]]

    embeddings = utils.get_embeddings_with_cache(
        client=MagicMock(),
        texts=["cached chunk", "new chunk"],
        content_hashes=["hash_cached", "hash_new"],
        tenant_id="tenant",
        model_name="model",
        cache_context="documents",
    )

    assert embeddings[0] == [0.5, 0.0]
    assert embeddings[1] == [1.0, 1.0]
    store_mock.assert_called_once()
    # ensure cache miss triggered embedding generation
    create_mock.assert_called_once_with(["new chunk"])


@patch("utils.flag_cache_entry_for_refresh")
@patch("utils.fetch_embedding_cache")
def test_get_embeddings_with_cache_flags_stale_entries(fetch_mock, flag_mock):
    stale_time = (datetime.now(timezone.utc) - timedelta(seconds=utils.EMBEDDING_CACHE_TTL_SECONDS + 10)).isoformat()
    fetch_mock.return_value = {
        "stale_hash": {
            "embedding": [0.1, 0.2],
            "refreshed_at": stale_time,
            "needs_refresh": False,
        }
    }

    embeddings = utils.get_embeddings_with_cache(
        client=MagicMock(),
        texts=["stale chunk"],
        content_hashes=["stale_hash"],
        tenant_id="tenant",
        model_name="model",
        cache_context="documents",
    )

    assert embeddings[0] == [0.1, 0.2]
    flag_mock.assert_called_once()
