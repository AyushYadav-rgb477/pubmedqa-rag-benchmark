import os

import pytest

from biomedqa import env
from biomedqa.config import load_config, resolve
from biomedqa.data.pubmedqa import example_from_row, examples_by_pmid
from biomedqa.hashing import config_hash, ids_checksum, sha256_bytes, sort_ids
from biomedqa.labels import INVALID, LABELS, check_gold, check_prediction


def test_labels_and_checks() -> None:
    assert LABELS == ("yes", "no", "maybe")
    assert check_gold("maybe") == "maybe"
    assert check_prediction(INVALID) == INVALID
    with pytest.raises(ValueError):
        check_gold(INVALID)
    with pytest.raises(ValueError):
        check_prediction("Yes")


def test_config_hash_is_key_order_independent() -> None:
    a = {"model": "x", "gen": {"temperature": 0, "max_new_tokens": 512}}
    b = {"gen": {"max_new_tokens": 512, "temperature": 0}, "model": "x"}
    assert config_hash(a) == config_hash(b)
    assert config_hash(a) != config_hash({**a, "model": "y"})


def test_ids_checksum_sorts_numerically_and_uses_lf() -> None:
    assert sort_ids(["10", "9", "100"]) == ["9", "10", "100"]
    assert ids_checksum(["10", "9"]) == sha256_bytes(b"9\n10\n")
    assert ids_checksum(["10", "9"]) == ids_checksum(["9", "10"])


def test_example_from_row_maps_fields(row_factory) -> None:
    ex = example_from_row(row_factory(123, "no"))
    assert ex.pmid == "123"
    assert ex.label == "no"
    assert ex.contexts == ("Background of 123.", "Results of 123.")
    assert ex.context_text == "Background of 123. Results of 123."


def test_context_text_never_contains_long_answer(row_factory) -> None:
    ex = example_from_row(row_factory(5, "yes"))
    assert "CONCLUSION" not in ex.context_text


def test_label_is_normalised_and_validated(row_factory) -> None:
    row = row_factory(1)
    row["final_decision"] = " Maybe "
    assert example_from_row(row).label == "maybe"
    row["final_decision"] = "unsure"
    with pytest.raises(ValueError):
        example_from_row(row)


def test_duplicate_pmids_raise(row_factory) -> None:
    with pytest.raises(ValueError, match="duplicate"):
        examples_by_pmid([row_factory(1), row_factory(1)])


def test_data_config_pins_revision() -> None:
    cfg = load_config(resolve("configs/data.yaml"))
    assert len(cfg["dataset"]["revision"]) == 40
    s = cfg["splits"]
    assert (s["n_train"], s["n_dev"], s["n_test"]) == (400, 100, 500)


def test_configure_safe_env_forces_jax_cpu(monkeypatch) -> None:
    monkeypatch.setenv("JAX_PLATFORMS", "cuda")
    monkeypatch.delenv("XLA_PYTHON_CLIENT_PREALLOCATE", raising=False)
    env.configure_safe_env()
    assert os.environ["JAX_PLATFORMS"] == "cpu"
    assert os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] == "false"


def test_collect_versions_records_python_and_packages() -> None:
    v = env.collect_versions(("numpy", "definitely-not-installed-pkg"))
    assert v["python"]
    assert "numpy" in v["packages"]
    assert "definitely-not-installed-pkg" not in v["packages"]
