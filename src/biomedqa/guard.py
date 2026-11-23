"""Test-set lock. The official test split stays sealed until Phase 4.

Every code path that loads, predicts on or scores questions goes through these checks. The only
way past them is an explicit ``allow_test=True`` (CLI: ``--allow-test``).
"""

from __future__ import annotations

from collections.abc import Iterable

TEST_SPLIT = "test"


class TestSetLockedError(RuntimeError):
    """Raised when the test split is touched without ``allow_test=True``."""

    __test__ = False  # not a pytest test class


def check_split_access(split: str, allow_test: bool = False) -> None:
    if split == TEST_SPLIT and not allow_test:
        raise TestSetLockedError(
            "The test split is locked until Phase 4. Pass allow_test=True / --allow-test "
            "only for the frozen final evaluation."
        )


def check_ids_not_test(
    ids: Iterable[str], test_ids: Iterable[str], allow_test: bool = False
) -> None:
    """Raise if any of ``ids`` belongs to the test split and the lock is not lifted."""
    if allow_test:
        return
    overlap = set(ids) & set(test_ids)
    if overlap:
        raise TestSetLockedError(
            f"{len(overlap)} id(s) belong to the locked test split. Pass allow_test=True / "
            "--allow-test only for the frozen final evaluation."
        )
