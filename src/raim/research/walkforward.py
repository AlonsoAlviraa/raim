"""Purged + embargoed walk-forward over time-ordered events.

Folds are contiguous blocks of book-event indices. For each test block, the train set is
all EARLIER data (expanding window, so no future data), minus a purge gap right before the
test block (labels/feature windows overlapping the boundary) and an embargo after."""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Fold:
    train: np.ndarray
    test: np.ndarray


def purged_walk_forward(n: int, n_folds: int = 5, purge: int = 200, embargo: int = 100,
                        min_train: int = 1000) -> list[Fold]:
    edges = np.linspace(0, n, n_folds + 2, dtype=int)
    folds = []
    for k in range(1, n_folds + 1):
        t0, t1 = edges[k], edges[k + 1]
        train_end = max(t0 - purge, 0)
        train = np.arange(0, train_end)
        if len(train) < min_train:
            continue
        test = np.arange(min(t0 + embargo, t1), t1)
        folds.append(Fold(train, test))
    return folds


def assert_no_leakage(fold: Fold, purge: int) -> None:
    if len(fold.train) and len(fold.test):
        assert fold.train.max() + purge <= fold.test.min(), "train overlaps test (leakage)"
