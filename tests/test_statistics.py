from __future__ import annotations

import numpy as np
import pytest

from virelion_cardioscore.analysis.statistics import bootstrap_ci, bootstrap_cluster_ci


def test_bootstrap_ci_is_reproducible():
    values = np.array([1.0, 2.0, 3.0, 4.0])
    first = bootstrap_ci(values, n_bootstrap=500, seed=42)
    second = bootstrap_ci(values, n_bootstrap=500, seed=42)

    assert first.to_dict() == second.to_dict()
    assert first.n_clusters is None


def test_cluster_bootstrap_preserves_cluster_structure():
    # Each independent cluster has two technical observations. Resampling
    # clusters must keep those observations together rather than treating all
    # six wells as six independent units.
    values = np.array([0.0, 0.0, 10.0, 10.0, 20.0, 20.0])
    clusters = np.array(["B1", "B1", "B2", "B2", "B3", "B3"])

    result = bootstrap_cluster_ci(values, clusters, n_bootstrap=500, seed=7)

    assert result.estimate == pytest.approx(10.0)
    assert result.n_observations == 6
    assert result.n_clusters == 3
    assert result.ci_low <= result.estimate <= result.ci_high


def test_cluster_bootstrap_weights_clusters_equally_when_sizes_differ():
    # B1 contributes one technical observation while B2/B3 contribute four
    # each. The cluster-level estimate must still give B1/B2/B3 equal weight.
    values = np.array([0.0, 10.0, 10.0, 10.0, 10.0, 20.0, 20.0, 20.0, 20.0])
    clusters = np.array([
        "B1",
        "B2", "B2", "B2", "B2",
        "B3", "B3", "B3", "B3",
    ])

    result = bootstrap_cluster_ci(values, clusters, n_bootstrap=500, seed=7)

    assert result.estimate == pytest.approx(10.0)
    assert result.n_clusters == 3


def test_cluster_bootstrap_requires_multiple_clusters():
    with pytest.raises(ValueError, match="At least two independent clusters"):
        bootstrap_cluster_ci(np.array([1.0, 2.0]), np.array(["B1", "B1"]), n_bootstrap=100)


def test_cluster_bootstrap_rejects_missing_cluster_identifier():
    with pytest.raises(ValueError, match="missing identifiers"):
        bootstrap_cluster_ci(np.array([1.0, 2.0]), np.array(["B1", None], dtype=object), n_bootstrap=100)
