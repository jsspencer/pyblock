from __future__ import annotations

import numpy as np
import pytest

import pyblock
import pyblock.tests.base as tests_base


def assert_stats_equal(benchmark, test):
    assert len(benchmark) == len(test)
    for b_item, t_item in zip(benchmark, test):
        for bench_val, test_val in zip(b_item, t_item):
            if isinstance(bench_val, np.ndarray) or isinstance(test_val, np.ndarray):
                np.testing.assert_array_almost_equal(test_val, bench_val, decimal=8)
            else:
                assert test_val == bench_val


class TestBlocking1D:
    @pytest.fixture(autouse=True)
    def setup_data(self):
        self.data = tests_base.data_1D
        self.weights = None
        self.benchmark = tests_base.reblock_1D
        self.benchmark_opt = tests_base.reblock_1D_opt
        self.benchmark_ddof = tests_base.reblock_1D_ddof

    def test_reblock(self):
        stats = pyblock.blocking.reblock(self.data, weights=self.weights)
        assert_stats_equal(self.benchmark, stats)

    def test_reblock_optimal(self):
        stats = pyblock.blocking.reblock(self.data, weights=self.weights)
        assert self.benchmark_opt == pyblock.blocking.find_optimal_block(
            len(self.data), stats
        )

    def test_reblock_ddof(self):
        stats = pyblock.blocking.reblock(self.data, ddof=0, weights=self.weights)
        assert_stats_equal(self.benchmark_ddof, stats)


class TestUnitWeightedBlocking1D(TestBlocking1D):
    @pytest.fixture(autouse=True)
    def setup_weights(self):
        self.weights = np.ones(self.data.shape[0])


class TestWeightedBlocking1D(TestBlocking1D):
    @pytest.fixture(autouse=True)
    def setup_weights(self):
        self.weights = tests_base.weights
        self.benchmark = tests_base.weighted_reblock_1D
        self.benchmark_opt = tests_base.weighted_reblock_1D_opt
        self.benchmark_ddof = tests_base.weighted_reblock_1D_ddof

    def test_weight_negative(self):
        with pytest.raises(RuntimeError):
            pyblock.blocking.reblock(self.data, weights=-self.weights)

    def test_weight_len(self):
        with pytest.raises(RuntimeError):
            pyblock.blocking.reblock(self.data, weights=self.weights[:-1])


class TestBlocking2D:
    @pytest.fixture(autouse=True)
    def setup_data(self):
        self.data = tests_base.data_2D
        self.weights = None
        self.benchmark = tests_base.reblock_2D
        self.benchmark_opt = tests_base.reblock_2D_opt

    def test_reblock(self):
        stats = pyblock.blocking.reblock(self.data, weights=self.weights)
        assert_stats_equal(self.benchmark, stats)

    def test_reblock_row(self):
        stats = pyblock.blocking.reblock(
            self.data.transpose(), rowvar=0, weights=self.weights
        )
        assert_stats_equal(self.benchmark, stats)

    def test_reblock_rowvar(self):
        with pytest.raises(ValueError):
            pyblock.blocking.reblock(self.data, ddof=1.2, weights=self.weights)

    def test_reblock_optimal(self):
        stats = pyblock.blocking.reblock(self.data, weights=self.weights)
        assert self.benchmark_opt == pyblock.blocking.find_optimal_block(
            len(self.data[0]), stats
        )


class TestUnitWeightedBlocking2D(TestBlocking2D):
    @pytest.fixture(autouse=True)
    def setup_weights(self):
        self.weights = np.ones(self.data.shape[1])


class TestWeightedBlocking2D(TestBlocking2D):
    @pytest.fixture(autouse=True)
    def setup_weights(self):
        self.weights = tests_base.weights
        self.benchmark = tests_base.weighted_reblock_2D
        self.benchmark_opt = tests_base.weighted_reblock_2D_opt

    def test_weight_ndim(self):
        weights_2d = np.vstack((self.weights, self.weights))
        with pytest.raises(RuntimeError):
            pyblock.blocking.reblock(self.data, weights=weights_2d)


class TestBlocking3D:
    @pytest.fixture(autouse=True)
    def setup_data(self):
        self.data = np.empty((3, 3, 3))

    def test_reblock(self):
        with pytest.raises(RuntimeError):
            pyblock.blocking.reblock(self.data)
