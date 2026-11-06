from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import pyblock
import pyblock.tests.base as tests_base


def assert_pd_stats_equal(data_len, reblock, cov, benchmark):
    bench_slice = np.array([x[0] for x in benchmark])
    np.testing.assert_array_equal(data_len.index.to_numpy(), bench_slice)

    bench_slice = np.array([x[1] for x in benchmark])
    np.testing.assert_array_equal(data_len.to_numpy(), bench_slice)

    for ind, key in [
        (2, "mean"),
        (4, "standard error"),
        (5, "standard error error"),
    ]:
        bench_slice = np.array([x[ind] for x in benchmark]).flatten()
        np.testing.assert_array_almost_equal(
            reblock.xs(key, axis=1, level=1).to_numpy().flatten(),
            bench_slice,
            decimal=8,
        )

    bench_slice = np.array([x[3] for x in benchmark]).flatten()
    np.testing.assert_array_almost_equal(
        cov.to_numpy().flatten(), bench_slice, decimal=8
    )


class TestBlocking1D:
    @pytest.fixture(autouse=True)
    def setup_data(self):
        self.data = pd.Series(tests_base.data_1D)

    def test_pdblock(self):
        data_len, reblock, cov = pyblock.pd_utils.reblock(self.data)
        assert_pd_stats_equal(data_len, reblock, cov, tests_base.reblock_1D)
        assert reblock[("data", "optimal block")].dtype == bool

    def test_pdblock_df(self):
        data_len, reblock, cov = pyblock.pd_utils.reblock(pd.DataFrame(self.data))
        assert_pd_stats_equal(data_len, reblock, cov, tests_base.reblock_1D)
        assert reblock[(0, "optimal block")].dtype == bool

    def test_pdblock_opt(self):
        data_len, reblock, cov = pyblock.pd_utils.reblock(self.data)
        assert pyblock.pd_utils.optimal_block(reblock) == tests_base.reblock_1D_opt[0]
        reblock.loc[tests_base.reblock_1D_opt[0] - 1, ("data", "optimal block")] = True
        with pytest.raises(ValueError):
            pyblock.pd_utils.optimal_block(reblock)
        reblock[("data", "optimal block")] = False
        assert pyblock.pd_utils.optimal_block(reblock) == float("inf")
        assert pyblock.pd_utils.reblock_summary(reblock).empty

    def test_weighted_pdblock(self):
        weights = pd.Series(tests_base.weights)
        data_len, reblock, cov = pyblock.pd_utils.reblock(self.data, weights=weights)
        assert_pd_stats_equal(data_len, reblock, cov, tests_base.weighted_reblock_1D)

    def test_weighted_df_pdblock(self):
        weights = pd.DataFrame({1: tests_base.weights})
        data_len, reblock, cov = pyblock.pd_utils.reblock(self.data, weights=weights)
        assert_pd_stats_equal(data_len, reblock, cov, tests_base.weighted_reblock_1D)


class TestBlocking2D:
    @pytest.fixture(autouse=True)
    def setup_data(self):
        self.data = pd.DataFrame({1: tests_base.data_2D[0], 2: tests_base.data_2D[1]})

    def test_pdblock(self):
        data_len, reblock, cov = pyblock.pd_utils.reblock(self.data)
        assert_pd_stats_equal(data_len, reblock, cov, tests_base.reblock_2D)

    def test_pdblock_rows(self):
        data = self.data.T
        data_len, reblock, cov = pyblock.pd_utils.reblock(data, axis=1)
        assert_pd_stats_equal(data_len, reblock, cov, tests_base.reblock_2D)

    def test_pdblock_opt(self):
        data_len, reblock, cov = pyblock.pd_utils.reblock(self.data)
        assert pyblock.pd_utils.optimal_block(reblock) == max(tests_base.reblock_2D_opt)
        for name, group in reblock.T.groupby(level=0):
            assert (
                pyblock.pd_utils.optimal_block(group.T)
                == tests_base.reblock_2D_opt[name - 1]
            )

    def test_weighted_pdblock(self):
        weights = pd.Series(tests_base.weights)
        data_len, reblock, cov = pyblock.pd_utils.reblock(self.data, weights=weights)
        assert_pd_stats_equal(data_len, reblock, cov, tests_base.weighted_reblock_2D)

    def test_2d_weighted_pdblock(self):
        weights = pd.DataFrame({1: tests_base.weights, 2: tests_base.weights})
        with pytest.raises(RuntimeError):
            pyblock.pd_utils.reblock(self.data, weights=weights)

    def test_pdblock_summary(self):
        data_len, reblock, cov = pyblock.pd_utils.reblock(self.data)
        summary = pyblock.pd_utils.reblock_summary(reblock)
        opt = tests_base.reblock_2D[max(tests_base.reblock_2D_opt)]
        opt = np.array([opt[2], opt[4], opt[5]]).transpose()
        np.testing.assert_array_almost_equal(
            summary.to_numpy().flatten(), opt.flatten(), decimal=8
        )
        summary = pyblock.pd_utils.reblock_summary(reblock[1])
        np.testing.assert_array_almost_equal(
            summary.to_numpy().flatten(), opt[0], decimal=8
        )


class TestOptimalError:
    def test_series_without_optimal_block_raises(self):
        data = pd.Series(np.zeros(5), name="rand")
        with pytest.raises(ValueError):
            pyblock.pd_utils.optimal_block(data)

    def test_dataframe_without_optimal_block_raises(self):
        data = pd.DataFrame({1: np.zeros(5), 2: np.zeros(5)})
        with pytest.raises(ValueError):
            pyblock.pd_utils.optimal_block(data)

    def test_invalid_type_raises(self):
        with pytest.raises(TypeError):
            pyblock.pd_utils.optimal_block([1, 2, 3])

        with pytest.raises(TypeError):
            pyblock.pd_utils.reblock("invalid_data")

    def test_optimal_block_typed_series(self):
        s = pd.Series([False, False, True], name=("var", "optimal block"))
        assert pyblock.pd_utils.optimal_block(s) == 2
