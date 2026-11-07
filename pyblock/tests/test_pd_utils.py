from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import pyblock
import pyblock.tests.base as tests_base


def assert_pd_stats_equal(reblock_df: pd.DataFrame, cov_df: pd.DataFrame, benchmark):
    expected_blocks = [x[0] for x in benchmark]
    expected_ndata = [x[1] for x in benchmark]

    variables = reblock_df["variable"].unique()

    for ivar, var in enumerate(variables):
        sub = reblock_df[reblock_df["variable"] == var]
        np.testing.assert_array_equal(sub["reblock"].to_numpy(), expected_blocks)
        np.testing.assert_array_equal(sub["ndata"].to_numpy(), expected_ndata)

        if len(variables) == 1:
            bench_mean = [np.asarray(x[2]).item() for x in benchmark]
            bench_se = [np.asarray(x[4]).item() for x in benchmark]
            bench_see = [np.asarray(x[5]).item() for x in benchmark]
        else:
            bench_mean = [x[2][ivar] for x in benchmark]
            bench_se = [x[4][ivar] for x in benchmark]
            bench_see = [x[5][ivar] for x in benchmark]

        np.testing.assert_array_almost_equal(
            sub["mean"].to_numpy(), bench_mean, decimal=8
        )
        np.testing.assert_array_almost_equal(
            sub["std_err"].to_numpy(), bench_se, decimal=8
        )
        np.testing.assert_array_almost_equal(
            sub["std_err_err"].to_numpy(), bench_see, decimal=8
        )

    # Check covariances
    for i, var_1 in enumerate(variables):
        for j, var_2 in enumerate(variables):
            cov_sub = cov_df[(cov_df["var_1"] == var_1) & (cov_df["var_2"] == var_2)]
            if len(variables) == 1:
                bench_cov = [np.asarray(x[3]).item() for x in benchmark]
            else:
                bench_cov = [np.array(x[3], ndmin=2)[i, j] for x in benchmark]
            np.testing.assert_array_almost_equal(
                cov_sub["cov"].to_numpy(), bench_cov, decimal=8
            )


class TestBlocking1D:
    @pytest.fixture(autouse=True)
    def setup_data(self):
        self.data = pd.Series(tests_base.data_1D)

    def test_pdblock(self):
        reblock_df, cov_df = pyblock.pd_utils.reblock(self.data)
        assert_pd_stats_equal(reblock_df, cov_df, tests_base.reblock_1D)
        assert reblock_df["optimal"].dtype == bool
        assert reblock_df["variable"].iloc[0] == "data"
        assert set(reblock_df.columns) == {
            "reblock",
            "ndata",
            "variable",
            "mean",
            "std_err",
            "std_err_err",
            "optimal",
        }
        assert set(cov_df.columns) == {"reblock", "var_1", "var_2", "cov"}

    def test_pdblock_df(self):
        reblock_df, cov_df = pyblock.pd_utils.reblock(pd.DataFrame(self.data))
        assert_pd_stats_equal(reblock_df, cov_df, tests_base.reblock_1D)
        assert reblock_df["optimal"].dtype == bool
        assert reblock_df["variable"].iloc[0] == 0

    def test_pdblock_opt(self):
        reblock_df, cov_df = pyblock.pd_utils.reblock(self.data)
        assert (
            pyblock.pd_utils.optimal_block(reblock_df) == tests_base.reblock_1D_opt[0]
        )
        # Simulate multiple optimal
        reblock_df.loc[tests_base.reblock_1D_opt[0] - 1, "optimal"] = True
        with pytest.raises(ValueError):
            pyblock.pd_utils.optimal_block(reblock_df)
        reblock_df["optimal"] = False
        assert pyblock.pd_utils.optimal_block(reblock_df) == float("inf")
        assert pyblock.pd_utils.reblock_summary(reblock_df).empty

    def test_weighted_pdblock(self):
        weights = pd.Series(tests_base.weights)
        reblock_df, cov_df = pyblock.pd_utils.reblock(self.data, weights=weights)
        assert_pd_stats_equal(reblock_df, cov_df, tests_base.weighted_reblock_1D)

    def test_weighted_df_pdblock(self):
        weights = pd.DataFrame({1: tests_base.weights})
        reblock_df, cov_df = pyblock.pd_utils.reblock(self.data, weights=weights)
        assert_pd_stats_equal(reblock_df, cov_df, tests_base.weighted_reblock_1D)


class TestBlocking2D:
    @pytest.fixture(autouse=True)
    def setup_data(self):
        self.data = pd.DataFrame({1: tests_base.data_2D[0], 2: tests_base.data_2D[1]})

    def test_pdblock(self):
        reblock_df, cov_df = pyblock.pd_utils.reblock(self.data)
        assert_pd_stats_equal(reblock_df, cov_df, tests_base.reblock_2D)

    def test_pdblock_rows(self):
        data = self.data.T
        reblock_df, cov_df = pyblock.pd_utils.reblock(data, axis=1)
        assert_pd_stats_equal(reblock_df, cov_df, tests_base.reblock_2D)

    def test_pdblock_opt(self):
        reblock_df, cov_df = pyblock.pd_utils.reblock(self.data)
        assert pyblock.pd_utils.optimal_block(reblock_df) == max(
            tests_base.reblock_2D_opt
        )
        for var in [1, 2]:
            sub = reblock_df[reblock_df["variable"] == var]
            assert (
                pyblock.pd_utils.optimal_block(sub)
                == tests_base.reblock_2D_opt[var - 1]
            )

    def test_weighted_pdblock(self):
        weights = pd.Series(tests_base.weights)
        reblock_df, cov_df = pyblock.pd_utils.reblock(self.data, weights=weights)
        assert_pd_stats_equal(reblock_df, cov_df, tests_base.weighted_reblock_2D)

    def test_2d_weighted_pdblock(self):
        weights = pd.DataFrame({1: tests_base.weights, 2: tests_base.weights})
        with pytest.raises(RuntimeError):
            pyblock.pd_utils.reblock(self.data, weights=weights)

    def test_pdblock_summary(self):
        reblock_df, cov_df = pyblock.pd_utils.reblock(self.data)
        summary = pyblock.pd_utils.reblock_summary(reblock_df)
        opt_step = max(tests_base.reblock_2D_opt)
        opt = tests_base.reblock_2D[opt_step]
        expected_means = opt[2]
        expected_se = opt[4]
        expected_see = opt[5]

        np.testing.assert_array_almost_equal(
            summary["mean"].to_numpy(), expected_means, decimal=8
        )
        np.testing.assert_array_almost_equal(
            summary["std_err"].to_numpy(), expected_se, decimal=8
        )
        np.testing.assert_array_almost_equal(
            summary["std_err_err"].to_numpy(), expected_see, decimal=8
        )

        # Single variable subset summary
        sub = reblock_df[reblock_df["variable"] == 1]
        summary_1 = pyblock.pd_utils.reblock_summary(sub)
        opt_1 = tests_base.reblock_2D[tests_base.reblock_2D_opt[0]]
        assert summary_1["reblock"].iloc[0] == tests_base.reblock_2D_opt[0]
        assert summary_1["mean"].iloc[0] == pytest.approx(opt_1[2][0], abs=1e-8)


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

        with pytest.raises(TypeError):
            pyblock.pd_utils.reblock_summary("invalid_data")

    def test_optimal_block_typed_series(self):
        s = pd.Series([False, False, True], name="optimal")
        assert pyblock.pd_utils.optimal_block(s) == 2
