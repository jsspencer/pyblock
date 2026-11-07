from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import pyblock


@pytest.fixture
def error_dataset():
    rand = np.random.RandomState(seed=5)
    ndata = 2**5
    data = pd.DataFrame({1: rand.randn(ndata), 2: rand.randn(ndata)})
    reblock_df, cov_df = pyblock.pd_utils.reblock(data)
    reblock_1 = reblock_df[reblock_df["variable"] == 1].reset_index(drop=True)
    reblock_2 = reblock_df[reblock_df["variable"] == 2].reset_index(drop=True)
    cov_12 = cov_df[(cov_df["var_1"] == 1) & (cov_df["var_2"] == 2)]["cov"].reset_index(
        drop=True
    )
    data_len = reblock_1["ndata"]

    expected_ratio = pd.DataFrame(
        {
            "std_err": {
                0: 0.66458146620536607,
                1: 0.75702195760340563,
                2: 0.71587945049229107,
                3: 0.68021031918966046,
                4: 0.16281610758487003,
            },
            "optimal": {0: False, 1: False, 2: False, 3: True, 4: False},
            "mean": {
                0: -0.086689548180912318,
                1: -0.086689548180912304,
                2: -0.086689548180912318,
                3: -0.08668954818091229,
                4: -0.08668954818091229,
            },
        }
    )
    return reblock_1, reblock_2, cov_12, data_len, expected_ratio


class TestError:
    def test_ratio(self, error_dataset):
        reblock_1, reblock_2, cov_12, data_len, expected_ratio = error_dataset
        ratio = pyblock.error.ratio(reblock_1, reblock_2, cov_12, data_len)
        test = ratio[["mean", "std_err"]]
        benchmark = expected_ratio[["mean", "std_err"]]
        np.testing.assert_array_almost_equal(benchmark, test, decimal=8)
        assert ratio.loc[ratio["optimal"]].index[0] == 3

    def test_optimal(self, error_dataset):
        reblock_1, reblock_2, cov_12, data_len, _ = error_dataset
        r2 = reblock_2.copy()
        r2["optimal"] = False
        r2.loc[1, "optimal"] = True
        ratio = pyblock.error.ratio(reblock_1, r2, cov_12, data_len)
        assert ratio.loc[ratio["optimal"]].index[0] == 3

    def test_ratio_single(self, error_dataset):
        reblock_1, reblock_2, cov_12, data_len, expected_ratio = error_dataset
        ratio = pyblock.error.ratio(
            reblock_1.loc[4],
            reblock_2.loc[4],
            cov_12[4],
            data_len[4],
        )
        for key in ("mean", "std_err"):
            assert ratio[key] == pytest.approx(expected_ratio.iloc[4][key], abs=1e-8)

    def test_product_single(self, error_dataset):
        reblock_1, reblock_2, cov_12, data_len, _ = error_dataset
        product = pyblock.error.product(
            reblock_1.iloc[4],
            reblock_2.iloc[4],
            cov_12[4],
            data_len[4],
        )
        benchmark = pd.Series({"mean": -0.00561329, "std_err": 0.01576336})
        for key in ("mean", "std_err"):
            assert product[key] == pytest.approx(benchmark[key], abs=1e-8)

    def test_subtraction_single(self, error_dataset):
        reblock_1, reblock_2, cov_12, data_len, _ = error_dataset
        subtraction = pyblock.error.subtraction(
            reblock_1.iloc[4],
            reblock_2.iloc[4],
            cov_12[4],
            data_len[4],
        )
        benchmark = pd.Series({"mean": -0.27652270, "std_err": 0.17002344})
        for key in ("mean", "std_err"):
            assert subtraction[key] == pytest.approx(benchmark[key], abs=1e-8)

    def test_addition_single(self, error_dataset):
        reblock_1, reblock_2, cov_12, data_len, _ = error_dataset
        addition = pyblock.error.addition(
            reblock_1.iloc[4],
            reblock_2.iloc[4],
            cov_12[4],
            data_len[4],
        )
        benchmark = pd.Series({"mean": 0.23240407, "std_err": 0.06664526})
        for key in ("mean", "std_err"):
            assert addition[key] == pytest.approx(benchmark[key], abs=1e-8)


@pytest.mark.parametrize(
    "val,err,expected",
    [
        (1.2345, 0.01, "1.23(1)"),
        (12331, 40, "12330(40)"),
        (float("inf"), 100, str(float("inf"))),
        (100.01, float("inf"), f"100.01({float('inf')})"),
        (100.01, 0.0, "100.01(0)"),
    ],
)
def test_pretty_fmt_err(val, err, expected):
    assert pyblock.error.pretty_fmt_err(val, err) == expected
