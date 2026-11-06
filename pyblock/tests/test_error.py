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
    data_len, reblock, cov_12 = pyblock.pd_utils.reblock(data)
    cov_12 = cov_12.xs(1, level=1)[2]
    ratio = pd.DataFrame(
        {
            "standard error": {
                0: 0.66458146620536607,
                1: 0.75702195760340563,
                2: 0.71587945049229107,
                3: 0.68021031918966046,
                4: 0.16281610758487003,
            },
            "optimal block": {0: False, 1: False, 2: False, 3: True, 4: False},
            "mean": {
                0: -0.086689548180912318,
                1: -0.086689548180912304,
                2: -0.086689548180912318,
                3: -0.08668954818091229,
                4: -0.08668954818091229,
            },
        }
    )
    ratio.index.name = "reblock"
    return data_len, reblock, cov_12, ratio


class TestError:
    def test_ratio(self, error_dataset):
        data_len, reblock, cov_12, expected_ratio = error_dataset
        ratio = pyblock.error.ratio(reblock[1], reblock[2], cov_12, data_len)
        # Compare numeric data in the same ordering
        test = ratio.drop("optimal block", axis=1).sort_index(axis=1)
        benchmark = expected_ratio.drop("optimal block", axis=1).sort_index(axis=1)
        np.testing.assert_array_almost_equal(benchmark, test, decimal=8)
        assert ratio[ratio["optimal block"]].index[0] == 3

    def test_optimal(self, error_dataset):
        data_len, reblock, cov_12, _ = error_dataset
        reblock[(2, "optimal block")] = False
        reblock.loc[1, (2, "optimal block")] = True
        ratio = pyblock.error.ratio(reblock[1], reblock[2], cov_12, data_len)
        assert ratio[ratio["optimal block"]].index[0] == 3

    def test_ratio_single(self, error_dataset):
        data_len, reblock, cov_12, expected_ratio = error_dataset
        ratio = pyblock.error.ratio(
            reblock[1].loc[4],
            reblock[2].loc[4],
            cov_12[4],
            data_len[4],
        )
        for key in ("mean", "standard error"):
            assert ratio[key] == pytest.approx(expected_ratio.iloc[4][key], abs=1e-8)

    def test_product_single(self, error_dataset):
        data_len, reblock, cov_12, _ = error_dataset
        product = pyblock.error.product(
            reblock[1].iloc[4],
            reblock[2].iloc[4],
            cov_12[4],
            data_len[4],
        )
        benchmark = pd.Series({"mean": -0.00561329, "standard error": 0.01576336})
        for key in ("mean", "standard error"):
            assert product[key] == pytest.approx(benchmark[key], abs=1e-8)

    def test_subtraction_single(self, error_dataset):
        data_len, reblock, cov_12, _ = error_dataset
        subtraction = pyblock.error.subtraction(
            reblock[1].iloc[4],
            reblock[2].iloc[4],
            cov_12[4],
            data_len[4],
        )
        benchmark = pd.Series({"mean": -0.27652270, "standard error": 0.17002344})
        for key in ("mean", "standard error"):
            assert subtraction[key] == pytest.approx(benchmark[key], abs=1e-8)

    def test_addition_single(self, error_dataset):
        data_len, reblock, cov_12, _ = error_dataset
        addition = pyblock.error.addition(
            reblock[1].iloc[4],
            reblock[2].iloc[4],
            cov_12[4],
            data_len[4],
        )
        benchmark = pd.Series({"mean": 0.23240407, "standard error": 0.06664526})
        for key in ("mean", "standard error"):
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
