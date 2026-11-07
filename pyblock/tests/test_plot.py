from __future__ import annotations

import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")
import matplotlib.figure
import pandas as pd

import pyblock.pd_utils as pd_utils
import pyblock.plot as plot
import pyblock.tests.base as tests_base


def test_plot_reblocking_1d():
    data = pd.Series(tests_base.data_1D)
    reblock_df, cov_df = pd_utils.reblock(data)
    fig = plot.plot_reblocking(reblock_df, plotshow=False)
    assert isinstance(fig, matplotlib.figure.Figure)
    assert len(fig.axes) == 1


def test_plot_reblocking_2d():
    data = pd.DataFrame({1: tests_base.data_2D[0], 2: tests_base.data_2D[1]})
    reblock_df, cov_df = pd_utils.reblock(data)
    fig = plot.plot_reblocking(reblock_df, plotshow=False)
    assert isinstance(fig, matplotlib.figure.Figure)
    assert len(fig.axes) == 2


def test_plot_reblocking_save_file(tmp_path):
    data = pd.Series(tests_base.data_1D)
    reblock_df, cov_df = pd_utils.reblock(data)
    output_file = tmp_path / "plot.png"
    fig = plot.plot_reblocking(reblock_df, plotfile=str(output_file), plotshow=False)
    assert output_file.exists()
    assert output_file.stat().st_size > 0
