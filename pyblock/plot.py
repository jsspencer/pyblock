"""Helper for plotting reblocking plots."""

# copyright: (c) 2015 James Spencer
# license: modified BSD license; see LICENSE for further details.

from __future__ import annotations

from typing import TYPE_CHECKING
import numpy
import matplotlib.pyplot as plt

if TYPE_CHECKING:
    import matplotlib.figure
    import pandas as pd


def plot_reblocking(
    stats_df: pd.DataFrame,
    plotfile: str | None = None,
    plotshow: bool = True,
) -> matplotlib.figure.Figure:
    """Plot the reblocking data.

    Args:
        stats_df: Tidy reblocking DataFrame as returned by `pyblock.pd_utils.reblock`.
        plotfile: If not null, save the plot to the given filename. If '-', then show the
            plot interactively.
        plotshow: If `plotfile` is not given or is '-', then show the plot interactively.

    Returns:
        Figure containing the plot of the reblocking data.
    """
    fig = plt.figure()

    if "variable" in stats_df.columns:
        groups = [(var, grp) for var, grp in stats_df.groupby("variable", sort=False)]
    else:
        groups = [(None, stats_df)]

    num_plots = len(groups)

    for i, (col, group) in enumerate(groups):
        ax = fig.add_subplot(num_plots, 1, i + 1)

        x = group["reblock"] if "reblock" in group.columns else group.index
        std_err = group["std_err"]
        std_err_err = (
            group["std_err_err"]
            if "std_err_err" in group.columns
            else numpy.zeros_like(std_err)
        )

        label = str(col) if col is not None else "data"
        line = ax.errorbar(x, std_err, std_err_err, marker="o", label=label)

        opt_col = "optimal" if "optimal" in group.columns else "optimal block"
        if opt_col in group.columns:
            opt_rows = group[group[opt_col].astype(bool)]
            if not opt_rows.empty:
                opt_x = (
                    opt_rows["reblock"].iloc[0]
                    if "reblock" in opt_rows.columns
                    else opt_rows.index[0]
                )
                opt_y = opt_rows["std_err"].iloc[0]
                opt_err = (
                    opt_rows["std_err_err"].iloc[0]
                    if "std_err_err" in opt_rows.columns
                    else 0.0
                )
                ax.annotate(
                    "",
                    (opt_x, opt_y - opt_err),
                    xytext=(0, -20),
                    textcoords="offset points",
                    arrowprops=dict(
                        arrowstyle="->",
                        linewidth=1.2 * line[0].get_linewidth(),
                    ),
                )

        ax.legend(loc="upper left")
        ax.set_ylabel("Standard Error")
        ax.set_xlabel("Reblock iteration")

    size = fig.get_size_inches()
    fig.set_size_inches(size[0], size[1] * num_plots)
    fig.tight_layout()

    if plotfile == "-" or (not plotfile and plotshow):
        plt.show()
    elif plotfile:
        fig.savefig(plotfile)

    return fig
