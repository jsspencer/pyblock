"""Helper for plotting reblocking plots."""

# copyright: (c) 2015 James Spencer
# license: modified BSD license; see LICENSE for further details.

from __future__ import annotations

from typing import TYPE_CHECKING
import matplotlib.pyplot as plt

if TYPE_CHECKING:
    import matplotlib.figure
    import pandas as pd


def plot_reblocking(
    block_info: pd.DataFrame,
    plotfile: str | None = None,
    plotshow: bool = True,
) -> matplotlib.figure.Figure:
    """Plot the reblocking data.

    Args:
        block_info: Reblocking data (i.e. the second item of the tuple returned by `reblock`).
        plotfile: If not null, save the plot to the given filename. If '-', then show the
            plot interactively.
        plotshow: If `plotfile` is not given or is '-', then show the plot interactively.

    Returns:
        Figure containing the plot of the reblocking data.
    """
    fig = plt.figure()
    data_sets = block_info.columns.get_level_values(0).unique()

    for i, col in enumerate(data_sets):
        ax = fig.add_subplot(len(data_sets), 1, i + 1)

        # There should only be (at most) one non-null value for optimal block.
        opt = block_info[block_info[(col, "optimal block")] != ""].index.values
        if len(opt) > 0:
            opt = opt[0]
        else:
            opt = None

        std_err = block_info[(col, "standard error")]
        if "standard error error" in block_info[col]:
            std_err_err = block_info[(col, "standard error error")]
        else:
            std_err_err = 0 * std_err
        line = ax.errorbar(
            block_info.index, std_err, std_err_err, marker="o", label=col
        )

        if opt is not None:
            ax.annotate(
                "",
                (opt, std_err[opt] - std_err_err[opt]),
                xytext=(0, -20),
                textcoords="offset points",
                arrowprops=dict(
                    arrowstyle="->",
                    linewidth=1.2 * line[0].get_linewidth(),
                ),
            )

        ax.legend(loc="upper left")
        ax.set_ylabel("standard error")
        ax.set_xlabel("Reblock iteration")

    size = fig.get_size_inches()
    fig.set_size_inches(size[0], size[1] * len(data_sets))
    fig.tight_layout()

    if plotfile == "-" or (not plotfile and plotshow):
        plt.show()
    elif plotfile:
        fig.savefig(plotfile)

    return fig
