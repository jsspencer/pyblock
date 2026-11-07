"""Pandas-based wrapper around :mod:`pyblock.blocking`."""

# copyright: (c) 2014 James Spencer
# license: modified BSD license; see LICENSE for further details.

from __future__ import annotations

import numpy
import pandas as pd
import pyblock.blocking


def reblock(
    data: pd.Series | pd.DataFrame,
    axis: int = 0,
    weights: pd.Series | pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Blocking analysis of correlated data.

    Args:
        data: Data to be blocked. If a Series, a single variable is analysed.
            If a DataFrame, variables are analysed along the given `axis`.
        axis: If non-zero, variables in data are in rows with columns
            corresponding to observations. Otherwise each column is a variable,
            observations are in rows, and blocking is performed down columns.
            Only used if data is a DataFrame.
        weights: A 1D weighting of the data to be reblocked. For multidimensional
            data, an identical weighting is applied to each variable.

    Returns:
        A tuple of `(stats_df, cov_df)`:
            - stats_df: A tidy DataFrame with columns:
                * reblock: Reblock iteration index (0, 1, 2, ...).
                * ndata: Number of data points in this blocking iteration.
                * variable: Label of the variable.
                * mean: Sample mean.
                * std_err: Standard error.
                * std_err_err: Estimated error in the standard error.
                * optimal: Boolean flag indicating if this iteration is optimal.
            - cov_df: A tidy DataFrame with columns:
                * reblock: Reblock iteration index.
                * var_1: First variable label.
                * var_2: Second variable label.
                * cov: Covariance between var_1 and var_2.

    Raises:
        TypeError: If data is not a pandas Series or DataFrame.
        RuntimeError: If multidimensional weights are provided.

    See Also:
        `pyblock.blocking.reblock`: NumPy-based implementation; see for
        documentation and notes on the reblocking procedure.
    """
    if isinstance(data, pd.Series):
        columns = [data.name if data.name is not None else "data"]
        axis = 0
    elif isinstance(data, pd.DataFrame):
        if axis:
            columns = data.index.to_numpy()
        else:
            columns = data.columns.to_numpy()
    else:
        raise TypeError(
            f"data must be a pandas Series or DataFrame, got {type(data).__name__}"
        )

    if weights is not None:
        if isinstance(weights, pd.DataFrame):
            if numpy.min(weights.shape) > 1:
                raise RuntimeError("cannot handle multidimensional weights")
            weights = numpy.array(weights.unstack())
        else:
            weights = weights.to_numpy()

    block_stats = pyblock.blocking.reblock(
        data.to_numpy(), rowvar=axis, weights=weights
    )
    data_size = data.shape[axis]
    optimal_blocks = pyblock.blocking.find_optimal_block(data_size, block_stats)

    nvars = len(columns)
    stats_rows = []
    cov_rows = []

    for stat in block_stats:
        b = stat.block
        n = stat.ndata
        cov_matrix = numpy.array(stat.cov, ndmin=2)

        for ivar, col in enumerate(columns):
            m = stat.mean.item() if nvars == 1 else stat.mean[ivar]
            se = stat.std_err.item() if nvars == 1 else stat.std_err[ivar]
            see = stat.std_err_err.item() if nvars == 1 else stat.std_err_err[ivar]
            is_opt = b == optimal_blocks[ivar]
            stats_rows.append(
                {
                    "reblock": b,
                    "ndata": n,
                    "variable": col,
                    "mean": m,
                    "std_err": se,
                    "std_err_err": see,
                    "optimal": is_opt,
                }
            )

        for i, var_1 in enumerate(columns):
            for j, var_2 in enumerate(columns):
                cov_rows.append(
                    {
                        "reblock": b,
                        "var_1": var_1,
                        "var_2": var_2,
                        "cov": cov_matrix[i, j].item(),
                    }
                )

    stats_df = pd.DataFrame(stats_rows)
    cov_df = pd.DataFrame(cov_rows)

    return stats_df, cov_df


def optimal_block(df: pd.DataFrame | pd.Series) -> int | float:
    """Get the optimal block value from the reblocking data.

    Args:
        df: Tidy reblocking DataFrame as returned by `reblock`, or a subset
            thereof for one or more variables.

    Returns:
        Optimal reblock iteration (or maximum across variables if multiple).
        Returns inf if no optimal block was found.

    Raises:
        TypeError: If df is not a DataFrame or Series.
        ValueError: If no optimal block column is present or if multiple entries
            are marked optimal for the same variable.
    """
    if isinstance(df, pd.Series):
        if "optimal" in df.index or "optimal block" in df.index:
            opt_col = "optimal" if "optimal" in df.index else "optimal block"
            return df["reblock"] if df[opt_col] else float("inf")
        name = str(df.name) if df.name is not None else ""
        if "optimal" not in name and "optimal block" not in name:
            raise ValueError("No optimal block data")
        opts = df[df.astype(bool)].index
        if len(opts) == 0:
            return float("inf")
        if len(opts) == 1:
            return opts[0]
        raise ValueError("Multiple entries listed as optimal.")

    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Expected pandas DataFrame or Series, got {type(df).__name__}")

    opt_col = (
        "optimal"
        if "optimal" in df.columns
        else ("optimal block" if "optimal block" in df.columns else None)
    )
    if opt_col is None:
        raise ValueError("No optimal block column found")

    var_col = "variable" if "variable" in df.columns else None
    groups = df.groupby(var_col, sort=False) if var_col else [(None, df)]

    max_opt: int | float = -1
    for _, group in groups:
        if "reblock" in group.columns:
            opts = group.loc[group[opt_col].astype(bool), "reblock"]
        else:
            opts = group.index[group[opt_col].astype(bool)]
        if len(opts) == 0:
            return float("inf")
        if len(opts) > 1:
            raise ValueError("Multiple entries listed as optimal.")
        opt_val = opts.iloc[0] if hasattr(opts, "iloc") else opts[0]
        max_opt = max(max_opt, opt_val)

    return max_opt


def reblock_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Get the data corresponding to the optimal block from the reblocking data.

    Args:
        df: Tidy reblocking statistics DataFrame as returned by `reblock`.

    Returns:
        DataFrame containing the statistics for each variable at the optimal block size
        (or largest optimal size if multiple variables are given).
        Returns an empty DataFrame if no optimal block was found.

    Raises:
        TypeError: If df is not a pandas DataFrame.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Expected pandas DataFrame, got {type(df).__name__}")
    opt = optimal_block(df)
    if opt < float("inf"):
        if "reblock" in df.columns:
            summary = df[df["reblock"] == opt].copy()
        else:
            summary = df.loc[[opt]].copy()
        for col in ("optimal", "optimal block"):
            if col in summary.columns:
                summary = summary.drop(columns=col)
        return summary.reset_index(drop=True)
    return pd.DataFrame()
