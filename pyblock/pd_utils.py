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
) -> tuple[pd.Series, pd.DataFrame, pd.DataFrame]:
    """Blocking analysis of correlated data.

    Args:
        data: Data to be blocked. See `axis` for order.
        axis: If non-zero, variables in data are in rows with the columns
            corresponding to the observation values. Blocking is then performed along
            the rows. Otherwise each column is a variable, observations are in the
            rows and blocking is performed down the columns. Only used if data is
            a DataFrame.
        weights: A 1D weighting of the data to be reblocked. For multidimensional
            data an identical weighting is applied to the data for each variable.

    Returns:
        A tuple containing:
            - data_len: Number of data points used in each reblocking iteration.
              Note some reblocking iterations discard a data point if there were
              an odd number of data points in the previous iteration.
            - block_info: Mean, standard error and estimated standard error for each
              variable at each reblock step.
            - covariance: Covariance matrix at each reblock step.

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

    iblock = [stat.block for stat in block_stats]
    data_len = pd.Series(
        [stat.ndata for stat in block_stats], index=iblock, name="data length"
    )
    data_len.index.name = "reblock"

    data_dict: dict[tuple, list] = {}
    for ivar, col in enumerate(columns):
        if len(columns) == 1:
            means = [
                (
                    numpy.asarray(s.mean).item()
                    if numpy.asarray(s.mean).ndim == 0
                    else s.mean[0]
                )
                for s in block_stats
            ]
            std_errs = [
                (
                    numpy.asarray(s.std_err).item()
                    if numpy.asarray(s.std_err).ndim == 0
                    else s.std_err[0]
                )
                for s in block_stats
            ]
            std_err_errs = [
                (
                    numpy.asarray(s.std_err_err).item()
                    if numpy.asarray(s.std_err_err).ndim == 0
                    else s.std_err_err[0]
                )
                for s in block_stats
            ]
        else:
            means = [s.mean[ivar] for s in block_stats]
            std_errs = [s.std_err[ivar] for s in block_stats]
            std_err_errs = [s.std_err_err[ivar] for s in block_stats]

        opt_flags = [False] * len(block_stats)
        opt_idx = optimal_blocks[ivar]
        if 0 <= opt_idx < len(block_stats):
            opt_flags[opt_idx] = True

        data_dict[(col, "mean")] = means
        data_dict[(col, "standard error")] = std_errs
        data_dict[(col, "standard error error")] = std_err_errs
        data_dict[(col, "optimal block")] = opt_flags

    multi_keys = pd.MultiIndex.from_tuples(data_dict.keys())
    block_info = pd.DataFrame(data_dict, index=iblock, columns=multi_keys)
    block_info.index.name = "reblock"

    covariance = []
    for stat in block_stats:
        cov = numpy.array(stat.cov, ndmin=2)
        covariance.append(pd.DataFrame(cov, index=columns, columns=columns))
    covariance = pd.concat(covariance, keys=iblock)
    covariance.index.names = ["reblock", ""]

    return (data_len, block_info, covariance)


def optimal_block(block_sub_info: pd.DataFrame | pd.Series) -> int | float:
    """Get the optimal block value from the reblocking data.

    Args:
        block_sub_info: Reblocking data (i.e. the second item of the tuple returned by
            `reblock`), or a subset thereof containing the statistics columns for
            one or more data items.

    Returns:
        Reblocking index corresponding to the reblocking iteration at which serial
        correlation has been removed (as estimated by the procedure in
        `pyblock.blocking.find_optimal_block`). If multiple data sets are passed
        in block_sub_info, this is the maximum index out of all data sets. Set to
        inf if an optimal block is not found for a data set.

    Raises:
        ValueError: If block_sub_info contains no Series or column in DataFrame
            named 'optimal block', or if multiple entries are listed as optimal.
        TypeError: If block_sub_info is neither a Series nor a DataFrame.
    """
    if isinstance(block_sub_info, pd.Series):
        name = block_sub_info.name
        is_opt = False
        if isinstance(name, tuple):
            is_opt = name[-1] == "optimal block"
        elif isinstance(name, str):
            is_opt = "optimal block" in name
        if not is_opt:
            raise ValueError("No optimal block data")
        iterator = [("optimal block", block_sub_info)]
    elif isinstance(block_sub_info, pd.DataFrame):
        level = block_sub_info.columns.nlevels - 1
        opt_cols = [
            col == "optimal block"
            for col in block_sub_info.columns.get_level_values(level)
        ]
        if not any(opt_cols):
            raise ValueError("No optimal block data")
        iterator = block_sub_info.loc[:, opt_cols].items()
    else:
        raise TypeError(
            f"Expected Series or DataFrame, got {type(block_sub_info).__name__}"
        )

    opt: int | float = -1
    for name, col in iterator:
        # Boolean check: works for bool True or non-empty string
        if col.dtype == bool:
            col_opt = col[col].index
        else:
            col_opt = col[col.astype(bool)].index
        if len(col_opt) == 0:
            opt = float("inf")
        elif len(col_opt) == 1:
            opt = max(col_opt[0], opt)
        else:
            raise ValueError("Multiple entries listed as optimal.")

    return opt


def reblock_summary(block_sub_info: pd.DataFrame | pd.Series) -> pd.DataFrame:
    """Get the data corresponding to the optimal block from the reblocking data.

    Args:
        block_sub_info: Reblocking data (i.e. the second item of the tuple returned by
            `reblock`), or a subset thereof containing the statistics columns for
            one or more data items.

    Returns:
        Mean, standard error and estimate of the error in the standard error
        corresponding to the optimal block size in the reblocking data (or largest
        optimal size if multiple data sets are given). The index is labelled with
        the data name, if known. An empty DataFrame is returned if no optimal block
        size was found.
    """
    opt = optimal_block(block_sub_info)
    if opt < float("inf"):
        summary = block_sub_info.loc[opt]
        # Convert to DataFrame, with statistics in columns.
        if summary.index.nlevels == 1:
            # Sadly don't know the data name; leave to user.
            summary = pd.DataFrame(summary).T
        else:
            # Have hierarchical index; can pivot into a DataFrame.
            # Each row will be labelled by the data name.
            summary = summary.unstack()
        summary = summary.drop(columns="optimal block")
    else:
        summary = pd.DataFrame()
    return summary
