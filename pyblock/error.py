"""Simple error propagation.

.. note::

    We only implement the functions as we need them...
"""

# copyright: (c) 2014 James Spencer
# license: modified BSD license; see LICENSE for further details.

from __future__ import annotations

import numpy
import pandas as pd
import pyblock.pd_utils as pd_utils


def ratio(
    stats_A: pd.Series | pd.DataFrame,
    stats_B: pd.Series | pd.DataFrame,
    cov_AB: float | pd.Series,
    data_len: int | pd.Series,
) -> pd.Series | pd.DataFrame:
    r"""Calculate the mean and standard error of :math:`f(A,B) = A/B`.

    Args:
        stats_A: Statistics (containing at least the 'mean' and 'standard error' fields)
            for variable :math:`A`. The rows contain different values of these statistics
            (e.g. from a reblocking analysis) if a DataFrame is passed.
        stats_B: Similarly for variable :math:`B`.
        cov_AB: Covariance between variables :math:`A` and :math:`B`. If `stats_A` and
            `stats_B` are DataFrames, then this must be a Series with the same index.
        data_len: Number of data points ('observations') used to obtain the statistics
            given in `stats_A` and `stats_B`. If `stats_A` and `stats_B` are DataFrames,
            then this must be a Series with the same index.

    Returns:
        Mean and standard error (and, if possible/relevant, optimal reblock
        iteration) for :math:`f(A,B)`. If `stats_A` and `stats_B` are DataFrames,
        this is a DataFrame with the same index; otherwise a Series is returned.
    """
    m_A, m_B = (stats_A["mean"], stats_B["mean"])
    mean = m_A / m_B
    se_A, se_B = (stats_A["standard error"], stats_B["standard error"])
    std_err = abs(
        mean
        * numpy.sqrt(
            (se_A / m_A) ** 2 + (se_B / m_B) ** 2 - 2 * cov_AB / (data_len * m_A * m_B)
        )
    )
    return _stats_summary(mean, std_err, stats_A, stats_B)


def product(
    stats_A: pd.Series | pd.DataFrame,
    stats_B: pd.Series | pd.DataFrame,
    cov_AB: float | pd.Series,
    data_len: int | pd.Series,
) -> pd.Series | pd.DataFrame:
    r"""Calculate the mean and standard error of :math:`f(A,B) = A \times B`.

    Args:
        stats_A: See `ratio`.
        stats_B: See `ratio`.
        cov_AB: See `ratio`.
        data_len: See `ratio`.

    Returns:
        See `ratio`.
    """
    m_A, m_B = (stats_A["mean"], stats_B["mean"])
    mean = m_A * m_B
    se_A, se_B = (stats_A["standard error"], stats_B["standard error"])
    std_err = abs(
        mean
        * numpy.sqrt(
            (se_A / m_A) ** 2 + (se_B / m_B) ** 2 + 2 * cov_AB / (data_len * m_A * m_B)
        )
    )
    return _stats_summary(mean, std_err, stats_A, stats_B)


def subtraction(
    stats_A: pd.Series | pd.DataFrame,
    stats_B: pd.Series | pd.DataFrame,
    cov_AB: float | pd.Series,
    data_len: int | pd.Series,
) -> pd.Series | pd.DataFrame:
    r"""Calculate the mean and standard error of :math:`f(A,B) = A - B`.

    Args:
        stats_A: See `ratio`.
        stats_B: See `ratio`.
        cov_AB: See `ratio`.
        data_len: See `ratio`.

    Returns:
        See `ratio`.
    """
    mean = stats_A["mean"] - stats_B["mean"]
    se = "standard error"
    std_err = stats_A[se] ** 2 + stats_B[se] ** 2 - 2 * cov_AB / data_len
    std_err = abs(numpy.sqrt(std_err))
    return _stats_summary(mean, std_err, stats_A, stats_B)


def addition(
    stats_A: pd.Series | pd.DataFrame,
    stats_B: pd.Series | pd.DataFrame,
    cov_AB: float | pd.Series,
    data_len: int | pd.Series,
) -> pd.Series | pd.DataFrame:
    r"""Calculate the mean and standard error of :math:`f(A,B) = A + B`.

    Args:
        stats_A: See `ratio`.
        stats_B: See `ratio`.
        cov_AB: See `ratio`.
        data_len: See `ratio`.

    Returns:
        See `ratio`.
    """
    mean = stats_A["mean"] + stats_B["mean"]
    se = "standard error"
    std_err = stats_A[se] ** 2 + stats_B[se] ** 2 + 2 * cov_AB / data_len
    std_err = abs(numpy.sqrt(std_err))
    return _stats_summary(mean, std_err, stats_A, stats_B)


def _stats_summary(
    mean: float | pd.Series,
    std_err: float | pd.Series,
    stats_A: pd.Series | pd.DataFrame,
    stats_B: pd.Series | pd.DataFrame,
) -> pd.Series | pd.DataFrame:
    r"""Summarise error propagation.

    Args:
        mean: Mean of :math:`f(A,B)`.
        std_err: Standard error of :math:`f(A,B)`.
        stats_A: Statistics for variable A.
        stats_B: Statistics for variable B.

    Returns:
        DataFrame or Series containing summarised statistics.
    """
    stats_dict = {"mean": mean, "standard error": std_err}
    if isinstance(mean, pd.Series):
        stats = pd.DataFrame(stats_dict)
        if isinstance(stats_A, pd.DataFrame) and isinstance(stats_B, pd.DataFrame):
            if (
                "optimal block" in stats_A.columns
                and "optimal block" in stats_B.columns
            ):
                opt_A = pd_utils.optimal_block(stats_A)
                opt_B = pd_utils.optimal_block(stats_B)
                if opt_A > opt_B:
                    stats["optimal block"] = stats_A["optimal block"]
                else:
                    stats["optimal block"] = stats_B["optimal block"]
    else:
        stats = pd.Series(stats_dict)
    return stats


def _calc_sig_figs(x: float) -> int:
    """Calculate the number of significant figures of decimal accuracy."""
    return -int(numpy.floor(numpy.log10(x)))


def pretty_fmt_err(val: float | int, err: float | int) -> str:
    """Pretty formatting of a value and associated error.

    Args:
        val: A (noisy) value.
        err: Error associated with the value.

    Returns:
        Value to the number of significant digits known, with the error in the
        last digit in brackets.

    Examples:
        >>> pretty_fmt_err(1.2345, 0.01)
        '1.23(1)'
        >>> pretty_fmt_err(12331, 40)
        '12330(40)'

    Notes:
        Rounding is handled with Python's `round` function, which handles rounding
        numbers at the midpoint in a range (e.g. 5 if round to the nearest 10) in
        a slightly odd way. As we're normally dealing with noisy data and rounding
        to remove more than just one significant figure, this is unlikely to
        impact us.
    """
    if numpy.isinf(val):
        return f"{val}"
    if numpy.isinf(err):
        return f"{val}({err})"
    if abs(err) < numpy.finfo(float(err)).eps:
        return f"{val}(0)"

    sig_err = round(err, _calc_sig_figs(err))
    sig_err_figs = _calc_sig_figs(sig_err)
    sig_val = round(val, sig_err_figs)

    if sig_err > 1:
        return f"{int(sig_val)}({int(round(sig_err))})"
    else:
        err_val = int(round(sig_err * 10**sig_err_figs))
        return f"{sig_val:.{sig_err_figs}f}({err_val})"
