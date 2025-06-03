from datapoke.enums import CoerceTypes
from datapoke.utils import force_boolcast

import numpy as np
import pandas as pd


import functools
from typing import Callable, Tuple, Union, Hashable


class PokeColumn:
    """
    A column wrapper for a pandas dataframe providing data quality profiling and safe coercion.

    This class is designed to support lightweight inspection and cleaning of individual
    columns in a DataFrame. It exposes common diagnostics (nulls, uniqueness, data types),
    along with a summary view and tools to attempt safe type coercion with error tracking.

    Parameters
    ----------
    df : pd.DataFrame
        The pandas dataframe containing the column to wrap and analyse.

    colname: Hashable
        The column name in df.

    Attributes
    ----------
    series : pd.Series
        The column within df as a series.
    dtypes : pd.Series
        Count of unique Python-level types in the column.
    ntypes : int
        Number of unique Python types in the column.
    nullcount : int
        Number of null values in the column.
    uniquevalues : int
        Number of unique values in the column.

    Methods
    -------
    summary(detailed=True, display=True)
        Returns a summary of column quality information.
    coerce_dtypes(target_dtype, copy=False)
        Attempts to coerce the column to a new type or callable function safely.
    """

    # TODO: write a __repr__ method
    # TODO: optimise duplicate function calls
    # TODO: add sampling
    # TODO: consider caching

    def __init__(self, df: pd.DataFrame, colname: Hashable):
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"Expected a pandas dataframe, but got {type(df).__name__}")
        if colname not in df.columns:
            raise KeyError(f"{colname} was not found in dataframe columns")

        self.df = df
        self.colname = colname

    # region Properties ------------------------

    @property
    def series(self) -> pd.Series:
        if self.colname not in self.df.columns:
            raise RuntimeError(f"Column '{self.colname}' no longer exists.")
        return self.df[self.colname]

    @property
    def dtypes(self):
        """Count of occurances of each pandas dtype in column, as a series."""
        return self.series.dropna().map(type).value_counts()

    @property
    def ntypes(self):
        """Count of distinct pandas dtypes in column"""
        return len(self.dtypes)

    @property
    def nullcount(self):
        """Count of nulls in column."""
        return self.series.isna().sum()

    @property
    def uniquevalues(self):
        """Count of unique values in column."""
        return len(self.series.drop_duplicates())

    # endregion properties

    # region Setters -------------------------

    @series.setter
    def series(self, new_series: pd.Series):
        if not isinstance(new_series, pd.Series):
            raise TypeError("Assigned value must be a pandas Series.")
        if len(new_series) != len(self.df):
            raise ValueError("Length of new Series does not match DataFrame.")
        self.df[self.colname] = new_series

    # endregion setters

    # region Public methods ------------------

    def summary(
        self, detailed: bool = True, display: bool = True
    ) -> Union[bool, pd.Series]:
        """
        Return a summary of data quality for the column.

        Parameters
        ----------
        detailed : bool, optional
            Whether to include detailed type breakdown.
        display : bool, optional
            If True, return a series that will print nicely, otherwise return a dict.

        Returns
        -------
        summary: dict or pd.Series
            Column metadata summary.
        """

        output = {
            "name": self.colname,
            "n_nulls": self.nullcount,
            "unique_val": self.uniquevalues,
            "most_freq_vals": self.series.value_counts().iloc[:5].index.tolist(),
            "dtypes": {t.__name__ for t in self.dtypes.index},
            "n_types": self.ntypes,
        }
        if detailed:
            output.update(
                {"n_" + key.__name__: value for key, value in self.dtypes.items()}
            )
        if display:
            return pd.Series(output)
        else:
            return output

    def coerce_dtypes(
        self,
        target_dtype: Union[CoerceTypes, Callable[[object], object]],
        copy: bool = False,
        aggresive_bools: bool = False,
    ) -> Tuple[pd.Series, pd.Index, list]:
        """
        Attempt to coerce the column to a specified data type, handling failures safely.

        Parameters
        ----------
        target_dtype : str or Callable
            The target type to convert to. Must be one of the following strings:
            - 'num' for numeric conversion via `pd.to_numeric`
            - 'datetime' for datetime conversion via `pd.to_datetime`
            - 'bool' for pandas' nullable boolean type
            - 'string' for pandas' native string type

            Alternatively, a custom function may be passed that accepts a single argument
            and returns a converted value. Any errors during conversion will be replaced with NaN.
        copy : bool, optional
            If True, return a new coerced Series. If False, update the column in place.
            Defaults to False.

        Returns
        -------
        coerced : pd.Series
            The Series after attempting type coercion. Failed conversions will be NaN.
        failed_rows : pd.Index
            Index of rows where coercion failed (original value was non-null, result is NaN).
        top_errors : list
            A list of up to 5 most common values that failed conversion.

        Raises
        ------
        ValueError
            If the `target_dtype` is not supported or coercion fails due to invalid data.

        Notes
        -----
        This function does not distinguish between float and int coercion at this stage.
        All numeric conversions default to float for compatibility with missing values.

        If passing a function as target_dtype, application is via pd.Series.map().
        """

        # TODO: add handling for ints vs. floats
        # TODO: add handling for unusual pandas categories
        # TODO: optimise failed memory usage a bit more

        if isinstance(target_dtype, Callable):
            coerced = self.series.map(self._nullonfailure_wrapper(target_dtype))
        else:
            converter = {
                "num": pd.to_numeric,
                "datetime": pd.to_datetime,
                "bool": lambda x: x.astype("boolean"),
                "string": lambda x: x.astype("string"),
            }.get(target_dtype)

            if converter is None:
                raise ValueError(
                    f"Unsupported type: {target_dtype}. Currently supported types are: num, datetime, bool, string."
                )

            try:
                if target_dtype in ["num", "datetime"]:
                    coerced = converter(self.series, errors="coerce")
                elif target_dtype == "boolean" and aggresive_bools:
                    coerced = self.series.map(force_boolcast).astype("boolean")
                else:
                    coerced = converter(self.series)

            except Exception as e:
                raise ValueError(
                    f"Failed to coerce column '{self.name}' to {target_dtype}: {e}"
                )

        failed = self.series[coerced.isna() & self.series.notna()]
        # need to calculate and return top errors before overwriting in place if copy = False:
        top_errors = failed.value_counts().iloc[:5].index.to_list()
        if not copy:
            self.series = coerced
        return coerced, failed.index, top_errors

    # endregion public methods

    # region Private methods ----------------

    @staticmethod
    def _nullonfailure_wrapper(
        func: Callable[[object], [object]],
    ) -> Callable[[object], [object]]:
        """Wraps a user defined function to return np.nan if there is an error.

        Parameters
        ----------
        func: Callable
            Any function that accepts and returns a single argument.

        Returns
        -------
        wrapper: Callable
            Wrapped version of func.
        """

        @functools.wraps(func)
        def wrapper(x):
            try:
                return func(x)
            except Exception:
                return np.nan

        return wrapper

    # endregion private methods
