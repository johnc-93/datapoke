from datapoke.series import PokeColumn


import pandas as pd


import functools
import logging
import os
import warnings
from datapoke.enums import CoerceTypes
from typing import Any, Callable, Literal, Tuple, Union




class PokeFrame:
    """
    A wrapper around a pandas DataFrame that provides lightweight data profiling,
    safe coercion, and structural integrity checks.

    PokeFrame is designed to support early data ingestion and quality assurance
    workflows. It allows summary inspection of all columns, attempts schema coercion
    with error tracking and quarantining, and provides integrity validation for
    downstream operations.

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe to wrap and inspect.
    safe_mode : bool, optional
        If True (default), tracks the index at load time to validate changes and
        raise warnings when row order or indexing might invalidate assumptions.
    """
    #TODO: add testing for single column data - ie. what if you throw a series in to pokeframe
    #TODO: different output types - ie. not just console
    #TODO: add docstrings
    #TODO: finish typing
    #TODO: integrate read_csv
    #TODO: lookat pypi setup
    #TODO: more info on null distribution
    #TODO: add testing
    #TODO: consider multiindexes if passed directly...

    def __init__(self, df: pd.DataFrame, safe_mode: bool = True):
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"Expected a pandas dataframe, but got {type(df).__name__}")
        self.df = df
        self.columns = {col_name: PokeColumn(df[col_name]) for col_name in df.columns}
        self._quarantine_mask = []
        self._safe_mode = safe_mode
        self._active_mask = None
        if self._safe_mode:
            self._index_hash = self._hash_index(df.index)

    def __getitem__(self,col_name):
        """
        Access the wrapped PokeColumn for a specific column name.

        Parameters
        ----------
        col_name : str
            Name of the column in the underlying dataframe.

        Returns
        -------
        PokeColumn
            The corresponding PokeColumn object.
        """
        return self.columns[col_name]

    def __repr__(self):
        return f"<PokeFrame: {len(self.df)} rows, {len(self.columns)} columns>"

    def __len__(self):
        return len(self.df)


    # Properties -----------------------------
    @property
    def index_hash(self):
        """Hash of dataframe index."""
        return self._index_hash

    @property
    def active_mask(self):
        """Index mask for active rows."""
        if self._active_mask is None:
            self._active_mask = self.df.index.difference(self.quarantine_mask)
        return self.active_mask

    #TODO: consider removal of this if we don't need it now sign has been reversed:
    @property
    def quarantine_mask(self):
        """Index mask for quarantined rows."""
        return self._quarantine_mask

    @property
    def safe_mode(self):
        """Enables tests to check underlying dataframe index hasn't been modified (unimplemented)."""
        #TODO: implement checks
        return self._safe_mode

    @property
    def schema(self):
        """
        Return the pandas dtypes of the underlying dataframe.

        Returns
        -------
        dict
            Dictionary of column names and their dtypes.
        """
        return self.df.dtypes.to_dict()

    # Setters ----------------------------------

    @quarantine_mask.setter
    def quarantine_mask(self, new_mask):
        #TODO: consider how to handle incremental updates to this, will need some more thinking if we have another mask type
        self._quarantine_mask = new_mask
        self._active_mask = ~new_mask
    
    @active_mask.setter
    def active_mask(self, new_mask):
        #TODO: consider if there should be some kind of mask check
        #TODO: consider if we need active and quarantine if there is no future unsynched behaviour
        #TODO: consider if we should allow the user access to this / enforce checks
        self._active_mask = new_mask
        self._quarantine_mask = ~new_mask


    @safe_mode.setter
    def safe_mode(self, newval):
        if newval != self._safe_mode:
            warnings.warn("Changing safe_mode after initialisation is not advised - integrity checks may not work as expected", RuntimeWarning)
            #if safe mode is turned on, recompute the index hash:
            if newval == True:
                self._index_hash = self._hash_index(self.df.index)
        self._safe_mode = newval




    #Public methods --------------

    def coerce_dtypes(self, schema: dict[Any, Union[CoerceTypes, Callable[[object], object]]], copy: bool = True, quarantine: Union[bool,Literal['detail']] = True, aggresive_bools: bool = False) -> Tuple[pd.DataFrame, dict]:

        """
        Attempt to coerce columns in the dataframe to specified types, with error handling and optional quarantining.

        Parameters
        ----------
        schema : dict
            A mapping of column names to coercion targets. Each value must be one of:
            - A `CoerceTypes` enum value ('num', 'datetime', 'bool', 'string'),
            - A callable that accepts a single value and returns the coerced version.
        copy : bool, optional
            If True (default), operate on a copy of the dataframe and return it.
            If False, coercions are performed in-place on the original dataframe. If quarantine
            is also False, failed coercions will be written to the dataframe as NaN,
            otherwise they will be left as the original values.
        quarantine : bool or {'detail'}, optional
            If True (default), rows with failed conversions are removed from the result
            and returned separately. If 'detail', adds a column noting which columns failed.
            The quarantine_mask property will also be updated to track failed coercions.
            If False, all rows are retained regardless of coercion outcome.

        Returns
        -------
        new_df : pd.DataFrame
            The coerced dataframe, potentially with problematic rows removed if quarantining is enabled.
        output : dict
            A dictionary containing:
            - 'summary': pd.DataFrame with per-column coercion results
            - 'coerce_errors': dict of column -> failed row indices
            - 'quarantine' (if enabled): pd.DataFrame of quarantined rows

        Raises
        ------
        KeyError
            If any columns in the schema are not found in the dataframe.
        ValueError
            If a coercion function or type string is unsupported.

        Notes
        -----
        - Failed conversions are treated as NaNs and tracked.
        - Top 5 failing values per column are included in the output summary.
        - If `safe_mode` is enabled, index integrity is checked before applying masks.
        - Currently all numeric coercions default to float to preserve nullability.
        - If all coercions successful, output["summary"].loc["total_errors","count"] will == 0.
        """
        #TODO: This will create a sliced dataframe so needs to reset the index hash
        #TODO: This creates a quarantine mask - decide logic for overwrites and sequential calls
        #TODO: checks helper function
        #TODO: check index hash every time we use a mask, but don't if we are just using a schema

        #region Validation and variable setup --------------------------
        CoerceTypes.validate_schema(schema, self.df)

        unexpected_keys = set(schema.keys()).difference(self.columns)
        if unexpected_keys:
            raise KeyError(f"The following columns were present in the schema but not the dataframe: {unexpected_keys}")

        if copy:
            new_df = self.df.copy()
        else:
            new_df = self.df
        startlen = len(new_df)
        #problemrows will be a dictionary of the indexes of failed coercions from each column
        problemrows = {}
        #c_errors tracks the top errors from each column (can't retrieve later if copy == quarantine == False)
        c_errors = {}

        #endregion end of: Validation and variable setup

        #region Try coercions ------------------------------------------

        for col, dtype in schema.items():
            #coerce_dtypes on pokecolumn is only called with copy = False if copy and quarantine are false
            #on pokeframe.coerce_dtypes - this is the only case where we want to modify the underlying values.
            try:
                coerced, failed, top_errors = self.columns[col].coerce_dtypes(dtype,copy = (copy or quarantine))
            except ValueError as e:
                logging.warning(f"Failed coercion in column {col} to type {dtype}")
                if copy is False and quarantine is False:
                    warnings.warn(f"(copy = False and quarantine = False, dataframe is being modified in place."
                                    f"Any columns processed before this point may have been modified", RuntimeWarning)
                    raise
            #handle update of new_df:
            if copy:
                #new_df needs to be modified in place:
                new_df[col] = coerced
            elif quarantine is True:
                #if quarantining but not copying, only update succesful coercions:
                mask = ~new_df.index.isin(failed)
                new_df.loc[mask,col] = coerced.loc[mask]

            if not failed.empty:
                logging.warning(f"Column {col} encountered {len(failed)} errors while converting to {dtype}, check second output for more info")
                problemrows[col] = failed
                c_errors[col] = top_errors


        outdict = {"summary": None, "coerce_errors": problemrows}
        #qindex is the index of any coercion errors
        qindex = functools.reduce(pd.Index.union, problemrows.values()) if problemrows else []

        #endregion end of: Try coercions

        #region Handle coercion failures -------------------------------

        #   region Setup quarantine output if needed--------------------

        if quarantine and problemrows:
                #qdf isn't a copy - so should be efficient, only needed for function return
                qdf = self.df.loc[qindex]
                #write an extra column in the output if detail is enabled:
                if quarantine == "detail":
                    if "coerce_errors" in self.df.columns:
                        logging.warning("Column 'corece_errors' already exists in the data, and will be overwritten in the quarantine")
                    qdf["coerce_errors"] = ""
                    for k, v in problemrows.items():
                        qdf.loc[v, "coerce_errors"] = qdf.loc[v, "coerce_errors"].map(lambda x: ", ".join([x, k]) if x else k)

                outdict["quarantine"] = qdf
        else:
            #add a 0 length dataframe of the same structure for consistency, even if quarantine is not set or there are no errors
            outdict["quarantine"] = pd.DataFrame(columns = new_df.columns)

        #set quarantine mask if quarantine != False
        if quarantine:
            self.quarantine_mask = new_df.index.isin(qindex)



        #   endregion: setup quarantine output if needed


        #region Setup outputs ------------------------------------------

        #build a summary output:
        od = {
            "total_rows": ["",startlen, ""],
            "successfully coerced": ["",startlen - len(qindex),""],
            "total_errors": ["", len(qindex), ""]
        }
        for sk, sv in schema.items():
            if sk not in problemrows:
                errs = [0, ""]
            else:
                errs = [len(problemrows[sk]), c_errors[sk]]
            convname = "func: "+sv.__name__ if isinstance(sv,Callable) else sv
            od.update({"colerr_" + sk: [convname] + errs})
        outdict["summary"] = pd.DataFrame.from_dict(od, orient= "index", columns = ["covert_to","count", "most_freq_errors"])


        #endregion: setup outputs

        #if we are quarantining, only return successful rows
        if quarantine:
            return new_df.loc[~self.quarantine_mask], outdict
        else:
            return new_df, outdict


    def summary(self) -> pd.DataFrame:
        """
        Return a summary of all columns in the dataframe.

        Returns
        -------
        pd.DataFrame
            A dataframe where each row summarizes a column's quality metrics.
        """
        output = []
        for column in self.columns.values():
            output.append(column.summary(detailed = False))
        return pd.DataFrame(output)


    # Private methods -------------------

    @staticmethod
    def _hash_index(index: pd.Index) -> int:
        """
        Compute a hash of the dataframe index.

        Used for integrity checking when `safe_mode` is enabled.

        Parameters
        ----------
        index : pd.Index
            The dataframe index to hash.

        Returns
        -------
        int
            A hash value for the index.
        """
        return hash(tuple(index))

    @staticmethod
    def _check_bom(filepath):
        with open(filepath, "rb") as f:
            start = f.read(4)
            if start.startswith(b'\xef\xbb\xbf'):
                return "utf-8-sig"
            elif start.startswith(b'\xff\xfe'):
                return "utf-16-le"
            elif start.startswith(b'\xfe\xff'):
                return "utf-16-be"
            else:
                return None

    def _validate_index(self) -> None:
        """
        Ensure that the dataframe index has not changed if safe_mode is enabled.

        Raises
        ------
        RuntimeError
            If the index hash differs from its original value.
        """
        #TODO: add tests for this
        if self._safe_mode and self._hash_index(self.df.index) != self._index_hash:
            raise RuntimeError("Dataframe index has changed since initialisation, methods involving row selection cannot be completed unless the index is reverted or the dataframe is reloaded")

    #Static & Class methods -------------

    @classmethod
    def load_csv(cls, filepath: Union[str, os.PathLike], encoding: str = None, safe_mode: bool = True, **kwargs):
        """Load a csv. Will try specified encoding (if supplied), and common fallback options, including BOM type encodings.

        Parameters
        ----------
        filepath: str or Path
            Path to CSV file.
        encoding: str, optional
            Encoding to try first (eg. utf-8), if not provided, a range of common encodings will be tried (including BOM encodings).
        **kwargs: dict, optional
            Additional arguments to pass to `pandas.read_csv()` (eg. `skiprows`, `header`).

        Returns
        -------
        DataFrame
            Loaded pandas DataFrame.

        Raises
        -------
        ValueError
            If none of the attempted encodings are successful.
        UnicodeDecodeError
            Re-raised as last known decode error if all encoding attempts fail.

        Notes
        -----
        Encodings attempted are:
        1. User supplied `encoding` argument (if provided).
        2. Encoding information from BOM ('utf-8-sig', 'utf-16-le', 'utf-16-be').
        3. Standard fallbacks: ('utf-8', 'cp1252', 'latin1').

        Will provide warnings if BOMs are detected that do not match the supplied encoding, or if another fallback is used.

        Examples
        --------

        >>> df = load_csv("data.csv")
        >>> df = load_csv("data.csv", encoding = "latin1")
        >>> df = load_csv("data.csv", skiprows = 3)
        """

        #TODO: tests to add:
        #various encoding tests
        #gibberish user supplied encoding
        #passthrough of **kwargs

        #setup encodings to try and add the user specified one if provided:
        encodings_totry = []
        if encoding:
            encodings_totry.append(encoding)

        #check for bom type csv encodings and add if needed:
        bomtype = cls._check_bom(filepath)
        if bomtype:
            if encoding != bomtype:
                logging.warning(f"{bomtype} bom detected, will try this encoding after any specified encoding")
                encodings_totry.append(bomtype)

        #add other standard encodings if they are not already there:
        for enc in ["utf-8", "latin1", "cp1252"]:
            if enc not in encodings_totry:
                encodings_totry.append(enc)


        fallback_needed = False
        last_exception = False


        for enc in encodings_totry:
            try:
                df = pd.read_csv(filepath, encoding = enc, **kwargs)
                if fallback_needed and encoding:
                    logging.warning(f"{filepath} could not be loaded with {encoding}, was loaded with {enc} as a fallback")
                else:
                    logging.info(f"{filepath} loaded with encoding: {enc}")
                return cls(df, safe_mode= safe_mode)
            except UnicodeDecodeError as e:
                if enc == encoding:
                    fallback_needed = True
                last_exception = e

        raise ValueError(
            f"Tried the following encodings: {encodings_totry}, was unable to decode {filepath}."
        ) from last_exception