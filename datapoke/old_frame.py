from datapoke.startup import CoerceTypes, PokeColumn, check_bom


import pandas as pd


import functools
import logging
import os
from typing import Any, Callable, Literal, Tuple


class PokeFrame:
    def __init__(self, df: pd.DataFrame, safe_mode: bool = True):
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"Expected a pandas dataframe, but got {type(df).__name__}")
        self.df = df
        self.columns = {col_name: PokeColumn(df[col_name]) for col_name in df.columns}
        self._quarantine_mask = None
        self._safe_mode = safe_mode
        if self._safe_mode:
            self._index_hash = self._hash_index(df.index)

    def __getitem__(self,col_name):
        return self.columns[col_name]



    # Properties -----------------------------
    @property
    def index_hash(self):
        return self._index_hash

    @property
    def quarantine_mask(self):
        if self._quarantine_mask is None:
            self._quarantine_mask = self.df.index
        return self._quarantine_mask

    @property
    def safe_mode(self):
        return self._safe_mode

    @property
    def schema(self):
        return df.dtypes.to_dict()

    # Setters ----------------------------------

    @quarantine_mask.setter
    def quarantine_mask(self, new_mask):
        self._quarantine_mask = new_mask

    @safe_mode.setter
    def safe_mode(self, newval):
        if newval != self._safe_mode:
            warnings.warn("Changing safe_mode after initialisation is not advised - integrity checks may not work as expected", RuntimeWarning)
            #if safe mode is turned on, recompute the index hash:
            if newval == True:
                self._index_hash = self._hash_index(df.index)
        self._safe_mode = newval
    def _validate_index(self) -> None:
        if self._safe_mode and self._hash_index(self.df.index) != self._index_hash:
            raise RuntimeError("Dataframe index has changed since initialisation, methods involving row selection cannot be completed unless the index is reverted or the dataframe is reloaded")


    #Public methods --------------

    def coerce_dtypes(self, schema: dict[Any, Union[CoerceTypes, Callable[[object], object]]], copy: bool = True, quarantine: Union[bool,Literal['detail']] = True) -> Tuple[pd.DataFrame, dict]:
        #TODO: This will create a sliced dataframe so needs to reset the index hash
        CoerceTypes.validate_schema(schema, self.df)

        unexpected_keys = set(schema.keys()).difference(self.columns)
        if unexpected_keys:
            raise KeyError(f"The following columns were present in the schema but not the dataframe: {unexpected_keys}")

        if copy:
            new_df = self.df.copy()
        else:
            new_df = self.df
        startlen = len(new_df)
        problemrows = {}
        c_errors = {}

        for col, dtype in schema.items():
            coerced, failed, top_errors = self.columns[col].coerce_dtypes(dtype,copy = copy)
            new_df[col] = coerced
            if not failed.empty:
                logging.warning(f"Column {col} encountered {len(failed)} errors while converting to {dtype}, check second output for more info")
                problemrows[col] = failed
                c_errors[col] = top_errors


        outdict = {"summary": None, "coerce_errors": problemrows}
        qindex = functools.reduce(pd.Index.union, problemrows.values()) if problemrows else []

        #handle coercion failures:
        if quarantine:
            if problemrows:
                qdf = self.df.loc[qindex]
                new_df.drop(index=qindex, inplace = True)
                if quarantine == "detail":
                    if "coerce_errors" in self.df.columns:
                        logging.warning("Column 'corece_errors' already exists in the data, and will be overwritten in the quarantine")
                    qdf["coerce_errors"] = ""
                    for k, v in problemrows.items():
                        qdf.loc[v, "coerce_errors"] = qdf.loc[v, "coerce_errors"].map(lambda x: ", ".join([x, k]) if x else k)


                outdict["quarantine"] = qdf
            else:
                outdict["quarantine"] = pd.DataFrame(columns = new_df.columns)





        #build a summary output:
        od = {
            "total_rows": ["",startlen, ""],
            "successfully coerced": ["",startlen - len(qindex),""]
        }
        for sk, sv in schema.items():
            if sk not in problemrows:
                errs = [0, ""]
            else:
                errs = [len(problemrows[sk]), c_errors[sk]]
            convname = "func: "+sv.__name__ if isinstance(sv,Callable) else sv
            od.update({"col_" + sk: [convname] + errs})
        outdict["summary"] = pd.DataFrame.from_dict(od, orient= "index", columns = ["covert_to","count/errors", "most_freq_errors"])

        return new_df, outdict
        #currently returns the coerced frame and a dict holding the problem rows dictionary, a summary, and the quarantined frame

        #TODO: checks helper function
        #TODO: check index hash every time we use a mask, but don't if we are just using a schema

    def summary(self):
        output = []
        for column in self.columns.values():
            output.append(column.summary(detailed = False))
        return pd.DataFrame(output)


    # Private methods -------------------

    @staticmethod
    def _hash_index(index):
        return hash(tuple(index))

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

        #setup encodings to try and add the user specified one if provided:
        encodings_totry = []
        if encoding:
            encodings_totry.append(encoding)

        #check for bom type csv encodings and add if needed:
        bomtype = check_bom(filepath)
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