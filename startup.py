import pandas as pd
import logging
from tabulate import tabulate
from typing import Literal, Union, Tuple, Mapping, Any, Callable
from enum import Enum
import functools
import numpy as np
import inspect

logging.basicConfig(level = logging.INFO)

testpath1 = r"Test data\pizza_sales\order_details.csv"
testpath2 = r"Test data\load tests\dtype_nulltest.csv"
testpath_cp = r"Test data\load tests\cp1252_encoded.csv"
testpath_lat1 = r"Test data\load tests\latin1_encoded.csv"
testpath_utf8sig = r"Test data\load tests\utf8_sig_break_utf8.csv"
testpath_inv = r"Test data\load tests\totally_invalid_encoding.csv"
testpath_messydtypes = r"Test data\load tests\messy_HR_data.csv"



class CoerceTypes(str, Enum):
    NUM = "num"
    DATETIME = "datetime"
    BOOL = "bool"
    STRING = "string"

    @classmethod
    def values(cls) -> set[str]:
        return {v.value for v in cls}
    
    @classmethod
    def is_valid(cls, value) -> bool:
        if value in cls.__members__.values():
            return True
        if callable(value):
            sig = inspect.signature(value)
            params = sig.parameters.values()    
            positional_args = [p for p in params if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)]
            return len(positional_args) == 1
        return False

    
    @classmethod
    def validate_schema(cls, schema: dict, df: pd.DataFrame) -> None:
        unexpected_keys = set(schema.keys()).difference(df.columns)
        if unexpected_keys:
            raise KeyError(f"The following columns were present in the schema but not the dataframe: {unexpected_keys}")

        invalid_target = {k: v for k, v in schema.items() if not cls.is_valid(v)}
        if invalid_target:
            raise ValueError(f"Unsupported type mappings in schema: {invalid_target}. Currently supported types are: {cls.values()} or a callable with one argument")
        
    #todo: write testing for these

def check_bom(filepath):
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
    
def nullonfailure_wrapper(func: Callable[[object],[object]]) -> Callable [[object], [object]]:
    """Wraps a user defined function to return np.nan if there is an error.

    Args:
        func (Callable[[object],[object]]): Any function that accepts and returns a single argument.

    Returns:
        Callable [[object], [object]]: Wrapped version of func.
    """
    @functools.wraps(func)
    def wrapper(x):
        try:
            return func(x)
        except:
            return np.nan
    return wrapper


def load_csv(filepath, encoding = None, **kwargs):  
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
            return df
        except UnicodeDecodeError as e:
            if enc == encoding:
                fallback_needed = True
            last_exception = e
    
    raise ValueError( 
        f"Tried the following encodings: {encodings_totry}, was unable to decode {filepath}."
    ) from last_exception

    #todo: tests to add:
    #various encoding tests
    #gibberish user supplied encoding
    #passthrough of **kwargs


def qual_col(column):
    if not isinstance(column, pd.Series):
        raise TypeError(f"Expected a pandas series, but got {type(column).__name__}")
    denulled = column.dropna()    
    nullcount = column.isna().sum()
    nondupes = column.drop_duplicates()
    dtypes = denulled.map(type).value_counts()
    output = {
        "name": column.name,
        "nullcount": int(nullcount),
        "unique values": len(nondupes),
        "dtypes": [t.__name__ for t in dtypes.index],
        "n_types":  len(dtypes)
    }
    return output


    #todo: tests to add: error check on dataframe rather than series

def qual_frame(df):
    output = []
    for col in df.columns:
        output.append(check_colqual(df[col]))
    return pd.DataFrame(output)

class PokeColumn:
    def __init__(self,series):
        if not isinstance(series, pd.Series):
            raise TypeError(f"Expected a pandas series, but got {type(series).__name__}")
        self.series = series
        self.name = self.series.name

    @property    
    def nullcount(self):
        return self.series.isna().sum()
    
    @property
    def uniquevalues(self):
        return len(self.series.drop_duplicates())

    @property
    def dtypes(self):
        return self.series.dropna().map(type).value_counts()

    @property
    def ntypes(self):
        return len(self.dtypes)



    def summary(self, detailed = True, display = True):
        output = {
            "name": self.name,
            "n_nulls": self.nullcount,
            "unique_val": self.uniquevalues,
            "most_freq_vals": self.series.value_counts().iloc[:5].index.tolist(),
            "dtypes": [t.__name__ for t in self.dtypes.index],
            "n_types":  self.ntypes
        }
        if detailed:
            output.update({"n_" + key.__name__: value for key, value in self.dtypes.items()})
        if display:
            return pd.Series(output)
        else:
            return output
    
    def coerce_dtypes(self,target_dtype, copy = False):
        if isinstance(target_dtype, Callable):
            coerced = self.series.map(nullonfailure_wrapper(target_dtype)) 
        else:
            converter = {
                "num": pd.to_numeric,
                "datetime": pd.to_datetime,
                "bool": lambda x: x.astype("boolean"), 
                "string": lambda x: x.astype("string")
            }.get(target_dtype)

            if converter is None:
                raise ValueError(f"Unsupported type: {target_dtype}. Currently supported types are: num, datetime, bool, string.")

            try:
                if target_dtype in ["num","datetime"]:
                    coerced = converter(self.series, errors="coerce")
                else:
                    coerced = converter(self.series)

            except Exception as e:
                raise ValueError(f"Failed to coerce column '{self.name}' to {target_dtype}: {e}")

        failed = self.series[coerced.isna() & self.series.notna()]
        top_errors = failed.value_counts().iloc[:5].index.to_list()
        if not copy:
            self.series = coerced
        return coerced, failed.index, top_errors
        #todo: add handling for ints vs. floats
        #todo: add handling for unusual pandas categories
        #todo: add docstring
        #todo: optimise failed memory usage a bit more
    
    
    

    #write a __repr__ function
    #optimise duplicate function calls
    #add sampling
    #add caching


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

    def _hash_index(self, index):
        return hash(tuple(index))

    @property
    def safe_mode(self):
        return self._safe_mode
    
    @safe_mode.setter
    def safe_mode(self, newval):
        if newval != self._safe_mode:
            warnings.warn("Changing safe_mode after initialisation is not advised - integrity checks may not work as expected", RuntimeWarning)
            #if safe mode is turned on, recompute the index hash:
            if newval == True:
                self._index_hash = self._hash_index(df.index)
        self._safe_mode = newval 

    @property
    def index_hash(self):
        return self._index_hash

    def validate_index(self) -> None:
        if self._safe_mode and self._hash_index(self.df.index) != self._index_hash:
            raise RuntimeError("Dataframe index has changed since initialisation, methods involving row selection cannot be completed unless the index is reverted or the dataframe is reloaded")


    def summary(self):
        output = []
        for column in self.columns.values():
            output.append(column.summary(detailed = False))
        return pd.DataFrame(output)

    @property
    def schema(self):
        return df.dtypes.to_dict()

    @property
    def quarantine_mask(self):
        if self._quarantine_mask is None:
            self._quarantine_mask = self.df.index
        return self._quarantine_mask
    
    @quarantine_mask.setter
    def quarantine_mask(self, new_mask):
        self._quarantine_mask = new_mask
    
    def coerce_dtypes(self, schema: dict[Any, Union[CoerceTypes, Callable[[object], object]]], copy: bool = True, quarantine: Union[bool,Literal['detail']] = True) -> Tuple[pd.DataFrame, dict]:
        #todo: This will create a sliced dataframe so needs to reset the index hash
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
        #currently returns the coerced frame and a dict holding the problem rows dictionary and the quarantined frame
        
        #poke coerce issues

        #todo: checks helper function
        #todo: check index hash every time we use a mask, but don't if we are just using a schema



    

    #todo: add handling for single column data 
    #todo: different output types
    #todo: add docstrings
    #todo: decide on final output for coercion mechanism - most likely will be quarantine problem records, can also include current functionality of coerce everything as copy
    #todo: finish typing


df = load_csv(testpath_lat1,encoding="utf-8")
df = load_csv(testpath_messydtypes)

headers = df.columns.tolist()

inferred_types = df.dtypes.to_dict()


logging.info(inferred_types)


df2 = PokeFrame(df)

print(df2[df2.df.columns[2]].summary())
print(df2.summary())
print(df2.df.head())

def temp(x):
    x.replace("/","-")
    if x == "2020/02/20":
        x = pd.to_datetime(x)
    return x
#dfc, probs = df2.coerce_dtypes({"Salary": "num", "Joining Date": temp}, quarantine= "detail")
a, b = df2.coerce_dtypes({"Joining Date": temp}, quarantine= False,copy= False)
print("""quarantine:
""")
#print(probs.get('quarantine',""))
#print(probs['summary'])
print(df2.summary())
print(df2.df.head())

print(len(df2.df))
print(b['summary'])
print(a.head())
