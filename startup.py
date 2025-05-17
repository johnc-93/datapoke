import pandas as pd
import logging
from tabulate import tabulate

logging.basicConfig(level = logging.INFO)

testpath1 = r"Test data\pizza_sales\order_details.csv"
testpath2 = r"Test data\load tests\dtype_nulltest.csv"
testpath_cp = r"Test data\load tests\cp1252_encoded.csv"
testpath_lat1 = r"Test data\load tests\latin1_encoded.csv"
testpath_utf8sig = r"Test data\load tests\utf8_sig_break_utf8.csv"
testpath_inv = r"Test data\load tests\totally_invalid_encoding.csv"


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
                logging.warning(f"CSV could not be loaded with {encoding}, was loaded with {enc} as a fallback")
            else:
                logging.info(f"CSV loaded with encoding: {enc}")
            return df
        except UnicodeDecodeError as e:
            if enc == encoding:
                fallback_needed = True
            last_exception = e
    
    raise ValueError( 
        f"Tried the following encodings: {encodings_totry}, was unable to decode CSV."
    ) from last_exception

    #tests to add:
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


    #tests to add: error check on dataframe rather than series

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
    def dtypes_list(self):
        return [t.__name__ for t in self.series.dropna().map(type).value_counts().index]

    @property
    def ntypes(self):
        return len(self.dtypes_list)


    def summary(self):
        output = {
            "name": self.name,
            "nullcount": self.nullcount,
            "unique values": self.uniquevalues,
            "dtypes": self.dtypes_list,
            "n_types":  self.ntypes
        }
        return output
    
    def try_coerce(self,target_dtype):
        converter = {
            "int": pd.to_numeric,
            "float": pd.to_numeric,
            "datetime": pd.to_datetime,
            "bool": lambda x: x.astype("boolean"), 
            "string": lambda x: x.astype("string")
        }.get(target_dtype)

        if converter is None:
            raise ValueError(f"Unsupported type: {target_dtype}. Currently supported types are: int, float, datetime, bool, string.")

        try:
            if target_dtype in ["int","float","datetime"]:
                coerced = converter(self.series, errors="coerce")
            else:
                coerced = converter(self.series)

        except Exception as e:
            raise ValueError(f"Failed to coerce column '{self.name}' to {target_dtype}: {e}")

        failed = self.series[coerced.isna() & self.series.notna()]
        return coerced, failed.index
        #todo: add handling for ints
        #todo: add handling for unusual pandas categories
    
    
    

    #write a __repr__ function
    #optimise duplicate function calls
    #add sampling
    #add caching
    #add coercion mechanism


class PokeFrame:
    def __init__(self, df):
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"Expected a pandas dataframe, but got {type(df).__name__}")
        self.df = df
        self.columns = {col_name: PokeColumn(df[col_name]) for col_name in df.columns}
    

    def __getitem__(self,col_name):
        return self.columns[col_name]

    def summary(self):
        output = []
        for column in self.columns.values():
            output.append(column.summary())
        return pd.DataFrame(output)

    @property
    def schema(self):
        return df.dtypes.to_dict()
    
    def coerce_dtypes(self, schema, copy=True):
        if not isinstance(schema, dict):
            raise TypeError(f"Expected a schema to be a dictionary, but got {type(schema).__name__}")
        unexpected_keys = set(schema.keys()).difference(self.columns)
        if unexpected_keys:
            raise KeyError("The following columns were present in the schema but not the dataframe: {unexpected_keys}")

        if copy:
            new_df = df.copy()
        else:
            new_df = df
        problemrows = {}
        
        for col, dtype in schema.items():
            coerced, failed = self.columns[col].try_coerce(dtype)
            new_df[col] = coerced
            if not failed.empty:
                logging.warn(f"Column {col} encountered {len(failed)} errors while converting to {dtype}, check second output for more info")
                problemrows[col] = failed
        return new_df, problemrows




    

    #todo: add handling for single column data 
    #todo: different output types


df = load_csv(testpath_lat1,encoding="utf-8")
df = load_csv(testpath2)

headers = df.columns.tolist()

inferred_types = df.dtypes.to_dict()


logging.info(inferred_types)

a = df['Department'].apply(type).value_counts()

dep = PokeColumn(df['Department'])
df2 = PokeFrame(df)

print(df.head())
a = df2.summary()
print(df2.summary())
#print(check_framequal(df))


