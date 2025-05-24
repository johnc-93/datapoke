import pandas as pd
import logging
from tabulate import tabulate
from datapoke.frame import PokeFrame
from utils import force_boolcast

logging.basicConfig(level = logging.INFO)

testpath1 = r"Test data\pizza_sales\order_details.csv"
testpath2 = r"Test data\load tests\dtype_nulltest.csv"
testpath_cp = r"Test data\load tests\cp1252_encoded.csv"
testpath_lat1 = r"Test data\load tests\latin1_encoded.csv"
testpath_utf8sig = r"Test data\load tests\utf8_sig_break_utf8.csv"
testpath_inv = r"Test data\load tests\totally_invalid_encoding.csv"
testpath_messydtypes = r"Test data\load tests\messy_HR_data.csv"



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


    #TODO: tests to add: error check on dataframe rather than series

def qual_frame(df):
    output = []
    for col in df.columns:
        output.append(check_colqual(df[col]))
    return pd.DataFrame(output)

#df = load_csv(testpath_lat1,encoding="utf-8")
#df = load_csv(testpath_messydtypes)




df2 = PokeFrame.load_csv(testpath_messydtypes)

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
