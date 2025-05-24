import pandas as pd
from datapoke.frame import PokeFrame
import pytest



corececasenames = "copy, quarantine, expected"
coercecases = [
    {"copy": False,"quarantine":False,"lendiff":0, "expected_agetype":{"float"}},
    {"copy": False,"quarantine": True,"lendiff": 1, "expected_agetype": {"str","float"}},
    {"copy":True,"quarantine": False,"lendiff":0, "expected_agetype": {"str"}},
    {"copy":True,"quarantine": True,"lendiff":1, "expected_agetype": {"str"}}]


@pytest.mark.parametrize("case", coercecases, ids= lambda c: f"copy: {c["copy"]}, quarantine: {c['quarantine']}")
def test_coerce_quarantinesplit(case):
    testpath = r"Test data\load tests\dtype_nulltest.csv"
    df =  PokeFrame.load_csv(testpath)
    outdf, detail = df.coerce_dtypes({"Age": "num"},copy = case["copy"], quarantine=case["quarantine"])
    assert len(df) - len(outdf) == case["lendiff"] 
    assert len(detail['quarantine']) == case["lendiff"]

@pytest.mark.parametrize("case", coercecases, ids= lambda c: f"copy: {c["copy"]}, quarantine: {c['quarantine']}")
def test_coerce_modifyinplace(case):
    testpath = r"Test data\load tests\dtype_nulltest.csv"
    df =  PokeFrame.load_csv(testpath)
    originallen = len(df)
    df.coerce_dtypes({"Age": "num"},copy = case["copy"], quarantine=case["quarantine"])
    assert originallen == len(df)
    sum = df.summary()
    assert sum.loc[sum['name'] == "Age","dtypes"].squeeze() == case['expected_agetype']


