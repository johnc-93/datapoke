from datapoke.frame import PokeFrame
import pytest
from pathlib import Path
from datapoke.enums import CoerceTypes

testdir = Path(__file__).parent.parent



#region test excel import

def test_excelimport():
    testpath = testdir / "Test data" / "load tests" / "MixedTypeDataTable.xlsx"
    df = PokeFrame.load_excel(testpath)

# region csv coercion tests

corececasenames = "copy, quarantine, expected"
coercecases = [
    {"copy": False, "quarantine": False, "lendiff": 0, "expected_agetype": {"float"}},
    {
        "copy": False,
        "quarantine": True,
        "lendiff": 1,
        "expected_agetype": {"str", "float"},
    },
    {"copy": True, "quarantine": False, "lendiff": 0, "expected_agetype": {"str"}},
    {"copy": True, "quarantine": True, "lendiff": 1, "expected_agetype": {"str"}},
]

@pytest.fixture
def null_testfile():
    testpath = testdir / "Test data" / "load tests" / "dtype_nulltest.csv"
    df = PokeFrame.load_csv(testpath)
    return df



@pytest.mark.parametrize(
    "input_dtype, enumed_dtype",
    [
        ("num", CoerceTypes.NUM),
        ("bool", CoerceTypes.BOOL),
        (CoerceTypes.NUM, CoerceTypes.NUM),
    ],
    ids=lambda c: f"input: {c[0]}, expected: {c[1]}",
)
def test_coerce_inputs(input_dtype, enumed_dtype, null_testfile):
    schema = CoerceTypes.validate_schema({"Age": input_dtype}, null_testfile)
    assert schema["Age"] == enumed_dtype

@pytest.mark.parametrize(
    "case",
    [
        {
            "target_col": "ID",
            "dtype": "bool",
            "agg_bool": True,
            "out_dtypes": {"bool"},
            "exception": False,
        },
        {
            "target_col": "ID",
            "dtype": "bool",
            "agg_bool": False,
            "out_dtypes": {},
            "exception": ValueError,
        },
    ],
    ids=lambda c: f"{c['target_col']} converting to {c['dtype']}, agg_bool={c['agg_bool']}, expected:{c['out_dtypes']}, expected exception:{c['exception']}",
)
def test_coercions(case, null_testfile):
    df = null_testfile.copy()
    if case["exception"]:
        with pytest.raises(case["exception"]):
            df.coerce_dtypes(
                schema={case["target_col"]: case["dtype"]},
                aggressive_bools=case["agg_bool"],
                copy=True,
                quarantine=False,
            )
    else:
        df.coerce_dtypes(
            schema={case["target_col"]: case["dtype"]},
            aggressive_bools=case["agg_bool"],
            copy=False,
            quarantine=False,
        )
        summary = df.summary()
        assert (
            summary.loc[summary["name"] == case["target_col"], "dtypes"].squeeze()
            == case["out_dtypes"]
        )


def test_schema_ref_errors(null_testfile):
    """check for key errors on schema columns that don't exist in df"""
    with pytest.raises(KeyError):
        CoerceTypes.validate_schema({"Honk": "num"}, null_testfile)


@pytest.mark.parametrize(
    "case",
    coercecases,
    ids=lambda c: f"copy: {c['copy']}, quarantine: {c['quarantine']}",
)
def test_coerce_quarantinesplit(case):
    testpath = testdir / "Test data" / "load tests" / "dtype_nulltest.csv"
    df = PokeFrame.load_csv(testpath)
    outdf, detail = df.coerce_dtypes(
        {"Age": CoerceTypes.NUM}, copy=case["copy"], quarantine=case["quarantine"]
    )
    assert len(df) - len(outdf) == case["lendiff"]
    assert len(detail["quarantine"]) == case["lendiff"]


@pytest.mark.parametrize(
    "case",
    coercecases,
    ids=lambda c: f"copy: {c['copy']}, quarantine: {c['quarantine']}",
)
def test_coerce_modifyinplace(case):
    testpath = testdir / "Test data" / "load tests" / "dtype_nulltest.csv"
    df = PokeFrame.load_csv(testpath)
    originallen = len(df)
    df.coerce_dtypes(
        {"Age": CoerceTypes.NUM}, copy=case["copy"], quarantine=case["quarantine"]
    )
    assert originallen == len(df)
    sum = df.summary()
    assert sum.loc[sum["name"] == "Age", "dtypes"].squeeze() == case["expected_agetype"]


#endregion CSV coercion tests