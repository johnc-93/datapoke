import pandas as pd
from datapoke.frame import PokeFrame
import pytest


@pytest.fixture
def coerceimport():
    testpath = "Test data\load tests\dtype_nulltest.csv"
    df = PokeFrame.load_csv(testpath)

@pytest.mark.parameterize()
