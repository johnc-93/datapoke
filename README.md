# datapoke

**Lightweight, flexible tools for inspecting and cleaning messy pandas DataFrames.**

`datapoke` is designed to help you interrogate your data quickly and handle schema coercion gracefully — even when the input is inconsistent or unpredictable.

---

## Features

- `PokeFrame` and `PokeColumn` wrappers for intuitive access to column-level insights
- Easy type coercion with row-level error tracking and optional quarantine
- Summaries of data types, nulls, and uniqueness
- Flexible handling of bad data (custom coercion functions, detailed error info)
- Safe mode to detect index changes and enforce data integrity

## Installation

```bash
pip install datapoke
```

## Example usage

```
import pandas as pd
from datapoke import PokeFrame

df = pd.read_csv("your_data.csv")
pf = PokeFrame(df)

# Get a quick summary of your data
print(pf.summary())

# Drill into a specific column
print(pf["age"].summary())

# Coerce columns to a schema
schema = {"age": "num", "joined": "datetime"}
df_cleaned, report = pf.coerce_dtypes(schema, copy=True, quarantine="detail")

print(report["summary"])
```

## Documentation

_API reference coming soon_

## License 

MIT license, see `license.txt` for more information

## Roadmap

In no particular order:

- Additions to datacleaning and load:
    - Ingestion of .xlsx and other formats
    - Easy recombination of quarantined rows post clean
- Lightweight data analysis ("driver analysis")
    - Designate a target variable and quickly identify associated or predictive features
    - Focus on explainability
    - Support for time series data
- Efficiency enhancements
    - Sampling options for large datasets
- UI
    - Streamlit based UI option
- Pipelining
    - Sequential data cleaning steps can 

