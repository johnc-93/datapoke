import pandas as pd
import numpy as np
from typing import Any, Union

def force_boolcast(x: Any) -> Union[bool,float]:
    """
    Convert a value to a bool, including simple string representations.
    
    Handles the following:
    - Strings: 'true', 'false', '1', '0' (case- and whitespace-insensitive)
    - Numbers: 0 is False, any other number is True
    - Booleans: returned as-is
    - Null-like values (None, np.nan): returns np.nan

    Any other type (e.g. dict, list, unrecognized string) returns np.nan.
    
    Parameters
    ----------
    x : any
        The value to convert to bool.

    Returns
    -------
    bool or np.nan
        The converted boolean, or np.nan if coercion is not possible.
    """
    if pd.isna(x):
        return np.nan
    
    if isinstance(x, str):
        x = x.strip().lower()
        if x in {"false", "0"}:
            return False
        elif x in {"true","1"}:
            return True
        else:
            return np.nan
    
    elif isinstance(x, (int, float, bool)):
        return bool(x)
    else:
        return np.nan