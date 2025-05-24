import pandas as pd


import inspect
from enum import Enum


class CoerceTypes(str, Enum):
    #TODO: write testing for these
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