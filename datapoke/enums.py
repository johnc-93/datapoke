import pandas as pd


import inspect
from enum import Enum
from typing import Hashable, Callable, Union


class CoerceTypes(str, Enum):
    # TODO: write testing for these
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
            positional_args = [
                p
                for p in params
                if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
            ]
            return len(positional_args) == 1
        return False

    @classmethod
    def validate_schema(
        cls,
        schema: dict[Hashable, Union["CoerceTypes", str, Callable[[object], object]]],
        df: pd.DataFrame,
    ) -> dict[Hashable, "CoerceTypes" | Callable[[object], object]]:
        unexpected_keys = set(schema.keys()).difference(df.columns)
        if unexpected_keys:
            raise KeyError(
                f"The following columns were present in the schema but not the dataframe: {unexpected_keys}"
            )
        validated_schema = dict()
        for k, v in schema.items():
            if cls.is_valid(v):
                if not callable(v):
                    validated_schema[k] = cls(v)
            else:
                raise ValueError(
                    f"Unsupported type mappings in schema: {v}. Currently supported types are: {cls.values()} or a callable with one argument"
                )
        return validated_schema
