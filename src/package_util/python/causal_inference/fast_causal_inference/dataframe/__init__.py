"""
This module provides the DataFrame class, which is a spark-dataframe-like wrapper for `all in sql`.
"""

__all__ = [
    "readClickHouse",
    "readStarRocks",
    "readSparkDf",
    "readCsv",
    "DataFrame",
    "readOlap",
]

from .dataframe import (
    readClickHouse,
    readStarRocks,
    readSparkDf,
    readCsv,
    DataFrame,
    readOlap,
)

from .provider import FCIProvider
