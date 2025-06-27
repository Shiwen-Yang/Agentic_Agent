import pandas as pd
import os

def load_dataset(path: str) -> pd.DataFrame:
    """
    Load a dataset from a supported file type into a pandas DataFrame.
    
    Supported formats: .csv, .tsv, .xlsx, .xls, .json, .parquet, .sas7bdat

    Raises:
        ValueError: if file extension is unsupported or file loading fails.
    """
    ext = os.path.splitext(path)[1].lower()

    try:
        if ext == ".csv":
            return pd.read_csv(path)
        elif ext == ".tsv":
            return pd.read_csv(path, sep="\t")
        elif ext in [".xlsx", ".xls"]:
            return pd.read_excel(path)
        elif ext == ".json":
            return pd.read_json(path)
        elif ext == ".parquet":
            return pd.read_parquet(path)
        elif ext == ".sas7bdat":
            return pd.read_sas(path)
        else:
            raise ValueError(f"Unsupported file type: {ext}")
    except Exception as e:
        raise ValueError(f"Failed to load dataset: {e}")

def preview_df(df: pd.DataFrame, n: int = 5) -> pd.DataFrame:
    """Return the first n rows of the DataFrame."""
    return df.head(n)

def infer_column_types(df: pd.DataFrame) -> dict:
    """Return a mapping from column name to inferred type."""
    return {col: pd.api.types.infer_dtype(df[col], skipna=True) for col in df.columns}

def get_missing_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Return a DataFrame of missing value counts and percentages."""
    total = df.isnull().sum()
    percent = (total / len(df)) * 100
    return pd.DataFrame({"missing_count": total, "missing_pct": percent})

def get_unique_counts(df: pd.DataFrame) -> dict:
    """Return number of unique values per column."""
    return {col: df[col].nunique() for col in df.columns}
