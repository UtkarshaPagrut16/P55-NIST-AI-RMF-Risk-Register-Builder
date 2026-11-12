"""U4 (part 2): CSV export of the complete risk register."""
import pandas as pd

from .config import REGISTER_COLUMNS


def register_to_csv(df: pd.DataFrame, classified: pd.DataFrame = None) -> bytes:
    """Return the register as UTF-8 CSV bytes (with BOM so Excel on Windows opens it correctly).

    If the RMF classification exists it is included, so the CSV is the complete register.
    """
    table = classified if classified is not None and not classified.empty else df
    base = [c for c in REGISTER_COLUMNS if c in table.columns]
    extra = [c for c in table.columns if c not in base]
    return table[base + extra].to_csv(index=False).encode("utf-8-sig")


def retrieval_to_csv(retrieval: pd.DataFrame) -> bytes:
    return retrieval.to_csv(index=False).encode("utf-8-sig")
