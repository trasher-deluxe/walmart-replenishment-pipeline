"""Loading, merging, and null-imputation for the retail transactions dataset.

Handles POS (point-of-sale) outages and connectivity gaps that leave nulls
in `cash_transactions`, `amount_cash`, `units_sold`, and `avg_ticket`.
"""

from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def load_raw(data_dir: Path = DATA_DIR) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load transactions, stores, and calendar CSVs as-is (no cleaning)."""
    transactions = pd.read_csv(data_dir / "transactions.csv", parse_dates=["date"])
    stores = pd.read_csv(data_dir / "stores.csv")
    calendar = pd.read_csv(data_dir / "calendar.csv", parse_dates=["date"])
    return transactions, stores, calendar


def merge_datasets(
    transactions: pd.DataFrame, stores: pd.DataFrame, calendar: pd.DataFrame
) -> pd.DataFrame:
    """Join transactions with store attributes and calendar variables."""
    df = transactions.merge(stores, on="store_id", how="left")
    df = df.merge(calendar, on="date", how="left")
    return df


def impute_pos_failures(df: pd.DataFrame) -> pd.DataFrame:
    """Impute nulls caused by POS/connectivity failures.

    - `cash_transactions` / `amount_cash`: derived from totals minus card,
      when the card side is present; otherwise 0.
    - `units_sold`, `avg_ticket`: filled with the store/category median.
    """
    df = df.copy()

    has_card = df["card_transactions"].notna()
    df.loc[has_card, "cash_transactions"] = df.loc[has_card, "cash_transactions"].fillna(
        df.loc[has_card, "total_transactions"] - df.loc[has_card, "card_transactions"]
    )
    df["cash_transactions"] = df["cash_transactions"].fillna(0)

    has_card_amt = df["amount_card"].notna()
    df.loc[has_card_amt, "amount_cash"] = df.loc[has_card_amt, "amount_cash"].fillna(
        df.loc[has_card_amt, "amount_total"] - df.loc[has_card_amt, "amount_card"]
    )
    df["amount_cash"] = df["amount_cash"].fillna(0)

    for col in ("units_sold", "avg_ticket"):
        df[col] = df.groupby(["store_id", "category"])[col].transform(
            lambda s: s.fillna(s.median())
        )

    return df


def load_and_prepare(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """Full loading pipeline: read, merge, impute."""
    transactions, stores, calendar = load_raw(data_dir)
    df = merge_datasets(transactions, stores, calendar)
    df = impute_pos_failures(df)
    return df
