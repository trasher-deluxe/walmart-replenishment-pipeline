"""Loading, grid completion, merging, and null-imputation for the retail transactions dataset.

Walmart Supply Chain Resilience:
1. Reconstructs missing POS sales metrics (amount_cash, cash_transactions, units_sold, avg_ticket).
2. Builds a complete Cartesian Grid (date x store_id x category) to preserve temporal continuity.
3. Merges static store attributes and dynamic calendar events cleanly.
"""

from pathlib import Path
import pandas as pd
import numpy as np

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

def load_raw(data_dir: Path = DATA_DIR) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load transactions, stores, and calendar CSVs with proper date parsing."""
    df_trans = pd.read_csv(data_dir / "transactions.csv", parse_dates=["date"])
    df_stores = pd.read_csv(data_dir / "stores.csv")
    df_cal = pd.read_csv(data_dir / "calendar.csv", parse_dates=["date"])
    return df_trans, df_stores, df_cal

def build_full_grid(df_trans: pd.DataFrame, df_stores: pd.DataFrame, df_cal: pd.DataFrame) -> pd.DataFrame:
    """Build a complete Cartesian Grid (dates x stores x categories) to handle temporal gaps."""
    all_dates = df_cal["date"].unique()
    all_stores = df_stores["store_id"].unique()
    all_categories = df_trans["category"].dropna().unique()

    # Cartesian product
    grid_index = pd.MultiIndex.from_product(
        [all_dates, all_stores, all_categories],
        names=["date", "store_id", "category"]
    )
    df_grid = pd.DataFrame(index=grid_index).reset_index()

    # Merge transactions onto full grid
    df_merged = df_grid.merge(df_trans, on=["date", "store_id", "category"], how="left")
    
    # Flag rows that were missing from original transactions due to POS outages
    df_merged["pos_outage_flag"] = df_merged["amount_total"].isnull().astype(int)
    
    return df_merged

def impute_pos_failures(df: pd.DataFrame, train_end: str | None = None) -> pd.DataFrame:
    """Impute missing metrics caused by POS/connectivity failures using accounting identities & medians.

    `train_end` (YYYY-MM-DD): median-based imputations are computed ONLY from rows on/before
    this date, so no information from the validation/holdout periods leaks into the imputed
    history. Accounting identities are row-wise and leak-free regardless.
    """
    df = df.copy()

    # 1. Accounting Identity: Cash Transactions = Total - Card
    has_card_tx = df["card_transactions"].notna()
    df.loc[has_card_tx, "cash_transactions"] = df.loc[has_card_tx, "cash_transactions"].fillna(
        df.loc[has_card_tx, "total_transactions"] - df.loc[has_card_tx, "card_transactions"]
    )
    
    # 2. Accounting Identity: Amount Cash = Amount Total - Amount Card
    has_card_amt = df["amount_card"].notna()
    df.loc[has_card_amt, "amount_cash"] = df.loc[has_card_amt, "amount_cash"].fillna(
        df.loc[has_card_amt, "amount_total"] - df.loc[has_card_amt, "amount_card"]
    )
    
    # 3. Recalculate Average Ticket where total_transactions > 0
    valid_tx = (df["total_transactions"] > 0) & (df["avg_ticket"].isnull())
    df.loc[valid_tx, "avg_ticket"] = df.loc[valid_tx, "amount_total"] / df.loc[valid_tx, "total_transactions"]

    # 4. Impute units_sold and remaining avg_ticket via median per (store_id, category, day_of_week).
    #    Medians come ONLY from the train window (train_end) to avoid leaking future stats.
    df["day_of_week"] = df["date"].dt.dayofweek
    ref = df if train_end is None else df[df["date"] <= pd.Timestamp(train_end)]
    for col in ["units_sold", "avg_ticket"]:
        grp_med = ref.groupby(["store_id", "category", "day_of_week"])[col].median().to_dict()
        keys = df.set_index(["store_id", "category", "day_of_week"]).index
        df[col] = df[col].fillna(pd.Series(keys.map(grp_med), index=df.index))
        # Fallback to category median (also train-only) if still null
        cat_med = ref.groupby("category")[col].median().to_dict()
        df[col] = df[col].fillna(df["category"].map(cat_med))

    df.drop(columns=["day_of_week"], inplace=True, errors="ignore")
    return df

def merge_master_dataset(data_dir: Path = DATA_DIR, train_end: str | None = None) -> pd.DataFrame:
    """End-to-End Master Dataset Creation for Model Training & Inference.
    
    Joins:
    - Full Grid (date x store_id x category)
    - Transactions (with POS imputation)
    - Stores metadata (on store_id)
    - Calendar metadata (on date)
    """
    df_trans, df_stores, df_cal = load_raw(data_dir)
    
    # 1. Impute POS failures on raw transactions (train-only medians to prevent leakage)
    df_trans_clean = impute_pos_failures(df_trans, train_end=train_end)
    
    # 2. Build complete grid
    df_grid = build_full_grid(df_trans_clean, df_stores, df_cal)
    
    # 3. Merge static store metadata
    df_master = df_grid.merge(df_stores, on="store_id", how="left")
    
    # 4. Merge dynamic calendar metadata
    df_master = df_master.merge(df_cal, on="date", how="left")
    
    # Fill remaining zero sales for empty grid rows
    for col in ["total_transactions", "cash_transactions", "card_transactions", "amount_total", "amount_cash", "amount_card"]:
        df_master[col] = df_master[col].fillna(0)
        
    return df_master

if __name__ == "__main__":
    df_master = merge_master_dataset()
    print("Master Dataset built successfully!")
    print(f"Shape: {df_master.shape}")
    print(f"Columns: {list(df_master.columns)}")
