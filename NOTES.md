# Notes

## Bug Fix: `filter_expiration_dte` crashes when `expiration` column is `String`

**File:** `thetadatars/options/screeners/_common.py`

**Function:** `filter_expiration_dte`

**Problem:** When `expiration='*'` (screen all expirations), the function computes DTE via a Polars column subtraction:

```python
(pl.col("expiration") - pl.lit(today)).dt.total_days().alias("_dte")
```

The `expiration` column returned by ThetaData is a `String` type, not a `Date` type. Polars raises `InvalidOperationError: - not allowed on str and date`.

**Fix:** Cast the column to `Date` before the subtraction when it is stored as a string:

```python
exp_col = pl.col("expiration")
if chain.schema.get("expiration") == pl.String or chain.schema.get("expiration") == pl.Utf8:
    exp_col = exp_col.str.to_date("%Y-%m-%d")
chain = chain.with_columns(
    (exp_col - pl.lit(today)).dt.total_days().alias("_dte")
)
```

Affects all strategies that pass `expiration='*'` with a DTE range (cash-secured puts, covered calls, iron condors, etc.).
