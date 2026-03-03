import pandas as pd

# Load legacy returns
legacy_returns = pd.read_csv("outputs/legacy_returns.csv")
print("=== Legacy Returns ===")
print(f"Columns: {legacy_returns.columns.tolist()}")
print(f"\nFirst 10 rows:")
print(legacy_returns.head(10))
print(f"\nStats for 'score' column:")
print(legacy_returns['score'].describe())
print(f"\nMin: {legacy_returns['score'].min()}")
print(f"\nMax: {legacy_returns['score'].max()}")
print(f"\nMean: {legacy_returns['score'].mean()}")

# Check if there are any extreme values
extreme = legacy_returns[abs(legacy_returns['score']) > 1]
if len(extreme) > 0:
    print(f"\n⚠️  Found {len(extreme)} extreme values (abs > 1):")
    print(extreme[['year_month', 'score']])
