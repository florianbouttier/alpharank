import pandas as pd
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent / "src"))

# Load legacy returns to check format
legacy_returns = pd.read_csv("outputs/legacy_returns.csv")
print("Legacy Returns Sample:")
print(legacy_returns.head(20))
print("\nLegacy Returns Stats:")
print(legacy_returns.describe())
print("\nColumn names:", legacy_returns.columns.tolist())
