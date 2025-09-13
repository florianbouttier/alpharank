# %%
import pandas as pd
import copy
from typing import Union, Sequence, Optional, Tuple
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

def detect_numeric_columns(df: pd.DataFrame, output_format: str = "dict") ->  Union[dict, pd.DataFrame] :
    """
    Scans object/string type columns and identifies those that can be converted
    to int or float without loss (no NaN values).
    
    Args:
        df: Input DataFrame
        output_format: "dict" returns {col: "int"|"float"}, 
                      "dataframe" returns a DataFrame with columns and their detected types
    
    Returns:
        dict or DataFrame depending on output_format parameter
    """
    convertible = {}
    df = df.copy()
    # Iterate through all columns
    for col in df.columns:
        # Check if column is object or string type
        if df[col].dtype == object or pd.api.types.is_string_dtype(df[col]) or pd.api.types.is_numeric_dtype(df[col]):
            # Attempt numeric conversion, dropping NaN values first
            # Drop null values, empty strings, and whitespace-only strings
            cleaned_series = df[col].dropna()
            cleaned_series = cleaned_series[cleaned_series != '']
            cleaned_series = cleaned_series[cleaned_series.astype(str).str.strip() != '']
            ser = pd.to_numeric(cleaned_series, errors="coerce")
            
            # Check if conversion was successful (no NaN values created)
            if ser.isnull().sum() == 0:
                # Check if all float values are actually integers
                if (ser % 1 == 0).all():
                    convertible[col] = "int"
                else:
                    convertible[col] = "float"
    
    # Return based on requested output format
    if output_format == "dataframe":
        return pd.DataFrame([
            {"column": col, "detected_type": dtype} 
            for col, dtype in convertible.items()
        ])
    else:
        return convertible

def convert_numeric_columns(df: pd.DataFrame, inplace: bool = True) -> tuple:
    """
    Convertit les colonnes détectées en int ou float.
    Renvoie le df (modifié ou copie) et le mapping.
    """
    if not inplace:
        df = df.copy()
    mapping = detect_numeric_columns(df)
    for col, dtype in mapping.items():
        df[col] = pd.to_numeric(df[col], errors="raise", downcast=dtype)
    return df, mapping

def convert_to_category(df : pd.DataFrame,categorical_features : list):  
    df = df.copy()
    
    if categorical_features is None:
        categorical_features = [c for c in df.columns if pd.api.types.is_categorical_dtype(df[c]) or pd.api.types.is_object_dtype(df[c])]
    
    for col in df :
        if col in categorical_features :        
            df[col] = df[col].astype('category')
    return df

def remove_columns_by_keywords(df, keywords_to_remove=['premium'], how='any'):
    """
    Remove columns from DataFrame that contain specified keywords.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    keywords_to_remove (list): List of keywords to search for in column names
    how (str): 'any' to remove columns containing keywords anywhere,
               'beginning' to remove columns starting with keywords
    
    Returns:
    pd.DataFrame: DataFrame with specified columns removed
    """
    if how == 'beginning':
        # Find columns that start with any of the keywords (case insensitive)
        cols_to_remove = [col for col in df.columns if any(col.lower().startswith(keyword.lower()) for keyword in keywords_to_remove)]
    if how == 'end':
        # Find columns that start with any of the keywords (case insensitive)
        cols_to_remove = [col for col in df.columns if any(col.lower().endswith(keyword.lower()) for keyword in keywords_to_remove)]
    else:
        # Find columns that contain any of the keywords (case insensitive)
        cols_to_remove = [col for col in df.columns if any(keyword.lower() in col.lower() for keyword in keywords_to_remove)]
    
    print(f"Deleted columns {cols_to_remove}")
    df_cleaned = df.drop(columns=cols_to_remove, errors='ignore')
    
    return df_cleaned

def select_columns_by_keywords(df, keywords_to_keep=['premium'], how='any'):
    """
    Remove columns from DataFrame that contain specified keywords.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    keywords_to_remove (list): List of keywords to search for in column names
    how (str): 'any' to remove columns containing keywords anywhere,
               'beginning' to remove columns starting with keywords
    
    Returns:
    pd.DataFrame: DataFrame with specified columns removed
    """
    if how == 'beginning':
        # Find columns that start with any of the keywords (case insensitive)
        cols_to_keep = [col for col in df.columns if any(col.lower().startswith(keyword.lower()) for keyword in keywords_to_keep)]
    else:
        # Find columns that contain any of the keywords (case insensitive)
        cols_to_keep = [col for col in df.columns if any(keyword.lower() in col.lower() for keyword in keywords_to_keep)]
    
    print(f"Selected columns {cols_to_keep}")
    df_cleaned = df[cols_to_keep]
    
    return df_cleaned

def remove_constant_columns(df):
    """
    Remove columns from DataFrame that have only one unique value.
    """
    # find columns with a single unique value (including NaN as a value)
    constant_cols = [col for col in df.columns if df[col].nunique(dropna=False) <= 1]
    print(f"Removed constant columns {constant_cols}")
    # drop them
    df_cleaned = df.drop(columns=constant_cols, errors='ignore')
    return df_cleaned

def fot(df,ticker) : 
    """
    Fonction pour filtrer les données par ticker.
    """
    df = df.copy()
    df = df[df['ticker'] == ticker]
    if 'date' in df.columns:
        df = df.sort_values('date', ascending=False)
    if 'year_month' in df.columns:
        df = df.sort_values('year_month', ascending=False)
    return df

def plot_kpis(
    df: pd.DataFrame,
    ticker: str,
    col_columns: str,
    cols_lines: Sequence[str],
    date_col: str = "year_month",
    title: Optional[str] = None,
    figsize: tuple = (12, 6),
    marker: str = "o",
    date_window: Optional[Tuple[Optional[object], Optional[object]]] = None,
):
    """
    Plot KPIs for a given ticker with dual y-axes:
    - Left y-axis: bar chart for `col_columns`
    - Right y-axis: line charts with dots for each KPI in `cols_lines`
    X-axis is a date column (default 'year_month'); handles Period dtype.
    Optionally filter by a date window: (start, end) inclusive; accepts str/Period/datetime.

    Returns (fig, ax) where ax is the left axis.
    """
    d = df.copy()

    # Choose a date column
    if date_col not in d.columns:
        if "date" in d.columns:
            date_col = "date"
        else:
            raise ValueError(f"date_col '{date_col}' not found and no 'date' column present.")

    # Filter like fot
    d = d[d["ticker"] == ticker].copy()
    if d.empty:
        raise ValueError(f"No data for ticker '{ticker}'.")

    # Prepare x-axis safely
    x = d[date_col]
    if pd.api.types.is_period_dtype(x):
        x = x.dt.to_timestamp()
    elif pd.api.types.is_datetime64_any_dtype(x):
        pass
    else:
        x_try = pd.to_datetime(x, errors="coerce")
        if x_try.notna().sum() >= max(1, int(0.5 * len(x))):
            x = x_try
        else:
            x = x.astype(str)

    # Build plotting frame and sort chronologically
    plot_df = d.copy()
    plot_df["_x_"] = x
    plot_df = plot_df.sort_values("_x_")

    # Optional date window filter
    if date_window is not None:
        start, end = date_window
        if pd.api.types.is_datetime64_any_dtype(plot_df["_x_"]):
            start_dt = pd.to_datetime(start, errors="coerce") if start is not None else None
            end_dt = pd.to_datetime(end, errors="coerce") if end is not None else None
            mask = pd.Series(True, index=plot_df.index)
            if start_dt is not None:
                mask &= plot_df["_x_"] >= start_dt
            if end_dt is not None:
                mask &= plot_df["_x_"] <= end_dt
            plot_df = plot_df[mask]
        else:
            # String-like x-axis: compare as strings if start/end provided as strings
            mask = pd.Series(True, index=plot_df.index)
            if isinstance(start, str):
                mask &= plot_df["_x_"].astype(str) >= start
            if isinstance(end, str):
                mask &= plot_df["_x_"].astype(str) <= end
            plot_df = plot_df[mask]
        if plot_df.empty:
            raise ValueError("No data after applying date_window filter.")

    # Y values
    if col_columns not in plot_df.columns:
        raise ValueError(f"Bar column '{col_columns}' not found in DataFrame.")
    y_bar = pd.to_numeric(plot_df[col_columns], errors="coerce")

    line_cols = [c for c in cols_lines if c in plot_df.columns]

    # Plot with dual axes
    fig, ax = plt.subplots(figsize=figsize)
    ax2 = ax.twinx()

    bar_container = ax.bar(plot_df["_x_"], y_bar, alpha=0.3, label=col_columns)

    line_handles = []
    for c in line_cols:
        y = pd.to_numeric(plot_df[c], errors="coerce")
        (ln,) = ax2.plot(plot_df["_x_"], y, marker=marker, label=c)
        line_handles.append(ln)

    # X-axis formatting
    if pd.api.types.is_datetime64_any_dtype(plot_df["_x_"]):
        locator = mdates.AutoDateLocator()
        formatter = mdates.ConciseDateFormatter(locator)
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(formatter)
    ax.tick_params(axis="x", rotation=45)

    # Labels, grid, legend
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.set_xlabel(date_col)
    ax.set_ylabel(col_columns)
    if line_cols:
        ax2.set_ylabel(", ".join(line_cols))
    if title is None:
        title = f"KPIs for {ticker}"
    ax.set_title(title)

    handles = [bar_container] + line_handles
    labels = [col_columns] + line_cols
    ax.legend(handles, labels, loc="best")

    fig.tight_layout()
    return fig, ax