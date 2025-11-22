import pandas as pd
from typing import Union, Sequence, Optional, Tuple

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
