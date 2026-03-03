# %%
from typing import List,Optional
import numpy as np
import pandas as pd
from .index_manager import IndexDataManager

def _to_group_sizes(group_ids: pd.Series) -> np.ndarray:
    """
    Calculates the size of each group in chronological order.
    This function correctly ensures the group order matches the sorted data.
    """
    # Using value_counts and reindexing by the sorted unique group IDs is a robust way
    # to maintain the correct chronological order for the groups.
    return group_ids.value_counts(sort=False).reindex(sorted(group_ids.unique())).to_numpy()

def make_rank_dataset(df: pd.DataFrame, features: List[str], target: str,remove_y_na : bool = True):
    """
    Prepares a dataset for XGBRanker using integer relevance ranks as the target.
    """
    d = df.copy()
    d = d.sort_values("year_month").reset_index(drop=True)
    
    if remove_y_na : 
        d.dropna(subset=[target], inplace=True)
    
    
    d["relevance"] = d.groupby("year_month")[target].rank(method="dense", ascending=True) - 1
    
    X = d[features]
    y = d["relevance"].astype(int).to_numpy() # Ensure the final type is integer.
    group = _to_group_sizes(d["year_month"])
    meta = d[["year_month", "ticker",target]].reset_index(drop=True)
    
    return X, y, group, meta


def make_prob_dataset(df: pd.DataFrame, features: List[str], target: str, remove_y_na: bool = True):
    """
    Prépare X (features), y (0/1 en array), et meta pour classification.
    """
    d = df.copy()
    d = d.sort_values("year_month").reset_index(drop=True)

    if remove_y_na:
        d.dropna(subset=[target], inplace=True)

    d[target] = (d[target] > 1).astype(int)

    X = d[features].to_numpy()  # numpy array (n_samples, n_features)
    y = d[target].to_numpy()   # numpy array (n_samples,)
    meta = d[["year_month", "ticker", target]].reset_index(drop=True)

    return X, y, meta

def clean_to_category(df : pd.DataFrame) -> pd.DataFrame : 
    df = df.copy()
    categorical_columns = [c for c in df.select_dtypes(include=['object', 'category']).columns]
    for col in categorical_columns : 
        df[col] = df[col].astype('category')
    return df

def convert_to_quantiles(df: pd.DataFrame,exclude_cols: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Transforms the numerical feature columns of a DataFrame into their
    cross-sectional quantiles for each time period.

    Args:
        df (pd.DataFrame): The input DataFrame, must contain a 'year_month' column.
        exclude_cols (Optional[List[str]], optional): A list of numerical columns
            to exclude from the transformation. Defaults to None.

    Returns:
        pd.DataFrame: A new DataFrame with numerical features converted to quantiles (0.0 to 1.0).
    """
    #print("Starting quantile transformation...")
    df_quantiles = df.copy()

    # Define a default list of columns that should never be transformed
    default_exclusions = ['year_month', 'monthly_return', 'future_return']
    if exclude_cols:
        default_exclusions.extend(exclude_cols)

    # Identify all numerical columns to be transformed
    cols_to_transform = df_quantiles.select_dtypes(include=np.number).columns.tolist()
    cols_to_transform = [col for col in cols_to_transform if col not in default_exclusions]

    #print(f"Transforming {len(cols_to_transform)} numerical columns into quantiles...")

    # Group by each month, then apply the rank-to-quantile transformation on each column
    # .rank(pct=True) calculates the percentile rank, which is exactly the quantile.
    for col in cols_to_transform:
        df_quantiles[col] = df_quantiles.groupby('year_month')[col].transform(lambda x: x.rank(pct=True))

    #print("Quantile transformation complete.")
    return df_quantiles


def transform_outliers_smoothly(
    series: pd.Series, iqr_multiplier: float = 1.5) -> pd.Series:
    """
    Handles outliers in a series using logarithmic soft clipping based on IQR.
    This is the base transformation function.
    """
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    
    # Handle cases where IQR is zero to avoid division by zero or no-op fences
    if iqr == 0:
        return series # No spread, no outliers to transform
        
    lower_fence = q1 - iqr_multiplier * iqr
    upper_fence = q3 + iqr_multiplier * iqr
    
    transformed_series = series.copy()
    
    upper_outliers_mask = series > upper_fence
    lower_outliers_mask = series < lower_fence
    
    if upper_outliers_mask.any():
        transformed_series.loc[upper_outliers_mask] = upper_fence + np.log1p(
            series.loc[upper_outliers_mask] - upper_fence
        )
        
    if lower_outliers_mask.any():
        transformed_series.loc[lower_outliers_mask] = lower_fence - np.log1p(
            -(series.loc[lower_outliers_mask] - lower_fence)
        )
        
    return transformed_series

def apply_smooth_outlier_transform(
    df: pd.DataFrame,
    exclude_cols: Optional[List[str]] = None,
    iqr_multiplier: float = 1.5
) -> pd.DataFrame:
    """
    Applies the smooth outlier transformation to all relevant numerical feature
    columns of a DataFrame, cross-sectionally for each time period.

    Args:
        df (pd.DataFrame): The input DataFrame, must contain a 'year_month' column.
        exclude_cols (Optional[List[str]], optional): A list of numerical columns
            to exclude from the transformation. Defaults to None.
        iqr_multiplier (float, optional): The multiplier for the IQR to define
            the outlier fences. Defaults to 1.5.

    Returns:
        pd.DataFrame: A new DataFrame with outliers in numerical features smoothly compressed.
    """
    print("Applying smooth outlier transformation...")
    df_transformed = df.copy()

    # Define a default list of columns that should never be transformed
    default_exclusions = ['year_month', 'monthly_return', 'future_return']
    if exclude_cols:
        default_exclusions.extend(exclude_cols)
    # Ensure uniqueness in case of overlap
    default_exclusions = list(set(default_exclusions))

    # Identify all numerical columns to be transformed
    cols_to_transform = df_transformed.select_dtypes(include=np.number).columns.tolist()
    cols_to_transform = [col for col in cols_to_transform if col not in default_exclusions]

    print(f"Transforming outliers in {len(cols_to_transform)} numerical columns...")

    # Group by each month, then apply the outlier transformation on each column
    for col in cols_to_transform:
        df_transformed[col] = df_transformed.groupby('year_month')[col].transform(
            lambda x: transform_outliers_smoothly(x, iqr_multiplier=iqr_multiplier)
        )

    print("Outlier transformation complete.")
    return df_transformed

def prepare_data_for_xgboost(kpi_df: pd.DataFrame,index : IndexDataManager,to_quantiles : bool = True,treshold_percentage_missing : float = 0.10) -> pd.DataFrame:
    
    df = kpi_df.copy()
    df = df.sort_values(['ticker', 'year_month']).reset_index(drop=True)
    df['future_return'] = df.groupby('ticker')['monthly_return'].shift(-1)
    selected_columns = df.columns
    df[selected_columns] = df[selected_columns].replace([np.inf, -np.inf], np.nan)
    df = df[selected_columns].copy()
    
    df = df.merge(index.components[['ticker','year_month']],left_on=['ticker','year_month'], right_on=['ticker','year_month'], how='inner')
    
    if to_quantiles : df = convert_to_quantiles(df, exclude_cols=['ticker', 'year_month', 'monthly_return', 'future_return'])
    else : df = apply_smooth_outlier_transform(df, exclude_cols=['ticker', 'year_month', 'monthly_return', 'future_return'], iqr_multiplier=2)
    
    df = df.dropna(subset = ['monthly_return'], axis=0)
    missing_df = (df.isnull().sum()/ len(df)).sort_values(ascending=False).to_frame('missing_percentage')
    removed_df = missing_df[missing_df['missing_percentage'] > treshold_percentage_missing].index
    df = df.drop(removed_df, axis=1)
    df = clean_to_category(df)
    
    print(f"Removed columns : {list(removed_df)} due to missing values above {treshold_percentage_missing*100}%")
    return df

def make_train_test_ranked(df,features,target,year_month_split) : 
    
    df = df.copy()
    train = df[df['year_month'] < year_month_split].reset_index(drop=True)
    test = df[df['year_month'] == year_month_split].reset_index(drop=True)

    X_train,y_train,group_train,meta_train = make_rank_dataset(df = train,features = features,target = target,remove_y_na = True)
    try : 
        X_test,y_test,group_test,meta_test = make_rank_dataset(test,features,target,remove_y_na=False)
    except : 
        X_test,y_test,group_test,meta_test = make_rank_dataset(test,features,target,remove_y_na=True)
    
    return X_train,y_train,group_train,meta_train,X_test,y_test,group_test,meta_test,test

def make_train_test_proba(df,features,target,year_month_split) : 
    
    df = df.copy()
    train = df[df['year_month'] < year_month_split].reset_index(drop=True)
    test = df[df['year_month'] == year_month_split].reset_index(drop=True)

    X_train,y_train,meta_train = make_prob_dataset(df = train,features = features,target = target,remove_y_na = True)
    try : 
        X_test,y_test,meta_test = make_prob_dataset(test,features,target,remove_y_na=False)
    except : 
        X_test,y_test,meta_test = make_prob_dataset(test,features,target,remove_y_na=True)
    
    return X_train,y_train,meta_train,X_test,y_test,meta_test,test

