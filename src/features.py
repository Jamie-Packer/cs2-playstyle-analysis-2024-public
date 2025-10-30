import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression


def fit_positioning_regressions(df, min_maps, verbose=True):
    """
    Fit linear regressions predicting ADAT from ADNT for T and CT sides.
    
    Uses only stable players (map_count >= min_maps) for fitting.
    
    Parameters
    ----------
    df : pd.DataFrame
        Player dataset with positioning features
    min_maps : int
        Minimum map count threshold for stable players
        
    Returns
    -------
    dict
        Dictionary with fitted models: {'t': model_t, 'ct': model_ct}
    """
    stable_df = df[df['map_count'] >= min_maps].copy()
    
    models = {}
    
    for side in ['t', 'ct']:
        X = stable_df[[f'adnt_rank_{side}']].values
        y = stable_df[f'adat_rank_{side}'].values
        
        model = LinearRegression()
        model.fit(X, y)
        
        models[side] = model
        
        coef = model.coef_[0]
        intercept = model.intercept_
        
        if verbose:
            print(f"{side.upper()}-side regression: ADAT = {coef:.3f} × ADNT + {intercept:.3f}")
    
    return models


def compute_positioning_residuals(df, models):
    """
    Compute ADAT residuals from fitted positioning regressions.
    
    Parameters
    ----------
    df : pd.DataFrame
        Full player dataset
    models : dict
        Fitted regression models from fit_positioning_regressions()
        
    Returns
    -------
    pd.DataFrame
        Clean dataframe with positioning features and computed residuals
    """
    result_df = df[[
        'steamid', 'player_name', 'team_clan_name', 'map_count',
        'adnt_rank_t', 'adat_rank_t', 'role_t',
        'adnt_rank_ct', 'adat_rank_ct', 'role_ct'
    ]].copy()
    
    for side in ['t', 'ct']:
        X = df[[f'adnt_rank_{side}']].values
        predicted_adat = models[side].predict(X)
        observed_adat = df[f'adat_rank_{side}'].values
        
        result_df[f'adat_residual_{side}'] = observed_adat - predicted_adat
    
    return result_df