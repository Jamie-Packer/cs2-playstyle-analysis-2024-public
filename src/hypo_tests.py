# src/hypo_tests.py

from __future__ import annotations
from typing import Optional, Tuple, Sequence
import numpy as np
import pandas as pd
from scipy.stats import t as student_t
from scipy import stats


def _bootstrap_mean_ci(
    values: np.ndarray,
    n_boot: int = 5000,
    ci: float = 0.95,
    random_state: Optional[int] = 42,
) -> Tuple[float, float]:
    """Basic nonparametric bootstrap CI for the mean; returns (lo, hi)."""
    x = np.asarray(values, dtype=float)
    x = x[~np.isnan(x)]
    n = x.size
    if n < 2:
        return (np.nan, np.nan)
    rng = np.random.default_rng(random_state)
    boot_means = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        sample = x[rng.integers(0, n, n, endpoint=False)]
        boot_means[b] = sample.mean()
    alpha = (1.0 - ci) / 2.0
    return float(np.quantile(boot_means, alpha)), float(np.quantile(boot_means, 1 - alpha))


def _paired_t_one_sided(deltas: np.ndarray) -> Tuple[float, float]:
    """
    One-sided paired t-test on deltas (T - CT).
    Tests H1: mean(delta) > 0. Returns (t_stat, p_one_sided).
    """
    x = np.asarray(deltas, dtype=float)
    x = x[~np.isnan(x)]
    n = x.size
    if n < 2:
        return (np.nan, np.nan)
    sd = x.std(ddof=1)
    if sd == 0.0:
        return (np.nan, np.nan)
    t_stat = x.mean() / (sd / np.sqrt(n))
    p = student_t.sf(t_stat, df=n - 1)
    return float(t_stat), float(p)


def test_side_trade_increase(
    df: pd.DataFrame,
    min_maps: int,
    map_count_col: str = "map_count",
    random_state: int = 42,
) -> pd.DataFrame:
    """
    #1 T trades more — paired, one-sided test on player-level deltas (T - CT)
    for PODT and POKT in a WIDE dataset.

    Assumes columns (case-insensitive match done internally):
      - 'podt_t','podt_ct'
      - 'pokt_t','pokt_ct'
      - 'map_count' (the same for both sides; used only to filter)

    Steps:
      1) Filter players with map_count >= min_maps.
      2) For each metric (PODT, POKT):
         - Keep rows where both side columns are non-null.
         - Compute delta = T - CT per player.
         - Report n, mean_delta, 95% bootstrap CI, one-sided paired t-test p-value.

    Returns DataFrame with columns:
      ['feature','n','mean_delta','ci95_lo','ci95_hi','t_stat','p_one_sided','alternative','test','note']
    """
    # Case-insensitive column resolver
    cols_lower = {c.lower(): c for c in df.columns}

    required = ["podt_t", "podt_ct", "pokt_t", "pokt_ct", map_count_col.lower()]
    missing = [r for r in required if r not in cols_lower]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    map_col = cols_lower[map_count_col.lower()]
    dff = df.loc[df[map_col] >= int(min_maps)].copy()

    def _compute(feature_base: str):
        col_t = cols_lower[f"{feature_base}_t"]
        col_ct = cols_lower[f"{feature_base}_ct"]
        sub = dff.loc[dff[col_t].notna() & dff[col_ct].notna(), [col_t, col_ct]]
        if sub.shape[0] < 2:
            return {
                "feature": feature_base.upper(),
                "n": int(sub.shape[0]),
                "mean_delta": np.nan,
                "ci95_lo": np.nan,
                "ci95_hi": np.nan,
                "t_stat": np.nan,
                "p_one_sided": np.nan,
                "alternative": "T > CT",
                "test": "paired_t (mean Δ)",
                "note": f"Players with ≥{min_maps} maps; insufficient paired rows",
            }
        deltas = sub[col_t].to_numpy(dtype=float) - sub[col_ct].to_numpy(dtype=float)
        mean_delta = float(np.mean(deltas))
        ci_lo, ci_hi = _bootstrap_mean_ci(deltas, n_boot=5000, ci=0.95, random_state=random_state)
        t_stat, p_one = _paired_t_one_sided(deltas)
        return {
            "feature": feature_base.upper(),
            "n": int(deltas.size),
            "mean_delta": mean_delta,
            "ci95_lo": ci_lo,
            "ci95_hi": ci_hi,
            "t_stat": t_stat,
            "p_one_sided": p_one,
            "alternative": "T > CT",
            "test": "paired_t (mean Δ)",
            "note": f"Players with ≥{min_maps} maps; Δ = {col_t} − {col_ct}",
        }

    rows = [
        _compute("podt"),
        _compute("pokt"),
    ]
    return pd.DataFrame(rows, columns=[
        "feature","n","mean_delta","ci95_lo","ci95_hi","t_stat","p_one_sided","alternative","test","note"
    ])


def _iqr(x: np.ndarray) -> float:
    """Interquartile range (nan-safe)."""
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if x.size == 0:
        return np.nan
    q1, q3 = np.quantile(x, [0.25, 0.75])
    return float(q3 - q1)


def _paired_bootstrap_log_ratio_sd(
    ct_vals: np.ndarray,
    t_vals: np.ndarray,
    n_boot: int = 5000,
    ci: float = 0.95,
    random_state: Optional[int] = 42,
) -> Tuple[float, float]:
    """Bootstrap CI for SD ratio (CT/T) via paired bootstrap on log-ratios."""
    assert ct_vals.shape == t_vals.shape
    x_ct = np.asarray(ct_vals, dtype=float)
    x_t = np.asarray(t_vals, dtype=float)
    mask = ~np.isnan(x_ct) & ~np.isnan(x_t)
    x_ct, x_t = x_ct[mask], x_t[mask]
    n = x_ct.size
    if n < 3:
        return (np.nan, np.nan)

    rng = np.random.default_rng(random_state)
    logs = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        idx = rng.integers(0, n, n, endpoint=False)
        sd_ct = np.std(x_ct[idx], ddof=1)
        sd_t = np.std(x_t[idx], ddof=1)
        if sd_ct <= 0 or sd_t <= 0:
            logs[b] = np.nan
        else:
            logs[b] = np.log(sd_ct / sd_t)

    logs = logs[~np.isnan(logs)]
    if logs.size == 0:
        return (np.nan, np.nan)

    alpha = (1.0 - ci) / 2.0
    lo_log = np.quantile(logs, alpha)
    hi_log = np.quantile(logs, 1 - alpha)
    return (float(np.exp(lo_log)), float(np.exp(hi_log)))


def _paired_permutation_p_iqr(
    ct_vals: np.ndarray,
    t_vals: np.ndarray,
    n_perm: int = 10000,
    alternative: str = "less",  # CT spread < T spread
    random_state: Optional[int] = 42,
) -> float:
    """
    One-sided paired permutation p-value on IQR difference d = IQR_CT - IQR_T.
    Paired sign flips: for each player, with p=0.5, swap (CT, T).
    Returns p = P(d_perm <= d_obs) if alternative='less'.
    """
    assert ct_vals.shape == t_vals.shape
    x_ct = np.asarray(ct_vals, dtype=float)
    x_t = np.asarray(t_vals, dtype=float)
    mask = ~np.isnan(x_ct) & ~np.isnan(x_t)
    x_ct, x_t = x_ct[mask], x_t[mask]
    n = x_ct.size
    if n < 3:
        return np.nan

    d_obs = _iqr(x_ct) - _iqr(x_t)

    rng = np.random.default_rng(random_state)
    count = 0
    total = 0

    for _ in range(n_perm):
        flips = rng.integers(0, 2, n, endpoint=False).astype(bool)
        ct_perm = np.where(flips, x_t, x_ct)
        t_perm = np.where(flips, x_ct, x_t)
        d_perm = _iqr(ct_perm) - _iqr(t_perm)
        total += 1
        if alternative == "less":
            if d_perm <= d_obs:
                count += 1
        elif alternative == "greater":
            if d_perm >= d_obs:
                count += 1
        else:
            raise ValueError("alternative must be 'less' or 'greater'")

    return float((count + 1) / (total + 1))  # small-sample correction
  

def test_side_variability_ct_less(
    df: pd.DataFrame,
    min_maps: int,
    features: Sequence[str] = ("oap", "tapd", "podt", "pokt"),
    map_count_col: str = "map_count",
    n_boot: int = 5000,
    n_perm: int = 10000,
    random_state: Optional[int] = 42,
) -> pd.DataFrame:
    """
    CT tighter spread — effect size via SD ratio (CT/T) + one-sided paired permutation on IQR.
    Assumes wide schema with '{feat}_ct', '{feat}_t' and 'map_count' for filtering.
    """
    cols_lower = {c.lower(): c for c in df.columns}
    if map_count_col.lower() not in cols_lower:
        raise KeyError(f"Column '{map_count_col}' not found for MIN_MAPS filtering.")
    map_col = cols_lower[map_count_col.lower()]
    dff = df.loc[df[map_col] >= int(min_maps)].copy()

    rows = []
    for base in features:
        base_l = base.lower()
        col_ct_l = f"{base_l}_ct"
        col_t_l = f"{base_l}_t"
        if col_ct_l not in cols_lower or col_t_l not in cols_lower:
            rows.append({
                "feature": base.upper(),
                "n_players": 0,
                "sd_ct": np.nan,
                "sd_t": np.nan,
                "sd_ratio_ct_over_t": np.nan,
                "ci95_lo_ratio": np.nan,
                "ci95_hi_ratio": np.nan,
                "iqr_ct": np.nan,
                "iqr_t": np.nan,
                "p_one_sided_perm_IQR": np.nan,
            })
            continue

        col_ct = cols_lower[col_ct_l]
        col_t = cols_lower[col_t_l]
        sub = dff.loc[dff[col_ct].notna() & dff[col_t].notna(), [col_ct, col_t]]
        x_ct = sub[col_ct].to_numpy(dtype=float)
        x_t = sub[col_t].to_numpy(dtype=float)
        n = x_ct.size

        if n < 3:
            rows.append({
                "feature": base.upper(),
                "n_players": int(n),
                "sd_ct": np.nan,
                "sd_t": np.nan,
                "sd_ratio_ct_over_t": np.nan,
                "ci95_lo_ratio": np.nan,
                "ci95_hi_ratio": np.nan,
                "iqr_ct": np.nan,
                "iqr_t": np.nan,
                "p_one_sided_perm_IQR": np.nan,
            })
            continue

        sd_ct = float(np.std(x_ct, ddof=1))
        sd_t = float(np.std(x_t, ddof=1))
        sd_ratio = sd_ct / sd_t if sd_t > 0 else np.nan
        ci_lo, ci_hi = _paired_bootstrap_log_ratio_sd(x_ct, x_t, n_boot=n_boot, ci=0.95, random_state=random_state)

        iqr_ct = _iqr(x_ct)
        iqr_t = _iqr(x_t)
        p_perm = _paired_permutation_p_iqr(x_ct, x_t, n_perm=n_perm, alternative="less", random_state=random_state)

        rows.append({
            "feature": base.upper(),
            "n_players": int(n),
            "sd_ct": sd_ct,
            "sd_t": sd_t,
            "sd_ratio_ct_over_t": sd_ratio,
            "ci95_lo_ratio": ci_lo,
            "ci95_hi_ratio": ci_hi,
            "iqr_ct": iqr_ct,
            "iqr_t": iqr_t,
            "p_one_sided_perm_IQR": p_perm,
        })

    res = pd.DataFrame(rows, columns=[
        "feature","n_players","sd_ct","sd_t","sd_ratio_ct_over_t",
        "ci95_lo_ratio","ci95_hi_ratio","iqr_ct","iqr_t","p_one_sided_perm_IQR"
    ])
    return res


# ======================================================== Role Based ========================================================

def test_role_contrast(
    df: pd.DataFrame,
    role_a: str,
    role_b: str,
    min_maps: int,
    map_count_col: str = "map_count",
    random_state: Optional[int] = 42,
) -> pd.DataFrame:
    """
    Compare two roles on behavioral features using Welch's t-test and Mann-Whitney U.
    
    Roles must include side suffix (_t or _ct) to handle AWPers on both sides.
    Tests all 6 behavioral features: OAP, TAPD, PODT, POKT, ADNT_rank, ADAT_rank.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Wide dataset with behavioral features and role labels.
    role_a, role_b : str
        Role labels WITH side suffix (e.g., 'Spacetaker_t', 'Lurker_t', 'Rotator_ct').
    min_maps : int
        Minimum map count threshold for player inclusion.
    map_count_col : str
        Column name for map count filtering.
    random_state : Optional[int]
        Seed for reproducibility (currently unused, reserved for future extensions).
    
    Returns:
    --------
    pd.DataFrame with columns:
        ['feature', 'n_{role_a}', 'n_{role_b}', 'mean_{role_a}', 'mean_{role_b}', 
         'cohens_d', 't_stat', 'p_welch', 'U_stat', 'p_mw']
    
    Example:
    --------
    >>> test_role_contrast(df, 'Spacetaker_t', 'Lurker_t', min_maps=40)
    """
    def _format_p_value(p: float) -> str:
        """Format p-value: use decimal notation if >= 0.0001, else scientific."""
        if np.isnan(p):
            return "NaN"
        if p >= 0.0001:
            return f"{p:.4f}"
        else:
            return f"{p:.2e}"
    
    # Validate role format
    if not (role_a.endswith('_t') or role_a.endswith('_ct')):
        raise ValueError(f"role_a '{role_a}' must end with '_t' or '_ct'")
    if not (role_b.endswith('_t') or role_b.endswith('_ct')):
        raise ValueError(f"role_b '{role_b}' must end with '_t' or '_ct'")
    
    # Extract side and role name
    if role_a.endswith('_ct'):
        side_a = '_ct'
        role_name_a = role_a[:-3]
    else:
        side_a = '_t'
        role_name_a = role_a[:-2]
    
    if role_b.endswith('_ct'):
        side_b = '_ct'
        role_name_b = role_b[:-3]
    else:
        side_b = '_t'
        role_name_b = role_b[:-2]
    
    if side_a != side_b:
        raise ValueError(f"Roles must be from same side: {role_a} vs {role_b}")
    
    side = side_a
    role_col = f'role{side}'
    
    # Define all behavioral features for this side
    features = [
        f'oap{side}',
        f'tapd{side}',
        f'podt{side}',
        f'pokt{side}',
        f'adnt_rank{side}',
        f'adat_rank{side}'
    ]
    
    cols_lower = {c.lower(): c for c in df.columns}
    
    if role_col.lower() not in cols_lower:
        raise KeyError(f"Role column '{role_col}' not found.")
    if map_count_col.lower() not in cols_lower:
        raise KeyError(f"Map count column '{map_count_col}' not found.")
    
    role_col_actual = cols_lower[role_col.lower()]
    map_col = cols_lower[map_count_col.lower()]
    
    dff = df.loc[df[map_col] >= int(min_maps)].copy()
    
    rows = []
    for feat in features:
        feat_lower = feat.lower()
        if feat_lower not in cols_lower:
            continue
        
        feat_col = cols_lower[feat_lower]
        
        mask_a = (dff[role_col_actual] == role_name_a) & dff[feat_col].notna()
        mask_b = (dff[role_col_actual] == role_name_b) & dff[feat_col].notna()
        
        vals_a = dff.loc[mask_a, feat_col].to_numpy(dtype=float)
        vals_b = dff.loc[mask_b, feat_col].to_numpy(dtype=float)
        
        n_a = vals_a.size
        n_b = vals_b.size
        
        if n_a < 2 or n_b < 2:
            rows.append({
                'feature': feat.upper(),
                f'n_{role_name_a}': int(n_a),
                f'n_{role_name_b}': int(n_b),
                f'mean_{role_name_a}': np.nan if n_a == 0 else round(float(np.mean(vals_a)), 3),
                f'mean_{role_name_b}': np.nan if n_b == 0 else round(float(np.mean(vals_b)), 3),
                'cohens_d': np.nan,
                't_stat': np.nan,
                'p_welch': "NaN",
                'U_stat': np.nan,
                'p_mw': "NaN",
            })
            continue
        
        mean_a = round(float(np.mean(vals_a)), 3)
        mean_b = round(float(np.mean(vals_b)), 3)
        sd_a = float(np.std(vals_a, ddof=1))
        sd_b = float(np.std(vals_b, ddof=1))
        
        pooled_sd = np.sqrt(((n_a - 1) * sd_a**2 + (n_b - 1) * sd_b**2) / (n_a + n_b - 2))
        cohens_d = round((mean_a - mean_b) / pooled_sd, 3) if pooled_sd > 0 else np.nan
        
        t_stat, p_welch = stats.ttest_ind(vals_a, vals_b, equal_var=False)
        t_stat = round(float(t_stat), 3)
        
        U_stat, p_mw = stats.mannwhitneyu(vals_a, vals_b, alternative='two-sided')
        
        rows.append({
            'feature': feat.upper(),
            f'n_{role_name_a}': int(n_a),
            f'n_{role_name_b}': int(n_b),
            f'mean_{role_name_a}': mean_a,
            f'mean_{role_name_b}': mean_b,
            'cohens_d': cohens_d,
            't_stat': t_stat,
            'p_welch': _format_p_value(p_welch),
            'U_stat': float(U_stat),
            'p_mw': _format_p_value(p_mw),
        })
    
    return pd.DataFrame(rows)

# ======================================================= Role Distinctiveness ========================================================

def compute_role_distinctiveness(
    df: pd.DataFrame,
    side: str,
    min_maps: int,
    map_count_col: str = "map_count",
) -> pd.DataFrame:
    """
    Compute within-side role distinctiveness using Mahalanobis distance.
    
    For each role on the specified side, calculates how distinct it is from
    all other roles on that side, plus per-feature contributions to distinctiveness.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Wide dataset with behavioral features and role labels.
    side : str
        Side to analyze: 't' or 'ct'.
    min_maps : int
        Minimum map count threshold for player inclusion.
    map_count_col : str
        Column name for map count filtering.
    
    Returns:
    --------
    pd.DataFrame with columns:
        ['role', 'n', 'mahal_d_sq', 'oap_contrib', 'tapd_contrib', 'podt_contrib',
         'pokt_contrib', 'adnt_contrib', 'adat_contrib']
    
    Sorted by mahal_d_sq descending (most distinct roles first).
    
    Notes:
    ------
    - mahal_d_sq: Squared Mahalanobis distance from role mean to other-roles mean
    - *_contrib: Squared standardized difference per feature (effect size contribution)
    - Contributions approximately sum to mahal_d_sq (exact match requires covariance matrix)
    """
    if side.lower() not in ['t', 'ct']:
        raise ValueError("side must be 't' or 'ct'")
    
    side_suffix = f"_{side.lower()}"
    role_col = f"role{side_suffix}"
    
    features = [
        f'oap{side_suffix}',
        f'tapd{side_suffix}',
        f'podt{side_suffix}',
        f'pokt{side_suffix}',
        f'adnt_rank{side_suffix}',
        f'adat_rank{side_suffix}'
    ]
    
    cols_lower = {c.lower(): c for c in df.columns}
    
    if role_col.lower() not in cols_lower:
        raise KeyError(f"Role column '{role_col}' not found.")
    if map_count_col.lower() not in cols_lower:
        raise KeyError(f"Map count column '{map_count_col}' not found.")
    
    role_col_actual = cols_lower[role_col.lower()]
    map_col = cols_lower[map_count_col.lower()]
    
    # Resolve feature columns
    feat_cols = []
    for feat in features:
        if feat.lower() not in cols_lower:
            raise KeyError(f"Feature '{feat}' not found.")
        feat_cols.append(cols_lower[feat.lower()])
    
    # Filter stable players and valid roles
    dff = df.loc[df[map_col] >= int(min_maps)].copy()
    dff = dff.loc[dff[role_col_actual].notna() & (dff[role_col_actual] != '')]
    
    # Drop rows with any missing features
    dff = dff.loc[dff[feat_cols].notna().all(axis=1)]
    
    if dff.shape[0] < 10:
        raise ValueError(f"Insufficient data: only {dff.shape[0]} valid players after filtering.")
    
    # Get feature matrix
    X = dff[feat_cols].to_numpy(dtype=float)
    roles = dff[role_col_actual].to_numpy()
    
    unique_roles = np.unique(roles)
    
    if len(unique_roles) < 2:
        raise ValueError(f"Need at least 2 roles for distinctiveness; found {len(unique_roles)}.")
    
    # Compute pooled covariance matrix across all players on this side
    cov_matrix = np.cov(X, rowvar=False)
    
    # Regularization in case of near-singularity
    reg = 1e-6
    cov_matrix_reg = cov_matrix + reg * np.eye(cov_matrix.shape[0])
    
    try:
        cov_inv = np.linalg.inv(cov_matrix_reg)
    except np.linalg.LinAlgError:
        raise ValueError("Covariance matrix is singular even after regularization.")
    
    rows = []
    for role in unique_roles:
        mask_role = roles == role
        mask_others = ~mask_role
        
        n_role = mask_role.sum()
        if n_role < 2:
            continue
        
        X_role = X[mask_role]
        X_others = X[mask_others]
        
        mean_role = X_role.mean(axis=0)
        mean_others = X_others.mean(axis=0)
        
        diff = mean_role - mean_others
        
        # Mahalanobis D²
        mahal_d_sq = float(diff @ cov_inv @ diff)
        
        # Per-feature contributions: squared standardized difference
        pooled_var = np.diag(cov_matrix)
        pooled_sd = np.sqrt(pooled_var)
        
        contribs = ((diff / pooled_sd) ** 2)
        
        rows.append({
            'role': str(role),
            'n': int(n_role),
            'mahal_d_sq': round(mahal_d_sq, 3),
            'oap_contrib': round(contribs[0], 3),
            'tapd_contrib': round(contribs[1], 3),
            'podt_contrib': round(contribs[2], 3),
            'pokt_contrib': round(contribs[3], 3),
            'adnt_contrib': round(contribs[4], 3),
            'adat_contrib': round(contribs[5], 3),
        })
    
    result = pd.DataFrame(rows, columns=[
        'role', 'n', 'mahal_d_sq', 'oap_contrib', 'tapd_contrib', 'podt_contrib',
        'pokt_contrib', 'adnt_contrib', 'adat_contrib'
    ])
    
    # Sort by distinctiveness descending
    result = result.sort_values('mahal_d_sq', ascending=False).reset_index(drop=True)
    
    return result