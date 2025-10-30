from __future__ import annotations
import numpy as np
import pandas as pd


# --------------------------- SE calculators ----------------------------------

def _se_prop(p: np.ndarray, trials: np.ndarray) -> np.ndarray:
    """
    Binomial-style SE for a proportion on 0..1 scale:
        SE = sqrt( p * (1 - p) / trials )
    """
    p = np.clip(p.astype(float), 1e-6, 1 - 1e-6)
    trials = np.maximum(trials.astype(float), 1.0)
    return np.sqrt(p * (1.0 - p) / trials)


def _se_inv_sqrt(maps: np.ndarray, scale: float = 1.0) -> np.ndarray:
    """
    Proxy for SE of a mean-like metric when per-event variance is unknown:
        SE ~ scale / sqrt(maps)
    """
    maps = np.maximum(maps.astype(float), 1.0)
    return scale / np.sqrt(maps)


# --------------------------- Public API: tables -------------------------------

def _bin_quantile(y: np.ndarray, x_maps: np.ndarray, bins, q: float):
    """
    Return (bin_centers, qth_values, counts) for y grouped by map bins.
    q in [0,1], e.g., 0.75 for p75.
    """
    cats = pd.cut(x_maps, bins=bins, right=False, include_lowest=True)
    grp = pd.Series(y).groupby(cats, observed=True)

    qval = grp.quantile(q, interpolation="linear")
    counts = grp.size()

    iv = cats.categories if hasattr(cats, "categories") else cats.cat.categories
    qval = qval.reindex(iv)
    counts = counts.reindex(iv).fillna(0).astype(int)
    centers = np.array([(b.left + b.right) / 2.0 for b in iv], dtype=float)
    return centers, qval.values.astype(float), counts.values.astype(int)


def rates_quantile_table(
    df: pd.DataFrame,
    bins,
    k_oap: float, k_podt: float, k_pokt: float,
    q: float = 0.75,
    map_col: str = "map_count",
) -> pd.DataFrame:
    """
    Quantile-by-bin SE for rate metrics, expressed in percentage points.
    Columns: metric, bin_center, y, n, q
    """
    maps = df[map_col].astype(float).values
    p_oap  = (df["oap_overall" ].astype(float) / 100.0).values
    p_podt = (df["podt_overall"].astype(float) / 100.0).values
    p_pokt = (df["pokt_overall"].astype(float) / 100.0).values

    se_oap  = _se_prop(p_oap,  k_oap  * maps) * 100.0
    se_podt = _se_prop(p_podt, k_podt * maps) * 100.0
    se_pokt = _se_prop(p_pokt, k_pokt * maps) * 100.0

    pieces = []
    for metric, se in [("oap_overall", se_oap), ("podt_overall", se_podt), ("pokt_overall", se_pokt)]:
        centers, qvals, counts = _bin_quantile(se, maps, bins, q=q)
        out = pd.DataFrame({"metric": metric, "bin_center": centers, "y": qvals, "n": counts, "q": q})
        pieces.append(out)
    return pd.concat(pieces, ignore_index=True)


def tapd_quantile_table(
    df: pd.DataFrame,
    bins,
    q: float = 0.75,
    map_col: str = "map_count",
    scale: str | None = None,  # None | "relative_ref"
    ref_mask: pd.Series | None = None,
) -> tuple[pd.DataFrame, float | None]:
    """
    Quantile-by-bin for tapd proxy. If scale=None, returns arbitrary units.
    If scale="relative_ref", divides by median tapd_overall on ref_mask and expresses %.
    Returns (table, reference_value or None).
    """
    maps = df[map_col].astype(float).values
    se_tapd = _se_inv_sqrt(maps, scale=1.0)

    centers, qvals, counts = _bin_quantile(se_tapd, maps, bins, q=q)

    ref_value = None
    if scale == "relative_ref":
        assert ref_mask is not None, "ref_mask required when scale='relative_ref'"
        ref_series = df.loc[ref_mask, "tapd_overall"].astype(float)
        ref_value = float(ref_series.median()) if not ref_series.empty else np.nan
        qvals = (qvals / ref_value) * 100.0

    tbl = pd.DataFrame({
        "metric": "tapd_overall",
        "bin_center": centers,
        "y": qvals,
        "n": counts,
        "q": q,
        "unit": "% of typical tapd" if scale == "relative_ref" else "proxy units",
    })
    return tbl, ref_value

def coverage_by_maps(df: pd.DataFrame, bins, map_col: str = "map_count") -> pd.DataFrame:
    """
    Count players per map_count bin. Returns: bin_left, bin_right, bin_center, n
    """
    x = df[map_col].to_numpy(dtype=float)
    cats = pd.cut(x, bins=bins, right=False, include_lowest=True)
    counts = pd.Series(1, index=df.index).groupby(cats, observed=True).sum()

    iv = cats.categories if hasattr(cats, "categories") else cats.cat.categories
    counts = counts.reindex(iv).fillna(0).astype(int)
    centers = np.array([(b.left + b.right) / 2.0 for b in iv], dtype=float)

    out = pd.DataFrame({
        "bin_left": [b.left for b in iv],
        "bin_right": [b.right for b in iv],
        "bin_center": centers,
        "n": counts.values,
    })
    return out


def retained_summary(df: pd.DataFrame, min_maps: int, map_col: str = "map_count") -> dict:
    """
    Return simple retention stats at MIN_MAPS.
    """
    total = int(df.shape[0])
    retained = int((df[map_col] >= min_maps).sum())
    return {
        "total": total,
        "retained": retained,
        "retained_pct": 100.0 * retained / total if total else 0.0,
        "min_maps": int(min_maps),
    }

