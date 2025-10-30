from __future__ import annotations
from typing import Optional, Literal, List
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import style
from style import get_role_colour

from viz_core import (
    FEATURE_LABELS_SHORT,
    FEATURE_LEGEND_FULL,
    _default_features_base,
    _feature_spec,
)

# ========================================================== EDA Stability ==========================================
# Plots for estimated standard errors of rates (opener/trade) and TAPD

def plot_rates_quantile(
    rates_tbl: pd.DataFrame,
    q: float = 0.75,
    tol_pp: float | None = 2.0,
    min_maps: int | None = None,
    savepath=None,
    savepath_svg=None,
):
    fig, ax = plt.subplots(figsize=(9.5, 5.2), constrained_layout=True)

    for metric in ["oap_overall","podt_overall","pokt_overall"]:
        d = rates_tbl.query("metric == @metric").sort_values("bin_center")
        ax.plot(d["bin_center"], d["y"], marker="o",
                label=f"{metric} — SE p{int(q*100)} (pp)")

    x_min, x_max = rates_tbl["bin_center"].min(), rates_tbl["bin_center"].max()
    if tol_pp is not None:
        ax.hlines(tol_pp, x_min, x_max, colors="red", linestyles="dashed", linewidth=1.2,
                  label=f"tolerance = {tol_pp:.1f} pp")
    if min_maps is not None:
        ax.vlines(min_maps, 0, ax.get_ylim()[1], colors="red", linestyles="dashed", linewidth=1.2,
                  label=f"MIN_MAPS = {min_maps}")

    ax.set_title(f"Estimated standard error (p{int(q*100)}) for opener/trade rates")
    ax.set_xlabel("Map Count (bin center)")
    ax.set_ylabel("Estimated standard error (percentage points)")
    ax.grid(True, alpha=0.25)
    ax.legend(ncol=2, frameon=False)
    if savepath is not None:
        plt.savefig(savepath)
    if savepath_svg is not None:
        plt.savefig(savepath_svg, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.show()


def plot_tapd_quantile(
    tapd_tbl: pd.DataFrame,
    q: float = 0.75,
    min_maps: int | None = None,
    savepath=None,
    savepath_svg=None,
):
    d = tapd_tbl.sort_values("bin_center")
    fig, ax = plt.subplots(figsize=(9, 4.5), constrained_layout=True)

    ax.plot(d["bin_center"], d["y"], marker="o", linestyle="--",
            label=f"tapd_overall — SE p{int(q*100)}")

    x_min, x_max = d["bin_center"].min(), d["bin_center"].max()
    if min_maps is not None:
        ax.vlines(min_maps, 0, ax.get_ylim()[1], colors="red", linestyles="dashed", linewidth=1.2,
                  label=f"MIN_MAPS = {min_maps}")

    ax.set_title(f"Estimated relative standard error (p{int(q*100)}) for 'Time Alive Per Death'")
    # shorter, wrapped label to avoid clipping
    ax.set_ylabel("Est. Relative SE\n(proxy units; 1/√maps)")
    ax.set_xlabel("Map Count (bin center)")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    if savepath is not None:
        plt.savefig(savepath)
    if savepath_svg is not None:
        plt.savefig(savepath_svg, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.show()


# plot histogram of map_count with optional MIN_MAPS line
def plot_mapcount_hist(df: pd.DataFrame, bins, min_maps: int | None = None, savepath=None, savepath_svg=None):
    """
    Histogram of players by map_count over provided bins. Optional MIN_MAPS line.
    """

    x = df["map_count"].to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(9.6, 4.6), constrained_layout=True)

    ax.hist(x, bins=bins, edgecolor="none", alpha=0.9)
    if min_maps is not None:
        ax.vlines(min_maps, 0, ax.get_ylim()[1], colors="red", linestyles="dashed", linewidth=1.2,
                  label=f"MIN_MAPS = {min_maps}")
        ax.legend(frameon=False)

    ax.set_title("Player coverage by map count")
    ax.set_xlabel("map_count (bin edges)")
    ax.set_ylabel("Players")
    ax.grid(True, alpha=0.25)

    if savepath is not None:
        plt.savefig(savepath)
    if savepath_svg is not None:
        plt.savefig(savepath_svg, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.show()


# ========================================================= Feature KDEs =================================================

# ---- internal KDE wrapper (seaborn) ----
def _kde(ax, x, *, clip=None, bw_adjust=1.0, label=None, color=None,
         lw=2.0, alpha=1.0, fill_alpha: float = 0.0, show_legend=True):
    vals = pd.to_numeric(pd.Series(x), errors="coerce").dropna().values
    if vals.size == 0:
        return
    # 1) filled area (no legend entry)
    if fill_alpha and fill_alpha > 0:
        sns.kdeplot(
            x=vals, ax=ax, clip=clip, bw_adjust=bw_adjust, fill=True,
            linewidth=0, color=color, alpha=fill_alpha, label=None
        )
    # 2) line on top (legend label here)
    sns.kdeplot(
        x=vals, ax=ax, clip=clip, bw_adjust=bw_adjust, fill=False,
        linewidth=lw, color=color, alpha=alpha, label=(label if show_legend else None)
    )



# ==========================================================
# 1) Single figure: 2×3 subplots, each shows CT vs T KDE
# ==========================================================

def plot_kdes_side_compare(
    df: pd.DataFrame,
    features=None,
    bw_adjust: float = 1.0,
    min_map_count: int | None = None,
    savepath: str | None = None,
    savepath_svg: str | None = None,
    mean_features: tuple[str, ...] = ("tapd", "oap", "podt", "pokt"),
):
    """Plot a 2×3 KDE grid comparing CT vs T for each feature.
    Filled densities; dashed mean lines only for `mean_features`."""
    data = df.copy()
    if min_map_count is not None and "map_count" in data.columns:
        data = data.loc[data["map_count"] >= int(min_map_count)].copy()

    ct_color = "#2060aaff"
    t_color  = "#de9b35ff"
    fill_a = 0.22

    feat_order = ["oap", "podt", "pokt", "tapd", "adnt_rank", "adat_rank"]
    feats = [f for f in (features or feat_order)]
    ct_spec = _feature_spec("ct", feats)
    t_spec  = _feature_spec("t",  feats)

    fig, axes = plt.subplots(2, 3, figsize=(13.2, 7.4), constrained_layout=True)
    axes = axes.ravel()
    fig.patch.set_facecolor(plt.rcParams["figure.facecolor"])
    for ax in axes:
        ax.set_facecolor(plt.rcParams["axes.facecolor"])

    for ax, ((ct_col, label, xlim), (t_col, _, _)), feat_name in zip(
        axes, zip(ct_spec, t_spec), feats
    ):
        ct_vals = pd.to_numeric(data[ct_col], errors="coerce").dropna()
        t_vals  = pd.to_numeric(data[t_col],  errors="coerce").dropna()

        # Filled KDEs + line
        _kde(ax, ct_vals, clip=xlim, bw_adjust=bw_adjust,
             label="CT side", color=ct_color, lw=2.2, alpha=0.95, fill_alpha=fill_a)
        _kde(ax, t_vals,  clip=xlim, bw_adjust=bw_adjust,
             label="T side",  color=t_color,  lw=2.2, alpha=0.95, fill_alpha=fill_a)

        # Means only for selected features (no legend entry)
        if feat_name in mean_features:
            if not ct_vals.empty:
                ax.axvline(ct_vals.mean(), color=ct_color, linestyle="--", linewidth=1.6, alpha=0.95, zorder=2)
            if not t_vals.empty:
                ax.axvline(t_vals.mean(),  color=t_color,  linestyle="--", linewidth=1.6, alpha=0.95, zorder=2)

        ax.set_title(label)
        if xlim is not None:
            ax.set_xlim(xlim)
        ax.grid(True, alpha=0.25)

        # Per-subplot legend (inside)
        ax.legend(frameon=False, fontsize=9, loc="upper right")

    fig.suptitle("Feature Distributions by Side (CT vs T) — KDEs",
                 fontsize=plt.rcParams.get("figure.titlesize", 18))

    if savepath:
        plt.savefig(savepath, bbox_inches="tight", facecolor=fig.get_facecolor())
    if savepath_svg:
        plt.savefig(savepath_svg, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.show()




def summarize_side_stats(
    df: pd.DataFrame,
    features=None,
    min_map_count: int | None = None,
    round_to: int | None = None,
) -> pd.DataFrame:
    """
    Return a compact side-by-side summary table for the KDE figures.
    Columns: Feature label, CT μ, CT σ, T μ, T σ, n (shown once).
    """
    data = df.copy()
    if min_map_count is not None and "map_count" in data.columns:
        data = data.loc[data["map_count"] >= int(min_map_count)].copy()

    feat_order = ["tapd", "oap", "podt", "pokt", "adnt_rank", "adat_rank"]
    feats = list(features) if features is not None else feat_order

    rows = []
    for f in feats:
        ct_col, label, _ = _feature_spec("ct", [f])[0]
        t_col,  _,     _ = _feature_spec("t",  [f])[0]

        ct = pd.to_numeric(data[ct_col], errors="coerce").dropna()
        t  = pd.to_numeric(data[t_col],  errors="coerce").dropna()

        rows.append({
            "Feature": label,
            "CT μ":   float(ct.mean()) if ct.size else float("nan"),
            "CT σ":   float(ct.std(ddof=1)) if ct.size > 1 else float("nan"),
            "T μ":    float(t.mean())  if t.size else float("nan"),
            "T σ":    float(t.std(ddof=1))  if t.size > 1 else float("nan"),
            "n":      int(min(ct.size, t.size)),  # show once; usually identical
        })

    out = pd.DataFrame(rows, index=feats)
    # keep the original logical order
    out = out.loc[feats].reset_index(drop=True)

    if round_to is not None:
        num_cols = ["CT μ", "CT σ", "T μ", "T σ"]
        out[num_cols] = out[num_cols].round(round_to)
    return out



# ==========================================================
# 2) One figure per side: roles overlaid in each subplot
# ==========================================================

def plot_kdes_roles_by_side(
    df: pd.DataFrame,
    side: str,
    role_col: str | None = None,
    roles_order=None,             # used as a TOGGLE list: only these roles are included (if provided)
    min_role_n: int = 8,
    bw_adjust: float = 1.0,
    min_map_count: int | None = None,
    savepath: str | None = None,
    savepath_svg: str | None = None,
):
    """ 
    Plot feature KDEs for a given side, overlaid by role.
    """
    s = str(side).lower()
    if s not in {"t", "ct"}:
        raise ValueError("side must be 't' or 'ct'")

    rcol = role_col or f"role_{s}"
    if rcol not in df.columns:
        raise KeyError(f"Role column '{rcol}' not found in df.")

    data = df.copy()
    if min_map_count is not None and "map_count" in data.columns:
        data = data.loc[data["map_count"] >= int(min_map_count)].copy()

    counts = data[rcol].value_counts(dropna=False).to_dict()
    eligible = [r for r, n in counts.items() if (pd.notna(r) and n >= min_role_n)]

    # roles_order == toggle: include only those listed (intersection with eligible)
    if roles_order is not None:
        roles = [r for r in roles_order if r in eligible]
    else:
        roles = sorted(eligible, key=lambda r: (-counts[r], str(r)))

    spec = _feature_spec(s, None)
    fill_a = 0.22  # semi-transparent fill

    fig, axes = plt.subplots(2, 3, figsize=(13.6, 7.8), constrained_layout=True)
    axes = axes.ravel()
    fig.set_constrained_layout_pads(w_pad=0.04, h_pad=0.1, wspace=0.08, hspace=0.10)
    fig.patch.set_facecolor(plt.rcParams["figure.facecolor"])
    for ax in axes:
        ax.set_facecolor(plt.rcParams["axes.facecolor"])

    for ax, (col, label, xlim) in zip(axes, spec):
        for role in roles:
            x = data.loc[data[rcol] == role, col]
            if x.notna().sum() == 0:
                continue
            _kde(ax, x, clip=xlim, bw_adjust=bw_adjust, label=str(role),
                 color=get_role_colour(role), lw=2.0, alpha=0.95, fill_alpha=fill_a,
                 show_legend=False)

        ax.set_title(label)
        if xlim is not None:
            ax.set_xlim(xlim)
        ax.grid(True, alpha=0.25)

    # top-center legend (only one entry per role)
    if roles:
        handles = []
        labels = []
        # build line handles in legend colors
        for r in roles:
            line = plt.Line2D([], [], color=get_role_colour(r), linewidth=2.0, label=str(r))
            handles.append(line)
            labels.append(str(r))
        ncol = min(len(labels), 6)
        fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.06),
                   ncol=ncol, frameon=False, title="Role")

    fig.suptitle(f"Feature Distributions by Role (KDEs) — Side = {s.upper()}", y=1.085)

    if savepath:
        plt.savefig(savepath, bbox_inches="tight", facecolor=fig.get_facecolor())
    if savepath_svg:
        plt.savefig(savepath_svg, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.show()


def summarize_role_stats_by_side(
    df: pd.DataFrame,
    side: str,
    role_col: str | None = None,
    roles_order=None,          # acts as a toggle list
    min_role_n: int = 8,
    features=None,
    min_map_count: int | None = None,
    round_to: int | None = None,
    wide: bool = False,        # True -> columns per role ("AWPer μ", "AWPer σ", ...)
) -> pd.DataFrame:
    """
    Summary stats by role for a given side ('t' or 'ct').
    Returns mean, std, n per feature x role. Set wide=True for a wide, report-style table.
    """
    s = str(side).lower()
    if s not in {"t", "ct"}:
        raise ValueError("side must be 't' or 'ct'")

    rcol = role_col or f"role_{s}"
    if rcol not in df.columns:
        raise KeyError(f"Role column '{rcol}' not found in df.")

    data = df.copy()
    if min_map_count is not None and "map_count" in data.columns:
        data = data.loc[data["map_count"] >= int(min_map_count)].copy()

    # choose roles
    counts = data[rcol].value_counts(dropna=False).to_dict()
    eligible = [r for r, n in counts.items() if (pd.notna(r) and n >= min_role_n)]
    roles = [r for r in (roles_order or []) if r in eligible] if roles_order is not None \
            else sorted(eligible, key=lambda r: (-counts[r], str(r)))

    feat_order = ["tapd", "oap", "podt", "pokt", "adnt_rank", "adat_rank"]
    feats = list(features) if features is not None else feat_order

    # build long table
    rows = []
    for f in feats:
        col, label, _ = _feature_spec(s, [f])[0]
        for role in roles:
            vals = pd.to_numeric(data.loc[data[rcol] == role, col], errors="coerce").dropna()
            rows.append({
                "Feature": label,
                "feature_key": f,   # keeps original order if labels change
                "Role": role,
                "μ": float(vals.mean()) if vals.size else float("nan"),
                "σ": float(vals.std(ddof=1)) if vals.size > 1 else float("nan"),
                "n": int(vals.size),
            })

    out = pd.DataFrame(rows)
    # keep logical ordering
    out["feature_order"] = out["feature_key"].map({k:i for i,k in enumerate(feats)})
    role_order_map = {r:i for i,r in enumerate(roles)}
    out["role_order"] = out["Role"].map(role_order_map)
    out = out.sort_values(["feature_order","role_order"], kind="stable")
    out = out.drop(columns=["feature_key","feature_order","role_order"])

    if round_to is not None:
        out[["μ","σ"]] = out[["μ","σ"]].round(round_to)

    if not wide:
        return out.reset_index(drop=True)

    # wide format: columns like "<Role> μ", "<Role> σ"
    wide_tbl = (out.pivot(index="Feature", columns="Role", values=["μ","σ"])
                  .sort_index(axis=1, level=1))
    # flatten columns
    flat_cols = []
    for stat, role in wide_tbl.columns:
        flat_cols.append(f"{role} {stat}")
    wide_tbl.columns = flat_cols
    wide_tbl = wide_tbl.reset_index()

    if round_to is not None:
        num_cols = [c for c in wide_tbl.columns if c != "Feature"]
        wide_tbl[num_cols] = wide_tbl[num_cols].round(round_to)

    return wide_tbl


# ========================================================== Radar Charts (modular, single-call API) ======================
# This is a hell of a lot of code for one plot, in the future it might be worth breaking it out into its own module or
# taking a simpler, static approach.


# ========================================================== Correlation Analysis ======================

def compute_feature_correlations(
    df: pd.DataFrame,
    min_maps: int,
    side: Literal["t", "ct"],
    features_base: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Compute Pearson correlation matrix for features on a given side.
    
    Parameters
    ----------
    df : pd.DataFrame
        Full dataset with map_count column and side-specific feature columns.
    min_maps : int
        Minimum map count threshold for stable players.
    side : Literal["t", "ct"]
        Which side to compute correlations for.
    features_base : Optional[List[str]]
        Base feature names (without side suffix). If None, uses default set.
        
    Returns
    -------
    pd.DataFrame
        Correlation matrix with uppercase feature labels as index and columns.
    """
    if features_base is None:
        features_base = _default_features_base()
    
    # Filter to stable players
    df_stable = df[df["map_count"] >= min_maps].copy()
    
    # Select side-specific columns
    feature_cols = [f"{feat}_{side}" for feat in features_base]
    
    # Compute Pearson correlations
    corr_matrix = df_stable[feature_cols].corr()
    
    # Rename to uppercase short labels for display
    labels = [feat.upper() for feat in features_base]
    corr_matrix.columns = labels
    corr_matrix.index = labels
    
    return corr_matrix


def plot_correlation_split_heatmap(
    corr_t: pd.DataFrame,
    corr_ct: pd.DataFrame,
    save_path: Optional[Path] = None,
    title: str = "CT vs T Side Feature Correlations",
    n_stable: Optional[int] = None,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Create a split-diagonal heatmap where each cell shows T-side correlations 
    in upper-right triangle and CT-side correlations in lower-left triangle.
    Only shows lower triangle of correlation matrix to avoid redundancy.
    
    Parameters
    ----------
    corr_t : pd.DataFrame
        Correlation matrix for T-side features (square, with matching index/columns).
    corr_ct : pd.DataFrame
        Correlation matrix for CT-side features (same shape as corr_t).
    save_path : Optional[Path]
        If provided, save figure as PNG and SVG to this path (without extension).
    title : str
        Figure title.
    n_stable : Optional[int]
        Number of stable players included (for subtitle).
        
    Returns
    -------
    tuple[plt.Figure, plt.Axes]
        The created figure and axes objects.
    """
    n = len(corr_t)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(11, 10))
    
    # Get colormap
    cmap = plt.cm.RdBu_r
    norm = plt.Normalize(vmin=-1, vmax=1)
    
    # Get side colors
    color_t = get_role_colour('T')
    color_ct = get_role_colour('CT')
    
    # Draw only lower triangle (including diagonal)
    for i in range(n):
        for j in range(i + 1):  # Only j <= i
            if i == j:
                # Diagonal: draw split square with T/CT colored triangles
                # Lower-left triangle (CT color)
                triangle_ct_diag = plt.Polygon(
                    [(j, n-1-i), (j+1, n-1-i), (j, n-i)],
                    facecolor=color_ct,
                    edgecolor='#666666',
                    linewidth=0.5
                )
                ax.add_patch(triangle_ct_diag)
                
                # Upper-right triangle (T color)
                triangle_t_diag = plt.Polygon(
                    [(j+1, n-1-i), (j+1, n-i), (j, n-i)],
                    facecolor=color_t,
                    edgecolor='#666666',
                    linewidth=0.5
                )
                ax.add_patch(triangle_t_diag)
                
                # Add T label (upper right)
                ax.text(j + 0.67, n-i-0.33, 'T',
                       ha='center', va='center',
                       fontsize=11, fontweight='bold',
                       color='white', alpha=0.9)
                
                # Add CT label (lower left)
                ax.text(j + 0.33, n-i-0.67, 'CT',
                       ha='center', va='center',
                       fontsize=11, fontweight='bold',
                       color='white', alpha=0.9)
            else:
                # Lower-left triangle (CT-side) - diagonal from bottom-left to top-right
                val_ct = corr_ct.iloc[i, j]
                color_ct_corr = cmap(norm(val_ct))
                triangle_ct = plt.Polygon(
                    [(j, n-1-i), (j+1, n-1-i), (j, n-i)],
                    facecolor=color_ct_corr,
                    edgecolor='#666666',
                    linewidth=0.5
                )
                ax.add_patch(triangle_ct)
                ax.text(j + 0.33, n-i-0.67, f'{val_ct:.2f}',
                       ha='center', va='center',
                       fontsize=9, color='white' if abs(val_ct) > 0.5 else 'black')
                
                # Upper-right triangle (T-side)
                val_t = corr_t.iloc[i, j]
                color_t_corr = cmap(norm(val_t))
                triangle_t = plt.Polygon(
                    [(j+1, n-1-i), (j+1, n-i), (j, n-i)],
                    facecolor=color_t_corr,
                    edgecolor='#666666',
                    linewidth=0.5
                )
                ax.add_patch(triangle_t)
                ax.text(j + 0.67, n-i-0.33, f'{val_t:.2f}',
                       ha='center', va='center',
                       fontsize=9, color='white' if abs(val_t) > 0.5 else 'black')
    
    # Set axis properties
    ax.set_xlim(0, n)
    ax.set_ylim(0, n)
    ax.set_aspect('equal')
    
    # Remove original feature suffixes for cleaner labels
    feature_labels = [label.replace('_RANK', '') for label in corr_t.columns]
    
    # Set ticks and labels (excluding unused positions)
    # X-axis: exclude last label (ADAT) since no squares appear in that column
    ax.set_xticks(np.arange(n - 1) + 0.5)
    ax.set_xticklabels(feature_labels[:-1], fontsize=11)
    
    # Y-axis: exclude top label (TAPD) since no squares appear in that row
    ax.set_yticks(np.arange(n - 1) + 0.5)
    ax.set_yticklabels(feature_labels[::-1][:-1], fontsize=11)
    
    # Remove grid
    ax.grid(False)
    ax.set_frame_on(True)
    
    # Title
    if n_stable:
        full_title = f"{title} (min 40 maps, n={n_stable})"
    else:
        full_title = title
    ax.set_title(full_title, pad=20, fontsize=14, fontweight='bold')
    
    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label('Pearson Correlation', fontsize=11)
    
    # Add feature legend below plot
    legend_lines = []
    for short, full in FEATURE_LEGEND_FULL.items():
        legend_lines.append(f"{short}: {full}")
    legend_text = "\n".join(legend_lines)
    
    fig.text(0.5, 0.02, legend_text,
            ha='center', va='top',
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.8",
                     facecolor=plt.rcParams["axes.facecolor"],
                     edgecolor=plt.rcParams["axes.edgecolor"],
                     alpha=0.8))
    
    # Add explanation below legend
    fig.text(0.5, -0.09, 
            "Each cell shows CT-side (lower left) and T-side (upper right) correlations",
            ha='center', va='top', fontsize=9, style='italic')
    
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        fig.savefig(f"{save_path}.png", dpi=150, bbox_inches="tight")
        fig.savefig(f"{save_path}.svg", bbox_inches="tight")
    
    return fig, ax


# ========================================================== Positional Residuals ====================================================================

def plot_positioning_residuals_by_role(
    df: pd.DataFrame,
    models: dict,
    min_maps: int,
    figsize: tuple[float, float] = (14, 8),
    save_path: str | None = None
) -> plt.Figure:
    """
    Plot role-stratified residuals from ADNT-ADAT linear regression.
    
    Creates a 2x2 figure showing mean residuals (left) and variance (right)
    for T-side (top) and CT-side (bottom).
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with positioning residuals (from compute_positioning_residuals)
    models : dict
        Fitted regression models from fit_positioning_regressions
    min_maps : int
        Minimum map count threshold for filtering
    figsize : tuple[float, float], default=(14, 8)
        Figure size in inches
    save_path : str or None, default=None
        If provided, save figure to this path (without extension)
        
    Returns
    -------
    matplotlib.figure.Figure
    """
    from scipy import stats
    
    d = df[df['map_count'] >= min_maps].copy()
    
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    sides = ['t', 'ct']
    side_labels = ['T-Side', 'CT-Side']
    T_COLOUR = style.ROLE_COLOURS.get('T', '#de9b35')
    CT_COLOUR = style.ROLE_COLOURS.get('CT', '#2060aa')
    side_colors = [T_COLOUR, CT_COLOUR]
    
    all_means = []
    all_stds = []
    
    for side in sides:
        resid_col = f'adat_residual_{side}'
        role_col = f'role_{side}'
        
        role_stats = d.groupby(role_col)[resid_col].agg([
            ('mean', 'mean'),
            ('std', 'std'),
            ('q25', lambda x: x.quantile(0.25)),
            ('q75', lambda x: x.quantile(0.75)),
            ('n', 'count')
        ]).reset_index()
        
        all_means.extend(role_stats['mean'].values)
        all_stds.extend(role_stats['std'].values)
    
    mean_lim = max(abs(np.min(all_means)), abs(np.max(all_means))) * 1.15
    std_lim = np.max(all_stds) * 1.15
    
    for row, (side, side_label, side_color) in enumerate(zip(sides, side_labels, side_colors)):
        resid_col = f'adat_residual_{side}'
        role_col = f'role_{side}'
        
        role_stats = d.groupby(role_col)[resid_col].agg([
            ('mean', 'mean'),
            ('std', 'std'),
            ('q25', lambda x: x.quantile(0.25)),
            ('q75', lambda x: x.quantile(0.75)),
            ('n', 'count')
        ]).reset_index()
        
        role_stats = role_stats.sort_values('mean', ascending=True)
        
        roles = role_stats[role_col].values
        means = role_stats['mean'].values
        stds = role_stats['std'].values
        q25s = role_stats['q25'].values
        q75s = role_stats['q75'].values
        ns = role_stats['n'].values
        
        iqr_lower = means - q25s
        iqr_upper = q75s - means
        
        colors = [style.get_role_colour(role) for role in roles]
        
        ax_mean = axes[row, 0]
        ax_std = axes[row, 1]
        
        y_pos = np.arange(len(roles))
        
        bars_mean = ax_mean.barh(y_pos, means, color=colors, alpha=0.85, edgecolor='white', linewidth=0.8)
        ax_mean.errorbar(means, y_pos, xerr=[iqr_lower, iqr_upper], fmt='none', ecolor='black', capsize=4, linewidth=1.5)
        
        ax_mean.axvline(0, color='white', linestyle='--', linewidth=1.5, alpha=0.7)
        
        for i, (bar, n) in enumerate(zip(bars_mean, ns)):
            width = bar.get_width()
            x_pos = width + (mean_lim * 0.02 if width >= 0 else -mean_lim * 0.02)
            ha = 'left' if width >= 0 else 'right'
            ax_mean.text(x_pos, bar.get_y() + bar.get_height()/2, f'n={int(n)}',
                        ha=ha, va='center', fontsize=9, color='white')
        
        ax_mean.set_yticks(y_pos)
        ax_mean.set_yticklabels(roles)
        ax_mean.set_xlabel('Average Residual (ADAT - Predicted ADAT)', fontsize=11)
        ax_mean.set_ylabel('Role', fontsize=11)
        ax_mean.set_xlim(-mean_lim, mean_lim)
        
        title_mean = ax_mean.set_title(
            f'{side_label}: Mean Role Positioning Residuals',
            fontsize=14, pad=10
        )
        title_mean.set_color(side_color)
        
        bars_std = ax_std.barh(y_pos, stds, color=colors, alpha=0.85, edgecolor='white', linewidth=0.8)
        
        for i, (bar, n) in enumerate(zip(bars_std, ns)):
            width = bar.get_width()
            ax_std.text(width + std_lim * 0.02, bar.get_y() + bar.get_height()/2, f'n={int(n)}',
                       ha='left', va='center', fontsize=9, color='white')
        
        ax_std.set_yticks(y_pos)
        ax_std.set_yticklabels(roles)
        ax_std.set_xlabel('Standard Deviation of Residuals', fontsize=11)
        ax_std.set_ylabel('Role', fontsize=11)
        ax_std.set_xlim(0, std_lim)
        
        title_std = ax_std.set_title(
            f'{side_label}: Variance in Role Positioning Residuals',
            fontsize=14, pad=10
        )
        title_std.set_color(side_color)
    
    fig.suptitle(
        'Role-Stratified Residuals from ADNT-ADAT Linear Regression',
        fontsize=16, y=0.99
    )
    
    plt.tight_layout(rect=[0, 0.01, 1, 0.99])
    
    if save_path:
        fig.savefig(f"{save_path}.png", dpi=150, bbox_inches='tight')
        fig.savefig(f"{save_path}.svg", bbox_inches='tight')
    
    return fig





