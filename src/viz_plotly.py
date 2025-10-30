"""
Interactive Plotly visualizations for CS2 playstyle analysis.
"""
from __future__ import annotations
from typing import Optional, Literal, List
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Project modules
import style
from viz_core import (
    FEATURE_LABELS_SHORT,
    FEATURE_LEGEND_FULL,
    _default_features_base,
)


# ========================================================== Radar Charts (modular, single-call API) ======================

# ---- constants & defaults (dark theme aware) ----
_RADIAL_RANGE = (-2.0, 2.0)
_CLIP_Z = 2.5
_MIN_ROLE_N = 8
_LINE_WIDTH = 2.2
_BAND_ALPHA = 0.18
_TITLE = "Role Profile Radars — T vs CT"

def _button_palette():
    """
    Pull button colors from style.py if present; otherwise, fall back to dark-friendly defaults.
    """
    active = getattr(style, "plotly_button_active_fill",   "#1f1f1f")   # slightly lighter than bg for focus
    inactive = getattr(style, "plotly_button_inactive_fill", "#030303") # deep charcoal
    hover = getattr(style, "plotly_button_hover_fill",      "#3a3a3a")
    border = getattr(style, "plotly_button_border",         "#3a3a3a")
    font = getattr(style, "plotly_button_font_color",       "#8d8d8d")
    return dict(active=active, inactive=inactive, hover=hover, border=border, font=font)

# ---- tiny utilities ----
def _features_base() -> List[str]:
    return _default_features_base()

def _rgba_from_role(role: str, alpha: float) -> str:
    hx = str(style.get_role_colour(role)).lstrip("#")
    if len(hx) == 3:
        hx = "".join(ch*2 for ch in hx)
    try:
        r = int(hx[0:2], 16); g = int(hx[2:4], 16); b = int(hx[4:6], 16)
        return f"rgba({r},{g},{b},{alpha})"
    except Exception:
        return f"rgba(120,120,120,{alpha})"

def _clip_arr(a: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return np.clip(np.asarray(a, dtype=float), lo, hi)

def _closed(a: np.ndarray) -> np.ndarray:
    return np.r_[a, a[:1]]

def _arrays_for_role(stats: pd.DataFrame, role: str, features_base: List[str], lo: float, hi: float) -> dict | None:
    s = stats.loc[stats["role"] == role].set_index("feature").reindex(features_base)
    if s.empty or s["mean"].isna().all(): 
        return None
    return {
        "mean": _clip_arr(s["mean"].to_numpy(float), lo, hi),
        "sd":   s["sd"].to_numpy(float),
        "q1":   _clip_arr(s["q1"].to_numpy(float), lo, hi),
        "q3":   _clip_arr(s["q3"].to_numpy(float), lo, hi),
    }

# ---- data prep (unchanged logic; just simplified surface) ----
def _prep_role_stats_by_side(
    df: pd.DataFrame,
    side: Literal["t","ct"],
    *,
    features_base: Optional[List[str]] = None,
    min_role_n: int = _MIN_ROLE_N,
    clip_z: float = _CLIP_Z,
) -> pd.DataFrame:
    """
    Compute side-wise z-scores for <feature>_<side> and aggregate per role_<side>.
    Returns tidy long: ['side','role','feature','mean','sd','q1','q3','n']
    """
    if features_base is None:
        features_base = _features_base()

    role_col = f"role_{side}"
    feat_cols = [f"{f}_{side}" for f in features_base]
    missing = [c for c in [role_col, *feat_cols] if c not in df.columns]
    if missing:
        raise KeyError(f"Missing columns for side='{side}': {missing}")

    d = df.loc[df[role_col].notna()].copy()
    X = d[feat_cols].astype(float)

    # drop rows with all-NaN across selected features
    mask_any = X.notna().any(axis=1)
    d = d.loc[mask_any]
    X = X.loc[mask_any]

    # side-global z-scores (ddof=0 to match your previous)
    means = X.mean(axis=0, skipna=True)
    stds  = X.std(axis=0, ddof=0, skipna=True).replace(0, 1.0)
    Z = ((X - means) / stds).clip(-abs(clip_z), abs(clip_z))

    long = Z.copy()
    long.columns = features_base
    long["role"] = d[role_col].astype(str).values
    long = long.melt(id_vars=["role"], var_name="feature", value_name="z").dropna()

    # keep roles with enough rows (players × ~half the features)
    role_counts = long.groupby("role")["z"].count()
    keep_roles = role_counts[role_counts >= min_role_n * max(1, len(features_base)//2)].index
    long = long[long["role"].isin(keep_roles)]

    out = (
        long.groupby(["role","feature"])["z"]
        .agg(mean="mean", sd="std",
             q1=lambda s: s.quantile(0.25),
             q3=lambda s: s.quantile(0.75),
             n="count")
        .reset_index()
    )
    out["side"] = side
    out["feature"] = pd.Categorical(out["feature"], categories=features_base, ordered=True)
    return out[["side","role","feature","mean","sd","q1","q3","n"]].sort_values(["role","feature"]).reset_index(drop=True)

# ---- figure building ----
def _init_two_polar(title: str) -> go.Figure:
    tpl = style.plotly_template() if hasattr(style, "plotly_template") else None
    fig = make_subplots(
        rows=1, cols=2, specs=[[{"type": "polar"}, {"type": "polar"}]],
        subplot_titles=("T side", "CT side"),
        horizontal_spacing=0.10,
    )
    fig.update_layout(
        template=tpl,
        width=1500, height=840,
        polar=dict(domain=dict(x=[0.03, 0.47], y=[0.22, 0.96]), bgcolor="#2b2b2b"),
        polar2=dict(domain=dict(x=[0.53, 0.97], y=[0.22, 0.96]), bgcolor="#2b2b2b"),
        title=dict(text=title, x=0.5, xanchor="center", font=dict(size=28)),
        font=dict(size=16),
        margin=dict(l=40, r=40, t=100, b=190),
    )

    # colourise subplot titles
    CT_COLOUR = "#2060aa"
    T_COLOUR  = "#de9b35"
    for ann in fig.layout.annotations:
        if ann.text == "T side":
            ann.update(
                x=0.25, y=1.02,
                xanchor="center", yanchor="bottom",
                font=dict(size=20, color=T_COLOUR)
            )
        elif ann.text == "CT side":
            ann.update(
                x=0.75, y=1.02,
                xanchor="center", yanchor="bottom",
                font=dict(size=20, color=CT_COLOUR)
            )
    return fig


def _style_axes(fig: go.Figure, radial_range: tuple[float,float]):
    fig.update_polars(dict(
        radialaxis=dict(
            range=list(radial_range),
            tickvals=[-2,-1,0,1,2],
            ticktext=["-2σ","-1σ","0","+1σ","+2σ"],
            tickfont=dict(size=15),
            gridcolor="rgba(255,255,255,0.10)",
            linecolor="rgba(255,255,255,0.20)",
        ),
        angularaxis=dict(
            tickfont=dict(size=15),
            gridcolor="rgba(255,255,255,0.10)",
            linecolor="rgba(255,255,255,0.20)",
            direction="clockwise",
        ),
    ))

def _add_legends(fig: go.Figure):
    fig.update_layout(
        legend=dict(
            orientation="h",
            y=0.088, yanchor="top",
            x=0.25, xanchor="center",
            traceorder="grouped",
            font=dict(size=14),
            itemwidth=60,
            bgcolor="rgba(43,43,43,0.85)",
            bordercolor="#3a3a3a", borderwidth=1,
        ),
        legend2=dict(
            orientation="h",
            y=0.088, yanchor="top",
            x=0.75, xanchor="center",
            traceorder="grouped",
            font=dict(size=14),
            itemwidth=60,
            bgcolor="rgba(43,43,43,0.85)",
            bordercolor="#3a3a3a", borderwidth=1,
        ),
    )
    # headers
    fig.add_trace(go.Scatterpolar(r=[], theta=[], mode="lines",
                                  name="<b>T roles</b>", showlegend=True,
                                  legendrank=10, hoverinfo="skip", legend="legend"))
    fig.add_trace(go.Scatterpolar(r=[], theta=[], mode="lines",
                                  name="<b>CT roles</b>", showlegend=True,
                                  legendrank=20, hoverinfo="skip", legend="legend2"))

def _add_role_traces(fig: go.Figure, *, role: str, arrays: dict, side_tag: str, col: int,
                     theta_closed: list[str], line_width: float, band_alpha: float,
                     band_iqr_idx: list[int], band_sd_idx: list[int]):
    if arrays is None: 
        return
    lg = f"{role}__{side_tag}"
    name = f"{role} — {side_tag}"
    fill_rgba = _rgba_from_role(role, band_alpha)
    line_color = style.get_role_colour(role)
    target_leg = "legend" if side_tag=="T" else "legend2"

    # IQR band (hidden by default; toggled via buttons)
    fig.add_trace(go.Scatterpolar(
        r=_closed(arrays["q1"]), theta=theta_closed, mode="lines",
        line=dict(width=0), fill=None, hoverinfo="skip",
        showlegend=False, legendgroup=lg, legend=target_leg, visible=False
    ), row=1, col=col); band_iqr_idx.append(len(fig.data)-1)

    fig.add_trace(go.Scatterpolar(
        r=_closed(arrays["q3"]), theta=theta_closed, mode="lines",
        line=dict(width=0), fill="tonext", fillcolor=fill_rgba,
        hoverinfo="skip", showlegend=False, legendgroup=lg, legend=target_leg, visible=False
    ), row=1, col=col); band_iqr_idx.append(len(fig.data)-1)

    # ±SD band (hidden by default)
    lower_sd = _clip_arr(arrays["mean"] - arrays["sd"], *_RADIAL_RANGE)
    upper_sd = _clip_arr(arrays["mean"] + arrays["sd"], *_RADIAL_RANGE)
    fig.add_trace(go.Scatterpolar(
        r=_closed(lower_sd), theta=theta_closed, mode="lines",
        line=dict(width=0), fill=None, hoverinfo="skip",
        showlegend=False, legendgroup=lg, legend=target_leg, visible=False
    ), row=1, col=col); band_sd_idx.append(len(fig.data)-1)

    fig.add_trace(go.Scatterpolar(
        r=_closed(upper_sd), theta=theta_closed, mode="lines",
        line=dict(width=0), fill="tonext", fillcolor=fill_rgba,
        hoverinfo="skip", showlegend=False, legendgroup=lg, legend=target_leg, visible=False
    ), row=1, col=col); band_sd_idx.append(len(fig.data)-1)

    # mean polygon (visible)
    fig.add_trace(go.Scatterpolar(
        r=_closed(arrays["mean"]), theta=theta_closed, mode="lines",
        line=dict(width=line_width, color=line_color),
        name=name, legendgroup=lg, showlegend=True, legend=target_leg,
        hovertemplate=f"<b>{name}</b><br>%{{theta}}<br>Mean: %{{r:.2f}}σ<extra></extra>",
    ), row=1, col=col)

def _wire_band_buttons(fig: go.Figure, band_iqr_idx: list[int], band_sd_idx: list[int],
                       baseline_idx: list[int]):
    total = len(fig.data)

    def vis_for(mode: str) -> list[bool]:
        vis = [True] * total
        if mode == "iqr":
            for i in band_sd_idx: vis[i] = False
        elif mode == "sd":
            for i in band_iqr_idx: vis[i] = False
        else:
            for i in band_iqr_idx + band_sd_idx: vis[i] = False
        return vis

    # bands
    fig.layout.updatemenus[0].buttons[0].args[0]["visible"] = vis_for("iqr")
    fig.layout.updatemenus[1].buttons[0].args[0]["visible"] = vis_for("sd")
    fig.layout.updatemenus[2].buttons[0].args[0]["visible"] = vis_for("off")

    # baseline toggle
    if baseline_idx:
        vis_on  = [True] * total
        vis_off = [True] * total
        for i in baseline_idx: vis_off[i] = False

        # respect current band mask (default "off")
        current_mode = "off"
        vis_on  = [a and b for a, b in zip(vis_on,  vis_for(current_mode))]
        vis_off = [a and b for a, b in zip(vis_off, vis_for(current_mode))]

        pal = _button_palette()
        btn = fig.layout.updatemenus[3].buttons[0]

        # When clicked (args): baseline ON → set bg active + label "Baseline: Off"
        btn.args  = [
            {"visible": vis_on},
            {"updatemenus[3].bgcolor": pal["active"],
             "updatemenus[3].buttons[0].label": "Baseline: Off"}
        ]
        # When clicked again (args2): baseline OFF → set bg inactive + label "Baseline: On"
        btn.args2 = [
            {"visible": vis_off},
            {"updatemenus[3].bgcolor": pal["inactive"],
             "updatemenus[3].buttons[0].label": "Baseline: On"}
        ]


def _add_feature_legend_annotation(fig: go.Figure):
    if not FEATURE_LEGEND_FULL:
        return
    lines = [f"<b>{k}</b>: {v}" for k, v in FEATURE_LEGEND_FULL.items()]
    fig.add_annotation(
        x=0.5, y=-0.03, xref="paper", yref="paper",
        xanchor="center", yanchor="top",
        text="<br>".join(lines),
        showarrow=False, align="center",
        font=dict(size=13, color="rgba(230,230,230,0.95)"),
        bgcolor="rgba(43,43,43,0.85)",
        bordercolor="#3a3a3a", borderwidth=1, borderpad=6,
    )

def _add_baseline(fig: go.Figure, theta_closed: list[str]) -> list[int]:
    base_r = [0.0] * len(theta_closed)
    base_style = dict(color="rgba(200,200,200,0.35)", width=1.2, dash="dot")
    baseline_idx = []
    for c in (1, 2):
        fig.add_trace(go.Scatterpolar(
            r=base_r, theta=theta_closed, mode="lines", line=base_style,
            name=None, showlegend=False, hoverinfo="skip",
        ), row=1, col=c); baseline_idx.append(len(fig.data)-1)
    return baseline_idx

def _apply_button_layout(fig: go.Figure):
    pal = _button_palette()
    fig.update_layout(
        updatemenus=[
            # IQR (left of center)
            dict(type="buttons", direction="right",
                 x=0.41, xanchor="center", y=1.06, yanchor="top",
                 bgcolor=pal["inactive"], bordercolor=pal["border"],
                 font=dict(color=pal["font"], size=13),
                 showactive=False,
                 buttons=[dict(label="Bands: IQR", method="update",
                               args=[{"visible": []},
                                     {"updatemenus[0].bgcolor": pal["active"],
                                      "updatemenus[1].bgcolor": pal["inactive"],
                                      "updatemenus[2].bgcolor": pal["inactive"]}])],
                 pad=dict(r=6,l=6,t=4,b=4)),
            # ±SD (center)
            dict(type="buttons", direction="right",
                 x=0.50, xanchor="center", y=1.06, yanchor="top",
                 bgcolor=pal["inactive"], bordercolor=pal["border"],
                 font=dict(color=pal["font"], size=13),
                 showactive=False,
                 buttons=[dict(label="Bands: ±SD", method="update",
                               args=[{"visible": []},
                                     {"updatemenus[0].bgcolor": pal["inactive"],
                                      "updatemenus[1].bgcolor": pal["active"],
                                      "updatemenus[2].bgcolor": pal["inactive"]}])],
                 pad=dict(r=6,l=6,t=4,b=4)),
            # Off (right of center; default highlighted)
            dict(type="buttons", direction="right",
                 x=0.59, xanchor="center", y=1.06, yanchor="top",
                 bgcolor=pal["active"], bordercolor=pal["border"],
                 font=dict(color=pal["font"], size=13),
                 showactive=False,
                 buttons=[dict(label="Bands: Off", method="update",
                               args=[{"visible": []},
                                     {"updatemenus[0].bgcolor": pal["inactive"],
                                      "updatemenus[1].bgcolor": pal["inactive"],
                                      "updatemenus[2].bgcolor": pal["active"]}])],
                 pad=dict(r=6,l=6,t=4,b=4)),
            # Baseline toggle (unchanged)
            dict(type="buttons", direction="right",
                 x=0.50, xanchor="center", y=0.20, yanchor="top",
                 bgcolor=pal["inactive"], bordercolor=pal["border"],
                 font=dict(color=pal["font"], size=13),
                 showactive=False,
                 buttons=[dict(label="Baseline: On", method="update",
                               args=[{"visible": []}])],
                 pad=dict(r=6,l=6,t=4,b=4)),
        ],
    )



# ---- public API (one-liner from the notebook) ----
def plot_role_radars_interactive(df: pd.DataFrame, min_map_count: int) -> "go.Figure":
    """
    Build the interactive two-panel (T vs CT) role radar figure from the raw dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        Requires <feature>_t / <feature>_ct, role_t / role_ct; optional map_count.
    min_map_count : int
        Keep only rows where map_count >= min_map_count (if column exists).

    Returns
    -------
    plotly.graph_objects.Figure
    """
    d = df.copy()
    if "map_count" in d.columns:
        d = d.loc[d["map_count"].astype(float) >= float(min_map_count)].copy()

    features_base = _features_base()
    # compute side-wise stats (z-space)
    stats_t  = _prep_role_stats_by_side(d, "t",  features_base=features_base)
    stats_ct = _prep_role_stats_by_side(d, "ct", features_base=features_base)

    roles = sorted(set(stats_t["role"]).union(stats_ct["role"]), key=str)
    theta_labels = [FEATURE_LABELS_SHORT.get(f, f.upper()) for f in features_base]
    theta_closed = theta_labels + [theta_labels[0]]

    fig = _init_two_polar(_TITLE)
    _style_axes(fig, _RADIAL_RANGE)
    _add_legends(fig)

    band_iqr_idx, band_sd_idx = [], []
    for r in roles:
        arr_t  = _arrays_for_role(stats_t,  r, features_base, *_RADIAL_RANGE)
        arr_ct = _arrays_for_role(stats_ct, r, features_base, *_RADIAL_RANGE)
        _add_role_traces(fig, role=r, arrays=arr_t,  side_tag="T",  col=1,
                         theta_closed=theta_closed, line_width=_LINE_WIDTH,
                         band_alpha=_BAND_ALPHA, band_iqr_idx=band_iqr_idx, band_sd_idx=band_sd_idx)
        _add_role_traces(fig, role=r, arrays=arr_ct, side_tag="CT", col=2,
                         theta_closed=theta_closed, line_width=_LINE_WIDTH,
                         band_alpha=_BAND_ALPHA, band_iqr_idx=band_iqr_idx, band_sd_idx=band_sd_idx)

    baseline_idx = _add_baseline(fig, theta_closed)
    _apply_button_layout(fig)             # dark-styled buttons
    _wire_band_buttons(fig, band_iqr_idx, band_sd_idx, baseline_idx)
    _add_feature_legend_annotation(fig)

    return fig



def plot_positioning_regression_interactive(
    df: pd.DataFrame,
    models: dict,
    min_maps: int
) -> go.Figure:
    """
    Interactive scatter plots showing ADNT vs ADAT with regression lines.
    
    Creates two side-by-side scatter plots (T and CT) displaying the positioning
    relationship between distance to nearest teammate and distance from team center,
    with fitted regression lines and role-colored points.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with positioning features and residuals (from compute_positioning_residuals)
    models : dict
        Fitted regression models from fit_positioning_regressions: {'t': model_t, 'ct': model_ct}
    min_maps : int
        Minimum map count threshold for filtering stable players
        
    Returns
    -------
    plotly.graph_objects.Figure
        Interactive figure with role toggles via legend
    """
    
    d = df[df['map_count'] >= min_maps].copy()
    
    tpl = style.plotly_template() if hasattr(style, "plotly_template") else None
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("T Side", "CT Side"),
        horizontal_spacing=0.15,
    )
    
    T_COLOUR = style.ROLE_COLOURS.get('T', '#de9b35')
    CT_COLOUR = style.ROLE_COLOURS.get('CT', '#2060aa')
    REGRESSION_COLOR = '#ff69b4'
    
    x_label = f"{FEATURE_LEGEND_FULL.get('ADNT', 'ADNT')} (ADNT)"
    y_label = f"{FEATURE_LEGEND_FULL.get('ADAT', 'ADAT')} (ADAT)"
    
    regression_indices = []
    
    for col, side in enumerate(['t', 'ct'], start=1):
        model = models[side]
        coef = model.coef_[0]
        intercept = model.intercept_
        
        x_col = f'adnt_rank_{side}'
        y_col = f'adat_rank_{side}'
        role_col = f'role_{side}'
        resid_col = f'adat_residual_{side}'
        
        x_line = np.array([0.2, 1.0])
        y_line = coef * x_line + intercept
        
        fig.add_trace(
            go.Scatter(
                x=x_line,
                y=y_line,
                mode='lines',
                line=dict(color=REGRESSION_COLOR, width=2.5, dash='solid'),
                name='Regression',
                showlegend=False,
                hoverinfo='skip',
                legendgroup='regression',
            ),
            row=1, col=col
        )
        regression_indices.append(len(fig.data) - 1)
        
        roles = sorted(d[role_col].dropna().unique())
        legend_name = 'legend' if side == 't' else 'legend2'
        
        for role in roles:
            role_data = d[d[role_col] == role]
            
            color = style.get_role_colour(role)
            
            fig.add_trace(
                go.Scatter(
                    x=role_data[x_col],
                    y=role_data[y_col],
                    mode='markers',
                    marker=dict(
                        color=color,
                        size=9,
                        line=dict(width=0.5, color='rgba(255,255,255,0.4)')
                    ),
                    name=role,
                    customdata=np.column_stack((
                        role_data['player_name'],
                        role_data[resid_col]
                    )),
                    hovertemplate=(
                        '<b>%{customdata[0]}</b><br>'
                        f'{x_label}: %{{x:.3f}}<br>'
                        f'{y_label}: %{{y:.3f}}<br>'
                        'Residual: %{customdata[1]:.3f}'
                        '<extra></extra>'
                    ),
                    legendgroup=f'{role}_{side}',
                    showlegend=True,
                    legend=legend_name,
                ),
                row=1, col=col
            )
    
    pal = _button_palette()
    
    total_traces = len(fig.data)
    vis_on = [True] * total_traces
    vis_off = [True] * total_traces
    for idx in regression_indices:
        vis_off[idx] = False
    
    fig.update_layout(
        template=tpl,
        width=1400,
        height=850,
        title=dict(
            text="ADNT vs ADAT Positioning Relationship by Role",
            x=0.5,
            xanchor='center',
            font=dict(size=26)
        ),
        margin=dict(l=70, r=70, t=110, b=180),
        legend=dict(
            orientation='h',
            title=dict(text='<b>T Roles</b>', font=dict(size=15, color=T_COLOUR)),
            x=0.2125, xanchor='center',
            y=-0.2, yanchor='middle',
            bgcolor='rgba(43,43,43,0.88)',
            bordercolor='#3a3a3a',
            borderwidth=1,
            font=dict(size=13),
        ),
        legend2=dict(
            orientation='h',
            title=dict(text='<b>CT Roles</b>', font=dict(size=15, color=CT_COLOUR)),
            x=0.7875, xanchor='center',
            y=-0.2, yanchor='middle',
            bgcolor='rgba(43,43,43,0.88)',
            bordercolor='#3a3a3a',
            borderwidth=1,
            font=dict(size=13),
        ),
        updatemenus=[
            dict(
                type='buttons',
                direction='right',
                x=0.5, xanchor='center',
                y=-0.2, yanchor='middle',
                bgcolor=pal['active'],
                bordercolor=pal['border'],
                font=dict(color=pal['font'], size=13),
                showactive=False,
                buttons=[
                    dict(
                        label='Regression: Off',
                        method='update',
                        args=[
                            {'visible': vis_on},
                            {'updatemenus[0].bgcolor': pal['active'],
                             'updatemenus[0].buttons[0].label': 'Regression: Off'}
                        ],
                        args2=[
                            {'visible': vis_off},
                            {'updatemenus[0].bgcolor': pal['inactive'],
                             'updatemenus[0].buttons[0].label': 'Regression: On'}
                        ]
                    )
                ],
                pad=dict(r=8, l=8, t=6, b=6)
            )
        ]
    )
    
    for ann in fig.layout.annotations:
        if ann.text == "T Side":
            ann.update(
                font=dict(size=22, color=T_COLOUR),
                x=0.2125, y=1.02,
                xanchor='center', yanchor='bottom'
            )
        elif ann.text == "CT Side":
            ann.update(
                font=dict(size=22, color=CT_COLOUR),
                x=0.7875, y=1.02,
                xanchor='center', yanchor='bottom'
            )
    
    fig.update_xaxes(
        title_text=x_label,
        title_font=dict(size=18),
        range=[0.25, 0.95],
        dtick=0.1,
        constrain='domain',
        row=1, col=1
    )
    fig.update_xaxes(
        title_text=x_label,
        title_font=dict(size=18),
        range=[0.25, 0.95],
        dtick=0.1,
        constrain='domain',
        row=1, col=2
    )
    fig.update_yaxes(
        title_text=y_label,
        title_font=dict(size=18),
        range=[0.25, 0.95],
        scaleanchor='x',
        scaleratio=1,
        constrain='domain',
        row=1, col=1
    )
    fig.update_yaxes(
        title_text=y_label,
        title_font=dict(size=18),
        range=[0.25, 0.95],
        scaleanchor='x2',
        scaleratio=1,
        constrain='domain',
        row=1, col=2
    )
    
    return fig





