# src/style.py
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import font_manager as fm
import matplotlib as mpl
import seaborn as sns
import plotly.graph_objects as go

# seaborn: 0.13.2 matplotlib: 3.10.6 plotly: 6.3.1
 
# ---- Role colours (unchanged) ----
ROLE_COLOURS = {
    'Spacetaker': '#e41a1c',
    'Lurker': '#000080',
    'AWPer': '#556b2f',
    'Half-Lurker': '#ff8c00',
    'Rotator': '#8b0000',
    'Anchor': '#00008b',
    'Mixed': '#a0522d',
    'IGL': '#ff69b4',
    'Flex': '#ffa07a',
    # Other / non-standard roles & sides
    'T': '#de9b35',
    'CT': '#2060aa',
    'Opener': '#ff6347',
    'Closer': '#87ceeb',
    'Support': '#ffd700',
    'Unknown': '#808080',
}

def get_role_colour(role_name: str) -> str:
    if pd.isna(role_name):
        return ROLE_COLOURS['Unknown']
    if isinstance(role_name, str) and role_name.startswith('IGL-'):
        base_role = role_name.replace('IGL-', '')
        return ROLE_COLOURS.get(base_role, ROLE_COLOURS['Unknown'])
    return ROLE_COLOURS.get(role_name, ROLE_COLOURS['Unknown'])

# ---- Palettes ----
_DARK_COLOR_CYCLE = [
    "#63B3ED", "#F6AD55", "#68D391", "#F687B3",
    "#3C39F5", "#F6E05E", "#81E6D9", "#F56565"
]
_LIGHT_COLOR_CYCLE = [
    "#1f77b4", "#ff7f0e", "#2ca02c",
    "#d62728", "#422bc2", "#8c564b",
    "#e377c2", "#7f7f7f"
]

def _resolve_font(preferred_font: str | None) -> str:
    """
    Return the first available font from [preferred_font, Verdana, DejaVu Sans].
    Avoids 'findfont not found' spam by not listing unavailable families.
    """
    candidates = []
    if preferred_font:
        candidates.append(preferred_font)
    candidates += ["Verdana", "DejaVu Sans"]
    for name in candidates:
        try:
            fm.findfont(name, fallback_to_default=False)
            return name
        except Exception:
            continue
    return "DejaVu Sans"

def set_mpl_theme(mode: str = "dark", preferred_font: str | None = None):
    """
    Apply a Matplotlib theme.
    - mode: "dark" (default) or "light"
    - preferred_font: try this installed font first; fallback = Verdana -> DejaVu Sans
    """
    chosen_font = _resolve_font(preferred_font)

    # Colors / palette
    dark = (mode.lower() == "dark")
    bg_fig   = "#161616" if dark else "#ffffff"
    bg_axes  = "#1e1e1e" if dark else "#ffffff"
    fg_text  = "#e6e6e6" if dark else "#1a1a1a"
    grid_col = "#3a3a3a" if dark else "#dddddd"
    spine_col = "#666666" if dark else "#888888"
    color_cycle = _DARK_COLOR_CYCLE if dark else _LIGHT_COLOR_CYCLE

    # Fonts
    mpl.rcParams["font.family"] = [chosen_font]
    mpl.rcParams["font.sans-serif"] = [chosen_font, "Verdana", "DejaVu Sans"]

    # Core rc
    plt.rcParams.update({
        "figure.figsize": (9.6, 5.2),
        "figure.dpi": 120,
        "savefig.dpi": 150,

        "figure.facecolor": bg_fig,
        "axes.facecolor": bg_axes,
        "axes.edgecolor": spine_col,
        "axes.labelcolor": fg_text,
        "axes.titlecolor": fg_text,

        "axes.grid": True,
        "grid.color": grid_col,
        "grid.alpha": 0.35,
        "axes.spines.top": False,
        "axes.spines.right": False,

        "text.color": fg_text,
        "xtick.color": fg_text,
        "ytick.color": fg_text,
        "figure.titlesize": 18,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,

        "axes.prop_cycle": plt.cycler(color=color_cycle),
        "lines.linewidth": 2.0,
        "lines.markersize": 5.5,

        "legend.frameon": True,
        "legend.edgecolor": spine_col,
        "legend.facecolor": (0, 0, 0, 0) if dark else "#ffffff",
        "legend.fontsize": 10,
        "legend.labelcolor": fg_text,
    })


def set_seaborn_theme(mode: str = "dark", preferred_font: str | None = None):
    """ Apply a Seaborn theme, based on our Matplotlib theme."""
    set_mpl_theme(mode=mode, preferred_font=preferred_font)

    rc = dict(plt.rcParams)  # make a plain dict (important for some versions)
    sns.set_theme(rc=rc) 



def plotly_template(mode: str = "dark", preferred_font: str | None = None):
    font_family = preferred_font or (plt.rcParams.get("font.family") or ["DejaVu Sans"])[0]
    fg = plt.rcParams.get("text.color", "#cfd3dc")
    axlab = plt.rcParams.get("axes.labelcolor", fg)
    xtick = plt.rcParams.get("xtick.color", fg)
    ytick = plt.rcParams.get("ytick.color", fg)
    grid = plt.rcParams.get("grid.color", "#7a8194")
    paper = plt.rcParams.get("figure.facecolor", "#0f1116")
    plot = plt.rcParams.get("axes.facecolor", "#0f1116")
    fsize = int(plt.rcParams.get("font.size", 11))

    return go.layout.Template(
        layout=go.Layout(
            paper_bgcolor=paper,
            plot_bgcolor=plot,
            font=dict(family=font_family, color=fg, size=fsize),
            legend=dict(font=dict(family=font_family, color=fg)),
            xaxis=dict(
                title=dict(font=dict(family=font_family, color=axlab, size=fsize)),
                tickfont=dict(family=font_family, color=xtick, size=fsize),
                gridcolor=grid
            ),
            yaxis=dict(
                title=dict(font=dict(family=font_family, color=axlab, size=fsize)),
                tickfont=dict(family=font_family, color=ytick, size=fsize),
                gridcolor=grid
            ),
            margin=dict(t=110, r=30, b=40, l=50)
        )
    )

