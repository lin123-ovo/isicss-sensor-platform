"""Shared visual primitives for the CSG124 Streamlit workbench."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Callable, TypeVar

import streamlit as st
from sensor_app.table_display import display_dataframe
import streamlit_shadcn_ui as ui


T = TypeVar("T")


LIGHT_TOKENS = {
    "canvas": "#F5F7FC",
    "sidebar": "#EEF2F8",
    "surface": "#FFFFFF",
    "surface_muted": "#F7F7FB",
    "text": "#202124",
    "muted": "#98A2B3",
    "border": "#E6EAF1",
    "primary": "#3547D3",
    "primary_soft": "#EAECFB",
    "cyan": "#10C5E8",
    "magenta": "#E526A8",
    "purple": "#711AC8",
    "success": "#2ECC55",
    "danger": "#F5222D",
}

DARK_TOKENS = {
    "canvas": "#0F1320",
    "sidebar": "#151A2A",
    "surface": "#1A2032",
    "surface_muted": "#20273A",
    "text": "#F5F7FC",
    "muted": "#A7B0C2",
    "border": "#30384E",
    "primary": "#7181FF",
    "primary_soft": "#252E57",
    "cyan": "#36D5F2",
    "magenta": "#F15CBD",
    "purple": "#A36BFF",
    "success": "#45D86B",
    "danger": "#FF5D68",
}


def _css_tokens(tokens: dict[str, str]) -> str:
    """Return CSS custom properties for one semantic theme."""

    return "\n".join(f"--nexrun-{name.replace('_', '-')}: {value};" for name, value in tokens.items())


def apply_nexrun_theme(selected_theme: str) -> None:
    """Apply the shared Nexrun-inspired workbench shell and semantic tokens."""

    tokens = DARK_TOKENS if selected_theme == "深色" else LIGHT_TOKENS
    system_override = ""
    if selected_theme == "跟随系统":
        system_override = (
            "@media (prefers-color-scheme: dark) { :root {"
            + _css_tokens(DARK_TOKENS)
            + "} }"
        )

    css = """
    <style>
    :root {
        __TOKENS__
        --nexrun-radius: 16px;
        --nexrun-shadow: 0 10px 30px rgba(44, 51, 73, .06);
    }
    __SYSTEM_OVERRIDE__
    html, body, .stApp,
    [data-testid="stAppViewContainer"],
    [data-testid="stMain"] {
        background: var(--nexrun-canvas) !important;
        color: var(--nexrun-text) !important;
        font-family: -apple-system, BlinkMacSystemFont, "SF Pro Text", "PingFang SC",
            "Microsoft YaHei", Arial, sans-serif;
    }
    [data-testid="stHeader"] {
        background: color-mix(in srgb, var(--nexrun-canvas) 92%, transparent) !important;
    }
    .block-container {
        max-width: 1680px;
        padding: 1.2rem 2rem 4rem;
    }
    h1, h2, h3, h4, h5, h6, p, label,
    [data-testid="stMarkdownContainer"] {
        color: var(--nexrun-text);
    }
    h1 {font-size: 2rem !important; letter-spacing: -.04em; font-weight: 750 !important;}
    h2 {font-size: 1.65rem !important; letter-spacing: -.025em; font-weight: 720 !important;}
    h3 {font-size: 1.25rem !important; font-weight: 700 !important;}
    [data-testid="stCaptionContainer"], .stCaption, small {
        color: var(--nexrun-muted) !important;
    }
    .sensor-eyebrow, .workspace-eyebrow {
        color: var(--nexrun-primary);
        font-size: .72rem;
        font-weight: 800;
        letter-spacing: .12em;
        text-transform: uppercase;
        margin-bottom: .28rem;
    }
    .sensor-page-note {color: var(--nexrun-muted); font-size: .9rem; margin: -.4rem 0 1rem;}
    .sensor-divider {
        height: 1px;
        background: var(--nexrun-border);
        margin: 1rem 0;
    }
    .workspace-title {font-size: 2rem; line-height: 1.12; font-weight: 750; color: var(--nexrun-text); margin: 0;}
    .workspace-subtitle {color: var(--nexrun-muted); margin: .45rem 0 0; font-size: .9rem;}
    .workspace-status {display: flex; flex-wrap: wrap; justify-content: flex-end; gap: .45rem; padding-top: .3rem;}
    .workspace-status span, .nexrun-pill {
        background: var(--nexrun-primary-soft);
        border: 1px solid color-mix(in srgb, var(--nexrun-primary) 18%, var(--nexrun-border));
        border-radius: 999px;
        color: var(--nexrun-primary);
        font-size: .76rem;
        font-weight: 700;
        padding: .38rem .68rem;
    }
    .nexrun-card, .nexrun-timeline {
        background: var(--nexrun-surface);
        border: 1px solid var(--nexrun-border);
        border-radius: var(--nexrun-radius);
        box-shadow: var(--nexrun-shadow);
    }
    .nexrun-card {padding: 1rem 1.05rem; margin-bottom: .85rem;}
    .nexrun-card-kicker {color: var(--nexrun-muted); font-size: .72rem; font-weight: 700; letter-spacing: .06em; text-transform: uppercase;}
    .nexrun-card-title {font-size: 1rem; font-weight: 750; margin: .3rem 0 .2rem;}
    .nexrun-card-copy {color: var(--nexrun-muted); font-size: .82rem; line-height: 1.55;}
    .nexrun-timeline {padding: .55rem 1rem;}
    .nexrun-timeline-item {position: relative; padding: .7rem .35rem .75rem 1.4rem; border-bottom: 1px solid var(--nexrun-border);}
    .nexrun-timeline-item:last-child {border-bottom: 0;}
    .nexrun-timeline-item::before {
        content: "";
        position: absolute;
        left: .05rem;
        top: 1rem;
        width: .58rem;
        height: .58rem;
        border-radius: 50%;
        background: var(--nexrun-primary);
        box-shadow: 0 0 0 4px var(--nexrun-primary-soft);
    }
    .nexrun-timeline-title {font-size: .84rem; font-weight: 700;}
    .nexrun-timeline-meta {color: var(--nexrun-muted); font-size: .72rem; margin-top: .2rem;}
    [data-testid="stMetric"] {
        background: var(--nexrun-surface);
        border: 1px solid var(--nexrun-border);
        border-radius: var(--nexrun-radius);
        box-shadow: var(--nexrun-shadow);
        padding: 1rem 1.05rem;
        min-height: 112px;
    }
    [data-testid="stMetricLabel"] {color: var(--nexrun-muted) !important; font-weight: 650;}
    [data-testid="stMetricValue"], [data-testid="stMetricValue"] * {
        color: var(--nexrun-text) !important;
        font-size: clamp(1.72rem, 2.15vw, 2.25rem) !important;
        line-height: 1.12 !important;
    }
    [data-testid="stMetricDelta"] {font-weight: 700;}
    div[data-testid="stVerticalBlockBorderWrapper"] {
        background: var(--nexrun-surface);
        border-color: var(--nexrun-border) !important;
        border-radius: var(--nexrun-radius) !important;
        box-shadow: var(--nexrun-shadow);
    }
    .stButton > button, .stDownloadButton > button {
        min-height: 2.65rem;
        border-radius: 10px;
        border-color: var(--nexrun-primary);
        font-weight: 700;
    }
    .stButton > button[kind="primary"], .stDownloadButton > button[kind="primary"] {
        background: var(--nexrun-primary);
        color: #FFFFFF;
    }
    .stButton > button:hover, .stDownloadButton > button:hover {
        border-color: var(--nexrun-cyan);
        color: var(--nexrun-primary);
    }
    [data-baseweb="select"] > div,
    [data-baseweb="input"] > div,
    [data-testid="stTextInputRootElement"],
    [data-testid="stFileUploaderDropzone"] {
        background: var(--nexrun-surface) !important;
        border-color: var(--nexrun-border) !important;
        border-radius: 11px !important;
    }
    div[data-testid="stNumberInput"] input,
    div[data-testid="stSelectbox"] [data-baseweb="select"] > div {min-height: 2.75rem;}
    div[data-testid="stSelectbox"] [role="group"][data-rac] {
        min-height: 2.75rem;
        background: var(--nexrun-surface) !important;
        border-color: var(--nexrun-border) !important;
        border-radius: 11px !important;
    }
    div[data-testid="stSelectbox"] [role="group"][data-rac] input,
    div[data-testid="stSelectbox"] [role="group"][data-rac] button {
        color: var(--nexrun-text) !important;
    }
    button[data-testid="stPopoverButton"] {
        background: var(--nexrun-surface) !important;
        border-color: var(--nexrun-border) !important;
        color: var(--nexrun-text) !important;
    }
    button[data-testid="stPopoverButton"] * {color: var(--nexrun-text) !important;}
    div[data-testid="stDataFrame"] {
        border: 1px solid var(--nexrun-border);
        border-radius: 13px;
        overflow: auto;
        background: var(--nexrun-surface);
        box-shadow: var(--nexrun-shadow);
    }
    [data-testid="stAlert"] {border-radius: 12px;}
    [data-testid="stExpander"] {
        background: var(--nexrun-surface);
        border-color: var(--nexrun-border);
        border-radius: 12px;
    }
    section[data-testid="stSidebar"] {
        width: 224px;
        min-width: 224px !important;
        max-width: 224px !important;
        flex-basis: 224px !important;
        position: sticky !important;
        top: 0;
        align-self: flex-start;
        height: 100vh;
        overflow: hidden !important;
        overscroll-behavior: none;
        background: var(--nexrun-sidebar);
        border-right: 1px solid var(--nexrun-border);
    }
    section[data-testid="stSidebar"] [data-testid="stSidebarUserContent"] {
        box-sizing: border-box;
        height: 100%;
        padding: 24px 18px;
        overflow-y: hidden !important;
        overscroll-behavior: none;
    }
    section[data-testid="stSidebar"] [data-testid="stRadio"] > label {display: none;}
    section[data-testid="stSidebar"] [data-testid="stRadio"] label {
        display: flex;
        box-sizing: border-box;
        width: 100%;
        min-height: 46px;
        padding: 0 .85rem;
        margin: 0 0 6px;
        align-items: center;
        justify-content: flex-start;
        border-radius: 10px;
        cursor: pointer;
        transition: background 140ms ease, color 140ms ease, transform 140ms ease;
    }
    section[data-testid="stSidebar"] [data-testid="stRadio"]
    label[data-testid="stRadioOption"] > div > div:first-child {display: none !important;}
    section[data-testid="stSidebar"] [data-testid="stRadio"] label p {
        display: flex;
        align-items: center;
        gap: .62rem;
        color: var(--nexrun-muted);
        font-size: .88rem;
        font-weight: 650;
    }
    section[data-testid="stSidebar"] [data-testid="stRadio"] label p span {color: var(--nexrun-muted); font-size: 20px;}
    section[data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked) {
        background: var(--nexrun-primary-soft);
        box-shadow: inset 3px 0 0 var(--nexrun-primary);
    }
    section[data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked) p,
    section[data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked) p span {color: var(--nexrun-primary);}
    section[data-testid="stSidebar"] [data-testid="stRadio"] label:hover {background: var(--nexrun-surface); transform: translateX(2px);}
    section[data-testid="stSidebar"] [data-testid="stRadio"] label:active {background: var(--nexrun-border);}
    @media (min-width: 901px) {
        [data-testid="stSidebarCollapseButton"], [data-testid="collapsedControl"] {display: none;}
    }
    @media (max-width: 1100px) {
        .block-container {padding-left: 1.25rem; padding-right: 1.25rem;}
        section[data-testid="stSidebar"] {width: 208px; min-width: 208px !important; max-width: 208px !important; flex-basis: 208px !important;}
        [data-testid="stMetric"] {padding: .82rem .72rem;}
        [data-testid="stMetricLabel"] p {font-size: .7rem !important;}
        [data-testid="stMetricValue"], [data-testid="stMetricValue"] * {font-size: 1.35rem !important;}
    }
    @media (max-width: 900px) {
        .block-container {padding: 1rem .9rem 3rem;}
        [data-testid="stHorizontalBlock"] {flex-wrap: wrap !important; gap: .8rem !important;}
        [data-testid="stColumn"] {min-width: min(100%, 290px) !important; flex: 1 1 290px !important;}
        .workspace-status {justify-content: flex-start;}
        .workspace-title {font-size: 1.55rem;}
        [data-testid="stSidebarCollapseButton"], [data-testid="collapsedControl"] {display: flex !important;}
        [data-testid="stSidebarCollapseButton"] button,
        [data-testid="collapsedControl"] button {
            background: var(--nexrun-surface) !important;
            border: 1px solid var(--nexrun-border) !important;
            border-radius: 9px !important;
            color: var(--nexrun-text) !important;
        }
        [data-testid="stSidebarCollapseButton"] button *,
        [data-testid="collapsedControl"] button * {color: var(--nexrun-text) !important;}
    }
    </style>
    """.replace("__TOKENS__", _css_tokens(tokens)).replace(
        "__SYSTEM_OVERRIDE__",
        system_override,
    )
    st.markdown(css, unsafe_allow_html=True)


def render_page_header(
    eyebrow: str,
    title: str,
    description: str,
    *,
    status: str | None = None,
    status_variant: str = "default",
) -> None:
    """Render the consistent page heading used by each workbench route."""

    st.markdown(f'<div class="sensor-eyebrow">{eyebrow}</div>', unsafe_allow_html=True)
    title_column, status_column = st.columns([4, 1], vertical_alignment="center")
    with title_column:
        st.header(title)
        st.caption(description)
    if status:
        with status_column:
            ui.badge(status, variant=status_variant, width="stretch")


def render_metric_card(
    label: str,
    value: Any,
    *,
    description: str | None = None,
    delta: str | None = None,
    variant: str = "default",
    key: str | None = None,
) -> None:
    """Render a shadcn metric card with a safe fallback for older runtimes."""

    try:
        ui.metric_card(
            label,
            value,
            description=description,
            delta=delta,
            variant=variant,
            key=key,
            width="stretch",
        )
    except (RuntimeError, TypeError, AttributeError):
        st.metric(label, value, delta=delta, help=description)


def render_status_badge(
    text: str,
    *,
    variant: str = "default",
    key: str | None = None,
) -> None:
    """Render a compact status badge."""

    try:
        ui.badge(text, variant=variant, key=key)
    except (RuntimeError, TypeError, AttributeError):
        st.caption(text)


def render_section_card(
    title: str,
    description: str | None = None,
    *,
    footer: str | None = None,
    key: str | None = None,
) -> None:
    """Render a visual section header; content remains in the native Streamlit flow."""

    try:
        ui.card(title=title, description=description, footer=footer, key=key)
    except (RuntimeError, TypeError, AttributeError):
        st.subheader(title)
        if description:
            st.caption(description)


def render_empty_state(title: str, description: str) -> None:
    """Show a calm empty state instead of a generic info alert."""

    try:
        ui.alert(title, description, variant="default", width="stretch")
    except (RuntimeError, TypeError, AttributeError):
        st.info(f"{title}：{description}")


def render_alert(
    title: str,
    description: str,
    *,
    variant: str = "default",
) -> None:
    """Render an alert using the shared status palette."""

    try:
        ui.alert(
            title,
            description,
            variant="destructive" if variant in {"destructive", "error"} else "default",
            width="stretch",
        )
    except (RuntimeError, TypeError, AttributeError, ValueError):
        if variant in {"destructive", "error"}:
            st.error(f"{title}：{description}")
        elif variant in {"warning", "secondary"}:
            st.warning(f"{title}：{description}")
        else:
            st.info(f"{title}：{description}")


def render_result_table(
    data: Any,
    *,
    caption: str | None = None,
    max_height: int | None = None,
    key: str | None = None,
) -> None:
    """Render a compact shadcn table while preserving native dataframe fallback."""

    try:
        ui.table(data, caption=caption, max_height=max_height, key=key, width="stretch")
    except (RuntimeError, TypeError, AttributeError):
        display_dataframe(data, width="stretch", hide_index=True, height=max_height)


def render_confirm_dialog(
    title: str,
    description: str,
    *,
    confirm_label: str = "我确认继续",
    key: str | None = None,
) -> bool:
    """Render a confirmation gate that keeps destructive actions explicit."""

    with st.container(border=True):
        render_alert(title, description, variant="destructive")
        return bool(st.checkbox(confirm_label, key=key))


def render_select(
    label: str,
    options: Iterable[T],
    *,
    value: T | None = None,
    index: int | None = 0,
    format_func: Callable[[T], str] = str,
    key: str | None = None,
) -> T | None:
    """Render a shadcn select with a native select fallback."""

    try:
        return ui.select(
            label,
            options,
            value=value,
            index=index,
            format_func=format_func,
            key=key,
            width="stretch",
        )
    except (RuntimeError, TypeError, AttributeError):
        options_list = list(options)
        native_index = index if index is not None else 0
        return st.selectbox(label, options_list, index=native_index, format_func=format_func, key=key)


def render_theme_selector(
    *,
    key: str = "ui_theme",
    show_help: bool = True,
) -> str:
    """Render the user-facing theme preference control."""

    options = ["跟随系统", "浅色", "深色"]
    current = st.session_state.get(key, options[0])
    selected = st.selectbox("界面主题", options, index=options.index(current), key=key)
    if show_help:
        st.caption("浅色/深色主题会在页面重新渲染后应用。")
    return str(selected)
