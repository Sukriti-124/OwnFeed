"""OwnFeed demo: the same reader's feed, side by side. Normal feed vs. OwnFeed following your sliders.

Run from the repo root:  .venv/bin/streamlit run app/streamlit_app.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from src.data import BUCKETS
from src.feed import DEMO_ASSETS, PERSONAS, FeedEngine, FeedSession, build_demo_assets

# Warm editorial palette: cream paper, ink text, earthy topic colors
INK, MUTED, RULE, PAPER, ACCENT = "#1E1B16", "#6B6255", "#E5DCCB", "#FFFDF8", "#C2410C"
BUCKET_COLORS = {"News": "#1F4E79", "Learn": "#2F6B4F", "Lifestyle": "#B7791F", "Entertain": "#9D3B6B", "Sports": "#C2410C"}
SLIDER_COLOR, PAGE_COLOR = "#CFC4B0", INK
SIM_COLORS = {"Normal feed": "#8C8273", "OwnFeed (sliders only)": "#B7791F", "OwnFeed + balancer": ACCENT}
SIM_PAGES, SIM_RUNS = 10, 20

st.set_page_config(page_title="OwnFeed", page_icon="📰", layout="wide")
st.markdown(f"""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500;9..144,700&family=Inter:wght@400;500;600&display=swap');
  html, body, .stMarkdown, .stCaption, label, button {{ font-family: 'Inter', sans-serif; }}
  h1, h2, h3 {{ font-family: 'Fraunces', Georgia, serif !important; color: {INK}; letter-spacing: -0.01em; }}
  header[data-testid="stHeader"] {{ display: none; }}
  .block-container {{ padding-top: 2.6rem; max-width: 1250px; }}
  .brand {{ font-family: 'Fraunces', Georgia, serif; font-size: 44px; font-weight: 700; color: {INK}; line-height: 1; }}
  .tagline {{ font-family: 'Fraunces', Georgia, serif; font-style: italic; color: {MUTED}; font-size: 18px; margin: 6px 0 4px; }}
  .lede {{ color: {INK}; font-size: 15px; max-width: 760px; }}
  .step {{ font-size: 11px; font-weight: 600; letter-spacing: .14em; text-transform: uppercase; color: {MUTED};
           border-top: 1px solid {RULE}; padding-top: 12px; margin: 18px 0 6px; }}
  .masthead {{ border-bottom: 3px double {INK}; padding-bottom: 6px; margin-bottom: 6px; }}
  .masthead .name {{ font-family: 'Fraunces', Georgia, serif; font-size: 26px; font-weight: 700; color: {INK}; }}
  .masthead .sub {{ font-size: 13px; color: {MUTED}; }}
  .story {{ border-top: 1px solid {RULE}; padding: 12px 2px 8px; }}
  .story:first-child {{ border-top: none; }}
  .kicker {{ font-size: 11px; font-weight: 600; letter-spacing: .12em; text-transform: uppercase; }}
  .headline {{ font-family: 'Fraunces', Georgia, serif; font-size: 17px; font-weight: 500; line-height: 1.25; color: {INK}; margin: 3px 0; }}
  .snippet {{ font-size: 12.5px; color: {MUTED}; line-height: 1.4; }}
  div[data-testid="stVerticalBlockBorderWrapper"] {{ background: {PAPER}; border-color: {RULE} !important; }}
  div.stButton > button {{ border-radius: 999px; border: 1px solid {INK}; background: {INK}; color: {PAPER}; font-weight: 500; }}
  div.stButton > button:hover, div.stButton > button:focus, div.stButton > button:active, div.stButton > button:focus-visible
    {{ background: {ACCENT} !important; border-color: {ACCENT} !important; color: {PAPER} !important; box-shadow: none !important; }}
  div.stButton > button p {{ color: inherit !important; }}
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Loading the model and today's articles…")
def load_engine() -> FeedEngine:
    if not DEMO_ASSETS.exists():
        build_demo_assets()
    return FeedEngine()


engine = load_engine()


def past_click_mix(persona: str) -> pd.Series:
    """Share of the reader's past clicks per topic."""
    history = engine.persona_history[persona]
    buckets = engine.article_buckets[history[history >= 0]]
    counts = np.bincount(buckets[buckets >= 0].numpy(), minlength=len(BUCKETS))
    return pd.Series(counts / counts.sum(), index=BUCKETS)


def mix_chart(page_share: np.ndarray, targets: np.ndarray) -> alt.Chart:
    """Two small bars per topic: your slider vs. this page."""
    data = pd.concat([
        pd.DataFrame({"Topic": BUCKETS, "Series": "Your slider", "Share": targets}),
        pd.DataFrame({"Topic": BUCKETS, "Series": "This page", "Share": page_share}),
    ])
    order = ["Your slider", "This page"]
    return (
        alt.Chart(data)
        .mark_bar(cornerRadiusEnd=3)
        .encode(
            y=alt.Y("Topic:N", sort=BUCKETS, title=None, scale=alt.Scale(paddingInner=0.35),
                    axis=alt.Axis(labelColor=INK, labelFontSize=12, ticks=False, domain=False, labelOverlap=False)),
            yOffset=alt.YOffset("Series:N", sort=order),
            x=alt.X("Share:Q", title=None, scale=alt.Scale(domain=[0, 1]),
                    axis=alt.Axis(format="%", labelColor=MUTED, gridColor=RULE, domain=False, tickCount=4)),
            color=alt.Color("Series:N", sort=order, scale=alt.Scale(domain=order, range=[SLIDER_COLOR, PAGE_COLOR]),
                            legend=alt.Legend(orient="top", title=None, labelColor=INK)),
            tooltip=["Topic", "Series", alt.Tooltip("Share:Q", format=".0%")],
        )
        .properties(height=alt.Step(13), background="transparent")
        .configure_view(stroke=None)
    )


def render_column(title: str, subtitle: str, session: FeedSession, targets: np.ndarray) -> None:
    st.markdown(f"<div class='masthead'><div class='name'>{title}</div><div class='sub'>{subtitle}</div></div>", unsafe_allow_html=True)
    st.altair_chart(mix_chart(session.page_mix(), targets), width="stretch")
    with st.container(height=560, border=True):
        stories = []
        for row in session.page.itertuples():
            snippet = row.abstract if len(row.abstract) <= 130 else row.abstract[:127] + "…"
            stories.append(
                f"<div class='story'><div class='kicker' style='color:{BUCKET_COLORS[row.bucket]}'>{row.bucket}</div>"
                f"<div class='headline'>{row.title}</div><div class='snippet'>{snippet}</div></div>"
            )
        st.markdown("".join(stories), unsafe_allow_html=True)


# ---------------- Header ----------------
st.markdown(
    "<div class='brand'>OwnFeed</div><div class='tagline'>Your feed, your rules.</div>"
    "<div class='lede'>Most feeds show you more of whatever you click. OwnFeed lets you set the topic mix you want, "
    "and builds the page to match it. Below: the <b>same reader</b> and the <b>same articles</b>, two different feeds.</div>",
    unsafe_allow_html=True,
)

# ---------------- Step 1 and 2: reader and sliders ----------------
st.markdown("<div class='step'>1 · Pick a reader</div>", unsafe_allow_html=True)
c_reader, c_about = st.columns([1, 2.4])
persona = c_reader.selectbox("Reader", list(engine.personas), label_visibility="collapsed")
past = past_click_mix(persona)
c_about.markdown(
    f"A real MIND reader. **{past.max():.0%} of their past clicks were {past.idxmax()}.** "
    f"Both feeds below know this from their click history."
)

st.markdown("<div class='step'>2 · Set the mix you want</div>", unsafe_allow_html=True)
slider_cols = st.columns(len(BUCKETS))
raw = np.array([col.slider(b, 0, 100, 20, 5, key=f"slider_{b}") for col, b in zip(slider_cols, BUCKETS)], dtype=float)
targets = raw / raw.sum() if raw.sum() else np.full(len(BUCKETS), 1 / len(BUCKETS))
st.caption("Mix used (scaled to 100%): " + " · ".join(f"{b} {t:.0%}" for b, t in zip(BUCKETS, targets)))

# ---------------- Step 3: side by side ----------------
st.markdown("<div class='step'>3 · Compare the two feeds</div>", unsafe_allow_html=True)
normal = FeedSession(engine, persona, targets, mode="Normal feed")
ownfeed = FeedSession(engine, persona, targets, mode="Sliders only")
normal.next_page()
ownfeed.next_page()

left, right = st.columns(2, gap="large")
with left:
    render_column("Normal feed", "Shows what this reader is most likely to click. Ignores your sliders.", normal, targets)
with right:
    render_column("OwnFeed", "Builds the page to match your sliders, picking the best articles in each topic.", ownfeed, targets)

# ---------------- Step 4: what they actually read ----------------
st.markdown("<div class='step'>4 · Go further: what does the reader actually read?</div>", unsafe_allow_html=True)
favorite = PERSONAS[persona]
st.markdown(
    f"Showing your mix is not the whole story: people still click their favorite topic more. "
    f"This runs **{SIM_RUNS} simulated sessions** of this reader scrolling **{SIM_PAGES} pages** "
    f"(a fake twin that clicks the way they do) and tracks how far their *reading* is from your mix. "
    f"**OwnFeed + balancer** watches the clicks and shows a little less {favorite} and more of what they skip, "
    f"so the reading lands closer to your sliders."
)


@st.cache_data(show_spinner="Simulating sessions…")
def simulate_reading(persona: str, targets: tuple) -> tuple[pd.DataFrame, pd.DataFrame]:
    gaps, reads = [], []
    for name, mode in [("Normal feed", "Normal feed"), ("OwnFeed (sliders only)", "Sliders only"), ("OwnFeed + balancer", "OwnFeed")]:
        for run in range(SIM_RUNS):
            session = FeedSession(engine, persona, np.array(targets), mode=mode, seed=run)
            for page in range(1, SIM_PAGES + 1):
                session.next_page()
                session.twin_reads()
                gap = session.gap()
                if gap is not None:
                    gaps.append({"Feed": name, "Page": page, "Gap": gap})
            reads.append(pd.Series(session.read_counts, index=BUCKETS, name=name))
    gap_df = pd.DataFrame(gaps).groupby(["Feed", "Page"], as_index=False).Gap.mean()
    read_df = pd.concat(reads, axis=1).T.groupby(level=0).sum()
    return gap_df, read_df.div(read_df.sum(1), axis=0)


if st.button(f"Simulate {SIM_PAGES} pages of reading"):
    st.session_state.simulated = (persona, tuple(targets))

if st.session_state.get("simulated") == (persona, tuple(targets)):
    gap_df, read_mix = simulate_reading(persona, tuple(targets))
    order = list(SIM_COLORS)
    line = (
        alt.Chart(gap_df)
        .mark_line(strokeWidth=2.5, point=alt.OverlayMarkDef(size=40, filled=True))
        .encode(
            x=alt.X("Page:O", title="Page scrolled", axis=alt.Axis(labelAngle=0, labelColor=MUTED, titleColor=MUTED)),
            y=alt.Y("Gap:Q", title="Gap between reading and your mix", axis=alt.Axis(format="%", labelColor=MUTED, titleColor=MUTED, gridColor=RULE)),
            color=alt.Color("Feed:N", sort=order, scale=alt.Scale(domain=order, range=list(SIM_COLORS.values())),
                            legend=alt.Legend(orient="top", title=None, labelColor=INK)),
            tooltip=["Feed", "Page", alt.Tooltip("Gap:Q", format=".0%")],
        )
        .properties(height=260, background="transparent")
        .configure_view(stroke=None)
    )
    c_line, c_table = st.columns([1.6, 1], gap="large")
    c_line.altair_chart(line, width="stretch")
    with c_table:
        st.markdown(f"**Share of reading that was {favorite}** (your slider: {targets[BUCKETS.index(favorite)]:.0%})")
        for name in order:
            st.markdown(f"- {name}: **{read_mix.loc[name, favorite]:.0%}**")
        st.caption("Lower gap = what they read is closer to the mix you asked for. Averaged over the simulated sessions.")
