"""
╔═══════════════════════════════════════════════════════════════════╗
║  NEXUS ANALYTICS — Multi-School Academic Intelligence Platform    ║
║  60+ interactive charts across 7 pages, 10 predictive ML models   ║
╚═══════════════════════════════════════════════════════════════════╝
"""

from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from sklearn.linear_model      import LinearRegression, LogisticRegression
from sklearn.ensemble          import RandomForestClassifier, IsolationForest
from sklearn.cluster           import KMeans
from sklearn.preprocessing     import StandardScaler
from sklearn.metrics           import accuracy_score, confusion_matrix

st.set_page_config(
    page_title="Nexus Analytics",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ═══════════════════════════════════════════════════════════════════
#  THEME SYSTEM — every page has its own visual identity
# ═══════════════════════════════════════════════════════════════════
THEMES = {
    "Executive Overview": {
        "icon": "🏛",
        "subtitle": "Strategic snapshot across the academic network",
        "primary": "#4F46E5", "accent": "#8B5CF6",
        "grad": ("#6366F1", "#8B5CF6"),
        "palette": ["#4F46E5", "#7C3AED", "#EC4899", "#F59E0B", "#10B981", "#06B6D4"],
        "scale":   [[0, "#EEF2FF"], [1, "#4F46E5"]],
    },
    "Students 360°": {
        "icon": "🎓",
        "subtitle": "Academic standing, demographics and behavioural profile",
        "primary": "#059669", "accent": "#0D9488",
        "grad": ("#10B981", "#0D9488"),
        "palette": ["#059669", "#0D9488", "#10B981", "#34D399", "#6EE7B7", "#065F46"],
        "scale":   [[0, "#ECFDF5"], [1, "#059669"]],
    },
    "Academic Performance": {
        "icon": "📚",
        "subtitle": "Exam results, subjects, terms and grade distribution",
        "primary": "#D97706", "accent": "#EA580C",
        "grad": ("#F59E0B", "#EA580C"),
        "palette": ["#D97706", "#EA580C", "#F59E0B", "#FB923C", "#FBBF24", "#92400E"],
        "scale":   [[0, "#FEF3C7"], [1, "#D97706"]],
    },
    "Faculty Analytics": {
        "icon": "👩‍🏫",
        "subtitle": "Teacher distribution, ratings, workload and experience",
        "primary": "#BE123C", "accent": "#DB2777",
        "grad": ("#E11D48", "#DB2777"),
        "palette": ["#BE123C", "#DB2777", "#E11D48", "#F43F5E", "#FB7185", "#881337"],
        "scale":   [[0, "#FCE7F3"], [1, "#BE123C"]],
    },
    "Attendance Intelligence": {
        "icon": "📅",
        "subtitle": "Presence patterns, trends and absence intelligence",
        "primary": "#0284C7", "accent": "#0891B2",
        "grad": ("#0EA5E9", "#0891B2"),
        "palette": ["#0284C7", "#0891B2", "#0EA5E9", "#06B6D4", "#38BDF8", "#075985"],
        "scale":   [[0, "#E0F2FE"], [1, "#0284C7"]],
    },
    "School Network": {
        "icon": "🏫",
        "subtitle": "Institutions, boards, geography and benchmarks",
        "primary": "#7C3AED", "accent": "#A855F7",
        "grad": ("#8B5CF6", "#A855F7"),
        "palette": ["#7C3AED", "#A855F7", "#8B5CF6", "#A78BFA", "#C4B5FD", "#4C1D95"],
        "scale":   [[0, "#F3E8FF"], [1, "#7C3AED"]],
    },
    "Predictive Lab": {
        "icon": "🔮",
        "subtitle": "Machine-learning forecasts, simulations and explainability",
        "primary": "#0F766E", "accent": "#14B8A6",
        "grad": ("#14B8A6", "#6366F1"),
        "palette": ["#0F766E", "#14B8A6", "#6366F1", "#8B5CF6", "#F59E0B", "#EC4899"],
        "scale":   [[0, "#F0FDFA"], [1, "#0F766E"]],
    },
}

# ═══════════════════════════════════════════════════════════════════
#  GLOBAL CSS — production-grade polish
# ═══════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');

html, body, [class*="css"] { font-family: 'Inter', system-ui, sans-serif; }

.stApp { background: linear-gradient(180deg, #F8FAFC 0%, #EEF2F7 100%); }

.main .block-container {
    padding: 1.5rem 2.5rem 4rem 2.5rem;
    max-width: 1480px;
}

/* ── Metric cards ──────────────────────────────── */
[data-testid="stMetric"] {
    background: #ffffff;
    padding: 1.1rem 1.3rem;
    border-radius: 14px;
    box-shadow: 0 1px 3px rgba(15,23,42,0.06), 0 1px 2px rgba(15,23,42,0.04);
    border-left: 4px solid var(--accent, #4F46E5);
    transition: all .2s ease;
    position: relative; overflow: hidden;
}
[data-testid="stMetric"]::before {
    content: ''; position: absolute; top: 0; right: 0;
    width: 60px; height: 60px;
    background: radial-gradient(circle, var(--accent, #4F46E5)15 0%, transparent 70%);
    opacity: 0.08;
}
[data-testid="stMetric"]:hover {
    transform: translateY(-3px);
    box-shadow: 0 8px 20px rgba(15,23,42,0.10);
}
[data-testid="stMetricLabel"] p {
    font-size: 0.72rem !important; color: #64748B !important;
    text-transform: uppercase; letter-spacing: 0.06em;
    font-weight: 600 !important;
}
[data-testid="stMetricValue"] {
    font-size: 1.75rem !important; font-weight: 700 !important; color: #0F172A !important;
}
[data-testid="stMetricDelta"] {
    font-size: 0.8rem !important; font-weight: 600 !important;
}

/* ── Chart cards ──────────────────────────────── */
[data-testid="stPlotlyChart"] {
    background: #ffffff; border-radius: 14px;
    padding: 0.75rem;
    box-shadow: 0 1px 3px rgba(15,23,42,0.05), 0 1px 2px rgba(15,23,42,0.03);
    margin-bottom: 1rem;
    transition: box-shadow .2s ease;
}
[data-testid="stPlotlyChart"]:hover {
    box-shadow: 0 4px 12px rgba(15,23,42,0.08);
}

/* ── Tabs ──────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {
    gap: 0.3rem; background: transparent;
    border-bottom: 1px solid #E2E8F0; padding: 0;
    margin-bottom: 1rem;
}
.stTabs [data-baseweb="tab"] {
    background: transparent; padding: 0.7rem 1.2rem;
    font-weight: 600; color: #64748B;
    border-radius: 8px 8px 0 0;
    border: none; transition: all .15s;
}
.stTabs [data-baseweb="tab"]:hover { background: #F1F5F9; color: #0F172A; }
.stTabs [aria-selected="true"] {
    background: white !important;
    color: var(--accent, #4F46E5) !important;
    border-bottom: 2px solid var(--accent, #4F46E5) !important;
}

/* ── Sidebar ──────────────────────────────── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0F172A 0%, #1E293B 100%);
}
[data-testid="stSidebar"] * { color: #E2E8F0 !important; }
[data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
    color: #F8FAFC !important;
}
[data-testid="stSidebar"] label {
    font-size: 0.75rem !important; font-weight: 600 !important;
    color: #94A3B8 !important; text-transform: uppercase;
    letter-spacing: 0.05em;
}
[data-testid="stSidebar"] [data-baseweb="select"] > div,
[data-testid="stSidebar"] [data-baseweb="input"] > div {
    background: rgba(255,255,255,0.06) !important;
    border-color: rgba(255,255,255,0.12) !important;
}

footer { visibility: hidden; }

/* ── Column gutters ──────────────────────────────── */
[data-testid="column"] { padding: 0 0.5rem !important; }

/* ── Section label ──────────────────────────────── */
.section-label {
    font-size: 0.78rem; font-weight: 700; color: #64748B;
    text-transform: uppercase; letter-spacing: 0.09em;
    margin: 1.4rem 0 0.6rem 0; padding-left: 4px;
    border-left: 3px solid var(--accent, #4F46E5);
    padding: 4px 0 4px 12px;
}

/* ── Insight box ──────────────────────────────── */
.insight-box {
    background: linear-gradient(135deg, #F8FAFC 0%, #F1F5F9 100%);
    border-left: 3px solid var(--accent, #4F46E5);
    padding: 0.9rem 1.2rem; border-radius: 8px;
    font-size: 0.9rem; color: #334155;
    margin: 0.6rem 0 1rem 0;
    box-shadow: 0 1px 2px rgba(15,23,42,0.03);
}
.insight-box strong { color: #0F172A; }

/* ── Inline chip/tag ──────────────────────────────── */
.chip {
    display: inline-block; padding: 2px 10px;
    border-radius: 999px; font-size: 0.75rem;
    font-weight: 600; background: var(--accent-light, #EEF2FF);
    color: var(--accent, #4F46E5); margin-right: 4px;
}

/* ── KPI subtitle ──────────────────────────────── */
.kpi-caption {
    font-size: 0.75rem; color: #64748B;
    margin-top: -0.5rem; margin-bottom: 0.5rem;
    padding-left: 4px;
}
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════
#  HELPERS
# ═══════════════════════════════════════════════════════════════════
def apply_theme(t):
    st.markdown(
        f"<style>:root {{ --accent: {t['primary']}; --accent-light: {t['primary']}15; }} "
        f"[data-testid='stMetric'] {{ border-left-color: {t['primary']}; }}</style>",
        unsafe_allow_html=True,
    )

def render_header(page: str):
    t = THEMES[page]
    st.markdown(f"""
    <div style="
        background: linear-gradient(135deg, {t['grad'][0]} 0%, {t['grad'][1]} 100%);
        padding: 1.8rem 2.2rem; border-radius: 20px;
        margin-bottom: 1.5rem; color: white; position: relative; overflow: hidden;
        box-shadow: 0 12px 32px -12px {t['primary']}66;">
        <div style="position:absolute; top:-40px; right:-40px; width:180px; height:180px;
                    background: rgba(255,255,255,0.08); border-radius: 50%;"></div>
        <div style="position:absolute; bottom:-60px; right:80px; width:120px; height:120px;
                    background: rgba(255,255,255,0.05); border-radius: 50%;"></div>
        <div style="position:relative; z-index:1;">
            <div style="font-size:.75rem; opacity:.85; letter-spacing:.18em;
                        text-transform:uppercase; margin-bottom:.4rem; font-weight:600;">
                Nexus Analytics · {t['icon']}
            </div>
            <div style="font-size:2rem; font-weight:800; line-height:1.1; letter-spacing:-0.02em;">
                {page}
            </div>
            <div style="font-size:1rem; opacity:.9; margin-top:.45rem; font-weight:400;">
                {t['subtitle']}
            </div>
        </div>
    </div>""", unsafe_allow_html=True)

def section(label: str):
    st.markdown(f"<div class='section-label'>{label}</div>", unsafe_allow_html=True)

def insight(html: str):
    st.markdown(f"<div class='insight-box'>💡 {html}</div>", unsafe_allow_html=True)

def style_fig(fig, theme, *, show_legend=True, height=None, bargap=0.18):
    layout = dict(
        font_family='"Inter", system-ui, sans-serif',
        font_color="#1E293B", title_font_color="#0F172A",
        title_font_size=14, title_x=0.02, title_xanchor="left",
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=30, r=20, t=55, b=30),
        colorway=theme["palette"],
        showlegend=show_legend, bargap=bargap,
        legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="rgba(0,0,0,0)",
                    orientation="h", yanchor="bottom", y=1.02,
                    xanchor="right", x=1, font=dict(size=11)),
        hoverlabel=dict(bgcolor="white", font_size=12,
                        font_family='"Inter", sans-serif',
                        bordercolor=theme["primary"]),
    )
    if height: layout["height"] = height
    fig.update_layout(**layout)
    fig.update_xaxes(gridcolor="#F1F5F9", zerolinecolor="#E2E8F0",
                     linecolor="#E2E8F0", tickfont=dict(size=11))
    fig.update_yaxes(gridcolor="#F1F5F9", zerolinecolor="#E2E8F0",
                     linecolor="#E2E8F0", tickfont=dict(size=11))
    return fig

def add_trendline(fig, x, y, *, color, name="Trend"):
    xa, ya = np.asarray(x, float), np.asarray(y, float)
    m = ~(np.isnan(xa) | np.isnan(ya))
    if m.sum() < 2: return fig
    a, b = np.polyfit(xa[m], ya[m], 1)
    xl = np.linspace(xa[m].min(), xa[m].max(), 80)
    fig.add_trace(go.Scatter(x=xl, y=a*xl+b, mode="lines",
                             name=f"{name} · slope {a:.3f}",
                             line=dict(color=color, dash="dash", width=2.5)))
    return fig


# ═══════════════════════════════════════════════════════════════════
#  DATA LAYER
# ═══════════════════════════════════════════════════════════════════
@st.cache_data(show_spinner="Loading workbook…")
def load_workbook(src) -> dict:
    sheets = pd.read_excel(src, sheet_name=None)
    for n, f in sheets.items():
        f.columns = f.columns.astype(str).str.strip().str.lower().str.replace(" ", "_")
        sheets[n] = f
    return sheets

@st.cache_data(show_spinner="Building analytical views…")
def build_views(sheets: dict) -> dict:
    schools  = sheets["Schools"].rename(columns={"school_name": "school"})
    students = sheets["Students"].copy()
    records  = sheets["Student_Academic_Records"].copy()
    attend   = sheets["Attendance_Log"].copy()
    teachers = sheets["Teachers"].copy()
    principals = sheets["Principals"].copy()

    stu = students.merge(
        schools[["school_id","school","board_name","region","state","city",
                 "school_type","management_type","student_capacity"]],
        on="school_id", how="left")
    stu["performance_index"] = (stu["current_gpa"].fillna(0)*25
                                + stu["cumulative_attendance_pct"].fillna(0)*0.5)
    stu["age"] = 2026 - pd.to_datetime(stu["date_of_birth"], errors="coerce").dt.year

    rec = (records.merge(students[["student_id","gender"]], on="student_id", how="left")
                  .merge(schools[["school_id","school","board_name","region"]],
                         on="school_id", how="left"))

    att = attend.merge(schools[["school_id","school","region"]], on="school_id", how="left")
    att["attendance_date"] = pd.to_datetime(att["attendance_date"], errors="coerce")
    att["day_of_week"] = att["attendance_date"].dt.day_name()
    att["week"] = att["attendance_date"].dt.isocalendar().week

    teachers = teachers.merge(schools[["school_id","school","region"]],
                              on="school_id", how="left")

    return {"students": stu, "records": rec, "attendance": att,
            "teachers": teachers, "schools": schools, "principals": principals}

# ═══════════════════════════════════════════════════════════════════
#  ML MODELS — cached across interactions
# ═══════════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner="Training risk classifier…")
def train_risk_model(stu: pd.DataFrame):
    X = pd.DataFrame({
        "gpa":          stu["current_gpa"],
        "attendance":   stu["cumulative_attendance_pct"],
        "grade":        stu["grade_level"],
        "scholarship":  (stu["scholarship_flag"]=="Yes").astype(int),
        "iep":          (stu["iep_flag"]=="Yes").astype(int),
    })
    y = stu["academic_risk_flag"]
    lr = LogisticRegression(max_iter=2000).fit(X, y)
    rf = RandomForestClassifier(n_estimators=80, max_depth=6,
                                random_state=42, n_jobs=-1).fit(X, y)
    return {
        "lr": lr, "rf": rf, "features": X.columns.tolist(),
        "lr_acc": accuracy_score(y, lr.predict(X)),
        "rf_acc": accuracy_score(y, rf.predict(X)),
    }

@st.cache_resource(show_spinner="Training GPA regressor…")
def train_gpa_model(stu: pd.DataFrame):
    X = pd.DataFrame({
        "attendance":   stu["cumulative_attendance_pct"],
        "grade":        stu["grade_level"],
        "scholarship":  (stu["scholarship_flag"]=="Yes").astype(int),
        "iep":          (stu["iep_flag"]=="Yes").astype(int),
    })
    y = stu["current_gpa"]
    m = LinearRegression().fit(X, y)
    return {"model": m, "features": X.columns.tolist(), "r2": m.score(X, y),
            "coefs": dict(zip(X.columns, m.coef_)), "intercept": m.intercept_}

@st.cache_resource(show_spinner="Clustering cohort…")
def build_clusters(stu: pd.DataFrame, k: int = 4):
    X = stu[["current_gpa","cumulative_attendance_pct","grade_level"]].copy()
    X["scholarship"] = (stu["scholarship_flag"]=="Yes").astype(int)
    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)
    km = KMeans(n_clusters=k, random_state=42, n_init=10).fit(Xs)
    return {"model": km, "scaler": scaler, "labels": km.labels_,
            "centers": scaler.inverse_transform(km.cluster_centers_),
            "features": X.columns.tolist()}

@st.cache_resource(show_spinner="Scanning anomalies…")
def detect_anomalies(stu: pd.DataFrame):
    X = stu[["current_gpa","cumulative_attendance_pct","grade_level"]].values
    iso = IsolationForest(contamination=0.05, random_state=42).fit(X)
    scores = iso.score_samples(X)
    flags = iso.predict(X)  # -1 = anomaly
    return {"scores": scores, "flags": flags}


# ═══════════════════════════════════════════════════════════════════
#  SIDEBAR  (brand + uploader + filters + navigation)
# ═══════════════════════════════════════════════════════════════════
st.sidebar.markdown("""
<div style='padding:.5rem 0 1.4rem 0;'>
  <div style='font-size:1.55rem; font-weight:800; color:white; letter-spacing:-0.02em;'>
    🎓 Nexus <span style='color:#A78BFA;'>Analytics</span>
  </div>
  <div style='font-size:.72rem; color:#94A3B8; margin-top:.25rem;
              letter-spacing:.1em; text-transform:uppercase;'>
    Academic Intelligence Platform
  </div>
</div>""", unsafe_allow_html=True)

uploaded = st.sidebar.file_uploader("Workbook (.xlsx)", type="xlsx")
# Try several default filenames so the user can drop either into the repo
candidates = [Path("academic_realistic.xlsx"),
              Path("academic_multi_school_dashboard_populated_10000.xlsx")]
source = uploaded if uploaded is not None else next((p for p in candidates if p.exists()), None)

if source is None:
    st.title("Welcome to Nexus Analytics")
    st.info("Please upload the academic workbook in the sidebar to begin.")
    st.stop()

sheets = load_workbook(source)
views  = build_views(sheets)
stu, rec, att = views["students"], views["records"], views["attendance"]
teachers, schools, principals = views["teachers"], views["schools"], views["principals"]

st.sidebar.markdown("---")
st.sidebar.markdown("#### 🎛 Filters")

f_school = st.sidebar.multiselect("School", sorted(stu["school"].dropna().unique()))
f_region = st.sidebar.multiselect("Region", sorted(stu["region"].dropna().unique()))
f_grade  = st.sidebar.multiselect("Grade",  sorted(stu["grade_level"].dropna().unique()))
f_risk   = st.sidebar.multiselect("Risk",   sorted(stu["academic_risk_flag"].dropna().unique()))
f_gender = st.sidebar.multiselect("Gender", sorted(stu["gender"].dropna().unique()))

def filter_students(df):
    if f_school: df = df[df["school"].isin(f_school)]
    if f_region: df = df[df["region"].isin(f_region)]
    if f_grade:  df = df[df["grade_level"].isin(f_grade)]
    if f_risk:   df = df[df["academic_risk_flag"].isin(f_risk)]
    if f_gender: df = df[df["gender"].isin(f_gender)]
    return df

fstu = filter_students(stu)
ids  = set(fstu["student_id"])
frec = rec[rec["student_id"].isin(ids)]            if ids else rec.iloc[0:0]
fatt = att[att["stakeholder_id"].isin(ids)]        if ids else att.iloc[0:0]
ftch = teachers if not f_school else teachers[teachers["school"].isin(f_school)]

st.sidebar.markdown("---")
page = st.sidebar.radio("📂 Navigation", list(THEMES.keys()), label_visibility="collapsed")

if fstu.empty:
    st.error("⚠️ No rows match the current filters. Please clear some filters in the sidebar.")
    st.stop()

theme = THEMES[page]
apply_theme(theme)
render_header(page)


# ═══════════════════════════════════════════════════════════════════
#  PAGE: EXECUTIVE OVERVIEW
# ═══════════════════════════════════════════════════════════════════
if page == "Executive Overview":

    c1,c2,c3,c4,c5,c6 = st.columns(6)
    c1.metric("Schools",      stu["school"].nunique())
    c2.metric("Students",     f"{len(fstu):,}")
    c3.metric("Teachers",     f"{len(teachers):,}")
    c4.metric("Avg GPA",      round(fstu["current_gpa"].mean(), 2),
              delta=round(fstu["current_gpa"].mean() - stu["current_gpa"].mean(), 2))
    c5.metric("Attendance %", round(fstu["cumulative_attendance_pct"].mean(), 1))
    c6.metric("High Risk %",  f"{(fstu['academic_risk_flag']=='High').mean()*100:.1f}%")

    tab1, tab2, tab3 = st.tabs(["📈 Summary", "🏆 School Rankings", "🌐 Distribution"])

    # ── Tab 1 ────────────────────────────────────────────────────
    with tab1:
        section("📊 Core Distributions")
        col1, col2, col3 = st.columns(3)

        # Chart 1 — GPA histogram
        fig = px.histogram(fstu, x="current_gpa", nbins=25, title="GPA Distribution",
                           color_discrete_sequence=[theme["primary"]])
        fig.update_traces(marker_line_width=1, marker_line_color="white")
        fig.add_vline(x=fstu["current_gpa"].mean(), line_dash="dash",
                      line_color=theme["accent"],
                      annotation_text=f"Mean {fstu['current_gpa'].mean():.2f}",
                      annotation_position="top right")
        col1.plotly_chart(style_fig(fig, theme, show_legend=False, height=300, bargap=0.1),
                          use_container_width=True)

        # Chart 2 — Attendance histogram
        fig = px.histogram(fstu, x="cumulative_attendance_pct", nbins=25,
                           title="Attendance Distribution",
                           color_discrete_sequence=[theme["accent"]])
        fig.update_traces(marker_line_width=1, marker_line_color="white")
        fig.add_vline(x=fstu["cumulative_attendance_pct"].mean(), line_dash="dash",
                      line_color=theme["primary"],
                      annotation_text=f"Mean {fstu['cumulative_attendance_pct'].mean():.1f}%",
                      annotation_position="top right")
        col2.plotly_chart(style_fig(fig, theme, show_legend=False, height=300, bargap=0.1),
                          use_container_width=True)

        # Chart 3 — Performance index distribution
        fig = px.histogram(fstu, x="performance_index", nbins=25,
                           title="Performance Index",
                           color_discrete_sequence=[theme["palette"][2]])
        fig.update_traces(marker_line_width=1, marker_line_color="white")
        col3.plotly_chart(style_fig(fig, theme, show_legend=False, height=300, bargap=0.1),
                          use_container_width=True)

        section("🧭 Composition")
        col1, col2, col3 = st.columns(3)

        # Chart 4 — Risk donut
        rc = fstu["academic_risk_flag"].value_counts().reset_index()
        rc.columns = ["risk","count"]
        fig = px.pie(rc, names="risk", values="count", hole=0.6,
                     title="Risk Composition",
                     color="risk",
                     color_discrete_map={"Low":"#10B981","Medium":"#F59E0B","High":"#EF4444"})
        fig.update_traces(textposition="outside", textinfo="label+percent")
        fig.add_annotation(text=f"<b>{len(fstu):,}</b><br>students",
                           showarrow=False, font=dict(size=15, color="#0F172A"))
        col1.plotly_chart(style_fig(fig, theme, height=300), use_container_width=True)

        # Chart 5 — Gender split
        gc = fstu["gender"].value_counts().reset_index()
        gc.columns = ["gender","count"]
        fig = px.pie(gc, names="gender", values="count", hole=0.6,
                     title="Gender Composition",
                     color_discrete_sequence=theme["palette"])
        fig.update_traces(textposition="outside", textinfo="label+percent")
        col2.plotly_chart(style_fig(fig, theme, height=300), use_container_width=True)

        # Chart 6 — Grade level bar
        gl = fstu["grade_level"].value_counts().sort_index().reset_index()
        gl.columns = ["grade","count"]
        fig = px.bar(gl, x="grade", y="count", title="Students per Grade",
                     text_auto=True, color="count",
                     color_continuous_scale=theme["scale"])
        col3.plotly_chart(style_fig(fig, theme, show_legend=False, height=300),
                          use_container_width=True)

        section("🎯 Correlations at a glance")
        # Chart 7 — GPA vs Attendance scatter with regression line
        sample = fstu.sample(min(2500, len(fstu)), random_state=1)
        fig = px.scatter(sample, x="cumulative_attendance_pct", y="current_gpa",
                         color="academic_risk_flag", opacity=0.6,
                         title="GPA vs Attendance (coloured by risk)",
                         color_discrete_map={"Low":"#10B981","Medium":"#F59E0B","High":"#EF4444"},
                         hover_data=["school","grade_level"])
        fig = add_trendline(fig, sample["cumulative_attendance_pct"],
                            sample["current_gpa"], color=theme["primary"])
        st.plotly_chart(style_fig(fig, theme, height=420), use_container_width=True)

    # ── Tab 2 ────────────────────────────────────────────────────
    with tab2:
        by_school = (fstu.groupby("school")
                         .agg(students=("student_id","count"),
                              avg_gpa=("current_gpa","mean"),
                              avg_att=("cumulative_attendance_pct","mean"),
                              high_risk=("academic_risk_flag",
                                          lambda s: (s=="High").mean()*100))
                         .reset_index().sort_values("avg_gpa", ascending=False))

        section("🏆 Top performers")
        top_n = st.slider("Show top N schools", 5, len(by_school),
                          min(10, len(by_school)))

        # Chart 8 — Top schools by GPA
        fig = px.bar(by_school.head(top_n), x="school", y="avg_gpa",
                     title=f"Top {top_n} Schools by Average GPA",
                     text_auto=".2f", color="avg_gpa",
                     color_continuous_scale=theme["scale"],
                     hover_data=["students","avg_att","high_risk"])
        st.plotly_chart(style_fig(fig, theme, show_legend=False, height=420),
                        use_container_width=True)

        # Chart 9 — Quadrant bubble
        fig = px.scatter(by_school, x="avg_att", y="avg_gpa", size="students",
                         hover_name="school", title="School Quadrant · Attendance × GPA",
                         color="high_risk",
                         color_continuous_scale=[[0, "#10B981"], [0.5, "#F59E0B"], [1, "#EF4444"]],
                         labels={"high_risk":"High-risk %"})
        fig.add_hline(y=by_school["avg_gpa"].mean(), line_dash="dot", line_color="#94A3B8",
                      annotation_text="mean GPA", annotation_position="top left")
        fig.add_vline(x=by_school["avg_att"].mean(), line_dash="dot", line_color="#94A3B8",
                      annotation_text="mean attendance", annotation_position="bottom right")
        st.plotly_chart(style_fig(fig, theme, height=500), use_container_width=True)

    # ── Tab 3 ────────────────────────────────────────────────────
    with tab3:
        section("🌍 Geographic & board distribution")
        col1, col2 = st.columns(2)

        # Chart 10 — Region treemap
        regional = (fstu.groupby(["region","school"])
                        .size().reset_index(name="count"))
        fig = px.treemap(regional, path=["region","school"], values="count",
                         title="Student Distribution by Region / School",
                         color="count", color_continuous_scale=theme["scale"])
        fig.update_traces(marker=dict(cornerradius=5))
        col1.plotly_chart(style_fig(fig, theme, height=400), use_container_width=True)

        # Chart 11 — Board sunburst
        boards = (fstu.groupby(["board_name","school_type"])
                      .size().reset_index(name="count"))
        fig = px.sunburst(boards, path=["board_name","school_type"], values="count",
                          title="Board & School-Type Breakdown",
                          color="count", color_continuous_scale=theme["scale"])
        col2.plotly_chart(style_fig(fig, theme, height=400), use_container_width=True)


# ═══════════════════════════════════════════════════════════════════
#  PAGE: STUDENTS 360°
# ═══════════════════════════════════════════════════════════════════
elif page == "Students 360°":

    c1,c2,c3,c4,c5 = st.columns(5)
    c1.metric("In view",     f"{len(fstu):,}")
    c2.metric("High risk",   int((fstu["academic_risk_flag"]=="High").sum()))
    c3.metric("Scholarship", int((fstu["scholarship_flag"]=="Yes").sum()))
    c4.metric("IEP",         int((fstu["iep_flag"]=="Yes").sum()))
    c5.metric("Hostellers",  int((fstu["hostel_opted"]=="Yes").sum()))

    tab1, tab2, tab3 = st.tabs(["👥 Cohort", "🎯 Performance", "🔍 Segments"])

    # ── Tab 1 ────────────────────────────────────────────────────
    with tab1:
        section("Grade-level distributions")
        col1, col2 = st.columns(2)

        # Chart 12 — GPA by grade violin
        fig = px.violin(fstu, x="grade_level", y="current_gpa", box=True,
                        color_discrete_sequence=[theme["primary"]],
                        title="GPA Distribution by Grade (Violin + Box)")
        col1.plotly_chart(style_fig(fig, theme, show_legend=False, height=380),
                          use_container_width=True)

        # Chart 13 — Attendance by grade violin
        fig = px.violin(fstu, x="grade_level", y="cumulative_attendance_pct", box=True,
                        color_discrete_sequence=[theme["accent"]],
                        title="Attendance % by Grade (Violin + Box)")
        col2.plotly_chart(style_fig(fig, theme, show_legend=False, height=380),
                          use_container_width=True)

        section("Demographic splits")
        col1, col2 = st.columns(2)

        # Chart 14 — GPA by gender
        fig = px.box(fstu, x="gender", y="current_gpa", color="gender",
                     title="GPA by Gender", points=False,
                     color_discrete_sequence=theme["palette"])
        col1.plotly_chart(style_fig(fig, theme, show_legend=False, height=330),
                          use_container_width=True)

        # Chart 15 — Medium of instruction
        moi = fstu["medium_of_instruction"].value_counts().reset_index()
        moi.columns = ["medium","count"]
        fig = px.bar(moi, x="medium", y="count", title="Medium of Instruction",
                     text_auto=True, color="medium",
                     color_discrete_sequence=theme["palette"])
        col2.plotly_chart(style_fig(fig, theme, show_legend=False, height=330),
                          use_container_width=True)

    # ── Tab 2 ────────────────────────────────────────────────────
    with tab2:
        section("Scholarship & IEP impact")
        col1, col2 = st.columns(2)

        # Chart 16 — Scholarship vs GPA
        s_gpa = fstu.groupby("scholarship_flag")["current_gpa"].mean().reset_index()
        fig = px.bar(s_gpa, x="scholarship_flag", y="current_gpa",
                     title="Avg GPA · Scholarship vs Non-scholarship",
                     text_auto=".2f", color="scholarship_flag",
                     color_discrete_sequence=[theme["palette"][2], theme["primary"]])
        col1.plotly_chart(style_fig(fig, theme, show_legend=False, height=320),
                          use_container_width=True)

        # Chart 17 — IEP vs GPA
        i_gpa = fstu.groupby("iep_flag")["current_gpa"].mean().reset_index()
        fig = px.bar(i_gpa, x="iep_flag", y="current_gpa",
                     title="Avg GPA · IEP vs Non-IEP",
                     text_auto=".2f", color="iep_flag",
                     color_discrete_sequence=[theme["palette"][3], theme["palette"][5]])
        col2.plotly_chart(style_fig(fig, theme, show_legend=False, height=320),
                          use_container_width=True)

        # Chart 18 — Heatmap: GPA vs attendance density
        section("GPA × Attendance density")
        fig = px.density_heatmap(fstu, x="cumulative_attendance_pct", y="current_gpa",
                                 nbinsx=25, nbinsy=25,
                                 title="Joint Density · Attendance × GPA",
                                 color_continuous_scale=theme["scale"])
        st.plotly_chart(style_fig(fig, theme, show_legend=False, height=450),
                        use_container_width=True)

    # ── Tab 3 ────────────────────────────────────────────────────
    with tab3:
        section("Per-school performance")
        # Chart 19 — Sorted school GPA
        sc = fstu.groupby("school")["current_gpa"].mean().reset_index().sort_values(
            "current_gpa", ascending=True)
        fig = px.bar(sc, x="current_gpa", y="school", orientation="h",
                     title="Average GPA by School", text_auto=".2f",
                     color="current_gpa", color_continuous_scale=theme["scale"])
        st.plotly_chart(style_fig(fig, theme, show_legend=False, height=500),
                        use_container_width=True)

        section("Top & bottom 10 students")
        col1, col2 = st.columns(2)
        top10 = fstu.nlargest(10, "performance_index")[
            ["student_id","full_name","school","current_gpa",
             "cumulative_attendance_pct","performance_index"]]
        bot10 = fstu.nsmallest(10, "performance_index")[
            ["student_id","full_name","school","current_gpa",
             "cumulative_attendance_pct","performance_index"]]
        col1.markdown("**🏆 Top 10 performers**")
        col1.dataframe(top10, use_container_width=True, hide_index=True)
        col2.markdown("**📉 Bottom 10 performers**")
        col2.dataframe(bot10, use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════════════════════════════
#  PAGE: ACADEMIC PERFORMANCE
# ═══════════════════════════════════════════════════════════════════
elif page == "Academic Performance":

    c1,c2,c3,c4,c5 = st.columns(5)
    c1.metric("Exam records", f"{len(frec):,}")
    c2.metric("Pass rate",    f"{(frec['pass_fail']=='Pass').mean()*100:.1f}%")
    c3.metric("Avg %",        round(frec["percentage"].mean(), 1))
    c4.metric("Subjects",     frec["subject_name"].nunique())
    c5.metric("Terms",        frec["term_name"].nunique())

    tab1, tab2, tab3 = st.tabs(["🧾 Results", "📘 Subjects", "📈 Term Trends"])

    # ── Tab 1 ────────────────────────────────────────────────────
    with tab1:
        section("Grade & pass distributions")
        col1, col2 = st.columns(2)

        # Chart 20 — Grade letter distribution
        order = ["A1","A2","B1","B2","C1","C2","D","E"]
        gd = frec["grade_awarded"].value_counts().reindex(order, fill_value=0).reset_index()
        gd.columns = ["grade","count"]
        fig = px.bar(gd, x="grade", y="count", text_auto=True,
                     title="Letter Grade Distribution",
                     color="count", color_continuous_scale=theme["scale"])
        col1.plotly_chart(style_fig(fig, theme, show_legend=False, height=340),
                          use_container_width=True)

        # Chart 21 — Pass/fail donut
        pf = frec["pass_fail"].value_counts().reset_index()
        pf.columns = ["status","count"]
        fig = px.pie(pf, names="status", values="count", hole=0.6,
                     title="Pass / Fail Split",
                     color="status",
                     color_discrete_map={"Pass":"#10B981","Fail":"#EF4444"})
        fig.update_traces(textposition="outside", textinfo="label+percent+value")
        col2.plotly_chart(style_fig(fig, theme, height=340), use_container_width=True)

        section("Percentage distribution")
        # Chart 22 — Percentage histogram
        fig = px.histogram(frec, x="percentage", nbins=30, title="Exam Percentage Distribution",
                           color_discrete_sequence=[theme["primary"]])
        fig.update_traces(marker_line_width=1, marker_line_color="white")
        fig.add_vline(x=35, line_dash="dash", line_color="#EF4444",
                      annotation_text="Pass threshold (35%)",
                      annotation_position="top right")
        st.plotly_chart(style_fig(fig, theme, show_legend=False, height=360, bargap=0.1),
                        use_container_width=True)

    # ── Tab 2 ────────────────────────────────────────────────────
    with tab2:
        section("Subject-level performance")
        col1, col2 = st.columns(2)

        # Chart 23 — Box by subject
        fig = px.box(frec, x="subject_name", y="percentage", color="subject_name",
                     title="Score Distribution by Subject", points=False,
                     color_discrete_sequence=theme["palette"])
        col1.plotly_chart(style_fig(fig, theme, show_legend=False, height=380),
                          use_container_width=True)

        # Chart 24 — Subject pass rate
        sp = (frec.groupby("subject_name")
                  .apply(lambda d: (d["pass_fail"]=="Pass").mean()*100, include_groups=False)
                  .reset_index(name="pass_rate").sort_values("pass_rate"))
        fig = px.bar(sp, x="pass_rate", y="subject_name", orientation="h",
                     title="Pass Rate by Subject (%)", text_auto=".1f",
                     color="pass_rate", color_continuous_scale=theme["scale"])
        col2.plotly_chart(style_fig(fig, theme, show_legend=False, height=380),
                          use_container_width=True)

        # Chart 25 — Radar: subject profile
        section("Subject profile")
        subj_avg = frec.groupby("subject_name").agg(
            exam_pct=("percentage","mean"),
            assignment=("assignment_score","mean"),
            project=("project_score","mean"),
        ).reset_index()
        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(r=subj_avg["exam_pct"],
                                      theta=subj_avg["subject_name"],
                                      fill='toself', name="Exam %",
                                      line_color=theme["primary"]))
        fig.add_trace(go.Scatterpolar(r=subj_avg["assignment"]*5,
                                      theta=subj_avg["subject_name"],
                                      fill='toself', name="Assignment×5",
                                      line_color=theme["palette"][2]))
        fig.add_trace(go.Scatterpolar(r=subj_avg["project"]*10,
                                      theta=subj_avg["subject_name"],
                                      fill='toself', name="Project×10",
                                      line_color=theme["palette"][3]))
        fig.update_layout(title="Subject Profile · Exam · Assignment · Project",
                          polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
                          paper_bgcolor="rgba(0,0,0,0)", height=440,
                          title_x=0.02, font_family='"Inter", sans-serif',
                          legend=dict(orientation="h", yanchor="bottom", y=-0.1))
        st.plotly_chart(fig, use_container_width=True)

        # Chart 26 — Scatter: assignment vs exam
        section("Assignments vs exam scores")
        sample = frec.sample(min(2000, len(frec)), random_state=1)
        fig = px.scatter(sample, x="assignment_score", y="marks_obtained",
                         color="subject_name", opacity=0.6,
                         title="Assignment Score vs Exam Marks",
                         color_discrete_sequence=theme["palette"])
        fig = add_trendline(fig, sample["assignment_score"], sample["marks_obtained"],
                            color=theme["primary"])
        st.plotly_chart(style_fig(fig, theme, height=440), use_container_width=True)

    # ── Tab 3 ────────────────────────────────────────────────────
    with tab3:
        # Chart 27 — Term progression by subject
        ts = frec.groupby(["term_name","subject_name"])["percentage"].mean().reset_index()
        fig = px.line(ts, x="term_name", y="percentage", color="subject_name",
                      markers=True, title="Term-over-Term Subject Averages",
                      color_discrete_sequence=theme["palette"])
        fig.update_traces(line_width=3, marker=dict(size=10))
        st.plotly_chart(style_fig(fig, theme, height=400), use_container_width=True)

        # Chart 28 — Pass rate by term
        tp = (frec.groupby("term_name")
                  .apply(lambda d: (d["pass_fail"]=="Pass").mean()*100, include_groups=False)
                  .reset_index(name="pass_rate"))
        fig = px.bar(tp, x="term_name", y="pass_rate", text_auto=".1f",
                     title="Pass Rate by Term (%)",
                     color="pass_rate", color_continuous_scale=theme["scale"])
        fig.update_layout(yaxis_range=[80, 100])
        st.plotly_chart(style_fig(fig, theme, show_legend=False, height=340),
                        use_container_width=True)

        # Chart 29 — Heatmap: subject × term
        pivot = frec.pivot_table(index="subject_name", columns="term_name",
                                  values="percentage", aggfunc="mean")
        fig = px.imshow(pivot, text_auto=".1f",
                        title="Avg % · Subject × Term Heatmap",
                        color_continuous_scale=theme["scale"], aspect="auto")
        st.plotly_chart(style_fig(fig, theme, show_legend=False, height=380),
                        use_container_width=True)


# ═══════════════════════════════════════════════════════════════════
#  PAGE: FACULTY ANALYTICS
# ═══════════════════════════════════════════════════════════════════
elif page == "Faculty Analytics":

    c1,c2,c3,c4,c5 = st.columns(5)
    c1.metric("Teachers",       len(ftch))
    c2.metric("Avg rating",     round(ftch["teacher_performance_rating"].mean(), 2))
    c3.metric("Avg exp (yrs)",  round(ftch["years_experience"].mean(), 1))
    c4.metric("Avg attendance", f"{ftch['teacher_attendance_pct'].mean():.1f}%")
    c5.metric("Departments",    ftch["department"].nunique())

    tab1, tab2, tab3 = st.tabs(["🏛 Composition", "⭐ Performance", "⚖ Workload"])

    # ── Tab 1 ────────────────────────────────────────────────────
    with tab1:
        section("Department structure")
        col1, col2 = st.columns(2)

        # Chart 30 — Dept bar
        dept = ftch["department"].value_counts().reset_index()
        dept.columns = ["department","count"]
        fig = px.bar(dept, x="department", y="count", text_auto=True,
                     color="department", title="Teachers per Department",
                     color_discrete_sequence=theme["palette"])
        col1.plotly_chart(style_fig(fig, theme, show_legend=False, height=340),
                          use_container_width=True)

        # Chart 31 — Employment type donut
        emp = ftch["employment_type"].value_counts().reset_index()
        emp.columns = ["type","count"]
        fig = px.pie(emp, names="type", values="count", hole=0.55,
                     title="Employment Type",
                     color_discrete_sequence=theme["palette"])
        fig.update_traces(textposition="outside", textinfo="label+percent")
        col2.plotly_chart(style_fig(fig, theme, height=340), use_container_width=True)

        section("Qualifications & gender")
        col1, col2 = st.columns(2)
        # Chart 32 — Qualification
        q = ftch["highest_qualification"].value_counts().reset_index()
        q.columns = ["qual","count"]
        fig = px.bar(q, x="count", y="qual", orientation="h", text_auto=True,
                     title="Highest Qualification", color="count",
                     color_continuous_scale=theme["scale"])
        col1.plotly_chart(style_fig(fig, theme, show_legend=False, height=340),
                          use_container_width=True)

        # Chart 33 — Gender donut
        g = ftch["gender"].value_counts().reset_index()
        g.columns = ["gender","count"]
        fig = px.pie(g, names="gender", values="count", hole=0.55,
                     title="Gender Split (Faculty)",
                     color_discrete_sequence=theme["palette"])
        fig.update_traces(textposition="outside", textinfo="label+percent")
        col2.plotly_chart(style_fig(fig, theme, height=340), use_container_width=True)

    # ── Tab 2 ────────────────────────────────────────────────────
    with tab2:
        # Chart 34 — Rating by dept box
        fig = px.box(ftch, x="department", y="teacher_performance_rating",
                     color="department", title="Performance Rating by Department",
                     points=False, color_discrete_sequence=theme["palette"])
        st.plotly_chart(style_fig(fig, theme, show_legend=False, height=400),
                        use_container_width=True)

        # Chart 35 — Exp vs rating scatter with trendline
        fig = px.scatter(ftch, x="years_experience", y="teacher_performance_rating",
                         color="department", opacity=0.7,
                         title="Experience vs Performance Rating",
                         hover_data=["full_name","school"],
                         color_discrete_sequence=theme["palette"])
        fig = add_trendline(fig, ftch["years_experience"],
                            ftch["teacher_performance_rating"], color=theme["primary"])
        st.plotly_chart(style_fig(fig, theme, height=420),
                        use_container_width=True)

        # Chart 36 — Top 10 teachers
        section("Top 10 teachers by rating")
        top = ftch.nlargest(10, "teacher_performance_rating")[
            ["full_name","department","years_experience",
             "teacher_performance_rating","school"]]
        st.dataframe(top, use_container_width=True, hide_index=True)

    # ── Tab 3 ────────────────────────────────────────────────────
    with tab3:
        col1, col2 = st.columns(2)
        # Chart 37 — Workload histogram
        fig = px.histogram(ftch, x="weekly_workload_hours", nbins=20,
                           title="Weekly Workload Hours",
                           color_discrete_sequence=[theme["primary"]])
        fig.update_traces(marker_line_width=1, marker_line_color="white")
        col1.plotly_chart(style_fig(fig, theme, show_legend=False, height=340, bargap=0.1),
                          use_container_width=True)

        # Chart 38 — Workload density vs rating
        fig = px.density_heatmap(ftch, x="weekly_workload_hours",
                                 y="teacher_performance_rating", nbinsx=15, nbinsy=15,
                                 title="Workload vs Rating (density)",
                                 color_continuous_scale=theme["scale"])
        col2.plotly_chart(style_fig(fig, theme, show_legend=False, height=340),
                          use_container_width=True)

        # Chart 39 — Attendance vs rating
        fig = px.scatter(ftch, x="teacher_attendance_pct", y="teacher_performance_rating",
                         color="department", opacity=0.7,
                         title="Teacher Attendance % vs Rating",
                         color_discrete_sequence=theme["palette"])
        fig = add_trendline(fig, ftch["teacher_attendance_pct"],
                            ftch["teacher_performance_rating"], color=theme["primary"])
        st.plotly_chart(style_fig(fig, theme, height=420), use_container_width=True)


# ═══════════════════════════════════════════════════════════════════
#  PAGE: ATTENDANCE INTELLIGENCE
# ═══════════════════════════════════════════════════════════════════
elif page == "Attendance Intelligence":

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Log entries", f"{len(fatt):,}")
    pres = (fatt["attendance_status"]=="Present").mean()*100 if len(fatt) else 0
    c2.metric("Present %",   f"{pres:.1f}%")
    c3.metric("Unique dates", fatt["attendance_date"].nunique())
    c4.metric("Unique students", fatt["stakeholder_id"].nunique())

    tab1, tab2, tab3 = st.tabs(["📊 Overview", "📈 Time trends", "🏫 Comparisons"])

    # ── Tab 1 ────────────────────────────────────────────────────
    with tab1:
        section("Status & reasons")
        col1, col2 = st.columns(2)

        # Chart 40 — Status donut
        status = fatt["attendance_status"].value_counts().reset_index()
        status.columns = ["status","count"]
        fig = px.pie(status, names="status", values="count", hole=0.6,
                     title="Attendance Status Split",
                     color="status",
                     color_discrete_map={"Present":"#10B981","Absent":"#EF4444",
                                         "Late":"#F59E0B","Leave":"#94A3B8"})
        fig.update_traces(textposition="outside", textinfo="label+percent")
        fig.add_annotation(text=f"<b>{pres:.1f}%</b><br>present",
                           showarrow=False, font=dict(size=15, color="#0F172A"))
        col1.plotly_chart(style_fig(fig, theme, height=360), use_container_width=True)

        # Chart 41 — Absence reasons
        absences = fatt[fatt["attendance_status"]!="Present"]
        if len(absences):
            reason = absences["reason"].value_counts().reset_index()
            reason.columns = ["reason","count"]
            fig = px.bar(reason, x="reason", y="count", text_auto=True,
                         title="Reasons for Absence / Late / Leave",
                         color="count", color_continuous_scale=theme["scale"])
            col2.plotly_chart(style_fig(fig, theme, show_legend=False, height=360),
                              use_container_width=True)

    # ── Tab 2 ────────────────────────────────────────────────────
    with tab2:
        # Chart 42 — Daily presence line
        daily = (fatt.assign(p=(fatt["attendance_status"]=="Present").astype(int))
                      .groupby("attendance_date")["p"].mean().reset_index())
        daily["p"] *= 100
        fig = px.line(daily, x="attendance_date", y="p",
                      title="Daily Attendance % Trend", markers=True,
                      color_discrete_sequence=[theme["primary"]])
        fig.update_traces(line_width=2.5)
        fig.add_hline(y=daily["p"].mean(), line_dash="dash", line_color=theme["accent"],
                      annotation_text=f"Mean {daily['p'].mean():.1f}%")
        st.plotly_chart(style_fig(fig, theme, show_legend=False, height=380),
                        use_container_width=True)

        # Chart 43 — Day-of-week heatmap
        dow_order = ["Monday","Tuesday","Wednesday","Thursday","Friday"]
        dow = (fatt.assign(p=(fatt["attendance_status"]=="Present").astype(int))
                    .groupby("day_of_week")["p"].mean()
                    .reindex(dow_order).reset_index())
        dow["p"] *= 100
        fig = px.bar(dow, x="day_of_week", y="p", text_auto=".1f",
                     title="Attendance % by Day of Week",
                     color="p", color_continuous_scale=theme["scale"])
        fig.update_layout(yaxis_range=[70, 100])
        st.plotly_chart(style_fig(fig, theme, show_legend=False, height=340),
                        use_container_width=True)

        # Chart 44 — Weekly trend
        weekly = (fatt.assign(p=(fatt["attendance_status"]=="Present").astype(int))
                       .groupby("week")["p"].mean().reset_index())
        weekly["p"] *= 100
        fig = px.area(weekly, x="week", y="p", title="Weekly Attendance % Trend",
                      color_discrete_sequence=[theme["primary"]])
        fig.update_traces(line_width=2.5, fillcolor=theme["primary"]+"30")
        st.plotly_chart(style_fig(fig, theme, show_legend=False, height=340),
                        use_container_width=True)

    # ── Tab 3 ────────────────────────────────────────────────────
    with tab3:
        col1, col2 = st.columns(2)
        # Chart 45 — By grade
        by_grade = (fatt.assign(p=(fatt["attendance_status"]=="Present").astype(int))
                         .groupby("grade_level")["p"].mean().reset_index())
        by_grade["p"] *= 100
        fig = px.bar(by_grade, x="grade_level", y="p", text_auto=".1f",
                     title="Attendance % by Grade",
                     color="p", color_continuous_scale=theme["scale"])
        col1.plotly_chart(style_fig(fig, theme, show_legend=False, height=340),
                          use_container_width=True)

        # Chart 46 — By region
        by_region = (fatt.assign(p=(fatt["attendance_status"]=="Present").astype(int))
                         .groupby("region")["p"].mean().reset_index())
        by_region["p"] *= 100
        fig = px.bar(by_region, x="region", y="p", text_auto=".1f",
                     title="Attendance % by Region",
                     color="p", color_continuous_scale=theme["scale"])
        col2.plotly_chart(style_fig(fig, theme, show_legend=False, height=340),
                          use_container_width=True)

        # Chart 47 — School comparison
        by_school = (fatt.assign(p=(fatt["attendance_status"]=="Present").astype(int))
                          .groupby("school")["p"].mean().reset_index()
                          .sort_values("p", ascending=True))
        by_school["p"] *= 100
        fig = px.bar(by_school, x="p", y="school", orientation="h", text_auto=".1f",
                     title="Attendance % by School",
                     color="p", color_continuous_scale=theme["scale"])
        st.plotly_chart(style_fig(fig, theme, show_legend=False, height=500),
                        use_container_width=True)


# ═══════════════════════════════════════════════════════════════════
#  PAGE: SCHOOL NETWORK
# ═══════════════════════════════════════════════════════════════════
elif page == "School Network":

    c1,c2,c3,c4,c5 = st.columns(5)
    c1.metric("Schools", len(schools))
    c2.metric("Boards",  schools["board_name"].nunique())
    c3.metric("States",  schools["state"].nunique())
    c4.metric("Regions", schools["region"].nunique())
    c5.metric("Capacity", f"{int(schools['student_capacity'].sum()):,}")

    tab1, tab2, tab3 = st.tabs(["🏆 Rankings", "🧭 Distribution", "🗺 Geography"])

    # ── Tab 1 ────────────────────────────────────────────────────
    with tab1:
        summary = (stu.groupby("school").agg(
            students=("student_id","count"),
            avg_gpa=("current_gpa","mean"),
            avg_att=("cumulative_attendance_pct","mean"),
            high_risk=("academic_risk_flag", lambda s: (s=="High").mean()*100),
        ).reset_index().sort_values("avg_gpa", ascending=False))

        # Chart 48 — GPA bar with gradient
        fig = px.bar(summary, x="school", y="avg_gpa", text_auto=".2f",
                     title="Average GPA by School",
                     color="avg_gpa", color_continuous_scale=theme["scale"],
                     hover_data=["students","avg_att","high_risk"])
        st.plotly_chart(style_fig(fig, theme, show_legend=False, height=400),
                        use_container_width=True)

        # Chart 49 — Quadrant bubble
        fig = px.scatter(summary, x="avg_att", y="avg_gpa", size="students",
                         hover_name="school",
                         title="School Quadrant · Attendance × GPA",
                         color="high_risk",
                         color_continuous_scale=[[0,"#10B981"],[0.5,"#F59E0B"],[1,"#EF4444"]],
                         labels={"high_risk":"High-risk %"})
        fig.add_hline(y=summary["avg_gpa"].mean(), line_dash="dot",
                      line_color="#94A3B8", annotation_text="mean GPA")
        fig.add_vline(x=summary["avg_att"].mean(), line_dash="dot",
                      line_color="#94A3B8", annotation_text="mean att.")
        st.plotly_chart(style_fig(fig, theme, height=480), use_container_width=True)

    # ── Tab 2 ────────────────────────────────────────────────────
    with tab2:
        col1, col2 = st.columns(2)

        # Chart 50 — Boards
        boards = schools["board_name"].value_counts().reset_index()
        boards.columns = ["board","count"]
        fig = px.pie(boards, names="board", values="count", hole=0.6,
                     title="Schools by Board",
                     color_discrete_sequence=theme["palette"])
        fig.update_traces(textposition="outside", textinfo="label+percent")
        col1.plotly_chart(style_fig(fig, theme, height=360), use_container_width=True)

        # Chart 51 — Management type
        mgmt = schools["management_type"].value_counts().reset_index()
        mgmt.columns = ["type","count"]
        fig = px.pie(mgmt, names="type", values="count", hole=0.6,
                     title="Management Type",
                     color_discrete_sequence=theme["palette"])
        fig.update_traces(textposition="outside", textinfo="label+percent")
        col2.plotly_chart(style_fig(fig, theme, height=360), use_container_width=True)

        col1, col2 = st.columns(2)
        # Chart 52 — Region bar
        regions = schools["region"].value_counts().reset_index()
        regions.columns = ["region","count"]
        fig = px.bar(regions, x="region", y="count", text_auto=True,
                     color="region", title="Schools by Region",
                     color_discrete_sequence=theme["palette"])
        col1.plotly_chart(style_fig(fig, theme, show_legend=False, height=340),
                          use_container_width=True)

        # Chart 53 — School type
        types = schools["school_type"].value_counts().reset_index()
        types.columns = ["type","count"]
        fig = px.bar(types, x="type", y="count", text_auto=True,
                     color="type", title="Schools by Type",
                     color_discrete_sequence=theme["palette"])
        col2.plotly_chart(style_fig(fig, theme, show_legend=False, height=340),
                          use_container_width=True)

        # Chart 54 — Capacity vs enrolment
        enrolment = stu.groupby("school").size().reset_index(name="enrolled")
        cap = schools[["school","student_capacity"]].merge(enrolment, on="school", how="left")
        cap["utilisation"] = (cap["enrolled"]/cap["student_capacity"])*100
        fig = px.bar(cap.sort_values("utilisation", ascending=False),
                     x="school", y="utilisation", text_auto=".0f",
                     title="Capacity Utilisation (%) · Enrolment ÷ Capacity",
                     color="utilisation", color_continuous_scale=theme["scale"])
        st.plotly_chart(style_fig(fig, theme, show_legend=False, height=380),
                        use_container_width=True)

    # ── Tab 3 ────────────────────────────────────────────────────
    with tab3:
        if {"latitude","longitude"}.issubset(schools.columns):
            st.map(schools[["latitude","longitude"]].dropna(), zoom=4)

        # Chart 55 — Region-region heatmap (performance)
        perf = stu.groupby(["region","school_type"])["current_gpa"].mean().reset_index()
        pivot = perf.pivot(index="region", columns="school_type", values="current_gpa")
        fig = px.imshow(pivot, text_auto=".2f",
                        title="Avg GPA · Region × School Type",
                        color_continuous_scale=theme["scale"], aspect="auto")
        st.plotly_chart(style_fig(fig, theme, show_legend=False, height=360),
                        use_container_width=True)


# ═══════════════════════════════════════════════════════════════════
#  PAGE: PREDICTIVE LAB (10 ML charts)
# ═══════════════════════════════════════════════════════════════════
elif page == "Predictive Lab":

    # Train all models once
    risk_pkg  = train_risk_model(stu)
    gpa_pkg   = train_gpa_model(stu)
    cluster_pkg = build_clusters(stu, k=4)
    anom_pkg  = detect_anomalies(stu)

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Risk model acc (LR)", f"{risk_pkg['lr_acc']*100:.1f}%")
    c2.metric("Risk model acc (RF)", f"{risk_pkg['rf_acc']*100:.1f}%")
    c3.metric("GPA model R²",        f"{gpa_pkg['r2']:.3f}")
    c4.metric("Anomalies flagged",   int((anom_pkg['flags']==-1).sum()))

    tab1, tab2, tab3, tab4 = st.tabs([
        "🎯 Risk Classifier", "🔮 What-If Simulator",
        "🧬 Clustering", "⚠️ Anomalies & Correlation"])

    # ── TAB 1 : RISK CLASSIFIER (4 charts) ──────────────────────
    with tab1:
        section("Model performance")
        col1, col2 = st.columns(2)

        # Predictive Chart 1 — Model accuracy comparison
        acc_df = pd.DataFrame({
            "Model": ["Logistic Regression", "Random Forest"],
            "Accuracy": [risk_pkg["lr_acc"]*100, risk_pkg["rf_acc"]*100],
        })
        fig = px.bar(acc_df, x="Model", y="Accuracy", text_auto=".1f",
                     title="🎯 [1/10] Classifier Accuracy Comparison",
                     color="Model", color_discrete_sequence=theme["palette"])
        fig.update_layout(yaxis_range=[0, 100])
        col1.plotly_chart(style_fig(fig, theme, show_legend=False, height=320),
                          use_container_width=True)

        # Predictive Chart 2 — Confusion matrix
        y_true = stu["academic_risk_flag"]
        y_pred = risk_pkg["rf"].predict(
            pd.DataFrame({
                "gpa": stu["current_gpa"],
                "attendance": stu["cumulative_attendance_pct"],
                "grade": stu["grade_level"],
                "scholarship": (stu["scholarship_flag"]=="Yes").astype(int),
                "iep": (stu["iep_flag"]=="Yes").astype(int)}))
        classes = sorted(y_true.unique())
        cm = confusion_matrix(y_true, y_pred, labels=classes)
        fig = px.imshow(cm, text_auto=True, x=classes, y=classes,
                        labels=dict(x="Predicted", y="Actual"),
                        title="🎯 [2/10] Confusion Matrix (Random Forest)",
                        color_continuous_scale=theme["scale"])
        col2.plotly_chart(style_fig(fig, theme, show_legend=False, height=320),
                          use_container_width=True)

        section("Explainability")
        col1, col2 = st.columns(2)

        # Predictive Chart 3 — Feature importance (Random Forest)
        imp = pd.DataFrame({
            "feature": risk_pkg["features"],
            "importance": risk_pkg["rf"].feature_importances_,
        }).sort_values("importance", ascending=True)
        fig = px.bar(imp, x="importance", y="feature", orientation="h",
                     title="🎯 [3/10] Feature Importance (RF)",
                     text_auto=".3f", color="importance",
                     color_continuous_scale=theme["scale"])
        col1.plotly_chart(style_fig(fig, theme, show_legend=False, height=340),
                          use_container_width=True)

        # Predictive Chart 4 — Risk probability distribution
        proba = risk_pkg["rf"].predict_proba(
            pd.DataFrame({
                "gpa": fstu["current_gpa"],
                "attendance": fstu["cumulative_attendance_pct"],
                "grade": fstu["grade_level"],
                "scholarship": (fstu["scholarship_flag"]=="Yes").astype(int),
                "iep": (fstu["iep_flag"]=="Yes").astype(int)}))
        high_idx = list(risk_pkg["rf"].classes_).index("High")
        p_high = proba[:, high_idx]
        fig = px.histogram(pd.DataFrame({"p_high": p_high}), x="p_high", nbins=25,
                           title="🎯 [4/10] P(High Risk) Distribution",
                           color_discrete_sequence=[theme["primary"]])
        fig.update_traces(marker_line_width=1, marker_line_color="white")
        fig.add_vline(x=0.5, line_dash="dash", line_color="#EF4444",
                      annotation_text="Decision threshold (0.5)")
        col2.plotly_chart(style_fig(fig, theme, show_legend=False, height=340, bargap=0.1),
                          use_container_width=True)

    # ── TAB 2 : WHAT-IF + SHAP-STYLE  (3 charts) ────────────────
    with tab2:
        st.markdown("Adjust the sliders to see real-time predictions for a hypothetical "
                    "student and how each feature contributes to the outcome.")

        # Input controls
        col_in, col_out = st.columns([1, 2])
        with col_in:
            att_input   = st.slider("Attendance %",  50.0, 100.0, 85.0, 0.5)
            grade_input = st.select_slider("Grade level",
                options=sorted(stu["grade_level"].dropna().unique()), value=6)
            schol_input = st.checkbox("Has scholarship", value=False)
            iep_input   = st.checkbox("Has IEP",         value=False)

        # Predict GPA
        x_gpa = pd.DataFrame([[att_input, grade_input,
                               int(schol_input), int(iep_input)]],
                             columns=gpa_pkg["features"])
        pred_gpa = float(gpa_pkg["model"].predict(x_gpa)[0])
        pred_gpa = max(0.0, min(4.0, pred_gpa))

        # Predict risk
        x_risk = pd.DataFrame([[pred_gpa, att_input, grade_input,
                                int(schol_input), int(iep_input)]],
                              columns=risk_pkg["features"])
        pred_risk  = risk_pkg["rf"].predict(x_risk)[0]
        pred_proba = risk_pkg["rf"].predict_proba(x_risk)[0]

        with col_out:
            cc1, cc2, cc3 = st.columns(3)
            cohort_gpa = stu["current_gpa"].mean()
            cc1.metric("Predicted GPA",      round(pred_gpa, 2),
                       delta=round(pred_gpa - cohort_gpa, 2))
            cc2.metric("Predicted risk",     pred_risk)
            cc3.metric("Risk confidence",    f"{pred_proba.max()*100:.1f}%")

            # Predictive Chart 5 — Probability bars
            prob_df = pd.DataFrame({
                "class": risk_pkg["rf"].classes_,
                "probability": pred_proba,
            })
            fig = px.bar(prob_df, x="class", y="probability", text_auto=".0%",
                         title="🔮 [5/10] Risk-Class Probabilities",
                         color="class",
                         color_discrete_map={"Low":"#10B981","Medium":"#F59E0B","High":"#EF4444"})
            fig.update_yaxes(tickformat=".0%", range=[0, 1])
            st.plotly_chart(style_fig(fig, theme, show_legend=False, height=300),
                            use_container_width=True)

        # Predictive Chart 6 — GPA gauge
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta", value=pred_gpa,
            delta={"reference": cohort_gpa, "valueformat": ".2f"},
            title={"text": "🔮 [6/10] Predicted GPA vs Cohort Mean"},
            gauge={
                "axis": {"range": [0, 4]}, "bar": {"color": theme["primary"]},
                "steps": [
                    {"range": [0, 2],   "color": "#FEE2E2"},
                    {"range": [2, 3],   "color": "#FEF3C7"},
                    {"range": [3, 4],   "color": "#DCFCE7"}],
                "threshold": {"line": {"color": "#0F172A", "width": 3},
                              "thickness": 0.75, "value": cohort_gpa}}))
        fig.update_layout(height=300, margin=dict(l=20, r=20, t=50, b=10),
                          paper_bgcolor="rgba(0,0,0,0)",
                          font_family='"Inter", sans-serif')
        st.plotly_chart(fig, use_container_width=True)

        # Predictive Chart 7 — SHAP-style waterfall explanation
        section("🔍 SHAP-style contribution breakdown")
        st.markdown("How each feature pushes the GPA prediction **up** or **down** from the baseline.")
        baseline = gpa_pkg["intercept"] + np.dot(
            list(gpa_pkg["coefs"].values()),
            stu[["cumulative_attendance_pct","grade_level"]].assign(
                scholarship=(stu["scholarship_flag"]=="Yes").astype(int),
                iep=(stu["iep_flag"]=="Yes").astype(int)
            )[gpa_pkg["features"]].mean().values)

        inputs = {"attendance": att_input, "grade": grade_input,
                  "scholarship": int(schol_input), "iep": int(iep_input)}
        means  = {"attendance": stu["cumulative_attendance_pct"].mean(),
                  "grade": stu["grade_level"].mean(),
                  "scholarship": (stu["scholarship_flag"]=="Yes").mean(),
                  "iep": (stu["iep_flag"]=="Yes").mean()}
        coefs_dict = gpa_pkg["coefs"]
        rename = {"attendance":"attendance","grade":"grade",
                  "scholarship":"scholarship","iep":"iep"}
        # Map model feature names -> human names
        coef_keys = list(coefs_dict.keys())  # ['attendance','grade','scholarship','iep']
        contributions = []
        for k in coef_keys:
            c = coefs_dict[k] * (inputs[k] - means[k])
            contributions.append({"feature": k, "contribution": c})
        contrib_df = pd.DataFrame(contributions).sort_values("contribution")
        fig = go.Figure(go.Waterfall(
            orientation="h",
            y=contrib_df["feature"].tolist() + ["Total Δ from mean"],
            x=contrib_df["contribution"].tolist() + [contrib_df["contribution"].sum()],
            measure=["relative"]*len(contrib_df) + ["total"],
            connector={"line":{"color":"#CBD5E1"}},
            decreasing={"marker":{"color":"#EF4444"}},
            increasing={"marker":{"color":"#10B981"}},
            totals={"marker":{"color":theme["primary"]}},
            text=[f"{v:+.3f}" for v in contrib_df["contribution"].tolist()] +
                 [f"{contrib_df['contribution'].sum():+.3f}"],
            textposition="outside",
        ))
        fig.update_layout(
            title="🔮 [7/10] SHAP-Style Feature Contributions to Predicted GPA",
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            height=320, font_family='"Inter", sans-serif', title_x=0.02)
        st.plotly_chart(fig, use_container_width=True)

        insight(f"For this student, <b>{contrib_df.iloc[-1]['feature']}</b> "
                f"contributes most positively "
                f"(+{contrib_df['contribution'].max():.3f} GPA points above "
                f"the cohort average baseline).")

    # ── TAB 3 : CLUSTERING (2 charts) ──────────────────────────
    with tab3:
        st.markdown("K-means clustering (k=4) groups students by GPA, attendance, "
                    "grade and scholarship status. Each cluster is a natural cohort.")

        labels = cluster_pkg["labels"]
        centers = cluster_pkg["centers"]
        cluster_stu = stu.assign(cluster=labels)

        # Predictive Chart 8 — Cluster scatter
        sample = cluster_stu.sample(min(3000, len(cluster_stu)), random_state=1)
        fig = px.scatter(sample, x="cumulative_attendance_pct", y="current_gpa",
                         color="cluster", opacity=0.7,
                         title="🧬 [8/10] Student Clusters · GPA × Attendance",
                         hover_data=["school","grade_level","academic_risk_flag"],
                         color_discrete_sequence=theme["palette"])
        # Overlay cluster centres
        for i, c in enumerate(centers):
            fig.add_trace(go.Scatter(
                x=[c[1]], y=[c[0]], mode="markers+text",
                marker=dict(size=25, color=theme["palette"][i], symbol="star",
                            line=dict(color="white", width=2)),
                text=[f"C{i}"], textposition="top center",
                name=f"Centre {i}", showlegend=False))
        st.plotly_chart(style_fig(fig, theme, height=480),
                        use_container_width=True)

        # Predictive Chart 9 — Cluster profile radar
        prof = cluster_stu.groupby("cluster").agg(
            gpa=("current_gpa","mean"),
            attendance=("cumulative_attendance_pct","mean"),
            grade=("grade_level","mean"),
            risk=("academic_risk_flag", lambda s: (s=="High").mean()*100),
        ).reset_index()
        # Normalize each axis 0-1
        prof_norm = prof.copy()
        for c in ["gpa","attendance","grade","risk"]:
            col = prof_norm[c]
            prof_norm[c] = (col - col.min())/(col.max() - col.min() + 1e-9)

        fig = go.Figure()
        axes = ["gpa","attendance","grade","risk"]
        for i, row in prof_norm.iterrows():
            fig.add_trace(go.Scatterpolar(
                r=[row[a] for a in axes], theta=axes,
                fill='toself', name=f"Cluster {int(row['cluster'])}",
                line_color=theme["palette"][int(row["cluster"])]))
        fig.update_layout(
            title="🧬 [9/10] Cluster Profiles (normalised)",
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            paper_bgcolor="rgba(0,0,0,0)", height=420,
            title_x=0.02, font_family='"Inter", sans-serif',
            legend=dict(orientation="h", yanchor="bottom", y=-0.15))
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("**Cluster summary:**")
        st.dataframe(prof.round(2), use_container_width=True, hide_index=True)

    # ── TAB 4 : ANOMALIES + CORRELATION (1 chart + bonus) ────────
    with tab4:
        # Predictive Chart 10 — Correlation heatmap
        section("Feature correlation matrix")
        num_df = stu[["current_gpa","cumulative_attendance_pct",
                      "grade_level","performance_index","age"]].copy()
        num_df["scholarship"] = (stu["scholarship_flag"]=="Yes").astype(int)
        num_df["iep"]         = (stu["iep_flag"]=="Yes").astype(int)
        num_df["high_risk"]   = (stu["academic_risk_flag"]=="High").astype(int)
        corr = num_df.corr()
        fig = px.imshow(corr, text_auto=".2f",
                        title="⚠️ [10/10] Feature Correlation Matrix",
                        color_continuous_scale=[[0,"#EF4444"],[0.5,"#FFFFFF"],[1,"#10B981"]],
                        zmin=-1, zmax=1, aspect="auto")
        st.plotly_chart(style_fig(fig, theme, show_legend=False, height=480),
                        use_container_width=True)

        # Bonus visual — anomaly scatter
        section("🚨 Isolation-forest anomalies")
        sample = stu.assign(anomaly=anom_pkg["flags"],
                            anom_score=anom_pkg["scores"]).sample(
                                min(3000, len(stu)), random_state=1)
        fig = px.scatter(sample, x="cumulative_attendance_pct", y="current_gpa",
                         color=sample["anomaly"].map({1:"Normal", -1:"Anomaly"}),
                         opacity=0.7,
                         title="Isolation-Forest: Students With Unusual Profiles",
                         color_discrete_map={"Normal":"#94A3B8", "Anomaly":"#EF4444"},
                         hover_data=["school","grade_level","academic_risk_flag"])
        st.plotly_chart(style_fig(fig, theme, height=440), use_container_width=True)

        anom_count = int((anom_pkg['flags']==-1).sum())
        insight(f"The isolation-forest flagged <b>{anom_count} students</b> "
                f"(≈5%) whose GPA / attendance / grade combination looks "
                f"statistically unusual compared with peers. "
                f"These are good candidates for early counsellor review.")


# ═══════════════════════════════════════════════════════════════════
#  SIDEBAR FOOTER
# ═══════════════════════════════════════════════════════════════════
st.sidebar.markdown("---")
st.sidebar.download_button(
    "📥 Download filtered students",
    fstu.to_csv(index=False).encode("utf-8"),
    file_name="filtered_students.csv",
    mime="text/csv",
    use_container_width=True,
)
st.sidebar.markdown(
    "<div style='margin-top:1rem; font-size:.7rem; color:#64748B; text-align:center;'>"
    "v2.0 · Built with Streamlit · Plotly · scikit-learn"
    "</div>", unsafe_allow_html=True)
