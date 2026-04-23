"""
╔═══════════════════════════════════════════════════════════════════╗
║  NEXUS ANALYTICS v3.0 — Multi-School Academic Intelligence        ║
║  10 pages · 90+ charts · 10 predictive ML models                  ║
║  Now with chart explanations + parents/comparison/benchmarking    ║
╚═══════════════════════════════════════════════════════════════════╝
"""

from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from sklearn.linear_model  import LinearRegression, LogisticRegression
from sklearn.ensemble      import RandomForestClassifier, IsolationForest
from sklearn.cluster       import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics       import accuracy_score, confusion_matrix

st.set_page_config(
    page_title="Nexus Analytics",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ═══════════════════════════════════════════════════════════════════
#  PAGE THEMES + OBJECTIVES
# ═══════════════════════════════════════════════════════════════════
THEMES = {
    "Executive Overview": {
        "icon": "🏛",
        "subtitle": "Strategic snapshot across the academic network",
        "objective": ("This page gives leadership a one-glance view of the "
                      "entire network: how many students, teachers, and schools "
                      "we operate, the overall GPA and attendance health, "
                      "and which schools are leading or trailing."),
        "primary": "#4F46E5", "accent": "#8B5CF6",
        "grad": ("#4338CA", "#7C3AED"),
        "palette": ["#4F46E5", "#7C3AED", "#EC4899", "#F59E0B", "#10B981", "#06B6D4", "#F43F5E", "#8B5CF6"],
        "scale":   [[0, "#EEF2FF"], [0.5, "#818CF8"], [1, "#3730A3"]],
    },
    "Students 360°": {
        "icon": "🎓",
        "subtitle": "Academic standing, demographics and behavioural profile",
        "objective": ("A deep dive into the student body: how performance "
                      "varies by grade and gender, how scholarships and IEP "
                      "programmes move outcomes, and who the top and bottom "
                      "performers are."),
        "primary": "#059669", "accent": "#10B981",
        "grad": ("#065F46", "#14B8A6"),
        "palette": ["#059669", "#0D9488", "#14B8A6", "#10B981", "#84CC16", "#65A30D", "#22C55E", "#047857"],
        "scale":   [[0, "#ECFDF5"], [0.5, "#34D399"], [1, "#064E3B"]],
    },
    "Academic Performance": {
        "icon": "📚",
        "subtitle": "Exam results, subjects, terms and grade distribution",
        "objective": ("Exam-level analytics: which subjects students struggle "
                      "with, where grades cluster, how pass rates change across "
                      "terms, and the relationship between classwork and exam "
                      "outcomes."),
        "primary": "#EA580C", "accent": "#F59E0B",
        "grad": ("#B91C1C", "#F59E0B"),
        "palette": ["#EA580C", "#F59E0B", "#DC2626", "#FB923C", "#FBBF24", "#EF4444", "#F97316", "#B45309"],
        "scale":   [[0, "#FFF7ED"], [0.5, "#FB923C"], [1, "#7C2D12"]],
    },
    "Faculty Analytics": {
        "icon": "👩‍🏫",
        "subtitle": "Teacher distribution, ratings, workload and experience",
        "objective": ("How the teaching workforce is structured: department "
                      "sizes, qualifications, employment types, and how "
                      "experience, workload, and attendance relate to "
                      "performance ratings."),
        "primary": "#BE123C", "accent": "#DB2777",
        "grad": ("#9F1239", "#C026D3"),
        "palette": ["#BE123C", "#DB2777", "#C026D3", "#E11D48", "#EC4899", "#A21CAF", "#F43F5E", "#86198F"],
        "scale":   [[0, "#FDF2F8"], [0.5, "#F472B6"], [1, "#831843"]],
    },
    "Attendance Intelligence": {
        "icon": "📅",
        "subtitle": "Presence patterns, trends and absence intelligence",
        "objective": ("Presence patterns over time and across cohorts: daily "
                      "and weekly trends, reasons for absence, and which "
                      "grades, regions, or schools show lower engagement."),
        "primary": "#0284C7", "accent": "#06B6D4",
        "grad": ("#0C4A6E", "#0891B2"),
        "palette": ["#0284C7", "#0891B2", "#06B6D4", "#0EA5E9", "#38BDF8", "#22D3EE", "#67E8F9", "#155E75"],
        "scale":   [[0, "#ECFEFF"], [0.5, "#22D3EE"], [1, "#0C4A6E"]],
    },
    "School Network": {
        "icon": "🏫",
        "subtitle": "Institutions, boards, geography and benchmarks",
        "objective": ("Information about the schools themselves: boards, "
                      "management types, regions, and how enrolment compares "
                      "against built capacity."),
        "primary": "#7C3AED", "accent": "#A855F7",
        "grad": ("#581C87", "#C026D3"),
        "palette": ["#7C3AED", "#A855F7", "#9333EA", "#C026D3", "#D946EF", "#E879F9", "#8B5CF6", "#6D28D9"],
        "scale":   [[0, "#FAF5FF"], [0.5, "#C084FC"], [1, "#4C1D95"]],
    },
    "Parents & Income": {
        "icon": "👨‍👩‍👧",
        "subtitle": "How family income and education shape student outcomes",
        "objective": ("Socio-economic analytics: how parental income, "
                      "education, and occupation influence student GPA, "
                      "attendance, and risk levels. Useful for equity "
                      "planning and scholarship allocation."),
        "primary": "#CA8A04", "accent": "#D97706",
        "grad": ("#854D0E", "#D97706"),
        "palette": ["#CA8A04", "#D97706", "#F59E0B", "#B45309", "#FBBF24", "#92400E", "#EAB308", "#78350F"],
        "scale":   [[0, "#FEFCE8"], [0.5, "#FBBF24"], [1, "#713F12"]],
    },
    "Student Comparison": {
        "icon": "🆚",
        "subtitle": "Compare 2–5 students head-to-head",
        "objective": ("Pick any 2–5 students from the sidebar filters and "
                      "compare them on every key metric — GPA, attendance, "
                      "performance index, subject-level scores, and fees. "
                      "Built for counselors, parent meetings, and "
                      "scholarship reviews."),
        "primary": "#DB2777", "accent": "#EC4899",
        "grad": ("#9F1239", "#BE185D"),
        "palette": ["#DB2777", "#BE185D", "#EC4899", "#F472B6", "#FB7185", "#F43F5E", "#E11D48", "#9F1239"],
        "scale":   [[0, "#FDF2F8"], [0.5, "#F472B6"], [1, "#831843"]],
    },
    "School Benchmarking": {
        "icon": "⚖️",
        "subtitle": "Side-by-side school comparison across academic + financial metrics",
        "objective": ("Direct school-vs-school comparison across academic "
                      "performance AND financial health: outstanding fees, "
                      "collection efficiency, payment defaults, and how "
                      "financial distress correlates with academic risk."),
        "primary": "#0891B2", "accent": "#0E7490",
        "grad": ("#164E63", "#0891B2"),
        "palette": ["#0891B2", "#0E7490", "#06B6D4", "#22D3EE", "#67E8F9", "#155E75", "#164E63", "#0284C7"],
        "scale":   [[0, "#ECFEFF"], [0.5, "#22D3EE"], [1, "#164E63"]],
    },
    "Predictive Lab": {
        "icon": "🔮",
        "subtitle": "Machine-learning forecasts, simulations and explainability",
        "objective": ("The crystal-ball page. Four ML models work together: "
                      "a classifier predicts each student's academic risk, a "
                      "regressor predicts GPA, K-means finds natural cohorts, "
                      "and an isolation forest flags statistical outliers. "
                      "Use the what-if simulator to test interventions."),
        "primary": "#0F766E", "accent": "#6366F1",
        "grad": ("#0F766E", "#4F46E5"),
        "palette": ["#0F766E", "#14B8A6", "#6366F1", "#8B5CF6", "#F59E0B", "#EC4899", "#10B981", "#3B82F6"],
        "scale":   [[0, "#F0FDFA"], [0.5, "#2DD4BF"], [1, "#134E4A"]],
    },
}

# ═══════════════════════════════════════════════════════════════════
#  CSS
# ═══════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');

html, body, [class*="css"] { font-family: 'Inter', system-ui, sans-serif; }
.stApp { background: linear-gradient(180deg, #F8FAFC 0%, #EEF2F7 100%); }
.main .block-container { padding: 1.5rem 2.5rem 4rem 2.5rem; max-width: 1480px; }

/* Metric cards */
[data-testid="stMetric"] {
    background: #ffffff; padding: 1.1rem 1.3rem; border-radius: 14px;
    box-shadow: 0 1px 3px rgba(15,23,42,0.06), 0 1px 2px rgba(15,23,42,0.04);
    border-left: 4px solid var(--accent, #4F46E5);
    transition: all .2s ease; position: relative; overflow: hidden;
}
[data-testid="stMetric"]::before {
    content: ''; position: absolute; top: 0; right: 0;
    width: 80px; height: 80px;
    background: radial-gradient(circle, var(--accent, #4F46E5) 0%, transparent 70%);
    opacity: 0.10;
}
[data-testid="stMetric"]:hover { transform: translateY(-3px); box-shadow: 0 8px 20px rgba(15,23,42,0.10); }
[data-testid="stMetricLabel"] p {
    font-size: 0.72rem !important; color: #475569 !important;
    text-transform: uppercase; letter-spacing: 0.06em; font-weight: 700 !important;
}
[data-testid="stMetricValue"] { font-size: 1.75rem !important; font-weight: 800 !important; color: #0F172A !important; }

/* Chart cards */
[data-testid="stPlotlyChart"] {
    background: #ffffff; border-radius: 14px; padding: 0.75rem;
    box-shadow: 0 1px 3px rgba(15,23,42,0.05), 0 1px 2px rgba(15,23,42,0.03);
    margin-bottom: 0.5rem; transition: box-shadow .2s ease;
}
[data-testid="stPlotlyChart"]:hover { box-shadow: 0 6px 16px rgba(15,23,42,0.10); }

/* Chart caption */
.chart-caption {
    font-size: 0.82rem; color: #475569;
    margin: -0.2rem 0 1.1rem 4px;
    padding: 0.4rem 0.8rem;
    background: #F8FAFC; border-left: 3px solid #CBD5E1;
    border-radius: 0 6px 6px 0; line-height: 1.4;
}

/* Page objective */
.page-objective {
    background: linear-gradient(135deg, #FFFFFF 0%, #F8FAFC 100%);
    border: 1px solid #E2E8F0;
    padding: 1rem 1.3rem; border-radius: 12px;
    font-size: 0.92rem; color: #334155; line-height: 1.55;
    margin-bottom: 1.4rem;
    box-shadow: 0 1px 2px rgba(15,23,42,0.02);
}
.page-objective strong { color: #0F172A; }

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    gap: 0.3rem; background: transparent; border-bottom: 1px solid #E2E8F0;
    padding: 0; margin-bottom: 1rem;
}
.stTabs [data-baseweb="tab"] {
    background: transparent; padding: 0.7rem 1.2rem;
    font-weight: 600; color: #475569; border-radius: 8px 8px 0 0;
    border: none; transition: all .15s;
}
.stTabs [data-baseweb="tab"]:hover { background: #F1F5F9; color: #0F172A; }
.stTabs [aria-selected="true"] {
    background: white !important; color: var(--accent, #4F46E5) !important;
    border-bottom: 2px solid var(--accent, #4F46E5) !important;
}

/* Sidebar */
[data-testid="stSidebar"] { background: linear-gradient(180deg, #0F172A 0%, #1E293B 100%); }
[data-testid="stSidebar"] * { color: #E2E8F0 !important; }
[data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 { color: #F8FAFC !important; }
[data-testid="stSidebar"] label {
    font-size: 0.75rem !important; font-weight: 600 !important;
    color: #CBD5E1 !important; text-transform: uppercase; letter-spacing: 0.05em;
}
[data-testid="stSidebar"] [data-baseweb="select"] > div,
[data-testid="stSidebar"] [data-baseweb="input"] > div {
    background: rgba(255,255,255,0.08) !important; border-color: rgba(255,255,255,0.15) !important;
}

footer { visibility: hidden; }
[data-testid="column"] { padding: 0 0.5rem !important; }

.section-label {
    font-size: 0.78rem; font-weight: 700; color: #334155;
    text-transform: uppercase; letter-spacing: 0.09em;
    margin: 1.4rem 0 0.6rem 0;
    border-left: 3px solid var(--accent, #4F46E5); padding: 4px 0 4px 12px;
}
.insight-box {
    background: linear-gradient(135deg, #F8FAFC 0%, #F1F5F9 100%);
    border-left: 3px solid var(--accent, #4F46E5);
    padding: 0.9rem 1.2rem; border-radius: 8px;
    font-size: 0.9rem; color: #1E293B; margin: 0.6rem 0 1rem 0;
    box-shadow: 0 1px 2px rgba(15,23,42,0.03);
}
.insight-box strong { color: #0F172A; }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════
#  HELPERS
# ═══════════════════════════════════════════════════════════════════
def apply_theme(t):
    st.markdown(
        f"<style>:root {{ --accent: {t['primary']}; }} "
        f"[data-testid='stMetric'] {{ border-left-color: {t['primary']}; }}</style>",
        unsafe_allow_html=True,
    )

def render_header(page: str):
    t = THEMES[page]
    st.markdown(f"""
    <div style="
        background: linear-gradient(135deg, {t['grad'][0]} 0%, {t['grad'][1]} 100%);
        padding: 1.8rem 2.2rem; border-radius: 20px;
        margin-bottom: 1rem; color: white; position: relative; overflow: hidden;
        box-shadow: 0 12px 32px -12px {t['primary']}66;">
        <div style="position:absolute; top:-40px; right:-40px; width:200px; height:200px;
                    background: rgba(255,255,255,0.08); border-radius: 50%;"></div>
        <div style="position:absolute; bottom:-80px; right:120px; width:140px; height:140px;
                    background: rgba(255,255,255,0.06); border-radius: 50%;"></div>
        <div style="position:relative; z-index:1;">
            <div style="font-size:.75rem; opacity:.9; letter-spacing:.18em;
                        text-transform:uppercase; margin-bottom:.4rem; font-weight:700;">
                Nexus Analytics · {t['icon']}
            </div>
            <div style="font-size:2rem; font-weight:800; line-height:1.1; letter-spacing:-0.02em;">
                {page}
            </div>
            <div style="font-size:1rem; opacity:.95; margin-top:.45rem; font-weight:400;">
                {t['subtitle']}
            </div>
        </div>
    </div>""", unsafe_allow_html=True)

    # Page-level objective card
    st.markdown(
        f"<div class='page-objective'><strong>🎯 Objective · </strong>{t['objective']}</div>",
        unsafe_allow_html=True,
    )

def section(label: str):
    st.markdown(f"<div class='section-label'>{label}</div>", unsafe_allow_html=True)

def caption(text: str):
    """Small explanation shown directly under a chart."""
    st.markdown(f"<div class='chart-caption'>📖 {text}</div>", unsafe_allow_html=True)

def insight(html: str):
    st.markdown(f"<div class='insight-box'>💡 {html}</div>", unsafe_allow_html=True)


def style_fig(fig, theme, *, show_legend=True, height=None, bargap=0.18):
    layout = dict(
        font_family='"Inter", system-ui, sans-serif',
        font_color="#0F172A",
        title_font_color="#0F172A",
        title_font_size=15, title_font_family='"Inter", sans-serif',
        title_x=0.02, title_xanchor="left",
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=50, r=25, t=55, b=45),
        colorway=theme["palette"],
        showlegend=show_legend, bargap=bargap,
        legend=dict(bgcolor="rgba(255,255,255,0.9)", bordercolor="#E2E8F0",
                    borderwidth=1, orientation="h", yanchor="bottom", y=1.02,
                    xanchor="right", x=1, font=dict(size=11, color="#0F172A")),
        hoverlabel=dict(bgcolor="white", font_size=12,
                        font_family='"Inter", sans-serif',
                        font_color="#0F172A", bordercolor=theme["primary"]),
    )
    if height: layout["height"] = height
    fig.update_layout(**layout)
    axis_style = dict(
        gridcolor="#E2E8F0", zerolinecolor="#CBD5E1",
        linecolor="#94A3B8", linewidth=1.5,
        tickfont=dict(size=12, color="#0F172A", family='"Inter", sans-serif'),
        title_font=dict(size=12, color="#334155", family='"Inter", sans-serif'),
    )
    fig.update_xaxes(**axis_style)
    fig.update_yaxes(**axis_style)
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
    parents  = sheets["Parents"].copy()
    principals = sheets["Principals"].copy()

    # Join students + school + parent info (income, occupation, education)
    parent_cols = parents[["parent_id","annual_income_band","occupation",
                           "education_level"]].rename(columns={
        "annual_income_band":"parent_income",
        "occupation":"parent_occupation",
        "education_level":"parent_education"})
    stu = (students
           .merge(schools[["school_id","school","board_name","region","state","city",
                           "school_type","management_type","student_capacity"]],
                  on="school_id", how="left")
           .merge(parent_cols, left_on="primary_parent_id", right_on="parent_id", how="left")
           .drop(columns=["parent_id"]))
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
            "teachers": teachers, "schools": schools, "principals": principals,
            "parents": parents}


# ═══════════════════════════════════════════════════════════════════
#  ML MODELS
# ═══════════════════════════════════════════════════════════════════
RISK_FEATURES = ["GPA", "Attendance %", "Grade Level", "Scholarship", "IEP"]
GPA_FEATURES  = ["Attendance %", "Grade Level", "Scholarship", "IEP"]

def make_risk_X(stu: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({
        "GPA":           stu["current_gpa"].values,
        "Attendance %":  stu["cumulative_attendance_pct"].values,
        "Grade Level":   stu["grade_level"].values,
        "Scholarship":   (stu["scholarship_flag"]=="Yes").astype(int).values,
        "IEP":           (stu["iep_flag"]=="Yes").astype(int).values,
    })

def make_gpa_X(stu: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({
        "Attendance %":  stu["cumulative_attendance_pct"].values,
        "Grade Level":   stu["grade_level"].values,
        "Scholarship":   (stu["scholarship_flag"]=="Yes").astype(int).values,
        "IEP":           (stu["iep_flag"]=="Yes").astype(int).values,
    })

@st.cache_resource(show_spinner="Training risk classifier…")
def train_risk_model(stu: pd.DataFrame):
    X = make_risk_X(stu)
    y = stu["academic_risk_flag"]
    lr = LogisticRegression(max_iter=2000).fit(X, y)
    rf = RandomForestClassifier(n_estimators=80, max_depth=6,
                                random_state=42, n_jobs=-1).fit(X, y)
    return {"lr": lr, "rf": rf, "features": RISK_FEATURES,
            "lr_acc": accuracy_score(y, lr.predict(X)),
            "rf_acc": accuracy_score(y, rf.predict(X))}

@st.cache_resource(show_spinner="Training GPA regressor…")
def train_gpa_model(stu: pd.DataFrame):
    X = make_gpa_X(stu)
    y = stu["current_gpa"]
    m = LinearRegression().fit(X, y)
    return {"model": m, "features": GPA_FEATURES, "r2": m.score(X, y),
            "coefs": dict(zip(GPA_FEATURES, m.coef_)),
            "intercept": m.intercept_,
            "means": {k: float(X[k].mean()) for k in GPA_FEATURES}}

@st.cache_resource(show_spinner="Clustering cohort…")
def build_clusters(stu: pd.DataFrame, k: int = 4):
    X = pd.DataFrame({
        "GPA": stu["current_gpa"].values,
        "Attendance %": stu["cumulative_attendance_pct"].values,
        "Grade Level": stu["grade_level"].values,
        "Scholarship": (stu["scholarship_flag"]=="Yes").astype(int).values,
    })
    scaler = StandardScaler().fit(X)
    km = KMeans(n_clusters=k, random_state=42, n_init=10).fit(scaler.transform(X))
    return {"labels": km.labels_,
            "centers": scaler.inverse_transform(km.cluster_centers_),
            "features": list(X.columns)}

@st.cache_resource(show_spinner="Scanning anomalies…")
def detect_anomalies(stu: pd.DataFrame):
    X = stu[["current_gpa","cumulative_attendance_pct","grade_level"]].values
    iso = IsolationForest(contamination=0.05, random_state=42).fit(X)
    return {"scores": iso.score_samples(X), "flags": iso.predict(X)}


# ═══════════════════════════════════════════════════════════════════
#  SIDEBAR
# ═══════════════════════════════════════════════════════════════════
st.sidebar.markdown("""
<div style='padding:.5rem 0 1.4rem 0;'>
  <div style='font-size:1.55rem; font-weight:800; color:white; letter-spacing:-0.02em;'>
    🎓 Nexus <span style='color:#A78BFA;'>Analytics</span>
  </div>
  <div style='font-size:.72rem; color:#CBD5E1; margin-top:.25rem;
              letter-spacing:.1em; text-transform:uppercase;'>
    Academic Intelligence Platform
  </div>
</div>""", unsafe_allow_html=True)

uploaded = st.sidebar.file_uploader("Workbook (.xlsx)", type="xlsx")
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
parents = views["parents"]

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
frec = rec[rec["student_id"].isin(ids)]       if ids else rec.iloc[0:0]
fatt = att[att["stakeholder_id"].isin(ids)]   if ids else att.iloc[0:0]
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

    with tab1:
        section("Core Distributions")
        col1, col2, col3 = st.columns(3)

        fig = px.histogram(fstu, x="current_gpa", nbins=25, title="GPA Distribution",
                           color_discrete_sequence=[theme["primary"]])
        fig.update_traces(marker_line_width=1, marker_line_color="white")
        fig.add_vline(x=fstu["current_gpa"].mean(), line_dash="dash",
                      line_color=theme["accent"], line_width=3,
                      annotation_text=f"Mean {fstu['current_gpa'].mean():.2f}",
                      annotation_position="top right", annotation_font_color="#0F172A")
        fig.update_layout(xaxis_title="GPA (0–4 scale)", yaxis_title="Number of students")
        col1.plotly_chart(style_fig(fig, theme, show_legend=False, height=300, bargap=0.1),
                          use_container_width=True)
        with col1: caption("Count of students in each GPA bucket. The dashed line is the "
                           "cohort mean — bars to its left are below-average performers.")

        fig = px.histogram(fstu, x="cumulative_attendance_pct", nbins=25,
                           title="Attendance Distribution",
                           color_discrete_sequence=[theme["accent"]])
        fig.update_traces(marker_line_width=1, marker_line_color="white")
        fig.add_vline(x=fstu["cumulative_attendance_pct"].mean(), line_dash="dash",
                      line_color=theme["primary"], line_width=3,
                      annotation_text=f"Mean {fstu['cumulative_attendance_pct'].mean():.1f}%",
                      annotation_position="top right", annotation_font_color="#0F172A")
        fig.update_layout(xaxis_title="Attendance %", yaxis_title="Number of students")
        col2.plotly_chart(style_fig(fig, theme, show_legend=False, height=300, bargap=0.1),
                          use_container_width=True)
        with col2: caption("How attendance is spread across the student body. Students "
                           "far to the left of the mean line are chronic-absence cases "
                           "worth investigating.")

        fig = px.histogram(fstu, x="performance_index", nbins=25,
                           title="Performance Index", color_discrete_sequence=[theme["palette"][2]])
        fig.update_traces(marker_line_width=1, marker_line_color="white")
        fig.update_layout(xaxis_title="Performance Index (GPA×25 + Attendance×0.5)",
                          yaxis_title="Number of students")
        col3.plotly_chart(style_fig(fig, theme, show_legend=False, height=300, bargap=0.1),
                          use_container_width=True)
        with col3: caption("A blended score combining GPA and attendance into a single "
                           "100-point metric. Useful for ranking when grades alone miss "
                           "engagement.")

        section("Composition")
        col1, col2, col3 = st.columns(3)

        rc = fstu["academic_risk_flag"].value_counts().reset_index(); rc.columns = ["risk","count"]
        fig = px.pie(rc, names="risk", values="count", hole=0.6, title="Risk Composition",
                     color="risk",
                     color_discrete_map={"Low":"#10B981","Medium":"#F59E0B","High":"#EF4444"})
        fig.update_traces(textposition="outside", textinfo="label+percent",
                          textfont=dict(size=13, color="#0F172A"),
                          marker=dict(line=dict(color="white", width=2)))
        fig.add_annotation(text=f"<b>{len(fstu):,}</b><br>students",
                           showarrow=False, font=dict(size=15, color="#0F172A"))
        col1.plotly_chart(style_fig(fig, theme, height=300), use_container_width=True)
        with col1: caption("Split of students into Low / Medium / High academic risk. "
                           "A growing red slice signals intervention is needed.")

        gc = fstu["gender"].value_counts().reset_index(); gc.columns = ["gender","count"]
        fig = px.pie(gc, names="gender", values="count", hole=0.6, title="Gender Composition",
                     color_discrete_sequence=theme["palette"])
        fig.update_traces(textposition="outside", textinfo="label+percent",
                          textfont=dict(size=13, color="#0F172A"),
                          marker=dict(line=dict(color="white", width=2)))
        col2.plotly_chart(style_fig(fig, theme, height=300), use_container_width=True)
        with col2: caption("Share of male vs female students. Check for imbalance that "
                           "could point to retention issues in one group.")

        gl = fstu["grade_level"].value_counts().sort_index().reset_index(); gl.columns = ["grade","count"]
        fig = px.funnel(gl.sort_values("count", ascending=True), x="count", y="grade",
                        title="Grade Funnel", color_discrete_sequence=[theme["primary"]])
        fig.update_traces(textfont=dict(color="white", size=12, family="Inter"))
        fig.update_layout(xaxis_title="Number of students", yaxis_title="Grade level")
        col3.plotly_chart(style_fig(fig, theme, show_legend=False, height=300),
                          use_container_width=True)
        with col3: caption("Students per grade shown as a funnel. A shrinking funnel toward "
                           "senior grades suggests drop-off or transfers.")

        section("Correlations")
        sample = fstu.sample(min(2500, len(fstu)), random_state=1)
        fig = px.scatter(sample, x="cumulative_attendance_pct", y="current_gpa",
                         color="academic_risk_flag", opacity=0.65,
                         title="GPA vs Attendance",
                         color_discrete_map={"Low":"#10B981","Medium":"#F59E0B","High":"#EF4444"},
                         hover_data=["school","grade_level"])
        fig = add_trendline(fig, sample["cumulative_attendance_pct"],
                            sample["current_gpa"], color=theme["primary"])
        fig.update_layout(xaxis_title="Attendance %", yaxis_title="GPA")
        st.plotly_chart(style_fig(fig, theme, height=420), use_container_width=True)
        caption("Each dot is a student. The dashed trendline shows the average "
                "relationship: a 10-point rise in attendance lifts GPA by ~0.4. "
                "Red (high-risk) students cluster in the bottom-left.")

    with tab2:
        by_school = (fstu.groupby("school").agg(
            students=("student_id","count"), avg_gpa=("current_gpa","mean"),
            avg_att=("cumulative_attendance_pct","mean"),
            high_risk=("academic_risk_flag", lambda s: (s=="High").mean()*100),
        ).reset_index().sort_values("avg_gpa", ascending=False))

        section("Top performers")
        top_n = st.slider("Show top N schools", 5, len(by_school), min(10, len(by_school)))

        fig = px.bar(by_school.head(top_n), x="school", y="avg_gpa",
                     title=f"Top {top_n} Schools by Average GPA",
                     text_auto=".2f", color="avg_gpa", color_continuous_scale=theme["scale"],
                     hover_data=["students","avg_att","high_risk"])
        fig.update_traces(textfont=dict(color="white", size=12))
        fig.update_layout(xaxis_title="School", yaxis_title="Average GPA")
        st.plotly_chart(style_fig(fig, theme, show_legend=False, height=420),
                        use_container_width=True)
        caption("Schools ranked by their cohort's mean GPA. Hover any bar to see enrolment, "
                "attendance, and high-risk percentage for that school.")

        fig = px.scatter(by_school, x="avg_att", y="avg_gpa", size="students",
                         hover_name="school", title="School Quadrant · Attendance × GPA",
                         color="high_risk",
                         color_continuous_scale=[[0,"#10B981"],[0.5,"#F59E0B"],[1,"#EF4444"]],
                         labels={"high_risk":"High-risk %"})
        fig.add_hline(y=by_school["avg_gpa"].mean(), line_dash="dot", line_color="#64748B",
                      annotation_text="mean GPA", annotation_position="top left",
                      annotation_font_color="#0F172A")
        fig.add_vline(x=by_school["avg_att"].mean(), line_dash="dot", line_color="#64748B",
                      annotation_text="mean attendance", annotation_position="bottom right",
                      annotation_font_color="#0F172A")
        fig.update_layout(xaxis_title="Average attendance %", yaxis_title="Average GPA")
        st.plotly_chart(style_fig(fig, theme, height=500), use_container_width=True)
        caption("Schools plotted as bubbles (size = enrolment). Top-right is the high-performer "
                "quadrant, bottom-left flags schools that need support on both axes.")

    with tab3:
        section("Geographic & board distribution")
        col1, col2 = st.columns(2)

        regional = fstu.groupby(["region","school"]).size().reset_index(name="count")
        fig = px.treemap(regional, path=["region","school"], values="count",
                         title="Students · Region → School",
                         color="count", color_continuous_scale=theme["scale"])
        fig.update_traces(marker=dict(cornerradius=6), textfont=dict(color="white", size=12))
        col1.plotly_chart(style_fig(fig, theme, height=420), use_container_width=True)
        with col1: caption("Hierarchical view: outer blocks are regions, inner blocks are "
                           "schools. Box size is proportional to student count.")

        boards = fstu.groupby(["board_name","school_type"]).size().reset_index(name="count")
        fig = px.sunburst(boards, path=["board_name","school_type"], values="count",
                          title="Board → School-Type Breakdown",
                          color="count", color_continuous_scale=theme["scale"])
        fig.update_traces(textfont=dict(color="white", size=12),
                          marker=dict(line=dict(color="white", width=2)))
        col2.plotly_chart(style_fig(fig, theme, height=420), use_container_width=True)
        with col2: caption("Inner ring is the examination board (CBSE, ICSE, State). "
                           "Outer ring shows primary/secondary split within each board.")

        rg = fstu.groupby("region")["current_gpa"].mean().reset_index()
        fig = px.bar_polar(rg, r="current_gpa", theta="region",
                           title="Average GPA by Region (Polar)",
                           color="current_gpa", color_continuous_scale=theme["scale"])
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", height=420, title_x=0.02,
                          font=dict(color="#0F172A", family="Inter"),
                          polar=dict(bgcolor="rgba(0,0,0,0)",
                                     angularaxis=dict(tickfont=dict(color="#0F172A", size=12)),
                                     radialaxis=dict(tickfont=dict(color="#334155", size=11))))
        st.plotly_chart(fig, use_container_width=True)
        caption("Average GPA shown in a circular (polar) layout. Longer arms mean higher "
                "regional performance — useful for geographic equity reviews.")


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

    with tab1:
        section("Grade-level distributions")
        col1, col2 = st.columns(2)

        fig = px.violin(fstu, x="grade_level", y="current_gpa", box=True, color="grade_level",
                        color_discrete_sequence=theme["palette"],
                        title="GPA Distribution by Grade")
        fig.update_layout(xaxis_title="Grade level", yaxis_title="GPA")
        col1.plotly_chart(style_fig(fig, theme, show_legend=False, height=380),
                          use_container_width=True)
        with col1: caption("Violin plots show the full shape of GPA per grade (wider = "
                           "more students at that level). The box inside marks quartiles "
                           "and the middle line is the median.")

        fig = px.violin(fstu, x="grade_level", y="cumulative_attendance_pct", box=True,
                        color="grade_level", color_discrete_sequence=theme["palette"],
                        title="Attendance % by Grade")
        fig.update_layout(xaxis_title="Grade level", yaxis_title="Attendance %")
        col2.plotly_chart(style_fig(fig, theme, show_legend=False, height=380),
                          use_container_width=True)
        with col2: caption("Same violin approach for attendance. Narrow violins mean the "
                           "grade is homogeneous; wide ones mean mixed engagement.")

        section("Demographic splits")
        col1, col2, col3 = st.columns(3)

        fig = px.box(fstu, x="gender", y="current_gpa", color="gender",
                     title="GPA by Gender", points="all",
                     color_discrete_sequence=theme["palette"])
        fig.update_traces(marker=dict(size=3, opacity=0.3))
        fig.update_layout(xaxis_title="Gender", yaxis_title="GPA")
        col1.plotly_chart(style_fig(fig, theme, show_legend=False, height=320),
                          use_container_width=True)
        with col1: caption("Every student shown as a small dot plus a box summarising each "
                           "group. Tells you both the distribution and how many outliers exist.")

        moi = fstu["medium_of_instruction"].value_counts().reset_index(); moi.columns = ["medium","count"]
        fig = px.bar(moi, x="medium", y="count", title="Medium of Instruction",
                     text_auto=True, color="medium", color_discrete_sequence=theme["palette"])
        fig.update_traces(textfont=dict(color="white", size=13))
        fig.update_layout(xaxis_title="Medium", yaxis_title="Number of students")
        col2.plotly_chart(style_fig(fig, theme, show_legend=False, height=320),
                          use_container_width=True)
        with col2: caption("Share of students by teaching medium — helps plan language "
                           "support and material translations.")

        cc = fstu["caste_category"].value_counts().reset_index(); cc.columns = ["cat","count"]
        fig = px.pie(cc, names="cat", values="count", hole=0.55, title="Caste Category",
                     color_discrete_sequence=theme["palette"])
        fig.update_traces(textposition="outside", textinfo="label+percent",
                          textfont=dict(size=12, color="#0F172A"),
                          marker=dict(line=dict(color="white", width=2)))
        col3.plotly_chart(style_fig(fig, theme, height=320), use_container_width=True)
        with col3: caption("Diversity breakdown — useful for reservation compliance and "
                           "inclusion reporting.")

    with tab2:
        section("Scholarship & IEP impact")
        col1, col2 = st.columns(2)

        s_gpa = fstu.groupby("scholarship_flag")["current_gpa"].mean().reset_index()
        fig = px.bar(s_gpa, x="scholarship_flag", y="current_gpa",
                     title="Avg GPA · Scholarship vs Non-scholarship",
                     text_auto=".2f", color="scholarship_flag",
                     color_discrete_sequence=[theme["palette"][2], theme["primary"]])
        fig.update_traces(textfont=dict(color="white", size=14))
        fig.update_layout(xaxis_title="Scholarship status", yaxis_title="Average GPA")
        col1.plotly_chart(style_fig(fig, theme, show_legend=False, height=320),
                          use_container_width=True)
        with col1: caption("Does giving a scholarship actually lift outcomes? This compares "
                           "the two cohorts' average GPA directly.")

        i_gpa = fstu.groupby("iep_flag")["current_gpa"].mean().reset_index()
        fig = px.bar(i_gpa, x="iep_flag", y="current_gpa",
                     title="Avg GPA · IEP vs Non-IEP",
                     text_auto=".2f", color="iep_flag",
                     color_discrete_sequence=[theme["palette"][3], theme["palette"][5]])
        fig.update_traces(textfont=dict(color="white", size=14))
        fig.update_layout(xaxis_title="IEP status", yaxis_title="Average GPA")
        col2.plotly_chart(style_fig(fig, theme, show_legend=False, height=320),
                          use_container_width=True)
        with col2: caption("Students on Individualised Education Plans (IEP) vs those not. "
                           "Expected lower GPA confirms IEP is correctly targeting "
                           "students who need extra support.")

        section("GPA × Attendance density")
        fig = px.density_heatmap(fstu, x="cumulative_attendance_pct", y="current_gpa",
                                 nbinsx=25, nbinsy=25,
                                 title="Joint Density · Attendance × GPA",
                                 color_continuous_scale=theme["scale"])
        fig.update_layout(xaxis_title="Attendance %", yaxis_title="GPA")
        st.plotly_chart(style_fig(fig, theme, show_legend=False, height=450),
                        use_container_width=True)
        caption("A heatmap showing where students concentrate. Darker cells = more students "
                "at that (attendance, GPA) combination. The diagonal pattern confirms "
                "attendance and GPA rise together.")

    with tab3:
        section("Per-school performance")
        sc = fstu.groupby("school")["current_gpa"].mean().reset_index().sort_values(
            "current_gpa", ascending=True)
        fig = px.bar(sc, x="current_gpa", y="school", orientation="h",
                     title="Average GPA by School", text_auto=".2f",
                     color="current_gpa", color_continuous_scale=theme["scale"])
        fig.update_traces(textfont=dict(color="#0F172A", size=11))
        fig.update_layout(xaxis_title="Average GPA", yaxis_title="School")
        st.plotly_chart(style_fig(fig, theme, show_legend=False, height=500),
                        use_container_width=True)
        caption("Horizontal bars let you rank all schools by GPA on one screen. "
                "The lightest bars at the bottom are the schools that need the most attention.")

        section("Multidimensional student profile")
        sample_pc = fstu.sample(min(500, len(fstu)), random_state=1).copy()
        sample_pc["risk_num"] = sample_pc["academic_risk_flag"].map({"Low":0, "Medium":1, "High":2})
        fig = px.parallel_coordinates(
            sample_pc, dimensions=["grade_level","cumulative_attendance_pct",
                                    "current_gpa","performance_index","risk_num"],
            color="risk_num",
            color_continuous_scale=[[0,"#10B981"],[0.5,"#F59E0B"],[1,"#EF4444"]],
            title="Parallel Coordinates · Cohort Profile (sample of 500)",
            labels={"risk_num":"Risk (0=Low, 2=High)",
                    "cumulative_attendance_pct":"Attendance %",
                    "performance_index":"Perf. Index", "current_gpa":"GPA",
                    "grade_level":"Grade"})
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", height=420, title_x=0.02,
                          font=dict(color="#0F172A"))
        st.plotly_chart(fig, use_container_width=True)
        caption("Each line is one student crossing five metrics. Red lines are high-risk "
                "students — you can literally trace their paths across attendance, GPA, "
                "and performance axes to see exactly where they lag.")

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

    with tab1:
        section("Grade & pass distributions")
        col1, col2 = st.columns(2)

        order = ["A1","A2","B1","B2","C1","C2","D","E"]
        gd = frec["grade_awarded"].value_counts().reindex(order, fill_value=0).reset_index()
        gd.columns = ["grade","count"]
        fig = px.bar(gd, x="grade", y="count", text_auto=True,
                     title="Letter Grade Distribution",
                     color="count", color_continuous_scale=theme["scale"])
        fig.update_traces(textfont=dict(color="#0F172A", size=12))
        fig.update_layout(xaxis_title="Letter grade", yaxis_title="Exam count")
        col1.plotly_chart(style_fig(fig, theme, show_legend=False, height=340),
                          use_container_width=True)
        with col1: caption("How many exams fell into each letter grade. A healthy bell-ish "
                           "curve centered around B/C is normal; a fat tail on D/E signals "
                           "widespread difficulty.")

        pf = frec["pass_fail"].value_counts().reset_index(); pf.columns = ["status","count"]
        fig = px.pie(pf, names="status", values="count", hole=0.6, title="Pass / Fail Split",
                     color="status", color_discrete_map={"Pass":"#10B981","Fail":"#EF4444"})
        fig.update_traces(textposition="outside", textinfo="label+percent+value",
                          textfont=dict(size=13, color="#0F172A"),
                          marker=dict(line=dict(color="white", width=2)))
        col2.plotly_chart(style_fig(fig, theme, height=340), use_container_width=True)
        with col2: caption("Simple pass vs fail donut. Red slice size is an immediate health "
                           "indicator — under 10% is generally acceptable.")

        section("Percentage distribution")
        fig = px.histogram(frec, x="percentage", nbins=30, title="Exam Percentage Distribution",
                           color_discrete_sequence=[theme["primary"]])
        fig.update_traces(marker_line_width=1, marker_line_color="white")
        fig.add_vline(x=35, line_dash="dash", line_color="#DC2626", line_width=3,
                      annotation_text="Pass threshold (35%)",
                      annotation_position="top right", annotation_font_color="#0F172A")
        fig.update_layout(xaxis_title="Exam percentage", yaxis_title="Exam count")
        st.plotly_chart(style_fig(fig, theme, show_legend=False, height=360, bargap=0.1),
                        use_container_width=True)
        caption("Full percentage distribution with the 35% pass line. Exams falling left "
                "of the red line are failures — the count of failing exams is readable "
                "directly off the chart.")

    with tab2:
        section("Subject-level performance")
        col1, col2 = st.columns(2)

        fig = px.box(frec, x="subject_name", y="percentage", color="subject_name",
                     title="Score Distribution by Subject", points=False,
                     color_discrete_sequence=theme["palette"])
        fig.update_layout(xaxis_title="Subject", yaxis_title="Exam percentage")
        col1.plotly_chart(style_fig(fig, theme, show_legend=False, height=380),
                          use_container_width=True)
        with col1: caption("Box per subject — the line inside is the median exam score. "
                           "Subjects whose entire box sits low (like Math here) are where "
                           "you likely need curriculum or teaching interventions.")

        sp = (frec.groupby("subject_name")
                  .apply(lambda d: (d["pass_fail"]=="Pass").mean()*100, include_groups=False)
                  .reset_index(name="pass_rate").sort_values("pass_rate"))
        fig = px.bar(sp, x="pass_rate", y="subject_name", orientation="h",
                     title="Pass Rate by Subject (%)", text_auto=".1f",
                     color="pass_rate", color_continuous_scale=theme["scale"])
        fig.update_traces(textfont=dict(color="white", size=13))
        fig.update_layout(xaxis_title="Pass rate %", yaxis_title="Subject")
        col2.plotly_chart(style_fig(fig, theme, show_legend=False, height=380),
                          use_container_width=True)
        with col2: caption("Percentage of students passing each subject. The shortest bar is "
                           "the hardest subject — double-check it for syllabus load or "
                           "teacher assignment issues.")

        section("Subject profile")
        subj_avg = frec.groupby("subject_name").agg(
            exam_pct=("percentage","mean"), assignment=("assignment_score","mean"),
            project=("project_score","mean"),
        ).reset_index()
        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(r=subj_avg["exam_pct"], theta=subj_avg["subject_name"],
                                      fill='toself', name="Exam %", line_color=theme["primary"]))
        fig.add_trace(go.Scatterpolar(r=subj_avg["assignment"]*5, theta=subj_avg["subject_name"],
                                      fill='toself', name="Assignment×5",
                                      line_color=theme["palette"][2]))
        fig.add_trace(go.Scatterpolar(r=subj_avg["project"]*10, theta=subj_avg["subject_name"],
                                      fill='toself', name="Project×10",
                                      line_color=theme["palette"][3]))
        fig.update_layout(
            title="Subject Profile Radar",
            polar=dict(bgcolor="rgba(248,250,252,0.5)",
                      radialaxis=dict(visible=True, range=[0, 100],
                                      tickfont=dict(color="#334155", size=11)),
                      angularaxis=dict(tickfont=dict(color="#0F172A", size=13))),
            paper_bgcolor="rgba(0,0,0,0)", height=440, title_x=0.02,
            font_family='"Inter", sans-serif', font_color="#0F172A",
            legend=dict(orientation="h", yanchor="bottom", y=-0.1, font=dict(color="#0F172A")))
        st.plotly_chart(fig, use_container_width=True)
        caption("Radar comparing three assessment types per subject. If one polygon is much "
                "smaller than others, students excel at that type (exam/assignment/project) "
                "in a way that's uneven across subjects.")

        section("Assignments vs exam scores")
        sample = frec.sample(min(2000, len(frec)), random_state=1)
        fig = px.scatter(sample, x="assignment_score", y="marks_obtained", color="subject_name",
                         opacity=0.6, title="Assignment Score vs Exam Marks",
                         color_discrete_sequence=theme["palette"])
        fig = add_trendline(fig, sample["assignment_score"], sample["marks_obtained"],
                            color="#0F172A")
        fig.update_layout(xaxis_title="Assignment score (out of 20)",
                          yaxis_title="Exam marks obtained (out of 100)")
        st.plotly_chart(style_fig(fig, theme, height=440), use_container_width=True)
        caption("Does classwork predict exam performance? A steep upward trendline says yes — "
                "assignment work is genuinely developing exam skills.")

    with tab3:
        ts = frec.groupby(["term_name","subject_name"])["percentage"].mean().reset_index()
        fig = px.line(ts, x="term_name", y="percentage", color="subject_name",
                      markers=True, title="Term-over-Term Subject Averages",
                      color_discrete_sequence=theme["palette"])
        fig.update_traces(line_width=3, marker=dict(size=11, line=dict(color="white", width=2)))
        fig.update_layout(xaxis_title="Term", yaxis_title="Average percentage")
        st.plotly_chart(style_fig(fig, theme, height=400), use_container_width=True)
        caption("How each subject's average score changes from Term 1 → 2 → 3. Lines trending "
                "upward mean students are improving; flat or declining lines flag subjects "
                "that need a pedagogy review.")

        tp = (frec.groupby("term_name")
                  .apply(lambda d: (d["pass_fail"]=="Pass").mean()*100, include_groups=False)
                  .reset_index(name="pass_rate"))
        fig = px.bar(tp, x="term_name", y="pass_rate", text_auto=".1f",
                     title="Pass Rate by Term (%)",
                     color="pass_rate", color_continuous_scale=theme["scale"])
        fig.update_layout(yaxis_range=[80, 100], xaxis_title="Term", yaxis_title="Pass rate %")
        fig.update_traces(textfont=dict(color="white", size=14))
        st.plotly_chart(style_fig(fig, theme, show_legend=False, height=340),
                        use_container_width=True)
        caption("Overall pass rate per term. If Term 3 is much higher than Term 1, students "
                "are adapting; a drop in Term 3 could mean exam fatigue.")

        pivot = frec.pivot_table(index="subject_name", columns="term_name",
                                  values="percentage", aggfunc="mean")
        fig = px.imshow(pivot, text_auto=".1f", title="Avg % · Subject × Term Heatmap",
                        color_continuous_scale=theme["scale"], aspect="auto")
        fig.update_traces(textfont=dict(size=14, color="#0F172A"))
        fig.update_layout(xaxis_title="Term", yaxis_title="Subject")
        st.plotly_chart(style_fig(fig, theme, show_legend=False, height=380),
                        use_container_width=True)
        caption("A grid cross-tabulating subject and term. Darker cells = higher average. "
                "Look for a single dark column (one great term) or a single light row "
                "(one weak subject) to diagnose where the action is.")


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

    with tab1:
        section("Department structure")
        col1, col2 = st.columns(2)

        dept = ftch["department"].value_counts().reset_index(); dept.columns = ["department","count"]
        fig = px.bar(dept, x="department", y="count", text_auto=True,
                     color="department", title="Teachers per Department",
                     color_discrete_sequence=theme["palette"])
        fig.update_traces(textfont=dict(color="white", size=13))
        fig.update_layout(xaxis_title="Department", yaxis_title="Teacher count")
        col1.plotly_chart(style_fig(fig, theme, show_legend=False, height=340),
                          use_container_width=True)
        with col1: caption("How teaching staff is distributed across departments. "
                           "A lopsided chart could indicate hiring gaps.")

        emp = ftch["employment_type"].value_counts().reset_index(); emp.columns = ["type","count"]
        fig = px.pie(emp, names="type", values="count", hole=0.55, title="Employment Type",
                     color_discrete_sequence=theme["palette"])
        fig.update_traces(textposition="outside", textinfo="label+percent",
                          textfont=dict(size=13, color="#0F172A"),
                          marker=dict(line=dict(color="white", width=2)))
        col2.plotly_chart(style_fig(fig, theme, height=340), use_container_width=True)
        with col2: caption("Split between permanent, contract, and visiting faculty. A high "
                           "contract share is a retention/risk signal.")

        section("Qualifications & gender")
        col1, col2 = st.columns(2)

        q = ftch["highest_qualification"].value_counts().reset_index(); q.columns = ["qual","count"]
        fig = px.bar(q, x="count", y="qual", orientation="h", text_auto=True,
                     title="Highest Qualification", color="count",
                     color_continuous_scale=theme["scale"])
        fig.update_traces(textfont=dict(color="white", size=12))
        fig.update_layout(xaxis_title="Teacher count", yaxis_title="Qualification")
        col1.plotly_chart(style_fig(fig, theme, show_legend=False, height=340),
                          use_container_width=True)
        with col1: caption("Faculty credentials at a glance. A higher ratio of M.Phil/PhD "
                           "generally correlates with stronger academic outcomes.")

        g = ftch["gender"].value_counts().reset_index(); g.columns = ["gender","count"]
        fig = px.pie(g, names="gender", values="count", hole=0.55, title="Gender Split (Faculty)",
                     color_discrete_sequence=theme["palette"])
        fig.update_traces(textposition="outside", textinfo="label+percent",
                          textfont=dict(size=13, color="#0F172A"),
                          marker=dict(line=dict(color="white", width=2)))
        col2.plotly_chart(style_fig(fig, theme, height=340), use_container_width=True)
        with col2: caption("Male/female teacher split — important for diversity targets "
                           "and role-modeling considerations.")

    with tab2:
        fig = px.box(ftch, x="department", y="teacher_performance_rating",
                     color="department", title="Performance Rating by Department",
                     points="all", color_discrete_sequence=theme["palette"])
        fig.update_traces(marker=dict(size=3, opacity=0.4))
        fig.update_layout(xaxis_title="Department", yaxis_title="Performance rating (1–5)")
        st.plotly_chart(style_fig(fig, theme, show_legend=False, height=400),
                        use_container_width=True)
        caption("Each dot is one teacher. Hovering reveals their rating; the box shows the "
                "department distribution. Compare medians (line in box) to see which "
                "departments are strongest.")

        fig = px.scatter(ftch, x="years_experience", y="teacher_performance_rating",
                         color="department", opacity=0.7, size="classes_assigned_count",
                         title="Experience × Rating (bubble size = classes assigned)",
                         hover_data=["full_name","school"],
                         color_discrete_sequence=theme["palette"])
        fig = add_trendline(fig, ftch["years_experience"],
                            ftch["teacher_performance_rating"], color="#0F172A")
        fig.update_layout(xaxis_title="Years of experience",
                          yaxis_title="Performance rating (1–5)")
        st.plotly_chart(style_fig(fig, theme, height=420), use_container_width=True)
        caption("Do senior teachers perform better? The trendline answers at a glance. "
                "Bubble size shows workload — big dots in the bottom-right would be "
                "experienced but overloaded teachers.")

        section("Top 10 teachers by rating")
        top = ftch.nlargest(10, "teacher_performance_rating")[
            ["full_name","department","years_experience","teacher_performance_rating","school"]]
        st.dataframe(top, use_container_width=True, hide_index=True)

    with tab3:
        col1, col2 = st.columns(2)

        fig = px.histogram(ftch, x="weekly_workload_hours", nbins=20,
                           title="Weekly Workload Hours", color_discrete_sequence=[theme["primary"]])
        fig.update_traces(marker_line_width=1, marker_line_color="white")
        fig.update_layout(xaxis_title="Weekly teaching hours", yaxis_title="Teacher count")
        col1.plotly_chart(style_fig(fig, theme, show_legend=False, height=340, bargap=0.1),
                          use_container_width=True)
        with col1: caption("Teacher workload spread. A long right tail means some teachers "
                           "carry far more than others — check those individuals for burnout risk.")

        fig = px.density_heatmap(ftch, x="weekly_workload_hours",
                                 y="teacher_performance_rating", nbinsx=15, nbinsy=15,
                                 title="Workload vs Rating (density)",
                                 color_continuous_scale=theme["scale"])
        fig.update_layout(xaxis_title="Weekly workload hours",
                          yaxis_title="Performance rating")
        col2.plotly_chart(style_fig(fig, theme, show_legend=False, height=340),
                          use_container_width=True)
        with col2: caption("Bright cells indicate common workload-rating combinations. If the "
                           "brightest zone sits at high-workload/low-rating, you have a "
                           "burnout correlation.")

        fig = px.scatter(ftch, x="teacher_attendance_pct", y="teacher_performance_rating",
                         color="department", opacity=0.7,
                         title="Teacher Attendance % × Rating", color_discrete_sequence=theme["palette"])
        fig = add_trendline(fig, ftch["teacher_attendance_pct"],
                            ftch["teacher_performance_rating"], color="#0F172A")
        fig.update_layout(xaxis_title="Teacher attendance %",
                          yaxis_title="Performance rating")
        st.plotly_chart(style_fig(fig, theme, height=420), use_container_width=True)
        caption("Is teacher attendance tied to their rating? A steep positive line confirms "
                "show-up-rate matters.")


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

    with tab1:
        section("Status & reasons")
        col1, col2 = st.columns(2)

        status = fatt["attendance_status"].value_counts().reset_index(); status.columns = ["status","count"]
        fig = px.pie(status, names="status", values="count", hole=0.6,
                     title="Attendance Status Split", color="status",
                     color_discrete_map={"Present":"#10B981","Absent":"#EF4444",
                                         "Late":"#F59E0B","Leave":"#94A3B8"})
        fig.update_traces(textposition="outside", textinfo="label+percent",
                          textfont=dict(size=13, color="#0F172A"),
                          marker=dict(line=dict(color="white", width=2)))
        fig.add_annotation(text=f"<b>{pres:.1f}%</b><br>present",
                           showarrow=False, font=dict(size=15, color="#0F172A"))
        col1.plotly_chart(style_fig(fig, theme, height=360), use_container_width=True)
        with col1: caption("Overall attendance breakdown across all logged days. The centre "
                           "number is the headline present-percentage for the current filter.")

        absences = fatt[fatt["attendance_status"]!="Present"]
        if len(absences):
            reason = absences["reason"].value_counts().reset_index(); reason.columns = ["reason","count"]
            fig = px.bar(reason, x="reason", y="count", text_auto=True,
                         title="Reasons for Absence / Late / Leave",
                         color="count", color_continuous_scale=theme["scale"])
            fig.update_traces(textfont=dict(color="white", size=13))
            fig.update_layout(xaxis_title="Reason", yaxis_title="Instance count")
            col2.plotly_chart(style_fig(fig, theme, show_legend=False, height=360),
                              use_container_width=True)
            with col2: caption("Why aren't students in class? Illness is expected, but a tall "
                               "'Family' or 'Travel' bar suggests calendar-planning "
                               "conversations are needed.")

    with tab2:
        daily = (fatt.assign(p=(fatt["attendance_status"]=="Present").astype(int))
                      .groupby("attendance_date")["p"].mean().reset_index())
        daily["p"] *= 100
        fig = px.line(daily, x="attendance_date", y="p", title="Daily Attendance % Trend",
                      markers=True, color_discrete_sequence=[theme["primary"]])
        fig.update_traces(line_width=3, marker=dict(size=7, line=dict(color="white", width=1.5)))
        fig.add_hline(y=daily["p"].mean(), line_dash="dash", line_color=theme["accent"],
                      line_width=2, annotation_text=f"Mean {daily['p'].mean():.1f}%",
                      annotation_font_color="#0F172A")
        fig.update_layout(xaxis_title="Date", yaxis_title="Present %")
        st.plotly_chart(style_fig(fig, theme, show_legend=False, height=380),
                        use_container_width=True)
        caption("Day-by-day attendance with a mean line. Sharp dips mark problem days — "
                "weather events, exam days, or local disruptions worth investigating.")

        dow_order = ["Monday","Tuesday","Wednesday","Thursday","Friday"]
        dow = (fatt.assign(p=(fatt["attendance_status"]=="Present").astype(int))
                    .groupby("day_of_week")["p"].mean().reindex(dow_order).reset_index())
        dow["p"] *= 100
        fig = px.bar(dow, x="day_of_week", y="p", text_auto=".1f",
                     title="Attendance % by Day of Week",
                     color="p", color_continuous_scale=theme["scale"])
        fig.update_traces(textfont=dict(color="white", size=14))
        fig.update_layout(yaxis_range=[70, 100], xaxis_title="Day of week",
                          yaxis_title="Present %")
        st.plotly_chart(style_fig(fig, theme, show_legend=False, height=340),
                        use_container_width=True)
        caption("Monday blues? Friday fade? This reveals weekly rhythm. Consistent "
                "Monday dips indicate a scheduling issue worth addressing.")

        weekly = (fatt.assign(p=(fatt["attendance_status"]=="Present").astype(int))
                       .groupby("week")["p"].mean().reset_index())
        weekly["p"] *= 100
        fig = px.area(weekly, x="week", y="p", title="Weekly Attendance % Trend",
                      color_discrete_sequence=[theme["primary"]])
        fig.update_traces(line_width=3, fillcolor=theme["primary"]+"30")
        fig.update_layout(xaxis_title="Week of year", yaxis_title="Present %")
        st.plotly_chart(style_fig(fig, theme, show_legend=False, height=340),
                        use_container_width=True)
        caption("Smoothed weekly view for spotting longer-term patterns without day-to-day "
                "noise.")

    with tab3:
        col1, col2 = st.columns(2)

        by_grade = (fatt.assign(p=(fatt["attendance_status"]=="Present").astype(int))
                         .groupby("grade_level")["p"].mean().reset_index())
        by_grade["p"] *= 100
        fig = px.bar(by_grade, x="grade_level", y="p", text_auto=".1f",
                     title="Attendance % by Grade",
                     color="p", color_continuous_scale=theme["scale"])
        fig.update_traces(textfont=dict(color="white", size=12))
        fig.update_layout(xaxis_title="Grade", yaxis_title="Present %")
        col1.plotly_chart(style_fig(fig, theme, show_legend=False, height=340),
                          use_container_width=True)
        with col1: caption("Attendance typically drops in higher grades as students juggle "
                           "coaching, exams, and social demands. Flag any unusually low grade.")

        by_region = (fatt.assign(p=(fatt["attendance_status"]=="Present").astype(int))
                         .groupby("region")["p"].mean().reset_index())
        by_region["p"] *= 100
        fig = px.bar(by_region, x="region", y="p", text_auto=".1f",
                     title="Attendance % by Region",
                     color="p", color_continuous_scale=theme["scale"])
        fig.update_traces(textfont=dict(color="white", size=12))
        fig.update_layout(xaxis_title="Region", yaxis_title="Present %")
        col2.plotly_chart(style_fig(fig, theme, show_legend=False, height=340),
                          use_container_width=True)
        with col2: caption("Regional attendance comparison. Big gaps between regions hint at "
                           "infrastructure, transport, or climate issues.")

        by_school = (fatt.assign(p=(fatt["attendance_status"]=="Present").astype(int))
                          .groupby("school")["p"].mean().reset_index()
                          .sort_values("p", ascending=True))
        by_school["p"] *= 100
        fig = px.bar(by_school, x="p", y="school", orientation="h", text_auto=".1f",
                     title="Attendance % by School", color="p",
                     color_continuous_scale=theme["scale"])
        fig.update_traces(textfont=dict(color="#0F172A", size=11))
        fig.update_layout(xaxis_title="Present %", yaxis_title="School")
        st.plotly_chart(style_fig(fig, theme, show_legend=False, height=500),
                        use_container_width=True)
        caption("All schools ranked by presence. The bottom of the list is where principals "
                "and parent councils should focus outreach.")


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

    with tab1:
        summary = (stu.groupby("school").agg(
            students=("student_id","count"), avg_gpa=("current_gpa","mean"),
            avg_att=("cumulative_attendance_pct","mean"),
            high_risk=("academic_risk_flag", lambda s: (s=="High").mean()*100),
        ).reset_index().sort_values("avg_gpa", ascending=False))

        fig = px.bar(summary, x="school", y="avg_gpa", text_auto=".2f",
                     title="Average GPA by School",
                     color="avg_gpa", color_continuous_scale=theme["scale"],
                     hover_data=["students","avg_att","high_risk"])
        fig.update_traces(textfont=dict(color="white", size=11))
        fig.update_layout(xaxis_title="School", yaxis_title="Average GPA")
        st.plotly_chart(style_fig(fig, theme, show_legend=False, height=400),
                        use_container_width=True)
        caption("All schools on one screen, ranked by their student's mean GPA. "
                "This is the quickest way to spot outliers on either end.")

        fig = px.scatter(summary, x="avg_att", y="avg_gpa", size="students",
                         hover_name="school", title="School Quadrant · Attendance × GPA",
                         color="high_risk",
                         color_continuous_scale=[[0,"#10B981"],[0.5,"#F59E0B"],[1,"#EF4444"]],
                         labels={"high_risk":"High-risk %"})
        fig.add_hline(y=summary["avg_gpa"].mean(), line_dash="dot", line_color="#64748B",
                      annotation_text="mean GPA", annotation_font_color="#0F172A")
        fig.add_vline(x=summary["avg_att"].mean(), line_dash="dot", line_color="#64748B",
                      annotation_text="mean att.", annotation_font_color="#0F172A")
        fig.update_layout(xaxis_title="Average attendance %", yaxis_title="Average GPA")
        st.plotly_chart(style_fig(fig, theme, height=480), use_container_width=True)
        caption("Each school as a bubble. Top-right quadrant = high performers on both axes. "
                "Red bubbles anywhere need urgent attention regardless of their position.")

    with tab2:
        col1, col2 = st.columns(2)

        boards = schools["board_name"].value_counts().reset_index(); boards.columns = ["board","count"]
        fig = px.pie(boards, names="board", values="count", hole=0.6, title="Schools by Board",
                     color_discrete_sequence=theme["palette"])
        fig.update_traces(textposition="outside", textinfo="label+percent",
                          textfont=dict(size=13, color="#0F172A"),
                          marker=dict(line=dict(color="white", width=2)))
        col1.plotly_chart(style_fig(fig, theme, height=360), use_container_width=True)
        with col1: caption("CBSE vs ICSE vs State mix across your network.")

        mgmt = schools["management_type"].value_counts().reset_index(); mgmt.columns = ["type","count"]
        fig = px.pie(mgmt, names="type", values="count", hole=0.6, title="Management Type",
                     color_discrete_sequence=theme["palette"])
        fig.update_traces(textposition="outside", textinfo="label+percent",
                          textfont=dict(size=13, color="#0F172A"),
                          marker=dict(line=dict(color="white", width=2)))
        col2.plotly_chart(style_fig(fig, theme, height=360), use_container_width=True)
        with col2: caption("Public / private / trust split — governance context for policy "
                           "rollouts.")

        col1, col2 = st.columns(2)
        regions = schools["region"].value_counts().reset_index(); regions.columns = ["region","count"]
        fig = px.bar(regions, x="region", y="count", text_auto=True, color="region",
                     title="Schools by Region", color_discrete_sequence=theme["palette"])
        fig.update_traces(textfont=dict(color="white", size=14))
        fig.update_layout(xaxis_title="Region", yaxis_title="School count")
        col1.plotly_chart(style_fig(fig, theme, show_legend=False, height=340),
                          use_container_width=True)
        with col1: caption("Geographic concentration of campuses.")

        types = schools["school_type"].value_counts().reset_index(); types.columns = ["type","count"]
        fig = px.bar(types, x="type", y="count", text_auto=True, color="type",
                     title="Schools by Type", color_discrete_sequence=theme["palette"])
        fig.update_traces(textfont=dict(color="white", size=14))
        fig.update_layout(xaxis_title="Type", yaxis_title="School count")
        col2.plotly_chart(style_fig(fig, theme, show_legend=False, height=340),
                          use_container_width=True)
        with col2: caption("Primary, secondary, senior secondary mix.")

        enrolment = stu.groupby("school").size().reset_index(name="enrolled")
        cap = schools[["school","student_capacity"]].merge(enrolment, on="school", how="left")
        cap["utilisation"] = (cap["enrolled"]/cap["student_capacity"])*100
        fig = px.bar(cap.sort_values("utilisation", ascending=False),
                     x="school", y="utilisation", text_auto=".0f",
                     title="Capacity Utilisation (%)",
                     color="utilisation", color_continuous_scale=theme["scale"])
        fig.update_traces(textfont=dict(color="white", size=11))
        fig.update_layout(xaxis_title="School", yaxis_title="Utilisation %")
        st.plotly_chart(style_fig(fig, theme, show_legend=False, height=380),
                        use_container_width=True)
        caption("Current enrolment as a percentage of built capacity. Over 100% means "
                "overcrowding; under 60% means wasted infrastructure.")

    with tab3:
        if {"latitude","longitude"}.issubset(schools.columns):
            st.map(schools[["latitude","longitude"]].dropna(), zoom=4)
            caption("Interactive map of school locations — zoom and pan to explore.")

        perf = stu.groupby(["region","school_type"])["current_gpa"].mean().reset_index()
        pivot = perf.pivot(index="region", columns="school_type", values="current_gpa")
        fig = px.imshow(pivot, text_auto=".2f",
                        title="Avg GPA · Region × School Type",
                        color_continuous_scale=theme["scale"], aspect="auto")
        fig.update_traces(textfont=dict(size=14, color="#0F172A"))
        fig.update_layout(xaxis_title="School type", yaxis_title="Region")
        st.plotly_chart(style_fig(fig, theme, show_legend=False, height=360),
                        use_container_width=True)
        caption("Regional performance cross-tabbed with school type. Dark cells are pockets "
                "of excellence; light cells flag where a specific (region, type) combo "
                "is underperforming.")


# ═══════════════════════════════════════════════════════════════════
#  PAGE: PARENTS & INCOME  (NEW — golden/amber theme)
# ═══════════════════════════════════════════════════════════════════
elif page == "Parents & Income":

    # Order income bands naturally instead of alphabetically
    income_order = ["<3 LPA", "3-6 LPA", "6-10 LPA", "10-20 LPA", "20-50 LPA", "50+ LPA"]
    available_bands = [b for b in income_order if b in fstu["parent_income"].unique()]

    c1,c2,c3,c4,c5 = st.columns(5)
    c1.metric("Students",    f"{len(fstu):,}")
    c2.metric("Income bands", len(available_bands))
    c3.metric("Occupations", fstu["parent_occupation"].nunique())
    c4.metric("Lowest-band GPA",
              f"{fstu[fstu['parent_income']==available_bands[0]]['current_gpa'].mean():.2f}"
              if available_bands else "—")
    c5.metric("Highest-band GPA",
              f"{fstu[fstu['parent_income']==available_bands[-1]]['current_gpa'].mean():.2f}"
              if available_bands else "—")

    tab1, tab2, tab3 = st.tabs(["💰 Income Impact", "🎓 Education & Occupation", "⚠️ Equity Lens"])

    with tab1:
        section("GPA across income bands")

        gpa_by_inc = (fstu.groupby("parent_income")["current_gpa"].mean()
                         .reindex(available_bands).reset_index())
        fig = px.bar(gpa_by_inc, x="parent_income", y="current_gpa", text_auto=".2f",
                     title="Average GPA by Parent Income Band",
                     color="current_gpa", color_continuous_scale=theme["scale"])
        fig.update_traces(textfont=dict(color="white", size=14))
        fig.update_layout(xaxis_title="Parent annual income band",
                          yaxis_title="Average GPA")
        st.plotly_chart(style_fig(fig, theme, show_legend=False, height=380),
                        use_container_width=True)
        caption("The core equity chart: does family income predict academic outcomes? "
                "A rising staircase means socioeconomic gap is real; a flat chart means "
                "the school system is successfully leveling the field.")

        section("Distribution within each band")
        col1, col2 = st.columns(2)

        fig = px.violin(fstu[fstu["parent_income"].isin(available_bands)],
                        x="parent_income", y="current_gpa", box=True,
                        category_orders={"parent_income": available_bands},
                        color="parent_income", color_discrete_sequence=theme["palette"],
                        title="GPA Distribution per Income Band (Violin)")
        fig.update_layout(xaxis_title="Parent income band", yaxis_title="GPA")
        col1.plotly_chart(style_fig(fig, theme, show_legend=False, height=380),
                          use_container_width=True)
        with col1: caption("Full spread of GPAs within each income group. Long violins "
                           "mean high variance — even within the same income group, "
                           "student outcomes differ widely, so income isn't destiny.")

        att_by_inc = (fstu.groupby("parent_income")["cumulative_attendance_pct"].mean()
                         .reindex(available_bands).reset_index())
        fig = px.bar(att_by_inc, x="parent_income", y="cumulative_attendance_pct",
                     text_auto=".1f", title="Average Attendance by Income Band",
                     color="cumulative_attendance_pct", color_continuous_scale=theme["scale"])
        fig.update_traces(textfont=dict(color="white", size=14))
        fig.update_layout(xaxis_title="Parent income band", yaxis_title="Attendance %")
        col2.plotly_chart(style_fig(fig, theme, show_legend=False, height=380),
                          use_container_width=True)
        with col2: caption("Attendance often correlates with income because low-income "
                           "families face transport, health, and work-related absences. "
                           "Check whether the income-attendance gap mirrors the income-GPA gap.")

        section("Risk composition by income")
        risk_inc = (fstu.groupby(["parent_income","academic_risk_flag"]).size()
                        .reset_index(name="count"))
        fig = px.bar(risk_inc, x="parent_income", y="count", color="academic_risk_flag",
                     category_orders={"parent_income": available_bands,
                                      "academic_risk_flag": ["Low","Medium","High"]},
                     title="Risk Composition by Income Band (Stacked)",
                     color_discrete_map={"Low":"#10B981","Medium":"#F59E0B","High":"#EF4444"},
                     barmode="stack")
        fig.update_layout(xaxis_title="Parent income band", yaxis_title="Student count")
        st.plotly_chart(style_fig(fig, theme, height=400),
                        use_container_width=True)
        caption("Stacked bars show how many Low/Medium/High-risk students fall in each "
                "income band. The red portion shrinking as income rises is a clear signal "
                "of an equity gap.")

    with tab2:
        section("Parent education and outcomes")

        edu_order = ["Below 10th","10th","12th","Graduate","Postgraduate","Doctorate"]
        available_edu = [e for e in edu_order if e in fstu["parent_education"].unique()]
        if not available_edu:
            available_edu = sorted(fstu["parent_education"].dropna().unique())

        edu_gpa = (fstu.groupby("parent_education")["current_gpa"].mean()
                       .reindex(available_edu).reset_index())
        fig = px.bar(edu_gpa, x="parent_education", y="current_gpa", text_auto=".2f",
                     title="Average Student GPA by Parent Education Level",
                     color="current_gpa", color_continuous_scale=theme["scale"])
        fig.update_traces(textfont=dict(color="white", size=14))
        fig.update_layout(xaxis_title="Parent highest education", yaxis_title="Student's average GPA")
        st.plotly_chart(style_fig(fig, theme, show_legend=False, height=360),
                        use_container_width=True)
        caption("Parental education often matters as much as income. A rising trend here "
                "means educated parents' involvement — homework help, reading at home, "
                "school engagement — shows up in their child's GPA.")

        section("Parent occupation")
        col1, col2 = st.columns(2)

        occ = fstu["parent_occupation"].value_counts().head(10).reset_index()
        occ.columns = ["occupation","count"]
        fig = px.bar(occ.sort_values("count"), x="count", y="occupation", orientation="h",
                     text_auto=True, title="Top 10 Parent Occupations",
                     color="count", color_continuous_scale=theme["scale"])
        fig.update_traces(textfont=dict(color="white", size=12))
        fig.update_layout(xaxis_title="Student count", yaxis_title="Occupation")
        col1.plotly_chart(style_fig(fig, theme, show_legend=False, height=400),
                          use_container_width=True)
        with col1: caption("Top 10 jobs among parents — useful for career-day planning "
                           "and understanding the network's demographic base.")

        occ_gpa = (fstu.groupby("parent_occupation")["current_gpa"].mean()
                       .reset_index().nlargest(10, "current_gpa"))
        fig = px.bar(occ_gpa.sort_values("current_gpa"), x="current_gpa", y="parent_occupation",
                     orientation="h", text_auto=".2f",
                     title="Top 10 Parent Occupations by Student Avg GPA",
                     color="current_gpa", color_continuous_scale=theme["scale"])
        fig.update_traces(textfont=dict(color="white", size=12))
        fig.update_layout(xaxis_title="Average student GPA", yaxis_title="Parent occupation")
        col2.plotly_chart(style_fig(fig, theme, show_legend=False, height=400),
                          use_container_width=True)
        with col2: caption("Which parent professions correlate with the strongest student "
                           "outcomes? Often professional/white-collar roles top this list — "
                           "but exceptions hint at motivated parents regardless of profession.")

        section("Income × Education heatmap")
        pivot = (fstu.groupby(["parent_income","parent_education"])["current_gpa"]
                     .mean().reset_index()
                     .pivot(index="parent_income", columns="parent_education", values="current_gpa")
                     .reindex(available_bands))
        fig = px.imshow(pivot, text_auto=".2f",
                        title="Avg GPA · Parent Income × Parent Education",
                        color_continuous_scale=theme["scale"], aspect="auto")
        fig.update_traces(textfont=dict(size=13, color="#0F172A"))
        fig.update_layout(xaxis_title="Parent education", yaxis_title="Parent income band")
        st.plotly_chart(style_fig(fig, theme, show_legend=False, height=420),
                        use_container_width=True)
        caption("The fully-joint view: does high income only help if parents are also "
                "educated? The cell with both high income AND high education is typically "
                "the darkest — confirming compounding effects.")

    with tab3:
        section("The equity chart: scholarship impact across income bands")

        sch_df = (fstu.groupby(["parent_income","scholarship_flag"])["current_gpa"]
                      .mean().reset_index())
        fig = px.bar(sch_df, x="parent_income", y="current_gpa",
                     color="scholarship_flag", barmode="group",
                     category_orders={"parent_income": available_bands},
                     title="Scholarship Effect · Grouped by Income Band",
                     color_discrete_map={"Yes":"#10B981","No":"#94A3B8"},
                     text_auto=".2f")
        fig.update_layout(xaxis_title="Parent income band", yaxis_title="Average GPA")
        fig.update_traces(textfont=dict(color="white", size=12))
        st.plotly_chart(style_fig(fig, theme, height=420), use_container_width=True)
        caption("Side-by-side bars for scholarship and non-scholarship students within each "
                "income band. The green bar being notably taller than grey in low-income "
                "bands confirms scholarships are working where most needed.")

        section("High-risk students per income band")
        high_by_inc = (fstu.assign(is_high=(fstu["academic_risk_flag"]=="High").astype(int))
                            .groupby("parent_income")["is_high"].mean()
                            .reindex(available_bands).reset_index())
        high_by_inc["is_high"] *= 100
        fig = px.bar(high_by_inc, x="parent_income", y="is_high", text_auto=".1f",
                     title="High-Risk Percentage by Income Band",
                     color="is_high", color_continuous_scale=[[0,"#10B981"],[0.5,"#F59E0B"],[1,"#EF4444"]])
        fig.update_traces(textfont=dict(color="white", size=14))
        fig.update_layout(xaxis_title="Parent income band",
                          yaxis_title="% of students flagged High Risk")
        st.plotly_chart(style_fig(fig, theme, show_legend=False, height=380),
                        use_container_width=True)
        caption("The clearest equity gauge — percentage of students flagged High Risk in "
                "each income bracket. The steeper the decline left-to-right, the larger "
                "the socio-economic achievement gap.")

        insight("This page exists to keep <b>equity visible</b>. Use it quarterly to check "
                "whether gaps between income bands are narrowing (good) or widening (bad), "
                "and to target scholarships and counseling where the need is greatest.")


# ═══════════════════════════════════════════════════════════════════
#  PAGE: STUDENT COMPARISON  (NEW — pink/rose theme)
# ═══════════════════════════════════════════════════════════════════
elif page == "Student Comparison":

    st.markdown("**Pick 2–5 students from the dropdown below to compare them across "
                "every metric.** The dropdown is searchable — type part of a name or ID.")

    # Build display label
    stu_label_df = fstu.copy()
    stu_label_df["label"] = (stu_label_df["full_name"] + " · "
                             + stu_label_df["student_id"] + " · "
                             + stu_label_df["school"] + " · G"
                             + stu_label_df["grade_level"].astype(str))

    options = stu_label_df["label"].tolist()
    default_picks = options[:3] if len(options) >= 3 else options

    picks = st.multiselect("Select students to compare (up to 5)", options,
                           default=default_picks, max_selections=5)

    if len(picks) < 2:
        st.info("👆 Pick at least 2 students to see the comparison charts.")
        st.stop()

    picked_df = stu_label_df[stu_label_df["label"].isin(picks)].copy()
    picked_ids = picked_df["student_id"].tolist()
    picked_names = picked_df["full_name"].tolist()

    # Snapshot cards
    section("Snapshot")
    cols = st.columns(len(picked_df))
    for col, (_, s) in zip(cols, picked_df.iterrows()):
        col.markdown(f"**{s['full_name']}**")
        col.metric("GPA", round(s["current_gpa"], 2))
        col.metric("Attendance %", round(s["cumulative_attendance_pct"], 1))
        col.metric("Risk", s["academic_risk_flag"])
        col.caption(f"{s['school']} · Grade {s['grade_level']} · {s['gender']}")

    section("Head-to-head comparison")

    # Chart 1: Grouped bar — GPA + Attendance side by side
    comp_df = picked_df[["full_name","current_gpa","cumulative_attendance_pct",
                          "performance_index"]].copy()
    melted = comp_df.melt(id_vars="full_name",
                          value_vars=["current_gpa","cumulative_attendance_pct","performance_index"],
                          var_name="metric", value_name="value")
    rename_metric = {"current_gpa":"GPA (0-4)",
                     "cumulative_attendance_pct":"Attendance %",
                     "performance_index":"Performance Index"}
    melted["metric"] = melted["metric"].map(rename_metric)
    fig = px.bar(melted, x="full_name", y="value", color="metric", barmode="group",
                 text_auto=".2f", title="Core Metrics Head-to-Head",
                 color_discrete_sequence=theme["palette"])
    fig.update_layout(xaxis_title="Student", yaxis_title="Value (note: different scales)")
    fig.update_traces(textfont=dict(color="white", size=12))
    st.plotly_chart(style_fig(fig, theme, height=420), use_container_width=True)
    caption("Three metrics per student on one chart. GPA is on the 0-4 scale, attendance "
            "and performance index are percentages — different magnitudes shown together "
            "so the relative strength of each student on each axis is instantly visible.")

    # Chart 2: Radar — normalized profile (0-1)
    section("Profile radar")
    metric_cols = ["current_gpa","cumulative_attendance_pct","performance_index",
                   "grade_level"]
    # Normalise against the full student population so each student's position is meaningful
    norm_ref = stu[metric_cols]
    normed = picked_df[metric_cols].copy()
    for c in metric_cols:
        lo, hi = norm_ref[c].min(), norm_ref[c].max()
        normed[c] = (picked_df[c] - lo)/(hi - lo + 1e-9)
    axis_labels = ["GPA","Attendance","Perf. Index","Grade"]
    fig = go.Figure()
    for i, (_, s) in enumerate(picked_df.iterrows()):
        fig.add_trace(go.Scatterpolar(
            r=normed.iloc[i].tolist(), theta=axis_labels,
            fill="toself", name=s["full_name"],
            line_color=theme["palette"][i % len(theme["palette"])]))
    fig.update_layout(
        title="Normalised Student Profiles (0-1 scale vs full cohort)",
        polar=dict(bgcolor="rgba(248,250,252,0.5)",
                   radialaxis=dict(visible=True, range=[0, 1],
                                   tickfont=dict(color="#334155", size=11)),
                   angularaxis=dict(tickfont=dict(color="#0F172A", size=13))),
        paper_bgcolor="rgba(0,0,0,0)", height=440, title_x=0.02,
        font_family='"Inter", sans-serif', font_color="#0F172A",
        legend=dict(orientation="h", yanchor="bottom", y=-0.15, font=dict(color="#0F172A")))
    st.plotly_chart(fig, use_container_width=True)
    caption("Radar comparing normalised scores: 1.0 means top of the full cohort on that "
            "axis, 0.0 means bottom. The larger a student's polygon, the stronger their "
            "all-round profile.")

    # Chart 3: Subject-level comparison
    section("Subject-level breakdown")
    sub_rec = rec[rec["student_id"].isin(picked_ids)].copy()
    if not sub_rec.empty:
        sub_rec = sub_rec.merge(picked_df[["student_id","full_name"]],
                                 on="student_id", how="left")
        subj_pivot = (sub_rec.groupby(["full_name","subject_name"])["percentage"]
                              .mean().reset_index())
        fig = px.bar(subj_pivot, x="subject_name", y="percentage",
                     color="full_name", barmode="group", text_auto=".0f",
                     title="Exam % per Subject (per student)",
                     color_discrete_sequence=theme["palette"])
        fig.update_layout(xaxis_title="Subject", yaxis_title="Exam percentage")
        fig.update_traces(textfont=dict(color="white", size=11))
        st.plotly_chart(style_fig(fig, theme, height=420), use_container_width=True)
        caption("Grouped bars showing each student's exam % in each subject. Reveals "
                "individual strengths: one student might excel at Math but struggle with "
                "English, another the reverse.")

    # Chart 4: Fees comparison
    section("Financial snapshot")
    col1, col2 = st.columns(2)
    fee_df = picked_df[["full_name","tuition_fee","fee_paid","fee_outstanding","payment_status"]].copy()

    fig = px.bar(fee_df.melt(id_vars="full_name",
                              value_vars=["fee_paid","fee_outstanding"],
                              var_name="bucket", value_name="amount"),
                 x="full_name", y="amount", color="bucket",
                 title="Tuition Paid vs Outstanding (₹)",
                 color_discrete_map={"fee_paid":"#10B981","fee_outstanding":"#EF4444"},
                 barmode="stack", text_auto=".2s")
    fig.update_traces(textfont=dict(color="white", size=11))
    fig.update_layout(xaxis_title="Student", yaxis_title="Amount (₹)")
    col1.plotly_chart(style_fig(fig, theme, height=380), use_container_width=True)
    with col1: caption("Stacked bars of tuition paid (green) vs outstanding (red). Students "
                        "with a large red segment are payment-risk cases worth flagging.")

    # Payment status table
    with col2:
        col2.markdown("**Payment status & amounts**")
        display_fees = fee_df.copy()
        display_fees["Tuition"]     = display_fees["tuition_fee"].map(lambda v: f"₹{v:,.0f}")
        display_fees["Paid"]        = display_fees["fee_paid"].map(lambda v: f"₹{v:,.0f}")
        display_fees["Outstanding"] = display_fees["fee_outstanding"].map(lambda v: f"₹{v:,.0f}")
        col2.dataframe(display_fees[["full_name","Tuition","Paid","Outstanding","payment_status"]]
                       .rename(columns={"full_name":"Student","payment_status":"Status"}),
                       use_container_width=True, hide_index=True)
        caption("Exact numbers for each student: what they owe, what they've paid, "
                "and their status bucket.")


# ═══════════════════════════════════════════════════════════════════
#  PAGE: SCHOOL BENCHMARKING  (NEW — cyan/teal theme)
# ═══════════════════════════════════════════════════════════════════
elif page == "School Benchmarking":

    # Build school-level summary WITH financials
    finance = (fstu.groupby("school").agg(
        students      = ("student_id","count"),
        tuition_total = ("tuition_fee","sum"),
        paid_total    = ("fee_paid","sum"),
        outstanding   = ("fee_outstanding","sum"),
        avg_gpa       = ("current_gpa","mean"),
        avg_att       = ("cumulative_attendance_pct","mean"),
        high_risk_pct = ("academic_risk_flag", lambda s: (s=="High").mean()*100),
        defaulted_pct = ("payment_status", lambda s: (s=="Defaulted").mean()*100),
    ).reset_index())
    finance["collection_rate"]    = finance["paid_total"]/finance["tuition_total"]*100
    finance["outstanding_pct"]    = finance["outstanding"]/finance["tuition_total"]*100

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Schools",       len(finance))
    c2.metric("Total billed",  f"₹{finance['tuition_total'].sum()/1e6:.1f}M")
    c3.metric("Total outstanding",  f"₹{finance['outstanding'].sum()/1e6:.1f}M")
    c4.metric("Collection rate", f"{finance['paid_total'].sum()/finance['tuition_total'].sum()*100:.1f}%")

    tab1, tab2, tab3 = st.tabs(["📚 Academic Comparison", "💰 Financial Comparison", "🔗 Joint View"])

    # ────────────── Academic comparison ──────────────────────
    with tab1:
        section("Head-to-head academic metrics")

        # Let user pick subset to compare, default all
        all_schools = sorted(finance["school"].tolist())
        picks = st.multiselect("Schools to compare (default: all)",
                               all_schools, default=all_schools[:8])
        if not picks:
            picks = all_schools
        sub_fin = finance[finance["school"].isin(picks)]

        col1, col2 = st.columns(2)
        fig = px.bar(sub_fin.sort_values("avg_gpa"), x="avg_gpa", y="school",
                     orientation="h", text_auto=".2f",
                     title="Average GPA Comparison",
                     color="avg_gpa", color_continuous_scale=theme["scale"])
        fig.update_traces(textfont=dict(color="white", size=12))
        fig.update_layout(xaxis_title="Average GPA", yaxis_title="School")
        col1.plotly_chart(style_fig(fig, theme, show_legend=False, height=420),
                          use_container_width=True)
        with col1: caption("Horizontal bars of average GPA — the simplest academic ranking. "
                           "Filter schools from the list above to focus on a peer group.")

        fig = px.bar(sub_fin.sort_values("avg_att"), x="avg_att", y="school",
                     orientation="h", text_auto=".1f",
                     title="Average Attendance % Comparison",
                     color="avg_att", color_continuous_scale=theme["scale"])
        fig.update_traces(textfont=dict(color="white", size=12))
        fig.update_layout(xaxis_title="Average attendance %", yaxis_title="School")
        col2.plotly_chart(style_fig(fig, theme, show_legend=False, height=420),
                          use_container_width=True)
        with col2: caption("Same view but for attendance. Pair this with the GPA chart to "
                           "see if schools are low on both axes or just one.")

        fig = px.bar(sub_fin.sort_values("high_risk_pct", ascending=False),
                     x="school", y="high_risk_pct", text_auto=".1f",
                     title="High-Risk Student Percentage",
                     color="high_risk_pct",
                     color_continuous_scale=[[0,"#10B981"],[0.5,"#F59E0B"],[1,"#EF4444"]])
        fig.update_traces(textfont=dict(color="white", size=12))
        fig.update_layout(xaxis_title="School", yaxis_title="% students flagged High Risk")
        st.plotly_chart(style_fig(fig, theme, show_legend=False, height=380),
                        use_container_width=True)
        caption("Share of each school's students flagged High Risk. Schools at the top are "
                "in the worst academic shape — priority intervention targets.")

        # Normalised multi-metric radar
        section("Normalised multi-metric comparison")
        radar_metrics = ["avg_gpa","avg_att","high_risk_pct"]
        radar_labels  = ["GPA","Attendance","Low Risk"]
        rd = sub_fin[["school"] + radar_metrics].copy()
        # For risk we want lower = better, so invert
        rd["high_risk_pct"] = 100 - rd["high_risk_pct"]
        # Normalise each column 0-1
        for c in radar_metrics:
            lo, hi = rd[c].min(), rd[c].max()
            rd[c] = (rd[c] - lo)/(hi - lo + 1e-9)

        fig = go.Figure()
        for i, (_, row) in enumerate(rd.iterrows()):
            fig.add_trace(go.Scatterpolar(
                r=[row[m] for m in radar_metrics], theta=radar_labels,
                fill="toself", name=row["school"],
                line_color=theme["palette"][i % len(theme["palette"])]))
        fig.update_layout(
            title="School Profiles (higher = better on all axes)",
            polar=dict(bgcolor="rgba(248,250,252,0.5)",
                      radialaxis=dict(visible=True, range=[0, 1],
                                      tickfont=dict(color="#334155", size=11)),
                      angularaxis=dict(tickfont=dict(color="#0F172A", size=13))),
            paper_bgcolor="rgba(0,0,0,0)", height=500, title_x=0.02,
            font_family='"Inter", sans-serif', font_color="#0F172A",
            legend=dict(orientation="h", yanchor="bottom", y=-0.2,
                        font=dict(color="#0F172A", size=10)))
        st.plotly_chart(fig, use_container_width=True)
        caption("Radar with risk inverted so 'bigger polygon = better school' on every axis. "
                "Schools with lopsided polygons have specific weaknesses — easy to spot at a glance.")

    # ────────────── Financial comparison ──────────────────────
    with tab2:
        section("Outstanding fees across schools")

        fig = px.bar(finance.sort_values("outstanding_pct", ascending=False),
                     x="school", y="outstanding_pct", text_auto=".1f",
                     title="Percentage of Tuition Outstanding · by School",
                     color="outstanding_pct",
                     color_continuous_scale=[[0,"#10B981"],[0.5,"#F59E0B"],[1,"#EF4444"]])
        fig.update_traces(textfont=dict(color="white", size=12))
        fig.update_layout(xaxis_title="School", yaxis_title="% of total tuition outstanding")
        st.plotly_chart(style_fig(fig, theme, show_legend=False, height=420),
                        use_container_width=True)
        caption("The headline financial chart: what percentage of each school's total tuition "
                "bill is still unpaid. Red bars are schools with severe collection problems.")

        col1, col2 = st.columns(2)
        fig = px.bar(finance.sort_values("collection_rate"),
                     x="collection_rate", y="school", orientation="h",
                     text_auto=".1f", title="Collection Rate (%) by School",
                     color="collection_rate", color_continuous_scale=theme["scale"])
        fig.update_traces(textfont=dict(color="white", size=12))
        fig.update_layout(xaxis_title="Collection rate %", yaxis_title="School")
        col1.plotly_chart(style_fig(fig, theme, show_legend=False, height=420),
                          use_container_width=True)
        with col1: caption("Same data, different angle: collection rate = paid ÷ billed. "
                           "High rates mean healthy cash flow; low rates mean aggressive "
                           "follow-up is needed.")

        fig = px.bar(finance.sort_values("defaulted_pct", ascending=False),
                     x="school", y="defaulted_pct", text_auto=".1f",
                     title="% Students in 'Defaulted' Payment Status",
                     color="defaulted_pct",
                     color_continuous_scale=[[0,"#10B981"],[0.5,"#F59E0B"],[1,"#EF4444"]])
        fig.update_traces(textfont=dict(color="white", size=12))
        fig.update_layout(xaxis_title="School", yaxis_title="% defaulted students")
        col2.plotly_chart(style_fig(fig, theme, show_legend=False, height=420),
                          use_container_width=True)
        with col2: caption("Proportion of each school's students classified as payment "
                           "defaulters (paid less than 40% of tuition). Highlights where "
                           "economic distress is concentrated.")

        section("Tuition billed vs collected")
        melted_fin = finance.melt(id_vars="school",
                                   value_vars=["paid_total","outstanding"],
                                   var_name="bucket", value_name="amount")
        melted_fin["bucket"] = melted_fin["bucket"].map(
            {"paid_total":"Collected","outstanding":"Outstanding"})
        fig = px.bar(melted_fin, x="school", y="amount", color="bucket",
                     barmode="stack", title="Tuition Stack · Collected vs Outstanding (₹)",
                     color_discrete_map={"Collected":"#10B981","Outstanding":"#EF4444"})
        fig.update_layout(xaxis_title="School", yaxis_title="Amount (₹)")
        st.plotly_chart(style_fig(fig, theme, height=440), use_container_width=True)
        caption("Each bar is a school's full tuition pie — green is money in the bank, red "
                "is money still owed. Total bar height shows total fee volume, so this "
                "chart reveals BOTH collection performance and relative school size.")

        # Payment status stacked
        section("Payment status mix per school")
        status_mix = (fstu.groupby(["school","payment_status"]).size().reset_index(name="count"))
        status_mix["pct"] = status_mix.groupby("school")["count"].transform(
            lambda x: x/x.sum()*100)
        fig = px.bar(status_mix, x="school", y="pct", color="payment_status",
                     barmode="stack", title="Payment Status Breakdown (%)",
                     category_orders={"payment_status":
                                      ["Paid in full","Minor dues","Partial","Defaulted"]},
                     color_discrete_map={"Paid in full":"#10B981","Minor dues":"#84CC16",
                                         "Partial":"#F59E0B","Defaulted":"#EF4444"})
        fig.update_layout(xaxis_title="School", yaxis_title="Percentage of students")
        st.plotly_chart(style_fig(fig, theme, height=440), use_container_width=True)
        caption("Each school's student body by payment status (100% = full school). Tells "
                "you whether a school's payment problems are concentrated in a few defaulters "
                "or spread across many partial-payers.")

    # ────────────── Joint view ──────────────────────
    with tab3:
        section("Does financial health correlate with academic outcomes?")

        fig = px.scatter(finance, x="outstanding_pct", y="avg_gpa", size="students",
                         hover_name="school", title="Outstanding % vs Average GPA",
                         color="high_risk_pct",
                         color_continuous_scale=[[0,"#10B981"],[0.5,"#F59E0B"],[1,"#EF4444"]],
                         labels={"high_risk_pct":"High-risk %"})
        fig = add_trendline(fig, finance["outstanding_pct"], finance["avg_gpa"],
                            color="#0F172A")
        fig.update_layout(xaxis_title="% tuition outstanding",
                          yaxis_title="Average GPA")
        st.plotly_chart(style_fig(fig, theme, height=500), use_container_width=True)
        caption("Each school plotted as a bubble: x-axis is outstanding %, y-axis is GPA, "
                "size is enrolment, colour is high-risk %. The trendline reveals the "
                "relationship — hover any bubble for full details. Use this to spot "
                "schools that are outliers in either direction.")

        section("Academic-financial composite ranking")
        # Composite score: higher GPA, higher collection rate, lower high_risk_pct
        rank = finance.copy()
        rank["academic_score"]  = (rank["avg_gpa"] - rank["avg_gpa"].min())/(rank["avg_gpa"].max() - rank["avg_gpa"].min() + 1e-9)
        rank["financial_score"] = (rank["collection_rate"] - rank["collection_rate"].min())/(rank["collection_rate"].max() - rank["collection_rate"].min() + 1e-9)
        rank["composite"]       = 0.6*rank["academic_score"] + 0.4*rank["financial_score"]
        rank = rank.sort_values("composite", ascending=False)

        fig = px.bar(rank, x="school", y="composite",
                     title="Composite Score (60% academic + 40% financial)",
                     color="composite", color_continuous_scale=theme["scale"],
                     text_auto=".2f",
                     hover_data=["avg_gpa","collection_rate","outstanding_pct"])
        fig.update_traces(textfont=dict(color="white", size=11))
        fig.update_layout(xaxis_title="School",
                          yaxis_title="Composite score (0-1 scale)")
        st.plotly_chart(style_fig(fig, theme, show_legend=False, height=420),
                        use_container_width=True)
        caption("A single 0-1 ranking that weights academic outcomes at 60% and financial "
                "health at 40%. Use this for board reviews and strategic resource allocation.")

        insight("If a school scores high academically but low financially (or vice versa), "
                "dig into the individual charts above to understand the imbalance. "
                "Truly troubled schools are low on <b>both</b> axes.")


# ═══════════════════════════════════════════════════════════════════
#  PAGE: PREDICTIVE LAB  (teal + indigo)
# ═══════════════════════════════════════════════════════════════════
elif page == "Predictive Lab":

    risk_pkg    = train_risk_model(stu)
    gpa_pkg     = train_gpa_model(stu)
    cluster_pkg = build_clusters(stu, k=4)
    anom_pkg    = detect_anomalies(stu)

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Risk model acc (LR)", f"{risk_pkg['lr_acc']*100:.1f}%")
    c2.metric("Risk model acc (RF)", f"{risk_pkg['rf_acc']*100:.1f}%")
    c3.metric("GPA model R²",        f"{gpa_pkg['r2']:.3f}")
    c4.metric("Anomalies flagged",   int((anom_pkg['flags']==-1).sum()))

    tab1, tab2, tab3, tab4 = st.tabs([
        "🎯 Risk Classifier", "🔮 What-If Simulator",
        "🧬 Clustering", "⚠️ Anomalies & Correlation"])

    with tab1:
        section("Model performance")
        col1, col2 = st.columns(2)

        acc_df = pd.DataFrame({
            "Model":    ["Logistic Regression", "Random Forest"],
            "Accuracy": [risk_pkg["lr_acc"]*100, risk_pkg["rf_acc"]*100]})
        fig = px.bar(acc_df, x="Model", y="Accuracy", text_auto=".1f",
                     title="🎯 [1/10] Classifier Accuracy Comparison",
                     color="Model", color_discrete_sequence=theme["palette"])
        fig.update_layout(yaxis_range=[0, 100],
                          xaxis_title="ML model", yaxis_title="Accuracy on training data (%)")
        fig.update_traces(textfont=dict(color="white", size=16))
        col1.plotly_chart(style_fig(fig, theme, show_legend=False, height=340),
                          use_container_width=True)
        with col1: caption("Two different algorithms trained on the same data. Random Forest "
                           "usually outperforms Logistic Regression because it captures "
                           "non-linear patterns.")

        X_all = make_risk_X(stu)
        y_true = stu["academic_risk_flag"]
        y_pred = risk_pkg["rf"].predict(X_all)
        classes = sorted(y_true.unique())
        cm = confusion_matrix(y_true, y_pred, labels=classes)
        fig = px.imshow(cm, text_auto=True, x=classes, y=classes,
                        labels=dict(x="Predicted risk class", y="Actual risk class", color="Count"),
                        title="🎯 [2/10] Confusion Matrix (Random Forest)",
                        color_continuous_scale=theme["scale"])
        fig.update_traces(textfont=dict(size=16, color="#0F172A"))
        col2.plotly_chart(style_fig(fig, theme, show_legend=False, height=340),
                          use_container_width=True)
        with col2: caption("Diagonal cells = correct predictions. Off-diagonal cells are errors: "
                           "a number at (Actual=High, Predicted=Low) means the model missed "
                           "a high-risk student — the most expensive error to make.")

        section("Explainability")
        col1, col2 = st.columns(2)

        imp = pd.DataFrame({
            "feature": risk_pkg["features"],
            "importance": risk_pkg["rf"].feature_importances_,
        }).sort_values("importance", ascending=True)
        fig = px.bar(imp, x="importance", y="feature", orientation="h",
                     title="🎯 [3/10] Feature Importance (Random Forest)",
                     text_auto=".3f", color="importance",
                     color_continuous_scale=theme["scale"])
        fig.update_traces(textfont=dict(color="#0F172A", size=13))
        fig.update_layout(xaxis_title="Importance (0-1)", yaxis_title="Feature")
        col1.plotly_chart(style_fig(fig, theme, show_legend=False, height=360),
                          use_container_width=True)
        with col1: caption("Which features does the model find most useful? The top bar drives "
                           "the most predictive power. If Attendance dominates, it's the "
                           "leading indicator to monitor in production.")

        X_f = make_risk_X(fstu)
        proba = risk_pkg["rf"].predict_proba(X_f)
        high_idx = list(risk_pkg["rf"].classes_).index("High")
        p_high = proba[:, high_idx]
        fig = px.histogram(pd.DataFrame({"p_high": p_high}), x="p_high", nbins=25,
                           title="🎯 [4/10] P(High Risk) Distribution",
                           color_discrete_sequence=[theme["primary"]])
        fig.update_traces(marker_line_width=1, marker_line_color="white")
        fig.add_vline(x=0.5, line_dash="dash", line_color="#DC2626", line_width=3,
                      annotation_text="Decision threshold (0.5)",
                      annotation_font_color="#0F172A")
        fig.update_layout(xaxis_title="Predicted probability of High-risk classification",
                          yaxis_title="Number of students")
        col2.plotly_chart(style_fig(fig, theme, show_legend=False, height=360, bargap=0.1),
                          use_container_width=True)
        with col2: caption("How confident the model is about each student. Bars near 0 or 1 "
                           "mean confident predictions; bars near 0.5 are borderline cases "
                           "worth manual review.")

    with tab2:
        st.markdown("Adjust the sliders to see real-time predictions for a hypothetical "
                    "student and how each feature contributes to the outcome.")

        col_in, col_out = st.columns([1, 2])
        with col_in:
            att_input   = st.slider("Attendance %",  50.0, 100.0, 85.0, 0.5)
            grade_input = st.select_slider("Grade level",
                options=sorted(stu["grade_level"].dropna().unique()), value=6)
            schol_input = st.checkbox("Has scholarship", value=False)
            iep_input   = st.checkbox("Has IEP",         value=False)

        x_gpa = pd.DataFrame(
            [[att_input, grade_input, int(schol_input), int(iep_input)]],
            columns=GPA_FEATURES)
        pred_gpa = float(gpa_pkg["model"].predict(x_gpa)[0])
        pred_gpa = max(0.0, min(4.0, pred_gpa))

        x_risk = pd.DataFrame(
            [[pred_gpa, att_input, grade_input, int(schol_input), int(iep_input)]],
            columns=RISK_FEATURES)
        pred_risk  = risk_pkg["rf"].predict(x_risk)[0]
        pred_proba = risk_pkg["rf"].predict_proba(x_risk)[0]
        cohort_gpa = float(stu["current_gpa"].mean())

        with col_out:
            cc1, cc2, cc3 = st.columns(3)
            cc1.metric("Predicted GPA",  round(pred_gpa, 2),
                       delta=round(pred_gpa - cohort_gpa, 2))
            cc2.metric("Predicted risk", pred_risk)
            cc3.metric("Confidence",     f"{pred_proba.max()*100:.1f}%")

            prob_df = pd.DataFrame({
                "class": risk_pkg["rf"].classes_, "probability": pred_proba})
            fig = px.bar(prob_df, x="class", y="probability", text_auto=".0%",
                         title="🔮 [5/10] Risk-Class Probabilities",
                         color="class",
                         color_discrete_map={"Low":"#10B981","Medium":"#F59E0B","High":"#EF4444"})
            fig.update_yaxes(tickformat=".0%", range=[0, 1])
            fig.update_traces(textfont=dict(color="white", size=16))
            fig.update_layout(xaxis_title="Risk class", yaxis_title="Predicted probability")
            st.plotly_chart(style_fig(fig, theme, show_legend=False, height=300),
                            use_container_width=True)
            caption("How confident is the model in each risk bucket? The tallest bar is the "
                    "predicted class; the second-tallest shows where the model is secondarily "
                    "considering.")

        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta", value=pred_gpa,
            delta={"reference": cohort_gpa, "valueformat": ".2f",
                   "font": {"color": "#0F172A"}},
            number={"font": {"color": "#0F172A", "size": 48}},
            title={"text": "🔮 [6/10] Predicted GPA vs Cohort Mean",
                   "font": {"color": "#0F172A", "size": 16}},
            gauge={
                "axis": {"range": [0, 4], "tickfont": {"color": "#0F172A", "size": 12}},
                "bar": {"color": theme["primary"]},
                "bgcolor": "white", "borderwidth": 2, "bordercolor": "#CBD5E1",
                "steps": [
                    {"range": [0, 2],   "color": "#FEE2E2"},
                    {"range": [2, 3],   "color": "#FEF3C7"},
                    {"range": [3, 4],   "color": "#DCFCE7"}],
                "threshold": {"line": {"color": "#0F172A", "width": 4},
                              "thickness": 0.85, "value": cohort_gpa}}))
        fig.update_layout(height=320, margin=dict(l=20, r=20, t=60, b=20),
                          paper_bgcolor="rgba(0,0,0,0)",
                          font_family='"Inter", sans-serif')
        st.plotly_chart(fig, use_container_width=True)
        caption("Gauge showing predicted GPA on a 0-4 scale. Red zone = at-risk GPA, "
                "amber = borderline, green = healthy. The triangle marker is the cohort "
                "average so you can see where this student sits relative to peers.")

        section("🔍 SHAP-style contribution breakdown")
        st.markdown("How each feature pushes the GPA prediction **up** or **down** "
                    "from the baseline.")

        inputs = {"Attendance %": att_input, "Grade Level": grade_input,
                  "Scholarship": int(schol_input), "IEP": int(iep_input)}
        means = gpa_pkg["means"]
        coefs = gpa_pkg["coefs"]

        contrib_rows = []
        for f in GPA_FEATURES:
            c = coefs[f] * (inputs[f] - means[f])
            contrib_rows.append({"feature": f, "contribution": c})
        contrib_df = pd.DataFrame(contrib_rows).sort_values("contribution")

        fig = go.Figure(go.Waterfall(
            orientation="h",
            y=contrib_df["feature"].tolist() + ["Total Δ from mean"],
            x=contrib_df["contribution"].tolist() + [contrib_df["contribution"].sum()],
            measure=["relative"]*len(contrib_df) + ["total"],
            connector={"line":{"color":"#CBD5E1"}},
            decreasing={"marker":{"color":"#EF4444"}},
            increasing={"marker":{"color":"#10B981"}},
            totals={"marker":{"color": theme["primary"]}},
            text=[f"{v:+.3f}" for v in contrib_df["contribution"].tolist()] +
                 [f"{contrib_df['contribution'].sum():+.3f}"],
            textposition="outside",
            textfont=dict(size=13, color="#0F172A")))
        fig.update_layout(
            title="🔮 [7/10] SHAP-Style Feature Contributions to Predicted GPA",
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            height=340, font_family='"Inter", sans-serif', title_x=0.02,
            font_color="#0F172A",
            xaxis=dict(title="GPA points pushed above / below baseline",
                       gridcolor="#E2E8F0", linecolor="#94A3B8",
                       tickfont=dict(color="#0F172A", size=12)),
            yaxis=dict(title="Feature",
                       gridcolor="#E2E8F0", linecolor="#94A3B8",
                       tickfont=dict(color="#0F172A", size=13)))
        st.plotly_chart(fig, use_container_width=True)
        caption("Waterfall explaining the prediction: green bars pushed GPA up, red bars "
                "pushed it down, and the total is the final difference from the cohort "
                "average. Attendance is usually the biggest lever.")

    with tab3:
        st.markdown("K-means clustering (k=4) groups students by GPA, attendance, grade "
                    "and scholarship status. Each cluster is a natural cohort.")

        labels = cluster_pkg["labels"]
        centers = cluster_pkg["centers"]
        cluster_stu = stu.assign(cluster=[f"Cluster {l}" for l in labels])

        sample = cluster_stu.sample(min(3000, len(cluster_stu)), random_state=1)
        fig = px.scatter(sample, x="cumulative_attendance_pct", y="current_gpa",
                         color="cluster", opacity=0.7,
                         title="🧬 [8/10] Student Clusters · GPA × Attendance",
                         hover_data=["school","grade_level","academic_risk_flag"],
                         color_discrete_sequence=theme["palette"])
        for i, c in enumerate(centers):
            fig.add_trace(go.Scatter(
                x=[c[1]], y=[c[0]], mode="markers+text",
                marker=dict(size=28, color=theme["palette"][i % len(theme["palette"])],
                            symbol="star", line=dict(color="#0F172A", width=2)),
                text=[f"C{i}"], textposition="top center",
                textfont=dict(size=13, color="#0F172A"),
                name=f"Centre {i}", showlegend=False))
        fig.update_layout(xaxis_title="Attendance %", yaxis_title="GPA")
        st.plotly_chart(style_fig(fig, theme, height=480), use_container_width=True)
        caption("Unsupervised learning finds 4 natural student segments. Star markers are "
                "each cluster's centre. Students near a star are most representative of "
                "that cluster's archetype.")

        prof = stu.assign(cluster=labels).groupby("cluster").agg(
            gpa=("current_gpa","mean"),
            attendance=("cumulative_attendance_pct","mean"),
            grade=("grade_level","mean"),
            risk=("academic_risk_flag", lambda s: (s=="High").mean()*100)).reset_index()
        prof_norm = prof.copy()
        for c in ["gpa","attendance","grade","risk"]:
            col = prof_norm[c]
            prof_norm[c] = (col - col.min())/(col.max() - col.min() + 1e-9)

        fig = go.Figure()
        axes = ["gpa","attendance","grade","risk"]
        axis_labels = ["GPA","Attendance %","Grade Level","High-Risk %"]
        for i, row in prof_norm.iterrows():
            fig.add_trace(go.Scatterpolar(
                r=[row[a] for a in axes], theta=axis_labels,
                fill='toself', name=f"Cluster {int(row['cluster'])}",
                line_color=theme["palette"][int(row["cluster"]) % len(theme["palette"])]))
        fig.update_layout(
            title="🧬 [9/10] Cluster Profiles (normalised 0–1)",
            polar=dict(bgcolor="rgba(248,250,252,0.5)",
                      radialaxis=dict(visible=True, range=[0, 1],
                                      tickfont=dict(color="#334155", size=11)),
                      angularaxis=dict(tickfont=dict(color="#0F172A", size=13))),
            paper_bgcolor="rgba(0,0,0,0)", height=420, title_x=0.02,
            font_family='"Inter", sans-serif', font_color="#0F172A",
            legend=dict(orientation="h", yanchor="bottom", y=-0.15,
                        font=dict(color="#0F172A")))
        st.plotly_chart(fig, use_container_width=True)
        caption("Radar comparing the four clusters on all dimensions. A cluster with low "
                "GPA/attendance and high risk is your at-risk segment — the target cohort "
                "for intensive intervention.")

        st.markdown("**Cluster summary:**")
        st.dataframe(prof.round(2), use_container_width=True, hide_index=True)

    with tab4:
        section("Feature correlation matrix")
        num_df = pd.DataFrame({
            "GPA":         stu["current_gpa"],
            "Attendance":  stu["cumulative_attendance_pct"],
            "Grade":       stu["grade_level"],
            "Performance": stu["performance_index"],
            "Age":         stu["age"],
            "Scholarship": (stu["scholarship_flag"]=="Yes").astype(int),
            "IEP":         (stu["iep_flag"]=="Yes").astype(int),
            "High Risk":   (stu["academic_risk_flag"]=="High").astype(int)})
        corr = num_df.corr()
        fig = px.imshow(corr, text_auto=".2f",
                        title="⚠️ [10/10] Feature Correlation Matrix",
                        color_continuous_scale=[[0,"#EF4444"],[0.5,"#FFFFFF"],[1,"#10B981"]],
                        zmin=-1, zmax=1, aspect="auto")
        fig.update_traces(textfont=dict(size=13, color="#0F172A"))
        fig.update_layout(xaxis_title="Feature", yaxis_title="Feature")
        st.plotly_chart(style_fig(fig, theme, show_legend=False, height=500),
                        use_container_width=True)
        caption("All feature pairs correlated simultaneously. Green = positive correlation, "
                "red = negative, white = unrelated. Strong green off-diagonal cells reveal "
                "the strongest drivers of student outcomes.")

        section("🚨 Isolation-forest anomalies")
        sample = stu.assign(
            anomaly_label=["Anomaly" if f==-1 else "Normal" for f in anom_pkg["flags"]],
            anom_score=anom_pkg["scores"]).sample(min(3000, len(stu)), random_state=1)
        fig = px.scatter(sample, x="cumulative_attendance_pct", y="current_gpa",
                         color="anomaly_label", opacity=0.7,
                         title="Isolation-Forest · Students with Unusual Profiles",
                         color_discrete_map={"Normal":"#94A3B8", "Anomaly":"#EF4444"},
                         hover_data=["school","grade_level","academic_risk_flag"])
        fig.update_layout(xaxis_title="Attendance %", yaxis_title="GPA")
        st.plotly_chart(style_fig(fig, theme, height=440), use_container_width=True)
        caption("Red dots are students whose attendance-GPA-grade combination doesn't match "
                "most peers — outliers worth a human review. Could be exceptional talents "
                "being missed or struggling students slipping under the radar.")

        anom_count = int((anom_pkg['flags']==-1).sum())
        insight(f"The isolation-forest flagged <b>{anom_count} students</b> (≈5%) "
                f"whose GPA / attendance / grade combination is statistically unusual "
                f"compared with peers. Good candidates for early counsellor review.")


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
    "<div style='margin-top:1rem; font-size:.7rem; color:#94A3B8; text-align:center;'>"
    "v3.0 · 10 pages · 90+ charts · Streamlit · Plotly · scikit-learn"
    "</div>", unsafe_allow_html=True)
