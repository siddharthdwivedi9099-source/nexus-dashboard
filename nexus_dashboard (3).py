"""
Nexus Analytics – Multi-School Academic Dashboard
Interactive + predictive. Per-page colour themes. Uses all workbook sheets.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score

st.set_page_config(
    page_title="Nexus Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",   # <-- fixes the hidden-sidebar issue
)


# ============================================================
# THEMES
# ============================================================
THEMES = {
    "Overview": {
        "icon": "📊", "subtitle": "A snapshot across schools, students and performance",
        "primary": "#4F46E5", "grad": ("#6366F1", "#8B5CF6"),
        "palette": ["#4F46E5", "#7C3AED", "#EC4899", "#F59E0B", "#10B981", "#06B6D4"],
    },
    "Students": {
        "icon": "🎓", "subtitle": "Academic standing, attendance and risk profile",
        "primary": "#059669", "grad": ("#10B981", "#0D9488"),
        "palette": ["#059669", "#0D9488", "#10B981", "#34D399", "#6EE7B7", "#065F46"],
    },
    "Academic Records": {
        "icon": "📚", "subtitle": "Exam outcomes, subjects and grade distribution",
        "primary": "#D97706", "grad": ("#F59E0B", "#EA580C"),
        "palette": ["#D97706", "#EA580C", "#F59E0B", "#FB923C", "#FBBF24", "#92400E"],
    },
    "Teachers": {
        "icon": "👩‍🏫", "subtitle": "Faculty distribution, ratings and experience",
        "primary": "#BE123C", "grad": ("#E11D48", "#DB2777"),
        "palette": ["#BE123C", "#DB2777", "#E11D48", "#F43F5E", "#FB7185", "#881337"],
    },
    "Attendance": {
        "icon": "📅", "subtitle": "Presence patterns by school and grade",
        "primary": "#0284C7", "grad": ("#0EA5E9", "#0891B2"),
        "palette": ["#0284C7", "#0891B2", "#0EA5E9", "#06B6D4", "#38BDF8", "#075985"],
    },
    "Schools": {
        "icon": "🏫", "subtitle": "Institutions, boards and geography",
        "primary": "#7C3AED", "grad": ("#8B5CF6", "#A855F7"),
        "palette": ["#7C3AED", "#A855F7", "#8B5CF6", "#A78BFA", "#C4B5FD", "#4C1D95"],
    },
}


# ============================================================
# GLOBAL CSS  (header no longer hidden — fixes sidebar toggle)
# ============================================================
BASE_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] { font-family:'Inter', system-ui, sans-serif; }

.stApp { background: linear-gradient(180deg, #F8FAFC 0%, #EEF2F7 100%); }

.main .block-container {
    padding-top: 1.5rem;  padding-bottom: 4rem;
    padding-left: 2.5rem; padding-right: 2.5rem;
    max-width: 1400px;
}

/* Metric cards */
[data-testid="stMetric"] {
    background: #ffffff;
    padding: 1.1rem 1.3rem;
    border-radius: 14px;
    box-shadow: 0 1px 3px rgba(15,23,42,0.06), 0 1px 2px rgba(15,23,42,0.04);
    border-left: 4px solid var(--accent, #4F46E5);
    transition: transform .15s ease, box-shadow .15s ease;
}
[data-testid="stMetric"]:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 12px rgba(15,23,42,0.08);
}
[data-testid="stMetricLabel"] p {
    font-size: .72rem !important; color: #64748B !important;
    text-transform: uppercase; letter-spacing: .06em; font-weight: 600 !important;
}
[data-testid="stMetricValue"] {
    font-size: 1.7rem !important; font-weight: 700 !important; color: #0F172A !important;
}

/* Chart cards */
[data-testid="stPlotlyChart"] {
    background: #ffffff; border-radius: 14px;
    padding: .75rem;
    box-shadow: 0 1px 3px rgba(15,23,42,0.05);
    margin-bottom: 1rem;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    gap: .5rem; background: transparent; border-bottom: 1px solid #E2E8F0;
    margin-bottom: 1rem;
}
.stTabs [data-baseweb="tab"] {
    background: transparent; padding: .6rem 1rem; font-weight: 600;
    color: #64748B; border-radius: 8px 8px 0 0;
}
.stTabs [aria-selected="true"] {
    background: white !important; color: var(--accent, #4F46E5) !important;
    border-bottom: 2px solid var(--accent, #4F46E5) !important;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0F172A 0%, #1E293B 100%);
}
[data-testid="stSidebar"] * { color: #E2E8F0 !important; }
[data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
    color: #F8FAFC !important;
}
[data-testid="stSidebar"] label {
    font-size: .78rem !important; font-weight: 600 !important;
    color: #94A3B8 !important; text-transform: uppercase; letter-spacing: .05em;
}
[data-testid="stSidebar"] [data-baseweb="select"] > div,
[data-testid="stSidebar"] [data-baseweb="input"] > div {
    background: rgba(255,255,255,0.06) !important;
    border-color: rgba(255,255,255,0.12) !important;
}

/* Keep the sidebar toggle (header) visible; only hide the Streamlit footer */
footer { visibility: hidden; }

/* Column gutters */
[data-testid="column"] { padding: 0 .5rem !important; }

/* Section label */
.section-label {
    font-size: .8rem; font-weight: 600; color: #64748B;
    text-transform: uppercase; letter-spacing: .08em;
    margin: 1.2rem 0 .6rem 0;
}
.insight-box {
    background: linear-gradient(135deg, #F1F5F9 0%, #E2E8F0 100%);
    border-left: 3px solid var(--accent, #4F46E5);
    padding: .85rem 1.1rem;
    border-radius: 10px;
    font-size: .9rem; color: #334155;
    margin: .5rem 0 1rem 0;
}
</style>
"""
st.markdown(BASE_CSS, unsafe_allow_html=True)


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
        padding: 1.8rem 2.2rem; border-radius: 18px;
        margin-bottom: 1.4rem; color: white;
        box-shadow: 0 10px 30px -10px {t['primary']}55;">
        <div style="font-size:.78rem; opacity:.85; letter-spacing:.14em;
                    text-transform:uppercase; margin-bottom:.35rem;">Nexus Analytics</div>
        <div style="font-size:1.9rem; font-weight:700; line-height:1.1;">
            {t['icon']}  {page}</div>
        <div style="font-size:1rem; opacity:.92; margin-top:.35rem;">{t['subtitle']}</div>
    </div>""", unsafe_allow_html=True)

def section(label: str):
    st.markdown(f"<div class='section-label'>{label}</div>", unsafe_allow_html=True)

def insight(text: str):
    st.markdown(f"<div class='insight-box'>💡 {text}</div>", unsafe_allow_html=True)

def style_fig(fig, theme, *, show_legend=True, bargap=0.15):
    fig.update_layout(
        font_family='"Inter", system-ui, sans-serif',
        font_color="#1E293B", title_font_color="#0F172A",
        title_font_size=14, title_x=0.02,
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=30, r=20, t=50, b=30),
        colorway=theme["palette"],
        showlegend=show_legend, bargap=bargap,
        legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="rgba(0,0,0,0)",
                    orientation="h", yanchor="bottom", y=1.02,
                    xanchor="right", x=1),
        hoverlabel=dict(bgcolor="white", font_size=12,
                        font_family='"Inter", system-ui, sans-serif',
                        bordercolor=theme["primary"]),
    )
    fig.update_xaxes(gridcolor="#F1F5F9", zerolinecolor="#E2E8F0", linecolor="#E2E8F0")
    fig.update_yaxes(gridcolor="#F1F5F9", zerolinecolor="#E2E8F0", linecolor="#E2E8F0")
    return fig

def add_trendline(fig, x, y, color, name="Trend"):
    """Add a least-squares trend line without needing statsmodels."""
    xa, ya = np.asarray(x, float), np.asarray(y, float)
    m = ~(np.isnan(xa) | np.isnan(ya))
    if m.sum() < 2: return fig
    a, b = np.polyfit(xa[m], ya[m], 1)
    xl = np.linspace(xa[m].min(), xa[m].max(), 60)
    fig.add_trace(go.Scatter(x=xl, y=a*xl + b, mode="lines",
                             name=f"{name} (slope {a:.3f})",
                             line=dict(color=color, dash="dash", width=2)))
    return fig


# ============================================================
# DATA LAYER
# ============================================================
@st.cache_data(show_spinner="Loading workbook…")
def load_workbook(src) -> dict:
    sheets = pd.read_excel(src, sheet_name=None)
    for n, f in sheets.items():
        f.columns = f.columns.astype(str).str.strip().str.lower().str.replace(" ", "_")
        sheets[n] = f
    return sheets

@st.cache_data(show_spinner="Building views…")
def build_views(sheets: dict) -> dict:
    schools  = sheets["Schools"].rename(columns={"school_name": "school"})
    students = sheets["Students"].copy()
    records  = sheets["Student_Academic_Records"].copy()
    attend   = sheets["Attendance_Log"].copy()
    teachers = sheets["Teachers"].copy()

    stu = students.merge(
        schools[["school_id","school","board_name","region","state","city","school_type"]],
        on="school_id", how="left")
    stu["performance_index"] = (stu["current_gpa"].fillna(0)*25
                                + stu["cumulative_attendance_pct"].fillna(0)*0.5)

    rec = (records.merge(students[["student_id","gender"]], on="student_id", how="left")
                  .merge(schools[["school_id","school","board_name","region"]],
                         on="school_id", how="left"))

    att = attend.merge(schools[["school_id","school","region"]], on="school_id", how="left")
    att["attendance_date"] = pd.to_datetime(att["attendance_date"], errors="coerce")

    return {"students": stu, "records": rec, "attendance": att,
            "teachers": teachers, "schools": schools}

# ---- ML models (cached so they only train once) ----
@st.cache_resource(show_spinner="Training risk model…")
def train_risk_model(stu: pd.DataFrame):
    feats = ["current_gpa","cumulative_attendance_pct","grade_level","scholarship","iep"]
    X = stu[["current_gpa","cumulative_attendance_pct","grade_level"]].copy()
    X["scholarship"] = (stu["scholarship_flag"] == "Yes").astype(int)
    X["iep"]         = (stu["iep_flag"] == "Yes").astype(int)
    y = stu["academic_risk_flag"]
    model = LogisticRegression(max_iter=1000).fit(X, y)
    acc = accuracy_score(y, model.predict(X))
    return model, feats, acc

@st.cache_resource(show_spinner="Training GPA model…")
def train_gpa_model(stu: pd.DataFrame):
    X = stu[["cumulative_attendance_pct","grade_level"]].copy()
    X["scholarship"] = (stu["scholarship_flag"] == "Yes").astype(int)
    y = stu["current_gpa"]
    model = LinearRegression().fit(X, y)
    return model, X.columns.tolist(), model.score(X, y)


# ============================================================
# SIDEBAR
# ============================================================
st.sidebar.markdown("""
<div style='padding:.5rem 0 1.5rem 0;'>
    <div style='font-size:1.5rem; font-weight:700; color:white;'>📊 Nexus Analytics</div>
    <div style='font-size:.75rem; color:#94A3B8; margin-top:.25rem;'>
        Multi-School Academic Dashboard</div>
</div>""", unsafe_allow_html=True)

uploaded = st.sidebar.file_uploader("Workbook (.xlsx)", type="xlsx")
default_path = Path("academic_multi_school_dashboard_populated_10000.xlsx")
source = uploaded if uploaded is not None else (default_path if default_path.exists() else None)

if source is None:
    st.title("Nexus Analytics")
    st.info("Upload the academic workbook in the sidebar to begin.")
    st.stop()

sheets = load_workbook(source)
views  = build_views(sheets)
stu, rec, att = views["students"], views["records"], views["attendance"]
teachers, schools = views["teachers"], views["schools"]

st.sidebar.markdown("---")

f_school = st.sidebar.multiselect("School", sorted(stu["school"].dropna().unique()))
f_grade  = st.sidebar.multiselect("Grade",  sorted(stu["grade_level"].dropna().unique()))
f_risk   = st.sidebar.multiselect("Academic risk",
                                  sorted(stu["academic_risk_flag"].dropna().unique()))

def filter_students(df):
    if f_school: df = df[df["school"].isin(f_school)]
    if f_grade:  df = df[df["grade_level"].isin(f_grade)]
    if f_risk:   df = df[df["academic_risk_flag"].isin(f_risk)]
    return df

fstu = filter_students(stu)
ids  = set(fstu["student_id"])
frec = rec[rec["student_id"].isin(ids)] if ids else rec.iloc[0:0]
fatt = att[att["stakeholder_id"].isin(ids)] if ids else att.iloc[0:0]

st.sidebar.markdown("---")
page = st.sidebar.radio("Navigation", list(THEMES.keys()))

if fstu.empty:
    st.error("No rows match the current filters. Clear some filters in the sidebar.")
    st.stop()

theme = THEMES[page]
apply_theme(theme)
render_header(page)


# ============================================================
# OVERVIEW
# ============================================================
if page == "Overview":
    c1,c2,c3,c4,c5 = st.columns(5)
    c1.metric("Schools",      stu["school"].nunique())
    c2.metric("Students",     f"{len(fstu):,}")
    c3.metric("Teachers",     f"{len(teachers):,}")
    c4.metric("Avg GPA",      round(fstu["current_gpa"].mean(), 2))
    c5.metric("Attendance %", round(fstu["cumulative_attendance_pct"].mean(), 1))

    tab1, tab2 = st.tabs(["📈 Summary", "🏆 Rankings"])

    with tab1:
        col1, col2 = st.columns(2)
        fig = px.histogram(fstu, x="current_gpa", nbins=15, title="GPA distribution",
                           color_discrete_sequence=[theme["primary"]])
        fig.update_traces(marker_line_width=1, marker_line_color="white")
        col1.plotly_chart(style_fig(fig, theme, show_legend=False, bargap=0.12),
                          use_container_width=True)

        fig = px.histogram(fstu, x="cumulative_attendance_pct", nbins=15,
                           title="Attendance % distribution",
                           color_discrete_sequence=[theme["palette"][1]])
        fig.update_traces(marker_line_width=1, marker_line_color="white")
        col2.plotly_chart(style_fig(fig, theme, show_legend=False, bargap=0.12),
                          use_container_width=True)

        col1, col2 = st.columns(2)
        risk_counts = fstu["academic_risk_flag"].value_counts().reset_index()
        risk_counts.columns = ["risk", "count"]
        fig = px.pie(risk_counts, names="risk", values="count",
                     title="Risk composition", hole=0.55,
                     color_discrete_sequence=theme["palette"])
        fig.update_traces(textinfo="label+percent", textposition="outside")
        fig.add_annotation(text=f"<b>{len(fstu):,}</b><br>students",
                           showarrow=False, font=dict(size=16, color="#0F172A"))
        col1.plotly_chart(style_fig(fig, theme), use_container_width=True)

        g = fstu["gender"].value_counts().reset_index(); g.columns = ["gender","count"]
        fig = px.bar(g, x="gender", y="count", title="Gender distribution",
                     color="gender", text_auto=True,
                     color_discrete_sequence=theme["palette"])
        col2.plotly_chart(style_fig(fig, theme, show_legend=False),
                          use_container_width=True)

    with tab2:
        by_school = (fstu.groupby("school")
                         .agg(students=("student_id","count"),
                              avg_gpa=("current_gpa","mean"),
                              avg_att=("cumulative_attendance_pct","mean"))
                         .reset_index().sort_values("avg_gpa", ascending=False))
        top_n = st.slider("Show top N schools", 5, len(by_school), min(10, len(by_school)))
        fig = px.bar(by_school.head(top_n), x="school", y="avg_gpa",
                     title=f"Top {top_n} schools by average GPA",
                     hover_data=["students","avg_att"], text_auto=".2f",
                     color="avg_gpa", color_continuous_scale=[[0, "#E0E7FF"], [1, theme["primary"]]])
        st.plotly_chart(style_fig(fig, theme, show_legend=False),
                        use_container_width=True)

        fig = px.scatter(by_school, x="avg_att", y="avg_gpa", size="students",
                         hover_name="school", title="Schools: attendance vs GPA (size = students)",
                         color="avg_gpa",
                         color_continuous_scale=[[0, "#E0E7FF"], [1, theme["primary"]]])
        st.plotly_chart(style_fig(fig, theme, show_legend=False),
                        use_container_width=True)


# ============================================================
# STUDENTS  (with risk predictor + what-if simulator)
# ============================================================
elif page == "Students":
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("In view",     f"{len(fstu):,}")
    c2.metric("High risk",   int((fstu["academic_risk_flag"]=="High").sum()))
    c3.metric("Scholarship", int((fstu["scholarship_flag"]=="Yes").sum()))
    c4.metric("Avg GPA",     round(fstu["current_gpa"].mean(), 2))

    tab1, tab2, tab3 = st.tabs(["👥 Cohort", "🎯 Risk predictor", "🔮 What-if simulator"])

    with tab1:
        col1, col2 = st.columns(2)
        fig = px.box(fstu, x="grade_level", y="current_gpa", title="GPA by grade",
                     color_discrete_sequence=[theme["primary"]], points=False)
        col1.plotly_chart(style_fig(fig, theme, show_legend=False),
                          use_container_width=True)

        fig = px.box(fstu, x="grade_level", y="cumulative_attendance_pct",
                     title="Attendance by grade",
                     color_discrete_sequence=[theme["palette"][1]], points=False)
        col2.plotly_chart(style_fig(fig, theme, show_legend=False),
                          use_container_width=True)

        sample = fstu.sample(min(2000, len(fstu)), random_state=1)
        fig = px.scatter(sample, x="cumulative_attendance_pct", y="current_gpa",
                         color="academic_risk_flag",
                         hover_data=["school","grade_level"], opacity=0.55,
                         title="GPA vs attendance — coloured by risk",
                         color_discrete_sequence=theme["palette"])
        fig = add_trendline(fig, sample["cumulative_attendance_pct"],
                            sample["current_gpa"], color=theme["primary"])
        st.plotly_chart(style_fig(fig, theme), use_container_width=True)

    with tab2:
        st.markdown("A logistic-regression classifier predicts each student's risk bucket "
                    "from GPA, attendance, grade, scholarship and IEP status.")

        model, feats, acc = train_risk_model(stu)

        c1, c2, c3 = st.columns(3)
        c1.metric("Model accuracy", f"{acc*100:.1f}%")
        c2.metric("Training rows",  f"{len(stu):,}")
        c3.metric("Classes",        len(model.classes_))

        X = fstu[["current_gpa","cumulative_attendance_pct","grade_level"]].copy()
        X["scholarship"] = (fstu["scholarship_flag"]=="Yes").astype(int)
        X["iep"]         = (fstu["iep_flag"]=="Yes").astype(int)
        preds = model.predict(X)
        proba = model.predict_proba(X)

        col1, col2 = st.columns(2)
        pred_counts = pd.Series(preds).value_counts().reset_index()
        pred_counts.columns = ["predicted_risk", "count"]
        fig = px.bar(pred_counts, x="predicted_risk", y="count", text_auto=True,
                     title="Predicted risk buckets for the current view",
                     color="predicted_risk", color_discrete_sequence=theme["palette"])
        col1.plotly_chart(style_fig(fig, theme, show_legend=False),
                          use_container_width=True)

        # Feature importance = mean absolute coefficient across classes
        importance = pd.DataFrame({
            "feature": feats,
            "importance": np.mean(np.abs(model.coef_), axis=0),
        }).sort_values("importance", ascending=True)
        fig = px.bar(importance, x="importance", y="feature", orientation="h",
                     title="Feature importance (mean |coefficient|)",
                     color="importance",
                     color_continuous_scale=[[0, "#E0E7FF"], [1, theme["primary"]]])
        col2.plotly_chart(style_fig(fig, theme, show_legend=False),
                          use_container_width=True)

        high_idx = np.where(model.classes_ == "High")[0]
        if len(high_idx):
            risk_score = proba[:, high_idx[0]]
            df_score = pd.DataFrame({"high_risk_probability": risk_score})
            fig = px.histogram(df_score, x="high_risk_probability", nbins=20,
                               title="Distribution of P(High risk) across the cohort",
                               color_discrete_sequence=[theme["primary"]])
            fig.update_traces(marker_line_width=1, marker_line_color="white")
            st.plotly_chart(style_fig(fig, theme, show_legend=False, bargap=0.08),
                            use_container_width=True)

    with tab3:
        st.markdown("Adjust the sliders to see the predicted GPA and risk bucket for a "
                    "hypothetical student. The model is trained on every student in the workbook.")

        gpa_model, gpa_feats, r2 = train_gpa_model(stu)
        risk_model, _, _ = train_risk_model(stu)

        col1, col2 = st.columns([1, 2])
        with col1:
            att = st.slider("Attendance %", 50.0, 100.0, 85.0, 0.5)
            grade = st.select_slider("Grade level",
                                     options=sorted(stu["grade_level"].dropna().unique()),
                                     value=6)
            scholarship = st.checkbox("Has scholarship", value=False)
            iep = st.checkbox("Has IEP (Individualized Education Programme)", value=False)

        x_gpa = pd.DataFrame([[att, grade, int(scholarship)]], columns=gpa_feats)
        predicted_gpa = float(gpa_model.predict(x_gpa)[0])
        predicted_gpa = max(0.0, min(4.0, predicted_gpa))

        x_risk = pd.DataFrame([[predicted_gpa, att, grade,
                                int(scholarship), int(iep)]],
                              columns=["current_gpa","cumulative_attendance_pct",
                                       "grade_level","scholarship","iep"])
        predicted_risk = risk_model.predict(x_risk)[0]
        risk_proba     = risk_model.predict_proba(x_risk)[0]
        cohort_gpa     = float(stu["current_gpa"].mean())

        with col2:
            cc1, cc2, cc3 = st.columns(3)
            cc1.metric("Predicted GPA",        round(predicted_gpa, 2),
                       delta=round(predicted_gpa - cohort_gpa, 2),
                       delta_color="normal")
            cc2.metric("Predicted risk",       predicted_risk)
            cc3.metric("Model fit (R²)",       round(r2, 3))

            # Probability bar for the three classes
            prob_df = pd.DataFrame({"class": risk_model.classes_,
                                    "probability": risk_proba})
            fig = px.bar(prob_df, x="class", y="probability", text_auto=".0%",
                         title="Risk-class probabilities",
                         color="class", color_discrete_sequence=theme["palette"])
            fig.update_yaxes(tickformat=".0%")
            st.plotly_chart(style_fig(fig, theme, show_legend=False),
                            use_container_width=True)

        # Gauge-style comparison against the cohort distribution
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=predicted_gpa,
            delta={'reference': cohort_gpa, 'valueformat': '.2f'},
            title={'text': "Predicted GPA vs cohort average"},
            gauge={
                'axis': {'range': [0, 4]},
                'bar': {'color': theme["primary"]},
                'steps': [
                    {'range':[0, 2],   'color': "#FEE2E2"},
                    {'range':[2, 3],   'color': "#FEF3C7"},
                    {'range':[3, 4],   'color': "#DCFCE7"},
                ],
                'threshold': {'line': {'color': "#0F172A", 'width': 3},
                              'thickness': 0.75, 'value': cohort_gpa},
            },
        ))
        fig.update_layout(height=280, margin=dict(l=20,r=20,t=40,b=10),
                          paper_bgcolor="rgba(0,0,0,0)",
                          font_family='"Inter", sans-serif')
        st.plotly_chart(fig, use_container_width=True)


# ============================================================
# ACADEMIC RECORDS
# ============================================================
elif page == "Academic Records":
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Exam records", f"{len(frec):,}")
    c2.metric("Pass rate",    f"{(frec['pass_fail']=='Pass').mean()*100:.1f}%")
    c3.metric("Avg %",        round(frec["percentage"].mean(), 1))
    c4.metric("Subjects",     frec["subject_name"].nunique())

    tab1, tab2, tab3 = st.tabs(["🧾 Results", "📘 By subject", "📈 Trends"])

    with tab1:
        col1, col2 = st.columns(2)
        order = sorted(frec["grade_awarded"].dropna().unique())
        grade_counts = frec["grade_awarded"].value_counts().reindex(order).reset_index()
        grade_counts.columns = ["grade","count"]
        fig = px.bar(grade_counts, x="grade", y="count", text_auto=True,
                     title="Grade distribution",
                     color="count",
                     color_continuous_scale=[[0, "#FEF3C7"], [1, theme["primary"]]])
        col1.plotly_chart(style_fig(fig, theme, show_legend=False),
                          use_container_width=True)

        pf = frec["pass_fail"].value_counts().reset_index()
        pf.columns = ["status","count"]
        fig = px.pie(pf, names="status", values="count", hole=0.55,
                     title="Pass / fail split",
                     color_discrete_sequence=theme["palette"])
        fig.update_traces(textinfo="label+percent", textposition="outside")
        col2.plotly_chart(style_fig(fig, theme), use_container_width=True)

    with tab2:
        fig = px.box(frec, x="subject_name", y="percentage",
                     color="subject_name", title="Score distribution by subject",
                     color_discrete_sequence=theme["palette"], points=False)
        st.plotly_chart(style_fig(fig, theme, show_legend=False),
                        use_container_width=True)

        subj_avg = frec.groupby("subject_name").agg(
            avg_pct=("percentage","mean"),
            avg_assign=("assignment_score","mean"),
            avg_project=("project_score","mean"),
        ).reset_index()
        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(r=subj_avg["avg_pct"],
                                      theta=subj_avg["subject_name"],
                                      fill='toself', name="Exam %",
                                      line_color=theme["primary"]))
        fig.add_trace(go.Scatterpolar(r=subj_avg["avg_assign"]*10,
                                      theta=subj_avg["subject_name"],
                                      fill='toself', name="Assign × 10",
                                      line_color=theme["palette"][2]))
        fig.update_layout(title="Subject profile (radar)",
                          polar=dict(radialaxis=dict(visible=True)),
                          paper_bgcolor="rgba(0,0,0,0)",
                          font_family='"Inter", sans-serif',
                          title_x=0.02)
        st.plotly_chart(fig, use_container_width=True)

        sample = frec.sample(min(2000, len(frec)), random_state=1)
        fig = px.scatter(sample, x="assignment_score", y="marks_obtained",
                         color="subject_name", opacity=0.55,
                         title="Assignment score vs exam marks",
                         color_discrete_sequence=theme["palette"])
        fig = add_trendline(fig, sample["assignment_score"],
                            sample["marks_obtained"], color=theme["primary"])
        st.plotly_chart(style_fig(fig, theme), use_container_width=True)

    with tab3:
        if frec["term_name"].nunique() > 1:
            term_subj = (frec.groupby(["term_name","subject_name"])["percentage"]
                             .mean().reset_index())
            fig = px.line(term_subj, x="term_name", y="percentage",
                          color="subject_name", markers=True,
                          title="Average % by subject across terms",
                          color_discrete_sequence=theme["palette"])
            st.plotly_chart(style_fig(fig, theme), use_container_width=True)
        else:
            subj_avg = frec.groupby("subject_name")["percentage"].mean().reset_index()
            fig = px.bar(subj_avg.sort_values("percentage", ascending=False),
                         x="subject_name", y="percentage", text_auto=".1f",
                         title=f"Average % per subject (term: {frec['term_name'].iloc[0]})",
                         color="percentage",
                         color_continuous_scale=[[0, "#FEF3C7"], [1, theme["primary"]]])
            st.plotly_chart(style_fig(fig, theme, show_legend=False),
                            use_container_width=True)
            insight("Only one term of data is available, so term-over-term trends "
                    "can't be drawn. Showing per-subject averages instead.")


# ============================================================
# TEACHERS
# ============================================================
elif page == "Teachers":
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Teachers",       len(teachers))
    c2.metric("Avg rating",     round(teachers["teacher_performance_rating"].mean(), 2))
    c3.metric("Avg attendance", f"{teachers['teacher_attendance_pct'].mean():.1f}%")
    c4.metric("Departments",    teachers["department"].nunique())

    tab1, tab2 = st.tabs(["🏛 Faculty", "⭐ Performance"])

    with tab1:
        col1, col2 = st.columns(2)
        dept = teachers["department"].value_counts().reset_index()
        dept.columns = ["department","count"]
        fig = px.bar(dept, x="department", y="count", text_auto=True,
                     color="department", title="Teachers by department",
                     color_discrete_sequence=theme["palette"])
        col1.plotly_chart(style_fig(fig, theme, show_legend=False),
                          use_container_width=True)

        fig = px.box(teachers, x="department", y="teacher_performance_rating",
                     color="department", title="Rating by department",
                     color_discrete_sequence=theme["palette"], points=False)
        col2.plotly_chart(style_fig(fig, theme, show_legend=False),
                          use_container_width=True)

        emp = teachers["employment_type"].value_counts().reset_index()
        emp.columns = ["type","count"]
        fig = px.pie(emp, names="type", values="count", hole=0.55,
                     title="Employment type split",
                     color_discrete_sequence=theme["palette"])
        fig.update_traces(textinfo="label+percent", textposition="outside")
        st.plotly_chart(style_fig(fig, theme), use_container_width=True)

    with tab2:
        fig = px.scatter(teachers, x="years_experience",
                         y="teacher_performance_rating", color="department",
                         hover_data=["subject_specialization","teacher_attendance_pct"],
                         opacity=0.75, title="Experience vs performance rating",
                         color_discrete_sequence=theme["palette"])
        fig = add_trendline(fig, teachers["years_experience"],
                            teachers["teacher_performance_rating"],
                            color=theme["primary"])
        st.plotly_chart(style_fig(fig, theme), use_container_width=True)

        fig = px.density_heatmap(teachers, x="weekly_workload_hours",
                                 y="teacher_performance_rating",
                                 title="Workload hours vs rating (density)",
                                 color_continuous_scale=[[0, "#FEE2E2"], [1, theme["primary"]]])
        st.plotly_chart(style_fig(fig, theme, show_legend=False),
                        use_container_width=True)


# ============================================================
# ATTENDANCE
# ============================================================
elif page == "Attendance":
    c1,c2,c3 = st.columns(3)
    c1.metric("Log entries", f"{len(fatt):,}")
    pres = (fatt["attendance_status"]=="Present").mean()*100 if len(fatt) else 0
    c2.metric("Present %",   f"{pres:.1f}%")
    c3.metric("Dates",       fatt["attendance_date"].nunique())

    tab1, tab2 = st.tabs(["📊 Summary", "🏫 By school / grade"])

    with tab1:
        col1, col2 = st.columns(2)
        status = fatt["attendance_status"].value_counts().reset_index()
        status.columns = ["status","count"]
        fig = px.pie(status, names="status", values="count", hole=0.55,
                     title="Attendance status split",
                     color_discrete_sequence=theme["palette"])
        fig.update_traces(textinfo="label+percent", textposition="outside")
        fig.add_annotation(text=f"<b>{pres:.1f}%</b><br>present",
                           showarrow=False, font=dict(size=16, color="#0F172A"))
        col1.plotly_chart(style_fig(fig, theme), use_container_width=True)

        reason = fatt[fatt["attendance_status"]!="Present"]["reason"].value_counts().reset_index()
        reason.columns = ["reason","count"]
        if not reason.empty:
            fig = px.bar(reason, x="reason", y="count", text_auto=True,
                         title="Reasons for absence",
                         color="count",
                         color_continuous_scale=[[0, "#E0F2FE"], [1, theme["primary"]]])
            col2.plotly_chart(style_fig(fig, theme, show_legend=False),
                              use_container_width=True)

    with tab2:
        by_grade = (fatt.assign(p=(fatt["attendance_status"]=="Present").astype(int))
                         .groupby("grade_level")["p"].mean().reset_index())
        by_grade["p"] *= 100
        fig = px.bar(by_grade, x="grade_level", y="p", text_auto=".1f",
                     title="Attendance % by grade", labels={"p":"present %"},
                     color="p",
                     color_continuous_scale=[[0, "#E0F2FE"], [1, theme["primary"]]])
        st.plotly_chart(style_fig(fig, theme, show_legend=False),
                        use_container_width=True)

        by_school = (fatt.assign(p=(fatt["attendance_status"]=="Present").astype(int))
                          .groupby("school")["p"].mean().reset_index()
                          .sort_values("p", ascending=True))
        by_school["p"] *= 100
        fig = px.bar(by_school, y="school", x="p", orientation="h", text_auto=".1f",
                     title="Attendance % by school",
                     labels={"p":"present %"},
                     color="p",
                     color_continuous_scale=[[0, "#E0F2FE"], [1, theme["primary"]]])
        st.plotly_chart(style_fig(fig, theme, show_legend=False),
                        use_container_width=True)


# ============================================================
# SCHOOLS
# ============================================================
elif page == "Schools":
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Schools", len(schools))
    c2.metric("Boards",  schools["board_name"].nunique())
    c3.metric("States",  schools["state"].nunique())
    c4.metric("Regions", schools["region"].nunique())

    tab1, tab2, tab3 = st.tabs(["🏆 Rankings", "🧭 Distribution", "🗺 Geography"])

    with tab1:
        summary = (stu.groupby("school")
                      .agg(students=("student_id","count"),
                           avg_gpa=("current_gpa","mean"),
                           avg_att=("cumulative_attendance_pct","mean"))
                      .reset_index().sort_values("avg_gpa", ascending=False))
        fig = px.bar(summary, x="school", y="avg_gpa",
                     hover_data=["students","avg_att"],
                     title="Average GPA by school", text_auto=".2f",
                     color="avg_gpa",
                     color_continuous_scale=[[0, "#EDE9FE"], [1, theme["primary"]]])
        st.plotly_chart(style_fig(fig, theme, show_legend=False),
                        use_container_width=True)

        fig = px.scatter(summary, x="avg_att", y="avg_gpa", size="students",
                         hover_name="school",
                         title="School quadrant: attendance vs GPA (size = students)",
                         color="avg_gpa",
                         color_continuous_scale=[[0, "#EDE9FE"], [1, theme["primary"]]])
        fig.add_hline(y=summary["avg_gpa"].mean(), line_dash="dot", line_color="#64748B",
                      annotation_text="mean GPA", annotation_position="top left")
        fig.add_vline(x=summary["avg_att"].mean(), line_dash="dot", line_color="#64748B",
                      annotation_text="mean attendance", annotation_position="bottom right")
        st.plotly_chart(style_fig(fig, theme, show_legend=False),
                        use_container_width=True)

    with tab2:
        col1, col2 = st.columns(2)
        boards = schools["board_name"].value_counts().reset_index()
        boards.columns = ["board","count"]
        fig = px.pie(boards, names="board", values="count", hole=0.55,
                     title="Schools by board",
                     color_discrete_sequence=theme["palette"])
        fig.update_traces(textinfo="label+percent", textposition="outside")
        col1.plotly_chart(style_fig(fig, theme), use_container_width=True)

        regions = schools["region"].value_counts().reset_index()
        regions.columns = ["region","count"]
        fig = px.bar(regions, x="region", y="count", text_auto=True,
                     color="region", title="Schools by region",
                     color_discrete_sequence=theme["palette"])
        col2.plotly_chart(style_fig(fig, theme, show_legend=False),
                          use_container_width=True)

        types = schools["school_type"].value_counts().reset_index()
        types.columns = ["type","count"]
        fig = px.bar(types, x="type", y="count", text_auto=True,
                     color="type", title="Schools by type",
                     color_discrete_sequence=theme["palette"])
        st.plotly_chart(style_fig(fig, theme, show_legend=False),
                        use_container_width=True)

    with tab3:
        if {"latitude","longitude"}.issubset(schools.columns):
            st.map(schools[["latitude","longitude"]].dropna(), zoom=3)
        else:
            insight("Latitude / longitude columns not present in this workbook.")


# ============================================================
# DOWNLOAD
# ============================================================
st.sidebar.markdown("---")
st.sidebar.download_button(
    "📥 Download filtered students",
    fstu.to_csv(index=False).encode("utf-8"),
    file_name="filtered_students.csv",
    mime="text/csv",
    use_container_width=True,
)
