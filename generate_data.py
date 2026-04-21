"""
Regenerate the academic workbook with realistic variance.
Keeps the exact same schema so the dashboard code doesn't need to change.

Key realism injections:
  - Each school has a "quality tier" that drives baseline performance
  - GPA = tier + grade_effect + scholarship_boost - iep_penalty + noise
  - Attendance correlated with GPA (r ~ 0.55) with own noise
  - Risk flag derived from joint GPA+attendance thresholds
  - Subject-level difficulty (Math/Science harder than PE/Art)
  - ~12% fail rate (instead of 0)
  - Attendance log expanded to 60 school days with per-student patterns
  - Teacher rating correlates with experience + small noise, varies by dept
  - Multiple terms with realistic progression
"""
from pathlib import Path
import numpy as np
import pandas as pd

SEED = 42
rng = np.random.default_rng(SEED)

OUT = Path("/home/claude/academic_realistic.xlsx")
SRC = Path("/mnt/user-data/uploads/academic_multi_school_dashboard_populated_10000.xlsx")

# ---- Read the original so we can keep stable columns we don't want to regenerate
orig = pd.read_excel(SRC, sheet_name=None)

# ==========================================================
# SCHOOLS — keep the original rows, add quality tier
# ==========================================================
schools = orig["Schools"].copy()
N_SCHOOLS = len(schools)

# Assign quality tier: 5 top-tier, 8 mid-tier, 7 lower-tier
tier_labels = np.array(["A"]*5 + ["B"]*8 + ["C"]*7)
rng.shuffle(tier_labels)
schools["_quality_tier"] = tier_labels   # internal use only, dropped before save
tier_baseline = {"A": 3.4, "B": 2.9, "C": 2.4}   # baseline GPA by tier
tier_noise    = {"A": 0.35, "B": 0.45, "C": 0.55}

# Spread latitude/longitude across real Indian city-like coordinates so map looks good
city_coords = [
    (28.61, 77.20), (19.07, 72.87), (12.97, 77.59), (13.08, 80.27),
    (22.57, 88.36), (17.38, 78.48), (23.02, 72.57), (18.52, 73.86),
    (26.91, 75.78), (21.17, 72.83), (28.46, 77.03), (30.73, 76.78),
    (11.01, 76.96), (22.71, 75.85), (25.32, 82.97), (31.10, 77.17),
    (15.49, 73.82), (10.82, 78.68), (34.08, 74.80), (26.45, 80.33),
]
for i in range(N_SCHOOLS):
    schools.loc[i, "latitude"]  = city_coords[i % len(city_coords)][0] + rng.normal(0, 0.15)
    schools.loc[i, "longitude"] = city_coords[i % len(city_coords)][1] + rng.normal(0, 0.15)

# Student capacity varies: tier A schools tend to be larger
schools["student_capacity"] = [
    int(rng.integers(800, 1500)) if t == "A" else
    int(rng.integers(500, 1000)) if t == "B" else
    int(rng.integers(300, 700))
    for t in schools["_quality_tier"]
]

# ==========================================================
# STUDENTS — realistic GPA / attendance driven by tier + noise
# ==========================================================
students = orig["Students"].copy()
N = len(students)

# Map each student to their school's tier
sch_tier = dict(zip(schools["school_id"], schools["_quality_tier"]))
students["_tier"] = students["school_id"].map(sch_tier)

# Scholarship & IEP: realistic prevalence
students["scholarship_flag"] = np.where(rng.random(N) < 0.15, "Yes", "No")
students["iep_flag"]         = np.where(rng.random(N) < 0.08, "Yes", "No")

# Grade effect: older grades tend to have slightly lower GPA (harder work)
grade = students["grade_level"].astype(float)
grade_effect = -0.02 * (grade - 6)  # grade 6 = neutral

# Generate GPA from tier baseline + grade effect + flags + noise
base = students["_tier"].map(tier_baseline).astype(float)
noise_std = students["_tier"].map(tier_noise).astype(float)
scholarship_boost = np.where(students["scholarship_flag"]=="Yes", 0.25, 0.0)
iep_penalty       = np.where(students["iep_flag"]=="Yes", 0.35, 0.0)

gpa = (base + grade_effect + scholarship_boost - iep_penalty
       + rng.normal(0, noise_std))
gpa = np.clip(gpa, 0.5, 4.0)
students["current_gpa"] = np.round(gpa, 2)

# Attendance: correlated with GPA (r ~ 0.55) + own noise
# latent = 0.55 * standardised(gpa) + 0.835 * N(0,1)   => corr ~ 0.55
gpa_z = (gpa - gpa.mean()) / gpa.std()
att_z = 0.55 * gpa_z + 0.835 * rng.standard_normal(N)
# Convert z-score to % with mean 84, sd 8, capped 45-100
att = 84 + att_z * 8
att = np.clip(att, 45, 100)
students["cumulative_attendance_pct"] = np.round(att, 1)

# Academic risk flag: derived from BOTH gpa and attendance
def risk_of(gpa, att):
    if gpa < 2.2 or att < 70:        return "High"
    if gpa < 2.9 or att < 82:        return "Medium"
    return "Low"
students["academic_risk_flag"] = [risk_of(g, a) for g, a in zip(gpa, att)]

# Gender — keep balanced
students["gender"] = rng.choice(["Male", "Female"], size=N, p=[0.51, 0.49])

# Transport/hostel realistic
students["transport_opted"] = rng.choice(["Yes","No"], size=N, p=[0.55, 0.45])
students["hostel_opted"]    = rng.choice(["Yes","No"], size=N, p=[0.12, 0.88])

# Medium of instruction varies by region
students["medium_of_instruction"] = rng.choice(
    ["English","Hindi","Regional"], size=N, p=[0.55, 0.30, 0.15]
)

students = students.drop(columns=["_tier"])

# ==========================================================
# STUDENT_ACADEMIC_RECORDS — subject-level variance, multi-term
# ==========================================================
rec_template = orig["Student_Academic_Records"].copy()
subjects = [
    ("MAT", "Mathematics",     -6),   # difficulty adjustment
    ("SCI", "Science",         -4),
    ("ENG", "English",          0),
    ("SOC", "Social Studies",  +2),
    ("CSC", "Computer Science",+1),
]
terms = ["Term 1", "Term 2", "Term 3"]
term_boost = {"Term 1": -3, "Term 2": 0, "Term 3": +3}   # students improve

# Build records: each student gets ~3 rows across subjects/terms (keep N≈10k)
sid_pick = rng.choice(students["student_id"].values, size=len(rec_template), replace=True)
records = []
for i, sid in enumerate(sid_pick):
    stu = students.loc[students["student_id"]==sid].iloc[0]
    subject = subjects[i % len(subjects)]
    term    = terms[i % len(terms)]

    # base percentage driven by student's GPA (0.5 -> 28%, 1.5 -> 41%, 3.5 -> 67%, 4.0 -> 74%)
    base_pct = 21 + stu["current_gpa"] * 13
    pct = base_pct + subject[2] + term_boost[term] + rng.normal(0, 11)
    pct = int(np.clip(round(pct), 10, 99))

    # Grade letter mapping
    if   pct >= 90: letter = "A1"
    elif pct >= 80: letter = "A2"
    elif pct >= 70: letter = "B1"
    elif pct >= 60: letter = "B2"
    elif pct >= 50: letter = "C1"
    elif pct >= 40: letter = "C2"
    elif pct >= 33: letter = "D"
    else:           letter = "E"
    pass_fail = "Pass" if pct >= 35 else "Fail"

    # Assignment & project scores correlated with percentage
    assign  = int(np.clip(pct/5 + rng.normal(0, 2), 0, 20))
    project = int(np.clip(pct/10 + rng.normal(0, 1.5), 0, 10))

    records.append({
        "record_id":         f"REC{i+1:07d}",
        "school_id":         stu["school_id"],
        "academic_year":     "2025-2026",
        "term_name":         term,
        "exam_type":         rng.choice(["Unit Test","Mid Term","Final Exam"]),
        "student_id":        sid,
        "grade_level":       stu["grade_level"],
        "section":           stu["section"],
        "subject_code":      subject[0],
        "subject_name":      subject[1],
        "teacher_id":        f"TCH{rng.integers(1, 701):05d}",
        "exam_date":         f"2026-0{1+i%3}-{(i%27)+1:02d}",
        "max_marks":         100,
        "marks_obtained":    pct,
        "percentage":        pct,
        "grade_awarded":     letter,
        "pass_fail":         pass_fail,
        "class_rank":        int(rng.integers(1, 40)),
        "attendance_pct_term": int(np.clip(stu["cumulative_attendance_pct"] + rng.normal(0,3), 40, 100)),
        "assignment_score":  assign,
        "project_score":     project,
        "remarks":           rng.choice([
            "Shows improvement","Consistent performer","Needs attention",
            "Excellent work","Regular effort","Can do better",
        ]),
    })
records = pd.DataFrame(records)

# ==========================================================
# TEACHERS — rating correlates with experience, varies by dept
# ==========================================================
teachers = orig["Teachers"].copy()
dept_bonus = {"Sciences": 0.2, "Mathematics": 0.15, "Computer Science": 0.1,
              "Languages": 0.0, "Social Science": -0.05}

exp = teachers["years_experience"].astype(float)
rating = 3.0 + 0.04 * exp + teachers["department"].map(dept_bonus).astype(float) + rng.normal(0, 0.35, len(teachers))
teachers["teacher_performance_rating"] = np.round(np.clip(rating, 1.0, 5.0), 2)

# Attendance % — experienced teachers slightly better, but not perfectly correlated
teachers["teacher_attendance_pct"] = np.clip(
    85 + 0.2 * exp + rng.normal(0, 4, len(teachers)), 60, 100
).round(0).astype(int)

# Workload: Sciences/Math slightly more loaded
dept_load = {"Sciences": 3, "Mathematics": 2, "Computer Science": 1,
             "Languages": 0, "Social Science": -1}
teachers["weekly_workload_hours"] = (28 + teachers["department"].map(dept_load).astype(int)
                                        + rng.integers(-3, 4, len(teachers)))

teachers["employment_type"] = rng.choice(
    ["Permanent","Contract","Visiting"], size=len(teachers), p=[0.70, 0.25, 0.05]
)

# ==========================================================
# ATTENDANCE_LOG — expand to 60 school days with realistic patterns
# ==========================================================
# Use a subset of students to keep this at a reasonable size (~15k rows)
dates = pd.bdate_range("2026-02-02", periods=60).strftime("%Y-%m-%d")
sampled_students = students.sample(250, random_state=SEED)

att_rows = []
aid = 1
for _, stu in sampled_students.iterrows():
    # Per-student presence probability derived from their attendance %
    p_present = stu["cumulative_attendance_pct"] / 100
    # Add day-of-week effect (Mondays lower, Fridays slight dip)
    for d_idx, date_str in enumerate(dates):
        dow_adj = [-0.03, 0, 0.01, 0, -0.02][d_idx % 5]  # M T W T F
        is_present = rng.random() < np.clip(p_present + dow_adj, 0, 1)
        status = "Present" if is_present else rng.choice(
            ["Absent","Late","Leave"], p=[0.65, 0.25, 0.10]
        )
        reason = ("Regular" if status == "Present"
                  else rng.choice(["Illness","Family","Travel","Other"],
                                  p=[0.45, 0.25, 0.15, 0.15]))
        att_rows.append({
            "attendance_id":     f"ATD{aid:07d}",
            "attendance_date":   date_str,
            "school_id":         stu["school_id"],
            "stakeholder_type":  "Student",
            "stakeholder_id":    stu["student_id"],
            "academic_year":     "2025-2026",
            "grade_level":       float(stu["grade_level"]),
            "section":           stu["section"],
            "attendance_status": status,
            "reason":            reason,
            "checkin_time":      "07:45" if status == "Present" else ("08:20" if status=="Late" else ""),
            "checkout_time":     "14:30" if status in ("Present","Late") else "",
        })
        aid += 1

attendance_log = pd.DataFrame(att_rows)

# ==========================================================
# WRITE WORKBOOK
# ==========================================================
schools_out = schools.drop(columns=["_quality_tier"])

output_sheets = {
    "README":                    orig["README"],
    "Data_Dictionary":           orig["Data_Dictionary"],
    "Lists":                     orig["Lists"],
    "Schools":                   schools_out,
    "Principals":                orig["Principals"],
    "Teachers":                  teachers,
    "Admins":                    orig["Admins"],
    "Students":                  students,
    "Parents":                   orig["Parents"],
    "Student_Parent_Map":        orig["Student_Parent_Map"],
    "Teacher_Class_Assignments": orig["Teacher_Class_Assignments"],
    "Student_Academic_Records":  records,
    "Attendance_Log":            attendance_log,
    "Generation_Summary":        orig["Generation_Summary"],
}

with pd.ExcelWriter(OUT, engine="openpyxl") as w:
    for name, df in output_sheets.items():
        df.to_excel(w, sheet_name=name, index=False)

print(f"✓ Wrote {OUT}")
print(f"  students: {len(students):,}  |  records: {len(records):,}  |  "
      f"attendance: {len(attendance_log):,}")
print(f"  GPA  — mean {students['current_gpa'].mean():.2f}  sd {students['current_gpa'].std():.2f}  "
      f"range {students['current_gpa'].min()}-{students['current_gpa'].max()}")
print(f"  Att% — mean {students['cumulative_attendance_pct'].mean():.1f}  sd {students['cumulative_attendance_pct'].std():.1f}")
print(f"  Risk — {students['academic_risk_flag'].value_counts().to_dict()}")
print(f"  Pass rate: {(records['pass_fail']=='Pass').mean()*100:.1f}%")
print(f"  GPA-Att correlation: {students['current_gpa'].corr(students['cumulative_attendance_pct']):.3f}")
