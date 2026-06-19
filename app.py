import os
import json
import ast
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from typing import List, Dict, Any

# Set page config
st.set_page_config(
    page_title="RedRob Candidate Analytics & Discovery",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom premium styling using glassmorphism and modern colors
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700&display=swap');
    
    /* Global Styles */
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    /* Main Background */
    .stApp {
        background: linear-gradient(135deg, #0e1117 0%, #161a24 100%);
        color: #e2e8f0;
    }
    
    /* Headers */
    h1, h2, h3 {
        color: #ffffff !important;
        font-weight: 700 !important;
        letter-spacing: -0.02em;
    }
    
    /* Card Container */
    .premium-card {
        background: rgba(30, 41, 59, 0.45);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 20px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.2);
    }
    
    /* Metrics */
    .metric-value {
        font-size: 2.2rem;
        font-weight: 700;
        color: #a78bfa;
        line-height: 1;
        margin-bottom: 4px;
    }
    
    .metric-label {
        font-size: 0.85rem;
        text-transform: uppercase;
        color: #94a3b8;
        letter-spacing: 0.05em;
    }
    
    /* Badge styling */
    .badge {
        display: inline-block;
        padding: 4px 10px;
        font-size: 0.75rem;
        font-weight: 600;
        border-radius: 20px;
        margin-right: 6px;
        margin-bottom: 6px;
    }
    .badge-primary { background: rgba(139, 92, 246, 0.25); color: #c084fc; border: 1px solid rgba(139, 92, 246, 0.4); }
    .badge-success { background: rgba(16, 185, 129, 0.25); color: #34d399; border: 1px solid rgba(16, 185, 129, 0.4); }
    .badge-warning { background: rgba(245, 158, 11, 0.25); color: #fbbf24; border: 1px solid rgba(245, 158, 11, 0.4); }
    .badge-danger { background: rgba(239, 68, 68, 0.25); color: #f87171; border: 1px solid rgba(239, 68, 68, 0.4); }
    .badge-info { background: rgba(59, 130, 246, 0.25); color: #60a5fa; border: 1px solid rgba(59, 130, 246, 0.4); }
    
    /* Career history timeline */
    .timeline-container {
        border-left: 2px solid rgba(139, 92, 246, 0.3);
        padding-left: 20px;
        margin-left: 10px;
        margin-top: 15px;
    }
    
    .timeline-item {
        position: relative;
        margin-bottom: 25px;
    }
    
    .timeline-item::before {
        content: '';
        position: absolute;
        width: 12px;
        height: 12px;
        background: #8b5cf6;
        border: 2px solid #0e1117;
        border-radius: 50%;
        left: -27px;
        top: 6px;
    }
    
    .timeline-title {
        font-weight: 600;
        font-size: 1rem;
        color: #ffffff;
        margin: 0;
    }
    
    .timeline-subtitle {
        font-size: 0.85rem;
        color: #a78bfa;
        margin: 2px 0 6px 0;
    }
    
    .timeline-desc {
        font-size: 0.9rem;
        color: #cbd5e1;
        margin: 0;
    }
</style>
""", unsafe_allow_html=True)

# Helper function to parse stringified arrays/dicts from CSV
def parse_literal(val: Any) -> Any:
    if not isinstance(val, str) or pd.isna(val) or not val.strip():
        return []
    try:
        return ast.literal_eval(val)
    except Exception:
        try:
            return json.loads(val)
        except Exception:
            return []

# Dynamic dataset loader
@st.cache_data
def load_sample_data() -> pd.DataFrame:
    sample_path = os.path.join("Redrobsdataset", "sample_candidates.json")
    if not os.path.exists(sample_path):
        return pd.DataFrame()
        
    try:
        with open(sample_path, "r", encoding="utf-8") as f:
            records = json.load(f)
            
        # Import pipeline components on-the-fly
        from src.cleaning import clean_candidate_record
        from src.classification import classify_candidate
        from src.quality_checks import check_single_candidate_quality
        from src.feature_engineering import engineer_features_dataframe
        
        cleaned_records = []
        for rec in records:
            cleaned_rec = clean_candidate_record(rec)
            # Classify segment
            segment = classify_candidate(cleaned_rec)
            cleaned_rec["candidate_segment"] = segment
            # Run quality check
            quality = check_single_candidate_quality(cleaned_rec)
            # Merge quality metrics
            for k, v in quality.items():
                if k != "candidate_id":
                    cleaned_rec[k] = v
            cleaned_records.append(cleaned_rec)
            
        df_clean = pd.DataFrame(cleaned_records)
        df_features = engineer_features_dataframe(df_clean)
        
        # Merge engineered features back into the clean DataFrame, dropping duplicates
        overlap_cols = [c for c in df_features.columns if c in df_clean.columns and c != "candidate_id"]
        df_features_clean = df_features.drop(columns=overlap_cols)
        df = pd.merge(df_clean, df_features_clean, on="candidate_id", how="inner")
        return df
    except Exception as e:
        st.error(f"Error executing pipeline on sample dataset: {e}")
        return pd.DataFrame()

@st.cache_data
def load_production_data(limit: int = 5000) -> pd.DataFrame:
    cleaned_path = os.path.join("output", "cleaned_candidates.csv")
    class_path = os.path.join("output", "candidate_classification.csv")
    features_path = os.path.join("output", "candidate_features.csv")
    
    if not os.path.exists(cleaned_path):
        return pd.DataFrame()
        
    try:
        # Load datasets
        df_clean = pd.read_csv(cleaned_path, nrows=limit)
        
        df_class = pd.read_csv(class_path) if os.path.exists(class_path) else pd.DataFrame()
        df_feat = pd.read_csv(features_path) if os.path.exists(features_path) else pd.DataFrame()
        
        # Merge tables
        df = df_clean
        if not df_class.empty:
            df = pd.merge(df, df_class, on="candidate_id", how="inner")
        if not df_feat.empty:
            # Drop matching columns except PK
            overlap_cols = [c for c in df_feat.columns if c in df.columns and c != "candidate_id"]
            df_feat_clean = df_feat.drop(columns=overlap_cols)
            df = pd.merge(df, df_feat_clean, on="candidate_id", how="inner")
            
        # Parse nested columns
        for col in ["career_history", "skills", "education", "certifications", "languages"]:
            if col in df.columns:
                df[col] = df[col].apply(parse_literal)
                
        return df
    except Exception as e:
        st.error(f"Error loading production data: {e}")
        return pd.DataFrame()

# Main Title Header
st.title("🧠 RedRob Intelligent Candidate Analytics & Discovery")
st.markdown("---")

# Sidebar Configuration
st.sidebar.image("https://redrob.com/assets/img/redrob-logo-dark.png", width=180)
st.sidebar.markdown("### ⚙️ Datasource & Processing Settings")

data_source = st.sidebar.selectbox(
    "Choose Dataset Source:",
    ["Sample Dataset (Fast & Rich Profiles)", "Full Production Dataset"]
)

# Load selected dataset
df = pd.DataFrame()
if data_source == "Sample Dataset (Fast & Rich Profiles)":
    with st.spinner("Processing sample candidates through pipeline..."):
        df = load_sample_data()
else:
    limit = st.sidebar.slider("Production Record Limit:", 500, 10000, 2000, 500)
    with st.spinner(f"Loading {limit} production candidate profiles..."):
        df = load_production_data(limit=limit)

if df.empty:
    st.error("No data could be loaded. Please ensure that the pipeline has been run or sample dataset exists.")
    st.info("To populate the full dataset, run the pipeline command: `python -m src.pipeline --format csv`")
    st.stop()

# Ensure unique records
df_unique = df[~df["is_duplicate"]].copy() if "is_duplicate" in df.columns else df.copy()
duplicate_count = len(df) - len(df_unique)

# Filter parameters in sidebar
st.sidebar.markdown("### 🔍 Filters")

# 1. Experience Filter
max_exp_val = float(df_unique["profile_years_experience"].max()) if "profile_years_experience" in df_unique.columns else 20.0
exp_range = st.sidebar.slider(
    "Years of Experience:",
    0.0, max(max_exp_val, 1.0), (0.0, max(max_exp_val, 1.0)), 0.5
)

# 2. Expected Salary Filter (INR LPA)
min_salary_col = "expected_salary_min"
max_salary_col = "expected_salary_max"
max_salary_val = float(df_unique[max_salary_col].max()) if max_salary_col in df_unique.columns else 100.0
salary_range = st.sidebar.slider(
    "Expected Salary (LPA INR):",
    0.0, max(max_salary_val, 10.0), (0.0, max(max_salary_val, 10.0)), 1.0
)

# 3. Candidate Segment Filter
segments = list(df_unique["candidate_segment"].unique()) if "candidate_segment" in df_unique.columns else []
selected_segments = st.sidebar.multiselect(
    "Candidate Segments:",
    options=segments,
    default=segments
)

# 4. Location Filter
locations = list(df_unique["location"].dropna().unique())
selected_locations = st.sidebar.multiselect(
    "Locations:",
    options=sorted(locations),
    default=[]
)

# 5. Quality Score Filter
quality_min = st.sidebar.slider("Min Quality Score:", 0, 100, 0, 5)

# 6. Skill search
# Extract all unique skill names for multiselect
all_skills = set()
for s_list in df_unique["skills"]:
    if isinstance(s_list, list):
        for s in s_list:
            if isinstance(s, dict) and "name" in s:
                all_skills.add(s["name"])
selected_skills = st.sidebar.multiselect(
    "Search Specific Skills:",
    options=sorted(list(all_skills)),
    default=[]
)

# Apply filters
filtered_df = df_unique.copy()

if "profile_years_experience" in filtered_df.columns:
    filtered_df = filtered_df[
        (filtered_df["profile_years_experience"] >= exp_range[0]) & 
        (filtered_df["profile_years_experience"] <= exp_range[1])
    ]

if min_salary_col in filtered_df.columns and max_salary_col in filtered_df.columns:
    filtered_df = filtered_df[
        (filtered_df[min_salary_col] >= salary_range[0]) & 
        (filtered_df[max_salary_col] <= salary_range[1])
    ]

if "candidate_segment" in filtered_df.columns and selected_segments:
    filtered_df = filtered_df[filtered_df["candidate_segment"].isin(selected_segments)]

if selected_locations:
    filtered_df = filtered_df[filtered_df["location"].isin(selected_locations)]

if "quality_score" in filtered_df.columns:
    filtered_df = filtered_df[filtered_df["quality_score"] >= quality_min]

if selected_skills:
    # Match candidate if they possess AT LEAST ONE of the selected skills
    def has_skills(skills_list):
        if not isinstance(skills_list, list):
            return False
        c_skills = {s["name"] for s in skills_list if isinstance(s, dict) and "name" in s}
        return any(s in c_skills for s in selected_skills)
        
    filtered_df = filtered_df[filtered_df["skills"].apply(has_skills)]

# UI Layout: Create tabs
tabs = st.tabs(["📊 Overview", "🗺️ Demographics & Companies", "🧠 Skillsets Analysis", "🔍 Candidate Profile Explorer", "🧼 Pipeline & Cleaning Logic"])

# ==========================================
# TAB 1: OVERVIEW DASHBOARD
# ==========================================
with tabs[0]:
    # Key KPI Metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""
        <div class="premium-card">
            <div class="metric-value">{len(filtered_df)}</div>
            <div class="metric-label">Filtered Candidates</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        avg_exp = filtered_df["profile_years_experience"].mean() if "profile_years_experience" in filtered_df.columns else 0.0
        st.markdown(f"""
        <div class="premium-card">
            <div class="metric-value">{avg_exp:.1f} yrs</div>
            <div class="metric-label">Average Experience</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        avg_q = filtered_df["quality_score"].mean() if "quality_score" in filtered_df.columns else 0.0
        st.markdown(f"""
        <div class="premium-card">
            <div class="metric-value">{avg_q:.1f}%</div>
            <div class="metric-label">Average Profile Quality</div>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown(f"""
        <div class="premium-card">
            <div class="metric-value">{duplicate_count}</div>
            <div class="metric-label">Duplicate Records Ignored</div>
        </div>
        """, unsafe_allow_html=True)

    col_chart1, col_chart2 = st.columns(2)
    
    with col_chart1:
        st.markdown("### Candidate Segments Breakdown")
        if "candidate_segment" in filtered_df.columns and len(filtered_df) > 0:
            segment_counts = filtered_df["candidate_segment"].value_counts().reset_index()
            segment_counts.columns = ["Segment", "Count"]
            
            fig = px.pie(
                segment_counts, 
                values="Count", 
                names="Segment", 
                hole=0.45,
                color_discrete_sequence=px.colors.qualitative.Pastel
            )
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font_color="#e2e8f0",
                legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5),
                margin=dict(t=10, b=50, l=10, r=10)
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No segment data available for plotting.")
            
    with col_chart2:
        st.markdown("### Expected Salary Distributions (LPA INR)")
        if "expected_salary_min" in filtered_df.columns and len(filtered_df) > 0:
            fig = go.Figure()
            fig.add_trace(go.Histogram(
                x=filtered_df["expected_salary_min"],
                name="Min Expected Salary",
                marker_color="#8b5cf6",
                opacity=0.75,
                nbinsx=20
            ))
            fig.add_trace(go.Histogram(
                x=filtered_df["expected_salary_max"],
                name="Max Expected Salary",
                marker_color="#34d399",
                opacity=0.75,
                nbinsx=20
            ))
            fig.update_layout(
                barmode="overlay",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font_color="#e2e8f0",
                xaxis_title="Salary (LPA INR)",
                yaxis_title="Count",
                margin=dict(t=20, b=20, l=20, r=20),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No salary data available for plotting.")

    # Quality Flags Audit
    st.markdown("### 🔍 Profile Data Quality & Integrity Checks")
    anomaly_cols = [
        ("inconsistent_experience_flag", "Inconsistent Experience", "badge-warning", "Profile experience vs career history duration mismatch"),
        ("impossible_timeline_flag", "Impossible Timeline", "badge-danger", "Start date after end date in jobs"),
        ("overlapping_jobs_flag", "Overlapping Jobs", "badge-warning", "Concurrent job timeline overlaps > 90 days"),
        ("worked_before_college_flag", "Worked Before College", "badge-info", "Candidate worked > 6 years before starting college"),
        ("suspicious_skill_durations_flag", "Suspicious Skill Duration", "badge-warning", "Skill experience exceeds total work experience"),
        ("missing_critical_info_flag", "Missing Critical Info", "badge-danger", "Empty summary, career, or skill sections")
    ]
    
    col_a, col_b = st.columns([1, 2])
    with col_a:
        st.markdown("#### Quality Audit Status")
        issues_summary = []
        for col_flag, label, badge, desc in anomaly_cols:
            if col_flag in filtered_df.columns:
                flagged_count = filtered_df[col_flag].sum()
                flagged_pct = (flagged_count / len(filtered_df)) * 100
                st.markdown(f"**{label}**: <span class='badge {badge}'>{flagged_count} ({flagged_pct:.1f}%)</span>", unsafe_allow_html=True)
                st.caption(desc)
    with col_b:
        st.markdown("#### Quality Scores Distribution")
        if "quality_score" in filtered_df.columns and len(filtered_df) > 0:
            fig = px.histogram(
                filtered_df,
                x="quality_score",
                nbins=20,
                labels={"quality_score": "Quality Score (out of 100)"},
                color_discrete_sequence=["#a78bfa"]
            )
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font_color="#e2e8f0",
                xaxis_title="Profile Quality Score (%)",
                yaxis_title="Count of Candidates",
                margin=dict(t=10, b=10, l=10, r=10)
            )
            st.plotly_chart(fig, use_container_width=True)

# ==========================================
# TAB 2: DEMOGRAPHICS & ROLES
# ==========================================
with tabs[1]:
    col_dem1, col_dem2 = st.columns(2)
    
    with col_dem1:
        st.markdown("### Top Candidate Locations")
        if "location" in filtered_df.columns and len(filtered_df) > 0:
            loc_counts = filtered_df["location"].value_counts().head(10).reset_index()
            loc_counts.columns = ["Location", "Candidates"]
            
            fig = px.bar(
                loc_counts,
                x="Candidates",
                y="Location",
                orientation="h",
                color="Candidates",
                color_continuous_scale="purples",
                labels={"Location": "Location", "Candidates": "Number of Candidates"}
            )
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font_color="#e2e8f0",
                margin=dict(t=10, b=10, l=10, r=10),
                coloraxis_showscale=False
            )
            # Invert y axis to show highest on top
            fig.update_yaxes(autorange="reversed")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No location data available.")
            
    with col_dem2:
        st.markdown("### Common Job Titles (Raw vs Normalized)")
        title_type = st.radio("Group by:", ["Normalized Current Titles", "Raw Current Titles"], horizontal=True)
        
        title_col = "normalized_current_title" if title_type == "Normalized Current Titles" else "current_title"
        
        if title_col in filtered_df.columns and len(filtered_df) > 0:
            title_counts = filtered_df[title_col].value_counts().head(10).reset_index()
            title_counts.columns = ["Job Title", "Candidates"]
            
            fig = px.bar(
                title_counts,
                x="Candidates",
                y="Job Title",
                orientation="h",
                color="Candidates",
                color_continuous_scale="teal",
                labels={"Job Title": "Job Title", "Candidates": "Number of Candidates"}
            )
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font_color="#e2e8f0",
                margin=dict(t=10, b=10, l=10, r=10),
                coloraxis_showscale=False
            )
            fig.update_yaxes(autorange="reversed")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No title data available.")

    # Company Profile Analytics (Product vs Consulting vs Startup)
    st.markdown("### 🏢 Previous Company Categories")
    col_c1, col_c2 = st.columns(2)
    with col_c1:
        st.markdown("#### Candidate Enterprise Experience")
        # Compute company score metrics
        product_score = filtered_df["product_company_score"].mean() * 100 if "product_company_score" in filtered_df.columns else 0.0
        consulting_score = filtered_df["consulting_company_score"].mean() * 100 if "consulting_company_score" in filtered_df.columns else 0.0
        startup_score = filtered_df["startup_score"].mean() * 100 if "startup_score" in filtered_df.columns else 0.0
        
        company_types = pd.DataFrame({
            "Category": ["Product Companies", "Consulting/Services", "Startups"],
            "Average Experience Index (%)": [product_score, consulting_score, startup_score]
        })
        
        fig = px.bar(
            company_types,
            x="Average Experience Index (%)",
            y="Category",
            orientation="h",
            color="Category",
            color_discrete_sequence=["#8b5cf6", "#60a5fa", "#34d399"]
        )
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#e2e8f0",
            margin=dict(t=10, b=10, l=10, r=10),
            showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption("A higher index represents that candidates spent a larger proportion of their career in these respective company categories.")

    with col_c2:
        st.markdown("#### Most Common Companies in Career History")
        # Extract all companies from career histories
        all_companies = []
        for jobs in filtered_df["career_history"]:
            if isinstance(jobs, list):
                for job in jobs:
                    if isinstance(job, dict) and "company" in job:
                        comp = job["company"].strip()
                        if comp and comp != "Unknown" and comp != "None":
                            all_companies.append(comp)
        if all_companies:
            comp_df = pd.Series(all_companies).value_counts().head(10).reset_index()
            comp_df.columns = ["Company", "Mentions"]
            
            fig = px.bar(
                comp_df,
                x="Mentions",
                y="Company",
                orientation="h",
                color="Mentions",
                color_continuous_scale="sunset",
            )
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font_color="#e2e8f0",
                margin=dict(t=10, b=10, l=10, r=10),
                coloraxis_showscale=False
            )
            fig.update_yaxes(autorange="reversed")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No career history company details available.")

# ==========================================
# TAB 3: SKILLSETS ANALYSIS
# ==========================================
with tabs[2]:
    st.markdown("### Core Skillsets and Proficiencies")
    
    # Process skill stats
    skills_expanded = []
    for idx, row in filtered_df.iterrows():
        c_id = row.get("candidate_id")
        skills = row.get("skills", [])
        if isinstance(skills, list):
            for s in skills:
                if isinstance(s, dict) and "name" in s:
                    skills_expanded.append({
                        "candidate_id": c_id,
                        "skill": s["name"],
                        "proficiency": s.get("proficiency", "unknown"),
                        "endorsements": s.get("endorsements", 0),
                        "duration_months": s.get("duration_months", 0)
                    })
                    
    if skills_expanded:
        skills_df = pd.DataFrame(skills_expanded)
        
        # 1. Most popular skills
        col_sk1, col_sk2 = st.columns(2)
        with col_sk1:
            st.markdown("#### Top 15 Most Prevalent Skills")
            top_skills = skills_df["skill"].value_counts().head(15).reset_index()
            top_skills.columns = ["Skill", "Candidates"]
            
            fig = px.bar(
                top_skills,
                x="Candidates",
                y="Skill",
                orientation="h",
                color="Candidates",
                color_continuous_scale="purples"
            )
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font_color="#e2e8f0",
                margin=dict(t=10, b=10, l=10, r=10),
                coloraxis_showscale=False
            )
            fig.update_yaxes(autorange="reversed")
            st.plotly_chart(fig, use_container_width=True)
            
        with col_sk2:
            st.markdown("#### Skill Proficiency Breakdowns")
            # Select skill to inspect
            selected_skill_inspect = st.selectbox(
                "Select a skill to inspect proficiency:",
                options=sorted(list(skills_df["skill"].unique())),
                index=0 if "python" not in skills_df["skill"].unique() else list(sorted(skills_df["skill"].unique())).index("python")
            )
            
            skill_sub = skills_df[skills_df["skill"] == selected_skill_inspect]
            prof_counts = skill_sub["proficiency"].value_counts().reset_index()
            prof_counts.columns = ["Proficiency", "Count"]
            
            # Order proficiencies
            order_mapping = {"expert": 4, "advanced": 3, "intermediate": 2, "beginner": 1, "unknown": 0}
            prof_counts["order"] = prof_counts["Proficiency"].map(order_mapping)
            prof_counts = prof_counts.sort_values(by="order", ascending=False)
            
            fig = px.bar(
                prof_counts,
                x="Proficiency",
                y="Count",
                color="Proficiency",
                color_discrete_sequence=px.colors.qualitative.Bold,
                labels={"Count": "Number of Candidates"}
            )
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font_color="#e2e8f0",
                margin=dict(t=10, b=10, l=10, r=10),
                showlegend=False
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Show stats
            avg_months = skill_sub["duration_months"].mean() / 12.0
            avg_endors = skill_sub["endorsements"].mean()
            st.markdown(f"""
            - **Average Experience in {selected_skill_inspect}**: {avg_months:.1f} years
            - **Average Endorsements**: {avg_endors:.1f}
            """)
            
        # Skill-segment correlation matrix
        st.markdown("#### Tech Stack Scores vs Candidate Segment")
        score_cols = [c for c in ["python_score", "retrieval_score", "ranking_score", "vector_db_score", "llm_score"] if c in filtered_df.columns]
        if score_cols and "candidate_segment" in filtered_df.columns:
            # Group by segment and average the scores
            scores_by_seg = filtered_df.groupby("candidate_segment")[score_cols].mean()
            # Normalize column labels
            scores_by_seg.columns = [c.replace("_score", "").upper() for c in scores_by_seg.columns]
            
            fig = px.imshow(
                scores_by_seg,
                labels=dict(x="Skill Area Score", y="Candidate Segment", color="Average Score"),
                x=scores_by_seg.columns,
                y=scores_by_seg.index,
                color_continuous_scale="plasma"
            )
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font_color="#e2e8f0",
                margin=dict(t=20, b=20, l=20, r=20)
            )
            st.plotly_chart(fig, use_container_width=True)
            st.caption("Heuristic scores engineered from skill proficiencies, endorsements, and career durations across segments.")
    else:
        st.info("No skill details available.")

# ==========================================
# TAB 4: CANDIDATE PROFILE EXPLORER
# ==========================================
with tabs[3]:
    st.markdown("### 🔍 Search & Discover Profiles")
    st.markdown("Select a candidate from the table below to explore their comprehensive profile details.")
    
    # Display table columns
    table_cols = ["candidate_id", "anonymized_name", "normalized_current_title", "location", "profile_years_experience", "quality_score", "candidate_segment"]
    display_df = filtered_df[[c for c in table_cols if c in filtered_df.columns]].copy()
    
    # Rename for cleaner table display
    rename_dict = {
        "candidate_id": "Candidate ID",
        "anonymized_name": "Name",
        "normalized_current_title": "Role Category",
        "location": "Location",
        "profile_years_experience": "Experience (Yrs)",
        "quality_score": "Quality Score",
        "candidate_segment": "Segment"
    }
    display_df = display_df.rename(columns=rename_dict)
    
    # Search box for names
    search_query = st.text_input("🔍 Quick Search by Name or ID:")
    if search_query:
        display_df = display_df[
            display_df["Name"].str.contains(search_query, case=False, na=False) |
            display_df["Candidate ID"].str.contains(search_query, case=False, na=False)
        ]
        
    # Render table selection
    selected_c_id = None
    if not display_df.empty:
        # User dropdown selection
        cand_list = [f"{row['Candidate ID']} - {row['Name']} ({row['Role Category']})" for _, row in display_df.iterrows()]
        selected_option = st.selectbox("Select Candidate to View Profile Details:", options=cand_list)
        selected_c_id = selected_option.split(" - ")[0]
    else:
        st.warning("No candidates match your current filter settings.")
        
    if selected_c_id:
        cand_data = filtered_df[filtered_df["candidate_id"] == selected_c_id].iloc[0].to_dict()
        
        st.markdown("---")
        
        # Profile detailed layout
        p_col1, p_col2 = st.columns([1, 2])
        
        with p_col1:
            st.markdown(f"## {cand_data.get('anonymized_name', 'Anonymous')}")
            st.markdown(f"**Headline**: `{cand_data.get('headline', 'N/A')}`")
            st.markdown(f"📍 **Location**: {cand_data.get('location', 'N/A')}, {cand_data.get('country', 'N/A')}")
            st.markdown(f"🏢 **Current Company**: {cand_data.get('current_company', 'N/A')} ({cand_data.get('current_industry', 'N/A')}, size {cand_data.get('current_company_size', 'N/A')})")
            
            # Show segment and salary
            salary_min = cand_data.get('expected_salary_min', 0.0)
            salary_max = cand_data.get('expected_salary_max', 0.0)
            
            st.markdown(f"""
            <div class="premium-card" style="margin-top: 15px;">
                <div class="metric-label">Assigned Segment</div>
                <div style="font-size: 1.3rem; font-weight: 700; color: #a78bfa; margin-bottom: 12px;">{cand_data.get('candidate_segment', 'SOFTWARE_ENGINEER')}</div>
                <div class="metric-label">Expected Salary Range</div>
                <div style="font-size: 1.3rem; font-weight: 700; color: #34d399; margin-bottom: 12px;">₹ {salary_min} - {salary_max} LPA</div>
                <div class="metric-label">Profile Quality Score</div>
                <div style="font-size: 1.3rem; font-weight: 700; color: #fbbf24;">{cand_data.get('quality_score', 100)} / 100</div>
            </div>
            """, unsafe_allow_html=True)
            
            # Show Quality Audit warnings
            st.markdown("### ⚠️ Data Integrity & Flag Audits")
            has_anomaly = False
            for col_flag, label, badge, desc in anomaly_cols:
                if cand_data.get(col_flag, False):
                    has_anomaly = True
                    st.markdown(f"<span class='badge badge-danger'>🚨 {label}</span> - {desc}", unsafe_allow_html=True)
                    
            if not has_anomaly:
                st.markdown("<span class='badge badge-success'>✓ Profile Verified Clean</span> - No logical anomalies found.", unsafe_allow_html=True)
            else:
                st.caption(f"**System Logs**: {cand_data.get('issues', 'No logs')}")

            # Skill Radar / Horizontal Bar
            st.markdown("### 🛠️ Skills & Assessments")
            skills_list = cand_data.get("skills", [])
            if skills_list:
                skill_names_p = [s["name"] for s in skills_list]
                skill_dur_p = [s.get("duration_months", 0) / 12.0 for s in skills_list]
                
                # Plotly horizontal bar chart of skills and durations
                fig_skill = px.bar(
                    x=skill_dur_p,
                    y=skill_names_p,
                    orientation="h",
                    labels={"x": "Duration (Years)", "y": "Skill"},
                    color=skill_dur_p,
                    color_continuous_scale="purples"
                )
                fig_skill.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font_color="#e2e8f0",
                    margin=dict(t=5, b=5, l=5, r=5),
                    coloraxis_showscale=False,
                    height=250
                )
                st.plotly_chart(fig_skill, use_container_width=True)
            else:
                st.info("No skills catalogued on this profile.")

        with p_col2:
            st.markdown("### Summary")
            st.write(cand_data.get("summary", "*No profile summary provided.*"))
            
            # Career history timeline
            st.markdown("### 💼 Career History Timeline")
            history = cand_data.get("career_history", [])
            if history:
                timeline_html = "<div class='timeline-container'>"
                for job in history:
                    co = job.get("company", "Unknown")
                    title = job.get("title", "Unknown")
                    dur_m = job.get("duration_months", 0)
                    dur_y = dur_m / 12.0
                    is_c = "Present" if job.get("is_current", False) else job.get("end_date", "N/A")
                    st_date = job.get("start_date", "N/A")
                    
                    timeline_html += f"""
                    <div class='timeline-item'>
                        <div class='timeline-title'>{title}</div>
                        <div class='timeline-subtitle'>{co} ({st_date} to {is_c} · {dur_y:.1f} yrs)</div>
                        <div class='timeline-desc'>{job.get('description', '')}</div>
                    </div>
                    """
                timeline_html += "</div>"
                st.markdown(timeline_html, unsafe_allow_html=True)
            else:
                st.info("No employment history provided.")

            # Education
            st.markdown("### 🎓 Education & Credentials")
            education = cand_data.get("education", [])
            if education:
                for edu in education:
                    inst = edu.get("institution", "Unknown")
                    deg = edu.get("degree", "Unknown")
                    fos = edu.get("field_of_study", "Unknown")
                    st_yr = edu.get("start_year", "N/A")
                    end_yr = edu.get("end_year", "N/A")
                    tier = edu.get("tier", "unknown").upper()
                    tier_badge = "badge-primary" if tier in ["TIER 1", "TIER_1", "1"] else "badge-info"
                    
                    st.markdown(f"""
                    **{deg} in {fos}**  
                    {inst} ({st_yr} - {end_yr}) · <span class="badge {tier_badge}">Tier: {tier}</span>
                    """, unsafe_allow_html=True)
            else:
                st.info("No educational qualifications catalogued.")
                
            # Certifications & Languages
            certs = cand_data.get("certifications", [])
            langs = cand_data.get("languages", [])
            
            c_c1, c_c2 = st.columns(2)
            with c_c1:
                st.markdown("### 📜 Certifications")
                if certs:
                    for cert in certs:
                        st.markdown(f"- {cert}")
                else:
                    st.info("No certifications listed.")
            with c_c2:
                st.markdown("### 🗣️ Languages")
                if langs:
                    for l in langs:
                        if isinstance(l, dict):
                            st.markdown(f"- **{l.get('language')}**: *{l.get('proficiency')}*")
                        else:
                            st.markdown(f"- {l}")
                else:
                    st.info("No languages listed.")

# ==========================================
# TAB 5: PIPELINE & CLEANING LOGIC
# ==========================================
with tabs[4]:
    st.markdown("## 🧼 Data Cleaning, Normalization & Quality Scoring Logic")
    st.markdown("""
    This tab details the backend data engineering pipeline implemented for the **RedRob Intelligent Candidate Discovery Challenge**.
    The codebase preprocesses raw candidate profiles, performs deduplication, normalizes job roles and skills, executes quality checks, and engineers scores.
    """)
    
    col_p1, col_p2 = st.columns(2)
    
    with col_p1:
        st.markdown("""
        ### 🔄 1. Deduplication Strategy
        To prevent bias in features and candidate discovery, a custom deduplication hash is generated for each profile:
        * **Hash Input**: Lowercased concatenation of: `anonymized_name` + `location` + first 150 characters of `summary`.
        * **Action**: Only the first instance of a duplicate profile is kept in the main pipeline. The duplicates are written to a separate audit log (`duplicate_records.jsonl`).
        """)
        
        st.markdown("""
        ### 📋 2. Job Title Normalization
        Raw titles from profiles and job histories are normalized to 8 standardized Tech & Business roles:
        1. **`AI_ML_ENGINEER`**: Matches machine learning, deep learning, artificial intelligence, CV, NLP, data science.
        2. **`DATA_SCIENTIST`**: Matches data scientist, applied scientist roles.
        3. **`DATA_ENGINEER`**: Matches big data, analytics engineer, ETL developer roles.
        4. **`RETRIEVAL_RANKING_ENGINEER`**: Matches search, ranking, recommender, information retrieval roles.
        5. **`DEVOPS_ENGINEER`**: Matches DevOps, SRE, site reliability, infrastructure, systems roles.
        6. **`PRODUCT_ENGINEER`**: Matches product manager, technical product manager (TPM), product owners.
        7. **`SOFTWARE_ENGINEER`**: Matches backend, frontend, full-stack, mobile, web developers.
        8. **`CONSULTANT`**: Matches consultant, advisory, partner roles.
        """)
        
    with col_p2:
        st.markdown("""
        ### 📊 3. Candidate Quality Scoring Rules
        Every candidate starts with a baseline Quality Score of **100**. Points are deducted for logical timeline anomalies and missing details:
        
        | Quality Check Flag | Rule / Logic | Penalty |
        | :--- | :--- | :---: |
        | **Missing Critical Info** | Empty summary, empty career history, or empty skills. | **-20 pts** |
        | **Impossible Timeline** | A job's start date is after its end date. | **-20 pts** |
        | **Inconsistent Experience** | Mismatch between profile experience vs sum of jobs in career history > 2 years. | **-15 pts** |
        | **Suspicious Skill Durations** | Reported duration of a skill exceeds total career experience by > 1.5 years. | **-15 pts** |
        | **Overlapping Jobs** | Non-current job timelines overlap by more than 90 days. | **-10 pts** |
        | **Worked Before College** | Candidate worked > 6 years before starting college (likely incorrect timeline). | **-10 pts** |
        """)
        
        st.markdown("""
        ### 🧠 4. Skill Normalization & Proficiency Scoring
        * **Normalization**: Maps thousands of variations (e.g. `ml`, `deep learning`, `artificial intelligence`) to a common key (e.g., `machine_learning`).
        * **Proficiency Scaling**: Converts descriptive proficiencies to numeric values for heuristic models:
          * `beginner` = 1.0 · `intermediate` = 2.0 · `advanced` = 3.0 · `expert` = 4.0
        * **Weighted Scores**: Calculates specific scores (e.g., Python score, LLM score) based on proficiency weights, endorsement counts, and skill durations.
        """)
