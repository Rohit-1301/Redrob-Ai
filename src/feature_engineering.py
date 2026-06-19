import math
import logging
import pandas as pd
from typing import List, Dict, Any
from src.config import SKILL_SETS, PRODUCT_INDUSTRIES, CONSULTING_COMPANIES

logger = logging.getLogger(__name__)

def compute_skill_score(skills: List[Dict[str, Any]], target_keywords: List[str]) -> float:
    """
    Computes a normalized score (0.0 to 1.0) for a group of target skills.
    Considers proficiency, endorsements, and duration.
    """
    if not skills:
        return 0.0
        
    proficiency_weights = {"beginner": 0.25, "intermediate": 0.5, "advanced": 0.75, "expert": 1.0}
    raw_score = 0.0
    
    for s in skills:
        name = s["name"]
        
        # Check if the skill matches any target keyword (direct or substring match)
        if any(keyword in name for keyword in target_keywords):
            prof_val = proficiency_weights.get(s.get("proficiency", "beginner"), 0.25)
            endorsements = s.get("endorsements", 0)
            duration_months = s.get("duration_months", 0)
            
            # log-scale endorsements factor to avoid extreme values
            endorsement_factor = 1.0 + math.log1p(endorsements)
            
            # duration in years (offset by 0.5 to give credit for presence)
            duration_factor = (duration_months / 12.0) + 0.5
            
            raw_score += prof_val * endorsement_factor * duration_factor
            
    # Normalize and cap the score between 0.0 and 1.0 (scaling by 10.0 as baseline)
    return min(1.0, raw_score / 10.0)

def extract_features_for_candidate(candidate: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generates engineered features for a single candidate profile.
    """
    candidate_id = candidate.get("candidate_id", "Unknown")
    
    # 1. Experience Features
    years_exp = float(candidate.get("profile_years_experience", 0.0))
    career_history = candidate.get("career_history", [])
    
    # AI Experience
    ai_keywords = ["machine learning", "ml", "artificial intelligence", "ai", "deep learning", 
                   "nlp", "computer vision", "llm", "transformers", "data scientist"]
    ai_duration_months = 0
    for job in career_history:
        title = job.get("title", "").lower()
        norm_title = job.get("normalized_title", "")
        desc = job.get("description", "").lower()
        
        is_ai_role = (
            norm_title in ["ML_ENGINEER", "DATA_SCIENTIST"] or
            any(kw in title for kw in ai_keywords) or
            any(kw in desc for kw in ai_keywords)
        )
        if is_ai_role:
            ai_duration_months += job.get("duration_months", 0)
            
    ai_experience_years = round(ai_duration_months / 12.0, 2)
    
    # Job Hop Score: 1.0 - (avg tenure in months / 36.0), capped between 0 and 1
    total_months = sum(job.get("duration_months", 0) for job in career_history)
    num_jobs = len(career_history)
    
    if num_jobs > 0:
        avg_tenure = total_months / num_jobs
        job_hop_score = round(max(0.0, min(1.0, 1.0 - (avg_tenure / 36.0))), 2)
    else:
        job_hop_score = 0.0
        
    # Promotion Score: number of jobs minus unique companies worked at (internal moves)
    companies = [job.get("company", "").lower() for job in career_history if job.get("company")]
    unique_companies = set(companies)
    promotion_score = max(0, len(companies) - len(unique_companies))
    
    # 2. Skills Features
    skills = candidate.get("skills", [])
    python_score = round(compute_skill_score(skills, SKILL_SETS["python"]), 2)
    retrieval_score = round(compute_skill_score(skills, SKILL_SETS["retrieval"]), 2)
    ranking_score = round(compute_skill_score(skills, SKILL_SETS["ranking"]), 2)
    vector_db_score = round(compute_skill_score(skills, SKILL_SETS["vector_db"]), 2)
    llm_score = round(compute_skill_score(skills, SKILL_SETS["llm"]), 2)
    
    # 3. Company Features
    product_months = 0
    startup_months = 0
    consulting_months = 0
    
    for job in career_history:
        dur = job.get("duration_months", 0)
        industry = job.get("industry", "")
        comp_size = job.get("company_size", "")
        company_name = job.get("company", "").lower()
        
        # Product company
        if industry in PRODUCT_INDUSTRIES:
            product_months += dur
            
        # Startup
        if comp_size in ["1-10", "11-50", "51-200"]:
            startup_months += dur
            
        # Consulting
        is_consulting = (
            "consulting" in industry.lower() or
            any(c in company_name for c in CONSULTING_COMPANIES)
        )
        if is_consulting:
            consulting_months += dur
            
    total_dur = max(1, total_months)
    product_company_score = round(product_months / total_dur, 2)
    startup_score = round(startup_months / total_dur, 2)
    consulting_company_score = round(consulting_months / total_dur, 2)
    
    # 4. Behavioral Features
    # Engagement score: completeness, connections, endorsements
    completeness = float(candidate.get("profile_completeness_score", 0.0)) / 100.0
    connections = float(candidate.get("connection_count", 0))
    endorsements = float(candidate.get("endorsements_received", 0))
    
    conn_factor = min(1.0, connections / 500.0)
    end_factor = min(1.0, endorsements / 100.0)
    engagement_score = round(0.4 * completeness + 0.3 * conn_factor + 0.3 * end_factor, 2)
    
    # Recruiter Response score
    response_rate = float(candidate.get("recruiter_response_rate", 0.0))
    response_time = float(candidate.get("avg_response_time_hours", 0.0))
    # Low response time is good. Cap at 168 hours (1 week)
    time_factor = max(0.0, min(1.0, 1.0 - (response_time / 168.0)))
    recruiter_response_score = round(0.6 * response_rate + 0.4 * time_factor, 2)
    
    # Activity score: views, applications, appearances, saves
    views = float(candidate.get("profile_views_received_30d", 0))
    apps = float(candidate.get("applications_submitted_30d", 0))
    search = float(candidate.get("search_appearance_30d", 0))
    saves = float(candidate.get("saved_by_recruiters_30d", 0))
    
    views_factor = min(1.0, views / 50.0)
    apps_factor = min(1.0, apps / 20.0)
    search_factor = min(1.0, search / 500.0)
    saves_factor = min(1.0, saves / 20.0)
    activity_score = round(0.25 * views_factor + 0.25 * apps_factor + 0.25 * search_factor + 0.25 * saves_factor, 2)
    
    return {
        "candidate_id": candidate_id,
        "years_experience": years_exp,
        "ai_experience_years": ai_experience_years,
        "job_hop_score": job_hop_score,
        "promotion_score": promotion_score,
        
        "python_score": python_score,
        "retrieval_score": retrieval_score,
        "ranking_score": ranking_score,
        "vector_db_score": vector_db_score,
        "llm_score": llm_score,
        
        "product_company_score": product_company_score,
        "startup_score": startup_score,
        "consulting_company_score": consulting_company_score,
        
        "engagement_score": engagement_score,
        "recruiter_response_score": recruiter_response_score,
        "activity_score": activity_score
    }

def engineer_features_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes features for all candidates in the DataFrame and returns a new DataFrame.
    """
    logger.info("Engineering candidate features...")
    
    feature_records = []
    for _, row in df.iterrows():
        features = extract_features_for_candidate(row.to_dict())
        feature_records.append(features)
        
    features_df = pd.DataFrame(feature_records)
    return features_df
