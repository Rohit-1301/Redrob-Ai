import datetime
import hashlib
import logging
import pandas as pd
from typing import List, Dict, Any, Tuple
from src.config import NON_TECH_TITLES
from src.normalization import normalize_title, normalize_skills

logger = logging.getLogger(__name__)

# Standard execution date representing "current time" for this dataset (based on 2026 logs)
CURRENT_DATE_STR = "2026-06-19"
CURRENT_DATE = datetime.datetime.strptime(CURRENT_DATE_STR, "%Y-%m-%d").date()

def parse_date(date_str: Any) -> Any:
    """
    Parses date strings to standard ISO date format (YYYY-MM-DD) or returns None if invalid/null.
    """
    if not date_str or not isinstance(date_str, str):
        return None
    try:
        # Standard formats: YYYY-MM-DD
        return datetime.datetime.strptime(date_str.strip()[:10], "%Y-%m-%d").date()
    except ValueError:
        # Try other common formats if any
        for fmt in ("%Y/%m/%d", "%d-%m-%Y", "%m/%d/%Y"):
            try:
                return datetime.datetime.strptime(date_str.strip(), fmt).date()
            except ValueError:
                pass
    return None

def calculate_duration_months(start_date: Any, end_date: Any, is_current: bool = False) -> int:
    """
    Calculates duration in months between start_date and end_date.
    If end_date is None and is_current is True, it uses CURRENT_DATE.
    """
    s_date = parse_date(start_date)
    e_date = parse_date(end_date)
    
    if not s_date:
        return 0
        
    if not e_date:
        if is_current:
            e_date = CURRENT_DATE
        else:
            return 0
            
    # Calculate difference in months
    diff = (e_date.year - s_date.year) * 12 + (e_date.month - s_date.month)
    return max(0, diff)

def clean_career_history(history: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Normalizes dates, titles, and calculates correct durations for career history.
    """
    if not history or not isinstance(history, list):
        return []
        
    cleaned_history = []
    for job in history:
        if not isinstance(job, dict):
            continue
            
        start_raw = job.get("start_date")
        end_raw = job.get("end_date")
        is_current = bool(job.get("is_current", False))
        
        s_date = parse_date(start_raw)
        e_date = parse_date(end_raw)
        
        # Recalculate duration_months to ensure correctness
        calculated_months = calculate_duration_months(s_date, e_date, is_current)
        
        cleaned_job = {
            "company": str(job.get("company", "Unknown")).strip(),
            "title": str(job.get("title", "Unknown")).strip(),
            "normalized_title": normalize_title(str(job.get("title", ""))),
            "start_date": str(s_date) if s_date else None,
            "end_date": str(e_date) if e_date else None,
            "is_current": is_current,
            "duration_months": calculated_months,
            "industry": str(job.get("industry", "Unknown")).strip(),
            "company_size": str(job.get("company_size", "Unknown")).strip(),
            "description": str(job.get("description", "")).strip()
        }
        cleaned_history.append(cleaned_job)
        
    # Sort history: current job first, then by start date descending
    cleaned_history.sort(key=lambda x: (x["is_current"], x["start_date"] or ""), reverse=True)
    return cleaned_history

def clean_education(education: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Standardizes education details, checking for valid years and tiers.
    """
    if not education or not isinstance(education, list):
        return []
        
    cleaned_edu = []
    for edu in education:
        if not isinstance(edu, dict):
            continue
            
        cleaned_edu.append({
            "institution": str(edu.get("institution", "Unknown")).strip(),
            "degree": str(edu.get("degree", "Unknown")).strip(),
            "field_of_study": str(edu.get("field_of_study", "Unknown")).strip(),
            "start_year": int(edu.get("start_year")) if edu.get("start_year") is not None else None,
            "end_year": int(edu.get("end_year")) if edu.get("end_year") is not None else None,
            "grade": str(edu.get("grade", "")) if edu.get("grade") is not None else None,
            "tier": str(edu.get("tier", "unknown")).strip().lower()
        })
    return cleaned_edu

def clean_candidate_record(record: Dict[str, Any]) -> Dict[str, Any]:
    """
    Flattens and cleans a single candidate record dictionary.
    """
    candidate_id = record.get("candidate_id", "Unknown")
    
    # Extract & clean profile
    profile = record.get("profile", {})
    years_exp = profile.get("years_of_experience", 0.0)
    
    # Extract & clean signals
    signals = record.get("redrob_signals", {})
    expected_salary = signals.get("expected_salary_range_inr_lpa", {})
    
    # Clean career history, skills, education
    raw_history = record.get("career_history", [])
    cleaned_history = clean_career_history(raw_history)
    
    raw_skills = record.get("skills", [])
    normalized_skills = normalize_skills(raw_skills)
    
    raw_edu = record.get("education", [])
    cleaned_edu = clean_education(raw_edu)
    
    # Parse certs & languages
    certs = record.get("certifications", [])
    languages = record.get("languages", [])

    # Calculate actual experience from history to double check
    history_exp_years = sum(job["duration_months"] for job in cleaned_history) / 12.0

    # Build the flattened clean candidate object
    cleaned_record = {
        "candidate_id": candidate_id,
        "anonymized_name": str(profile.get("anonymized_name", "Unknown")).strip(),
        "headline": str(profile.get("headline", "")).strip(),
        "summary": str(profile.get("summary", "")).strip(),
        "location": str(profile.get("location", "Unknown")).strip(),
        "country": str(profile.get("country", "Unknown")).strip(),
        "profile_years_experience": float(years_exp) if years_exp is not None else 0.0,
        "history_years_experience": round(history_exp_years, 2),
        "current_title": str(profile.get("current_title", "Unknown")).strip(),
        "normalized_current_title": normalize_title(str(profile.get("current_title", ""))),
        "current_company": str(profile.get("current_company", "Unknown")).strip(),
        "current_company_size": str(profile.get("current_company_size", "Unknown")).strip(),
        "current_industry": str(profile.get("current_industry", "Unknown")).strip(),
        
        # Nested fields (kept as clean structured objects)
        "career_history": cleaned_history,
        "skills": normalized_skills,
        "education": cleaned_edu,
        "certifications": certs,
        "languages": languages,
        
        # Flattened signals
        "profile_completeness_score": float(signals.get("profile_completeness_score", 0.0)),
        "signup_date": str(parse_date(signals.get("signup_date"))) if signals.get("signup_date") else None,
        "last_active_date": str(parse_date(signals.get("last_active_date"))) if signals.get("last_active_date") else None,
        "open_to_work_flag": bool(signals.get("open_to_work_flag", False)),
        "profile_views_received_30d": int(signals.get("profile_views_received_30d", 0)),
        "applications_submitted_30d": int(signals.get("applications_submitted_30d", 0)),
        "recruiter_response_rate": float(signals.get("recruiter_response_rate", 0.0)),
        "avg_response_time_hours": float(signals.get("avg_response_time_hours", 0.0)),
        "skill_assessment_scores": signals.get("skill_assessment_scores", {}),
        "connection_count": int(signals.get("connection_count", 0)),
        "endorsements_received": int(signals.get("endorsements_received", 0)),
        "notice_period_days": int(signals.get("notice_period_days", 0)),
        "expected_salary_min": float(expected_salary.get("min", 0.0)) if expected_salary else 0.0,
        "expected_salary_max": float(expected_salary.get("max", 0.0)) if expected_salary else 0.0,
        "preferred_work_mode": str(signals.get("preferred_work_mode", "flexible")).strip().lower(),
        "willing_to_relocate": bool(signals.get("willing_to_relocate", False)),
        "github_activity_score": float(signals.get("github_activity_score", -1.0)),
        "search_appearance_30d": int(signals.get("search_appearance_30d", 0)),
        "saved_by_recruiters_30d": int(signals.get("saved_by_recruiters_30d", 0)),
        "interview_completion_rate": float(signals.get("interview_completion_rate", 0.0)),
        "offer_acceptance_rate": float(signals.get("offer_acceptance_rate", -1.0)),
        "verified_email": bool(signals.get("verified_email", False)),
        "verified_phone": bool(signals.get("verified_phone", False)),
        "linkedin_connected": bool(signals.get("linkedin_connected", False))
    }
    
    return cleaned_record

def generate_dup_hash(record: Dict[str, Any]) -> str:
    """
    Generates a hash based on candidate details to detect duplicates.
    We combine name, location, and the summary text (first 150 chars) or career companies.
    """
    name = str(record.get("anonymized_name", "")).strip().lower()
    loc = str(record.get("location", "")).strip().lower()
    sum_snippet = str(record.get("summary", ""))[:150].strip().lower()
    
    # Hash it
    raw_str = f"{name}|{loc}|{sum_snippet}"
    return hashlib.mdxd5(raw_str.encode("utf-8")).hexdigest() if hasattr(hashlib, "mdxd5") else hashlib.md5(raw_str.encode("utf-8")).hexdigest()

def clean_dataset(df_raw: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Cleans the raw candidates DataFrame.
    Returns:
    - cleaned_df: Cleaned and flattened DataFrame.
    - duplicates_df: DataFrame containing the identified duplicate records.
    """
    logger.info("Cleaning records and flattening JSON...")
    
    # Process each row
    cleaned_records = []
    for _, row in df_raw.iterrows():
        # Keep it as dictionary to pass to cleaning function
        record_dict = row.to_dict()
        cleaned_rec = clean_candidate_record(record_dict)
        cleaned_records.append(cleaned_rec)
        
    cleaned_df = pd.DataFrame(cleaned_records)
    
    # Duplicate detection
    logger.info("Detecting duplicates...")
    # Add a custom hash for grouping
    cleaned_df["dup_hash"] = cleaned_df.apply(generate_dup_hash, axis=1)
    
    # Identify duplicate rows (we keep the first one, flag the rest)
    cleaned_df["is_duplicate"] = cleaned_df.duplicated(subset=["dup_hash"], keep="first")
    
    duplicates_df = cleaned_df[cleaned_df["is_duplicate"]].copy()
    
    # Remove the temporary dup_hash column
    cleaned_df = cleaned_df.drop(columns=["dup_hash"])
    duplicates_df = duplicates_df.drop(columns=["dup_hash"])
    
    logger.info(f"Deduplication complete. Found {len(duplicates_df)} duplicate records.")
    return cleaned_df, duplicates_df
