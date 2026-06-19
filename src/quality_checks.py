import logging
import datetime
import pandas as pd
from typing import List, Dict, Any, Tuple

logger = logging.getLogger(__name__)

def check_single_candidate_quality(candidate: Dict[str, Any]) -> Dict[str, Any]:
    """
    Evaluates data anomalies for a single candidate.
    Calculates flags and a quality_score out of 100.
    """
    candidate_id = candidate.get("candidate_id", "Unknown")
    
    # 1. Fetch values
    profile_exp = float(candidate.get("profile_years_experience", 0.0))
    history_exp = float(candidate.get("history_years_experience", 0.0))
    summary = candidate.get("summary", "")
    skills = candidate.get("skills", [])
    career_history = candidate.get("career_history", [])
    education = candidate.get("education", [])
    
    # Flags
    has_inconsistent_experience = False
    has_impossible_timeline = False
    has_overlapping_jobs = False
    has_suspicious_skill_durations = False
    has_missing_critical_info = False
    has_timeline_education_anomaly = False
    
    # List of issues detected for logging/debugging
    issues_detected = []
    
    # 2. Check: Missing critical information
    if not summary or len(summary.strip()) < 10:
        has_missing_critical_info = True
        issues_detected.append("Profile summary is missing or extremely short.")
    if not skills:
        has_missing_critical_info = True
        issues_detected.append("No skills listed on the profile.")
    if not career_history:
        has_missing_critical_info = True
        issues_detected.append("Career history is empty.")
        
    # 3. Check: Inconsistent experience
    # If the difference between profile experience and sum of history durations is > 2.0 years
    if career_history:
        diff = abs(profile_exp - history_exp)
        if diff > 2.0:
            has_inconsistent_experience = True
            issues_detected.append(f"Inconsistent experience: profile states {profile_exp} yrs, history indicates {history_exp} yrs (diff {round(diff, 2)} yrs).")

    # 4. Check: Impossible timelines & overlaps
    parsed_jobs = []
    for job in career_history:
        s_str = job.get("start_date")
        e_str = job.get("end_date")
        is_curr = job.get("is_current", False)
        
        if s_str:
            try:
                s_date = datetime.datetime.strptime(s_str, "%Y-%m-%d").date()
                e_date = datetime.datetime.strptime(e_str, "%Y-%m-%d").date() if e_str else datetime.date.today()
                
                parsed_jobs.append((s_date, e_date, job.get("company"), job.get("title"), is_curr))
                
                # Check start date after end date
                if s_date > e_date:
                    has_impossible_timeline = True
                    issues_detected.append(f"Impossible timeline: job at {job.get('company')} start date ({s_str}) is after end date ({e_str}).")
            except ValueError:
                pass
                
    # Check overlaps between non-concurrent roles
    # Sort jobs by start date
    parsed_jobs.sort(key=lambda x: x[0])
    for i in range(len(parsed_jobs) - 1):
        job1_start, job1_end, c1, t1, curr1 = parsed_jobs[i]
        job2_start, job2_end, c2, t2, curr2 = parsed_jobs[i+1]
        
        # If job1 ends after job2 starts (meaning they overlap) and both are not current roles
        if job1_end > job2_start and not curr1 and not curr2:
            # Check overlap size in days. If overlap is > 90 days (3 months), flag it
            overlap_days = (job1_end - job2_start).days
            if overlap_days > 90:
                has_overlapping_jobs = True
                issues_detected.append(f"Overlapping jobs: role at {c1} ({t1}) overlaps with {c2} ({t2}) by {overlap_days} days.")

    # 5. Check: Timeline vs Education (worked way before college)
    edu_years = [edu.get("start_year") for edu in education if edu.get("start_year")]
    if edu_years and parsed_jobs:
        earliest_edu_year = min(edu_years)
        earliest_job_start = parsed_jobs[0][0]
        
        # If they started working more than 6 years before starting college, it's anomalous
        if earliest_job_start.year < earliest_edu_year - 6:
            has_timeline_education_anomaly = True
            issues_detected.append(f"Timeline-education anomaly: candidate worked in {earliest_job_start.year} but started college in {earliest_edu_year}.")

    # 6. Check: Suspicious skill durations
    total_exp = max(profile_exp, history_exp)
    for s in skills:
        dur_months = s.get("duration_months", 0)
        dur_years = dur_months / 12.0
        
        # Skill duration exceeds total experience by more than 1.5 years
        if dur_years > total_exp + 1.5:
            has_suspicious_skill_durations = True
            issues_detected.append(f"Suspicious skill duration: '{s.get('name')}' duration is {round(dur_years, 1)} yrs, but total experience is {round(total_exp, 1)} yrs.")
            
        # Skill duration is unreasonably long (e.g. > 40 years)
        if dur_years > 40.0:
            has_suspicious_skill_durations = True
            issues_detected.append(f"Suspicious skill duration: '{s.get('name')}' duration is {round(dur_years, 1)} yrs (unreasonably high).")

    # 7. Compute Quality Score
    quality_score = 100
    
    if has_missing_critical_info:
        quality_score -= 20
    if has_impossible_timeline:
        quality_score -= 20
    if has_inconsistent_experience:
        quality_score -= 15
    if has_suspicious_skill_durations:
        quality_score -= 15
    if has_overlapping_jobs:
        quality_score -= 10
    if has_timeline_education_anomaly:
        quality_score -= 10
        
    quality_score = max(0, quality_score)
    
    return {
        "candidate_id": candidate_id,
        "quality_score": int(quality_score),
        "inconsistent_experience_flag": has_inconsistent_experience,
        "impossible_timeline_flag": has_impossible_timeline,
        "overlapping_jobs_flag": has_overlapping_jobs,
        "worked_before_college_flag": has_timeline_education_anomaly,
        "suspicious_skill_durations_flag": has_suspicious_skill_durations,
        "missing_critical_info_flag": has_missing_critical_info,
        "issues": "; ".join(issues_detected) if issues_detected else "No issues detected"
    }

def check_dataset_quality(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies quality checks to the entire DataFrame.
    Returns a DataFrame containing candidate quality metrics.
    """
    logger.info("Performing data quality checks...")
    
    quality_records = []
    for _, row in df.iterrows():
        quality_info = check_single_candidate_quality(row.to_dict())
        quality_records.append(quality_info)
        
    quality_df = pd.DataFrame(quality_records)
    
    # Calculate dataset quality summary
    avg_score = quality_df["quality_score"].mean()
    anomalies_count = (quality_df["quality_score"] < 100).sum()
    logger.info(f"Quality checks complete. Average quality score: {avg_score:.2f}/100. Anomalies found: {anomalies_count} records.")
    
    return quality_df
