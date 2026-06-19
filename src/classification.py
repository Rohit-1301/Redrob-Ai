import re
import logging
import pandas as pd
from typing import List, Dict, Any, Tuple
from src.config import CLASSIFICATION_KEYWORDS, CONSULTING_COMPANIES, MIN_TECH_SKILL_COUNT

logger = logging.getLogger(__name__)

def evaluate_candidate_scores(candidate: Dict[str, Any]) -> Dict[str, float]:
    """
    Computes classification scores for a single candidate across tech and non-tech dimensions.
    """
    # 1. Combine profile texts for keyword searching
    headline = candidate.get("headline", "").lower()
    summary = candidate.get("summary", "").lower()
    
    career_history = candidate.get("career_history", [])
    history_titles = [job.get("title", "").lower() for job in career_history]
    history_norm_titles = [job.get("normalized_title", "") for job in career_history]
    history_desc = [job.get("description", "").lower() for job in career_history]
    
    combined_text = " ".join([headline, summary] + history_titles + history_desc)
    
    # 2. Extract skill names & proficiencies
    skills = candidate.get("skills", [])
    skill_names = [s["name"] for s in skills]
    
    # Proficiency weights: beginner=1, intermediate=2, advanced=3, expert=4
    proficiency_map = {"beginner": 1.0, "intermediate": 2.0, "advanced": 3.0, "expert": 4.0}
    skill_weights = {s["name"]: proficiency_map.get(s["proficiency"], 1.0) for s in skills}
    
    scores = {
        "AI_ML_ENGINEER": 0.0,
        "RETRIEVAL_RANKING_ENGINEER": 0.0,
        "DATA_ENGINEER": 0.0,
        "DEVOPS_ENGINEER": 0.0,
        "PRODUCT_ENGINEER": 0.0,
        "SOFTWARE_ENGINEER": 0.0,
        "CONSULTING_ONLY": 0.0,
    }
    
    # Check normalized current title
    curr_title_norm = candidate.get("normalized_current_title", "")
    
    # ==========================================================
    # Score AI_ML_ENGINEER
    # ==========================================================
    # Current Title Bonus
    if curr_title_norm in ["ML_ENGINEER", "DATA_SCIENTIST"]:
        scores["AI_ML_ENGINEER"] += 20.0
    # Historic Titles Bonus
    for title in history_norm_titles:
        if title in ["ML_ENGINEER", "DATA_SCIENTIST"]:
            scores["AI_ML_ENGINEER"] += 5.0
            
    # Skill matches
    ai_skills = CLASSIFICATION_KEYWORDS["AI_ML_ENGINEER"]
    for s_name in skill_names:
        # Check direct match or substring match
        if any(keyword in s_name for keyword in ai_skills):
            scores["AI_ML_ENGINEER"] += 4.0 * skill_weights.get(s_name, 1.0)
            
    # Keyword text matches
    for keyword in ai_skills:
        # Avoid double-counting skills but search descriptions/summaries
        matches = len(re.findall(rf"\b{re.escape(keyword)}\b", combined_text))
        scores["AI_ML_ENGINEER"] += min(5.0, matches * 0.5)
        
    # ==========================================================
    # Score RETRIEVAL_RANKING_ENGINEER
    # ==========================================================
    if curr_title_norm == "RETRIEVAL_RANKING_ENGINEER":
        scores["RETRIEVAL_RANKING_ENGINEER"] += 20.0
    for title in history_norm_titles:
        if title == "RETRIEVAL_RANKING_ENGINEER":
            scores["RETRIEVAL_RANKING_ENGINEER"] += 5.0
            
    retrieval_skills = CLASSIFICATION_KEYWORDS["RETRIEVAL_RANKING_ENGINEER"]
    for s_name in skill_names:
        if any(keyword in s_name for keyword in retrieval_skills):
            scores["RETRIEVAL_RANKING_ENGINEER"] += 5.0 * skill_weights.get(s_name, 1.0)
            
    for keyword in retrieval_skills:
        matches = len(re.findall(rf"\b{re.escape(keyword)}\b", combined_text))
        scores["RETRIEVAL_RANKING_ENGINEER"] += min(5.0, matches * 0.5)

    # ==========================================================
    # Score DATA_ENGINEER
    # ==========================================================
    if curr_title_norm == "DATA_ENGINEER":
        scores["DATA_ENGINEER"] += 20.0
    for title in history_norm_titles:
        if title == "DATA_ENGINEER":
            scores["DATA_ENGINEER"] += 5.0
            
    data_skills = CLASSIFICATION_KEYWORDS["DATA_ENGINEER"]
    for s_name in skill_names:
        if any(keyword in s_name for keyword in data_skills):
            scores["DATA_ENGINEER"] += 4.0 * skill_weights.get(s_name, 1.0)
            
    for keyword in data_skills:
        matches = len(re.findall(rf"\b{re.escape(keyword)}\b", combined_text))
        scores["DATA_ENGINEER"] += min(5.0, matches * 0.5)

    # ==========================================================
    # Score DEVOPS_ENGINEER
    # ==========================================================
    if curr_title_norm == "DEVOPS_ENGINEER":
        scores["DEVOPS_ENGINEER"] += 20.0
    for title in history_norm_titles:
        if title == "DEVOPS_ENGINEER":
            scores["DEVOPS_ENGINEER"] += 5.0
            
    devops_skills = CLASSIFICATION_KEYWORDS["DEVOPS_ENGINEER"]
    for s_name in skill_names:
        if any(keyword in s_name for keyword in devops_skills):
            scores["DEVOPS_ENGINEER"] += 4.0 * skill_weights.get(s_name, 1.0)
            
    for keyword in devops_skills:
        matches = len(re.findall(rf"\b{re.escape(keyword)}\b", combined_text))
        scores["DEVOPS_ENGINEER"] += min(5.0, matches * 0.5)

    # ==========================================================
    # Score PRODUCT_ENGINEER
    # ==========================================================
    if curr_title_norm == "PRODUCT_ENGINEER":
        scores["PRODUCT_ENGINEER"] += 20.0
    for title in history_norm_titles:
        if title == "PRODUCT_ENGINEER":
            scores["PRODUCT_ENGINEER"] += 5.0
            
    prod_skills = CLASSIFICATION_KEYWORDS["PRODUCT_ENGINEER"]
    for s_name in skill_names:
        if any(keyword in s_name for keyword in prod_skills):
            scores["PRODUCT_ENGINEER"] += 3.0 * skill_weights.get(s_name, 1.0)
            
    for keyword in prod_skills:
        matches = len(re.findall(rf"\b{re.escape(keyword)}\b", combined_text))
        scores["PRODUCT_ENGINEER"] += min(5.0, matches * 0.5)

    # ==========================================================
    # Score SOFTWARE_ENGINEER
    # ==========================================================
    if curr_title_norm == "SOFTWARE_ENGINEER":
        scores["SOFTWARE_ENGINEER"] += 15.0
    for title in history_norm_titles:
        if title == "SOFTWARE_ENGINEER":
            scores["SOFTWARE_ENGINEER"] += 3.0
            
    se_skills = CLASSIFICATION_KEYWORDS["SOFTWARE_ENGINEER"]
    # Generic SE skills (languages, git)
    generic_se = ["python", "javascript", "typescript", "nodejs", "java", "cpp", "csharp", "go", "rust", "git", "html", "css", "sql"]
    for s_name in skill_names:
        if s_name in generic_se or any(keyword in s_name for keyword in se_skills):
            scores["SOFTWARE_ENGINEER"] += 2.0 * skill_weights.get(s_name, 1.0)
            
    for keyword in se_skills:
        matches = len(re.findall(rf"\b{re.escape(keyword)}\b", combined_text))
        scores["SOFTWARE_ENGINEER"] += min(5.0, matches * 0.3)

    # ==========================================================
    # Score CONSULTING_ONLY
    # ==========================================================
    curr_company = candidate.get("current_company", "").lower()
    
    if curr_title_norm == "CONSULTANT":
        scores["CONSULTING_ONLY"] += 25.0
    elif any(comp in curr_company for comp in CONSULTING_COMPANIES):
        scores["CONSULTING_ONLY"] += 15.0
        
    for title in history_titles:
        if "consultant" in title or "advisor" in title or "consulting" in title:
            scores["CONSULTING_ONLY"] += 5.0
            
    consulting_keywords = ["consultant", "consulting", "advisory", "advisor", "strategy consultant", "management consulting"]
    for keyword in consulting_keywords:
        matches = len(re.findall(rf"\b{re.escape(keyword)}\b", combined_text))
        scores["CONSULTING_ONLY"] += min(10.0, matches * 1.5)

    return scores

def classify_candidate(candidate: Dict[str, Any]) -> str:
    """
    Classifies a candidate into one of the 8 segments based on computed heuristic scores.
    """
    scores = evaluate_candidate_scores(candidate)
    
    # Determine general technical viability
    skills = candidate.get("skills", [])
    current_title = candidate.get("current_title", "").lower()
    norm_current_title = candidate.get("normalized_current_title", "")
    
    # List of technical skills
    tech_skills_count = 0
    all_tech_keywords = []
    for val in CLASSIFICATION_KEYWORDS.values():
        all_tech_keywords.extend(val)
    all_tech_keywords.extend(["python", "javascript", "typescript", "nodejs", "java", "cpp", "csharp", "go", "rust", "git", "sql"])
    
    for s in skills:
        s_name = s["name"]
        if any(kw in s_name for kw in all_tech_keywords):
            tech_skills_count += 1
            
    # Check if candidate is non-technical (Marketing, HR, Accountant, Graphic Design, Customer Support, etc.)
    non_tech_roles = [
        "hr manager", "human resources", "operations manager", "marketing manager",
        "graphic designer", "accountant", "project manager", "sales executive",
        "customer support", "civil engineer", "mechanical engineer", "support specialist"
    ]
    
    # Strong indicator for non-tech if they are in these roles and have very low tech skill profile
    is_non_tech_role = any(role in current_title for role in non_tech_roles)
    
    # 1. Check for CONSULTING_ONLY first
    # If they are currently in a consultant role or have massive consulting scores and weak tech scores
    if scores["CONSULTING_ONLY"] >= 20.0 and (tech_skills_count < MIN_TECH_SKILL_COUNT or is_non_tech_role):
        return "CONSULTING_ONLY"
        
    # 2. Check for IRRELEVANT
    # If they are in a non-tech role and have very low tech skills count
    if is_non_tech_role and tech_skills_count < MIN_TECH_SKILL_COUNT:
        # Double check that we don't accidentally categorize real backend developers with non-tech titles
        max_tech_score = max(scores[cat] for cat in ["AI_ML_ENGINEER", "RETRIEVAL_RANKING_ENGINEER", "DATA_ENGINEER", "DEVOPS_ENGINEER", "PRODUCT_ENGINEER", "SOFTWARE_ENGINEER"])
        if max_tech_score < 8.0:
            return "IRRELEVANT"
            
    # 3. Choose the highest technical segment score
    tech_segments = [
        "RETRIEVAL_RANKING_ENGINEER",
        "AI_ML_ENGINEER",
        "DATA_ENGINEER",
        "DEVOPS_ENGINEER",
        "PRODUCT_ENGINEER",
        "SOFTWARE_ENGINEER"
    ]
    
    # Find the segment with the highest score
    best_segment = "SOFTWARE_ENGINEER"
    best_score = -1.0
    
    for segment in tech_segments:
        score = scores[segment]
        if score > best_score:
            best_score = score
            best_segment = segment
            
    # Fallback to IRRELEVANT if even software engineer has no matching signals and they have very low tech skills
    if best_score < 4.0 and tech_skills_count == 0:
        return "IRRELEVANT"
        
    return best_segment

def classify_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies candidate classification to the entire DataFrame.
    """
    logger.info("Classifying candidates...")
    
    classifications = []
    for _, row in df.iterrows():
        classifications.append(classify_candidate(row.to_dict()))
        
    df["candidate_segment"] = classifications
    
    # Summarize classification stats
    summary = df["candidate_segment"].value_counts()
    logger.info(f"Classification distribution:\n{summary.to_string()}")
    
    return df
