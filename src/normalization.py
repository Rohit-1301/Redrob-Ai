import re
from typing import List, Dict, Any
from src.config import TITLE_MAPPINGS, SKILL_MAPPINGS

def normalize_title(title: str) -> str:
    """
    Standardizes job titles to standard categories.
    Returns standard categories: ML_ENGINEER, DATA_ENGINEER, RETRIEVAL_RANKING_ENGINEER,
    DEVOPS_ENGINEER, PRODUCT_ENGINEER, SOFTWARE_ENGINEER, DATA_SCIENTIST, CONSULTANT, or OTHER.
    """
    if not title or not isinstance(title, str):
        return "OTHER"
        
    title_clean = title.strip().lower()
    
    # Check regular expression mappings
    for pattern, std_title in TITLE_MAPPINGS.items():
        if re.search(pattern, title_clean):
            return std_title
            
    # Fallback to OTHER if no matches
    return "OTHER"

def clean_skill_name(name: str) -> str:
    """
    Cleans and standardizes raw skill names.
    - Matches against SKILL_MAPPINGS config.
    - If not found, lowercases, replaces special chars, and normalizes spacing.
    """
    if not name or not isinstance(name, str):
        return "unknown"
        
    name_clean = name.strip().lower()
    
    # Check exact mapped translation
    if name_clean in SKILL_MAPPINGS:
        return SKILL_MAPPINGS[name_clean]
        
    # Generic normalization if not in standard mapping
    # Replace & with and
    name_clean = name_clean.replace("&", "and")
    # Replace non-alphanumeric (except underscores and spaces) with space
    name_clean = re.sub(r"[^a-z0-9_\-\s]", "", name_clean)
    # Replace hyphens and spaces with underscores
    name_clean = re.sub(r"[\s\-]+", "_", name_clean)
    # Strip leading/trailing underscores
    name_clean = name_clean.strip("_")
    
    return name_clean or "unknown"

def normalize_skills(skills_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Takes a raw list of skill dictionaries and returns normalized skill dicts.
    Each skill in raw input contains: name, proficiency, endorsements, duration_months
    """
    if not skills_list or not isinstance(skills_list, list):
        return []
        
    normalized = []
    for skill in skills_list:
        if not isinstance(skill, dict) or "name" not in skill:
            continue
        
        normalized_skill = {
            "name": clean_skill_name(skill["name"]),
            "proficiency": skill.get("proficiency", "beginner").strip().lower(),
            "endorsements": int(skill.get("endorsements", 0)),
            "duration_months": int(skill.get("duration_months", 0))
        }
        normalized.append(normalized_skill)
        
    return normalized
