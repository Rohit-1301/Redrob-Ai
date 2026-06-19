import os

# ============================================================================
# Paths Configuration
# ============================================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "Redrobsdataset")
DEFAULT_INPUT_FILE = os.path.join(DATA_DIR, "candidates.jsonl")
DEFAULT_SAMPLE_FILE = os.path.join(DATA_DIR, "sample_candidates.json")
DEFAULT_OUTPUT_DIR = os.path.join(BASE_DIR, "output")

# Error logs
FAILED_RECORDS_LOG = "failed_records.jsonl"
DUPLICATE_RECORDS_LOG = "duplicate_records.jsonl"

# ============================================================================
# Job Title Normalization Configuration
# ============================================================================
# Map regex patterns to standard job roles
TITLE_MAPPINGS = {
    r"\b(ml|machine learning|ai|artificial intelligence|deep learning|nlp|computer vision|cv|transformers|reinforcement learning)\b\s*(engineer|developer|researcher|scientist|specialist|intern)": "ML_ENGINEER",
    r"\b(data scientist|data science|applied scientist)\b": "DATA_SCIENTIST",
    r"\b(data engineer|analytics engineer|big data engineer|data architect|etl developer)\b": "DATA_ENGINEER",
    r"\b(search|retrieval|ranking|recommender|recommendation|information retrieval)\b\s*(engineer|developer|researcher|specialist)": "RETRIEVAL_RANKING_ENGINEER",
    r"\b(devops|sre|site reliability|infrastructure|platform|cloud|systems|system)\b\s*(engineer|administrator|admin|architect)": "DEVOPS_ENGINEER",
    r"\b(product manager|product owner|product engineer|technical product manager|tpm)\b": "PRODUCT_ENGINEER",
    r"\b(software|backend|frontend|full\s*stack|web|application|app|mobile|android|ios|fullstack|developer|programmer)\b\s*(engineer|developer|architect|intern|specialist)": "SOFTWARE_ENGINEER",
    r"\b(consultant|advisory|advisor|partner|associate)\b": "CONSULTANT",
}

# Default title fallback for unmapped tech roles and non-tech roles
NON_TECH_TITLES = [
    "hr manager", "human resources", "operations manager", "operations analyst",
    "marketing manager", "marketing associate", "graphic designer", "designer",
    "mechanical engineer", "civil engineer", "accountant", "project manager",
    "sales executive", "sales manager", "customer support", "support specialist",
    "business analyst", "finance analyst", "office manager"
]

# ============================================================================
# Skill Normalization Configuration
# ============================================================================
# Maps lowercased raw skill names to standard normalized forms
SKILL_MAPPINGS = {
    # Python
    "python": "python", "python3": "python", "py": "python", "python programming": "python",
    
    # Machine Learning / AI
    "machine learning": "machine_learning", "ml": "machine_learning", 
    "deep learning": "deep_learning", "dl": "deep_learning",
    "nlp": "nlp", "natural language processing": "nlp",
    "computer vision": "computer_vision", "cv": "computer_vision",
    "pytorch": "pytorch", "tensorflow": "tensorflow", "keras": "tensorflow",
    "scikit-learn": "scikit_learn", "scikit learn": "scikit_learn", "sklearn": "scikit_learn",
    "numpy": "numpy", "pandas": "pandas", "jax": "jax", "spacy": "spacy", "nltk": "nltk",
    "opencv": "opencv", "huggingface": "hugging_face", "hugging face": "hugging_face",
    "xgboost": "xgboost", "lightgbm": "lightgbm", "catboost": "catboost",
    
    # LLMs & RAG
    "llm": "llm", "large language models": "llm", "large language model": "llm",
    "fine-tuning llms": "llm", "fine-tuning": "llm", "prompt engineering": "llm",
    "langchain": "langchain", "llamaindex": "llamaindex", "rag": "rag",
    "retrieval-augmented generation": "rag", "transformers": "transformers",
    
    # Vector Search
    "vector search": "vector_database", "vector db": "vector_database", 
    "vector database": "vector_database", "vector databases": "vector_database",
    "milvus": "vector_database", "pinecone": "vector_database", 
    "qdrant": "vector_database", "weaviate": "vector_database",
    "chromadb": "vector_database", "faiss": "vector_database",
    
    # Data Engineering
    "spark": "spark", "pyspark": "spark", "apache spark": "spark",
    "airflow": "airflow", "apache airflow": "airflow",
    "dbt": "dbt", "data build tool": "dbt",
    "kafka": "kafka", "apache kafka": "kafka",
    "hadoop": "hadoop", "hive": "hive",
    "snowflake": "snowflake", "redshift": "redshift", "bigquery": "bigquery",
    "databricks": "databricks", "etl": "etl", "data pipeline": "etl", "data pipelines": "etl",
    "sql": "sql", "postgresql": "sql", "postgres": "sql", "mysql": "sql", "sqlite": "sql",
    
    # DevOps & Cloud
    "docker": "docker", "kubernetes": "kubernetes", "k8s": "kubernetes",
    "terraform": "terraform", "ansible": "ansible", "jenkins": "jenkins", "git": "git",
    "ci/cd": "cicd", "github actions": "cicd",
    "aws": "aws", "amazon web services": "aws",
    "gcp": "gcp", "google cloud": "gcp", "google cloud platform": "gcp",
    "azure": "azure", "microsoft azure": "azure",
    
    # Web & UI/UX
    "javascript": "javascript", "js": "javascript", "typescript": "typescript", "ts": "typescript",
    "react": "react", "react.js": "react", "reactjs": "react",
    "node.js": "nodejs", "nodejs": "nodejs", "express": "nodejs",
    "html": "html", "css": "css", "tailwind": "tailwind",
    "angular": "angular", "angularjs": "angular", "vue": "vue", "vuejs": "vue",
    "figma": "figma", "ui/ux": "ui_ux", "product design": "ui_ux",
}

# ============================================================================
# Candidate Classification Configuration
# ============================================================================
# Classification keywords for segment matching in text (headline, summary, career description)
CLASSIFICATION_KEYWORDS = {
    "AI_ML_ENGINEER": [
        "machine learning", "ml", "artificial intelligence", "ai", "deep learning", 
        "neural network", "computer vision", "nlp", "reinforcement learning", 
        "tensorflow", "pytorch", "huggingface", "llm", "transformers", "data scientist"
    ],
    "RETRIEVAL_RANKING_ENGINEER": [
        "search", "retrieval", "ranking", "recommender", "recommendation", 
        "information retrieval", "vector search", "milvus", "pinecone", "qdrant", 
        "weaviate", "faiss", "elasticsearch", "solr", "lucene"
    ],
    "DATA_ENGINEER": [
        "data engineer", "data warehouse", "data pipeline", "etl", "spark", "pyspark", 
        "airflow", "dbt", "kafka", "hadoop", "snowflake", "redshift", "bigquery", 
        "databricks", "analytics engineer"
    ],
    "DEVOPS_ENGINEER": [
        "devops", "sre", "site reliability", "infrastructure", "platform engineer", 
        "cloud engineer", "kubernetes", "docker", "terraform", "ansible", "jenkins", 
        "ci/cd", "sysadmin", "system administrator"
    ],
    "PRODUCT_ENGINEER": [
        "product manager", "product owner", "product engineer", "technical product manager", 
        "tpm", "product-led", "ui/ux", "product design", "figma"
    ],
    "SOFTWARE_ENGINEER": [
        "software engineer", "software developer", "backend engineer", "frontend engineer", 
        "full stack developer", "fullstack", "web developer", "developer", "programmer"
    ]
}

# Classification rules thresholds
MIN_TECH_SKILL_COUNT = 2  # Min tech skills to be considered software engineer/tech
MIN_TECH_ROLES_COUNT = 1  # Min tech roles in career history to be considered tech

# ============================================================================
# Feature Engineering Configuration
# ============================================================================
# Skill sets for score calculations
SKILL_SETS = {
    "python": ["python"],
    "retrieval": ["vector_database", "elasticsearch", "search", "retrieval", "rag", "langchain", "llamaindex"],
    "ranking": ["ranking", "recommender", "recommendation"],
    "vector_db": ["vector_database"],
    "llm": ["llm", "transformers", "rag", "langchain", "llamaindex"]
}

# Profitability/Type categorization for companies
PRODUCT_INDUSTRIES = ["Software", "Internet", "SaaS", "Software Products", "Technology", "E-commerce"]
CONSULTING_COMPANIES = [
    "deloitte", "accenture", "pwc", "ey", "kpmg", "mckinsey", "bcg", "bain", 
    "wipro", "infosys", "tcs", "cognizant", "capgemini", "mindtree"
]

# Max thresholds for normalization
MAX_YEARS_EXPERIENCE = 20.0
MAX_ENDORSEMENTS = 100.0
MAX_SKILL_DURATION_MONTHS = 120.0
MAX_CONNECTION_COUNT = 1000
MAX_PROFILE_VIEWS_30D = 500
MAX_APPLICATIONS_30D = 100
MAX_SEARCH_APPEARANCE_30D = 1000
MAX_SAVED_BY_RECRUITERS_30D = 100
