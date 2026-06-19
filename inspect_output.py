import os
import pandas as pd

def inspect():
    output_dir = "output"
    
    print("=" * 60)
    print("Inspecting Redrob Pipeline Outputs")
    print("=" * 60)
    
    files = {
        "Cleaned Candidates": "cleaned_candidates.parquet",
        "Candidate Features": "candidate_features.parquet",
        "Candidate Classification": "candidate_classification.parquet"
    }
    
    for label, filename in files.items():
        filepath = os.path.join(output_dir, filename)
        print(f"\n[FILE] File: {label} ({filepath})")
        print("-" * 50)
        
        if not os.path.exists(filepath):
            print("[ERROR] File does not exist yet. Please run the pipeline first.")
            continue
            
        # Load parquet file
        df = pd.read_parquet(filepath)
        print(f"  Shape: {df.shape[0]} rows, {df.shape[1]} columns")
        print("  Columns:")
        for i, col in enumerate(df.columns, 1):
            print(f"    {i}. {col}")
            
        print("\n  Sample Data (First 2 rows):")
        # For readability, show a subset of columns or basic preview
        preview_cols = [c for c in ["candidate_id", "anonymized_name", "normalized_current_title", "candidate_segment", "quality_score", "python_score", "llm_score"] if c in df.columns]
        if not preview_cols:
            preview_cols = list(df.columns[:5])
            
        print(df[preview_cols].head(2).to_string(index=False))
        print("-" * 50)

if __name__ == "__main__":
    inspect()
