import os
import json
import pandas as pd
from src.pipeline import run_pipeline

def test_pipeline():
    print("=" * 60)
    print("Redrob Pipeline Verification & Testing")
    print("=" * 60)
    
    # 1. Paths
    sample_json = os.path.join("Redrobsdataset", "sample_candidates.json")
    temp_jsonl = os.path.join("Redrobsdataset", "sample_candidates_converted.jsonl")
    output_dir = os.path.join("output_test")
    
    print(f"Reading sample JSON array from: {sample_json}")
    with open(sample_json, "r", encoding="utf-8") as f:
        candidates = json.load(f)
        
    print(f"Loaded {len(candidates)} candidates from sample JSON. Writing to JSONL...")
    with open(temp_jsonl, "w", encoding="utf-8") as f:
        for c in candidates:
            f.write(json.dumps(c) + "\n")
            
    print(f"Created temporary JSONL file at: {temp_jsonl}")
    
    # 2. Run pipeline
    print("Executing pipeline on sample dataset...")
    run_pipeline(temp_jsonl, output_dir)
    
    # 3. Verify outputs
    print("\n" + "=" * 40)
    print("Validating Pipeline Outputs...")
    print("=" * 40)
    
    cleaned_file = os.path.join(output_dir, "cleaned_candidates.parquet")
    features_file = os.path.join(output_dir, "candidate_features.parquet")
    class_file = os.path.join(output_dir, "candidate_classification.parquet")
    
    # Assert files exist
    assert os.path.exists(cleaned_file), "cleaned_candidates.parquet was not created!"
    assert os.path.exists(features_file), "candidate_features.parquet was not created!"
    assert os.path.exists(class_file), "candidate_classification.parquet was not created!"
    
    print("[OK] All 3 Parquet files successfully generated.")
    
    # Load files to verify schemas & records
    df_clean = pd.read_parquet(cleaned_file)
    df_feat = pd.read_parquet(features_file)
    df_class = pd.read_parquet(class_file)
    
    print(f"Cleaned candidates record count: {len(df_clean)}")
    print(f"Candidate features record count: {len(df_feat)}")
    print(f"Candidate classification record count: {len(df_class)}")
    
    # Verify primary keys
    assert set(df_clean["candidate_id"]) == set(df_feat["candidate_id"]), "Keys mismatch between cleaned and features!"
    assert set(df_clean["candidate_id"]) == set(df_class["candidate_id"]), "Keys mismatch between cleaned and classification!"
    print("[OK] Primary keys (candidate_id) match across all tables.")
    
    # Check classification distribution
    print("\nSegment Distribution:")
    print("-" * 30)
    dist = df_class["candidate_segment"].value_counts()
    for seg, count in dist.items():
        print(f"  {seg:<30}: {count}")
        
    # Check quality checks
    print("\nData Quality Summary:")
    print("-" * 30)
    mean_q = df_class["quality_score"].mean()
    min_q = df_class["quality_score"].min()
    max_q = df_class["quality_score"].max()
    print(f"  Average Quality Score: {mean_q:.2f}/100")
    print(f"  Min Quality Score    : {min_q}/100")
    print(f"  Max Quality Score    : {max_q}/100")
    
    anomalous = (df_class["quality_score"] < 100).sum()
    print(f"  Total Anomalous Candidates: {anomalous} ({(anomalous / len(df_class)) * 100:.1f}%)")
    
    # Clean up temporary JSONL file
    if os.path.exists(temp_jsonl):
        os.remove(temp_jsonl)
        print("\n[OK] Cleaned up temporary JSONL file.")
        
    print("\nVerification Test Completed Successfully!")
    print("=" * 60)

if __name__ == "__main__":
    test_pipeline()
