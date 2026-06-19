import os
import pandas as pd

def convert():
    output_dir = "output"
    print("=" * 60)
    print("Exporting Parquet Outputs to CSV and JSON")
    print("=" * 60)
    
    files = {
        "cleaned_candidates": "cleaned_candidates.parquet",
        "candidate_features": "candidate_features.parquet",
        "candidate_classification": "candidate_classification.parquet"
    }
    
    for base_name, filename in files.items():
        parquet_path = os.path.join(output_dir, filename)
        if not os.path.exists(parquet_path):
            print(f"⚠️ {parquet_path} does not exist. Skipping.")
            continue
            
        print(f"Loading {parquet_path}...")
        df = pd.read_parquet(parquet_path)
        
        # Save as CSV
        csv_path = os.path.join(output_dir, f"{base_name}.csv")
        print(f"  Saving as CSV to {csv_path}...")
        df.to_csv(csv_path, index=False)
        
        # Save as JSON (Indented JSON array)
        json_path = os.path.join(output_dir, f"{base_name}.json")
        print(f"  Saving as JSON to {json_path}...")
        df.to_json(json_path, orient="records", indent=2, force_ascii=False)
        
    print("\n[OK] Export completed successfully!")
    print("=" * 60)

if __name__ == "__main__":
    convert()
