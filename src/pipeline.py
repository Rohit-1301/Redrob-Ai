import os
import argparse
import logging
import json
import pandas as pd
from src.config import DEFAULT_INPUT_FILE, DEFAULT_OUTPUT_DIR, FAILED_RECORDS_LOG, DUPLICATE_RECORDS_LOG
from src.data_loader import load_candidates_to_df
from src.cleaning import clean_dataset
from src.classification import classify_dataframe
from src.feature_engineering import engineer_features_dataframe
from src.quality_checks import check_dataset_quality

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def run_pipeline(input_file: str, output_dir: str, limit: int = None, output_format: str = "parquet"):
    """
    Executes the entire candidates pipeline end-to-end.
    """
    logger.info("=" * 60)
    logger.info("Redrob Intelligent Candidate Pipeline Execution Started")
    logger.info("=" * 60)
    logger.info(f"Input file: {input_file}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Output format: {output_format}")
    if limit:
        logger.info(f"Record limit: {limit}")
        
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Define error logs paths inside output directory
    failed_log_path = os.path.join(output_dir, FAILED_RECORDS_LOG)
    duplicate_log_path = os.path.join(output_dir, DUPLICATE_RECORDS_LOG)
    
    # 1. Load data
    try:
        df_raw = load_candidates_to_df(input_file, failed_log_path=failed_log_path, limit=limit)
    except Exception as e:
        logger.error(f"Pipeline aborted. Data loading failed: {e}")
        return
        
    if df_raw.empty:
        logger.warning("Pipeline completed early because no records were loaded.")
        return

    # 2. Cleaning and Deduplication
    cleaned_df, duplicates_df = clean_dataset(df_raw)
    
    # Write duplicates to a JSONL file for audit trail
    if not duplicates_df.empty:
        try:
            duplicates_df.to_json(duplicate_log_path, orient="records", lines=True)
            logger.info(f"Duplicate records written to: {duplicate_log_path}")
        except Exception as e:
            logger.error(f"Failed to write duplicates log: {e}")
            
    # Remove duplicates from the primary processing flow to avoid bias in features and classification
    # Keep is_duplicate column in output, but we segment only unique candidates.
    unique_cleaned_df = cleaned_df[~cleaned_df["is_duplicate"]].copy()
    logger.info(f"Proceeding with {len(unique_cleaned_df)} unique candidates for feature engineering and classification.")

    # 3. Candidate Classification
    classified_df = classify_dataframe(unique_cleaned_df)

    # 4. Feature Engineering
    features_df = engineer_features_dataframe(classified_df)

    # 5. Data Quality Checks
    quality_df = check_dataset_quality(classified_df)

    # 6. Prepare and Save Output Files
    logger.info(f"Saving processed datasets...")
    
    # A. Cleaned Candidates
    cleaned_final_df = classified_df.drop(columns=["candidate_segment"])
    
    # B. Candidate Features (features_df)
    
    # C. Candidate Classification
    # Merge classification segment and quality checks
    classification_final_df = pd.merge(
        classified_df[["candidate_id", "candidate_segment"]],
        quality_df,
        on="candidate_id",
        how="inner"
    )

    formats = [output_format] if output_format != "all" else ["parquet", "csv", "json"]
    
    for fmt in formats:
        fmt = fmt.lower()
        
        cleaned_file = os.path.join(output_dir, f"cleaned_candidates.{fmt}")
        features_file = os.path.join(output_dir, f"candidate_features.{fmt}")
        class_file = os.path.join(output_dir, f"candidate_classification.{fmt}")
        
        try:
            if fmt == "parquet":
                cleaned_final_df.to_parquet(cleaned_file, engine="pyarrow", index=False)
                features_df.to_parquet(features_file, engine="pyarrow", index=False)
                classification_final_df.to_parquet(class_file, engine="pyarrow", index=False)
            elif fmt == "csv":
                cleaned_final_df.to_csv(cleaned_file, index=False)
                features_df.to_csv(features_file, index=False)
                classification_final_df.to_csv(class_file, index=False)
            elif fmt == "json":
                # Convert records to JSON array
                cleaned_final_df.to_json(cleaned_file, orient="records", indent=2, force_ascii=False)
                features_df.to_json(features_file, orient="records", indent=2, force_ascii=False)
                classification_final_df.to_json(class_file, orient="records", indent=2, force_ascii=False)
            logger.info(f"Saved outputs successfully in {fmt.upper()} format to {output_dir}")
        except Exception as e:
            logger.error(f"Failed to save outputs in {fmt.upper()} format: {e}")

    logger.info("=" * 60)
    logger.info("Redrob Candidate Pipeline Completed Successfully")
    logger.info("=" * 60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Redrob Candidate Pipeline")
    parser.add_argument("--input", type=str, default=DEFAULT_INPUT_FILE, help="Path to input jsonl/jsonl.gz file")
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR, help="Path to output directory")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of candidate records to process")
    parser.add_argument("--format", type=str, default="parquet", choices=["parquet", "csv", "json", "all"], 
                        help="Format of the output tables (parquet, csv, json, all)")
    
    args = parser.parse_args()
    
    run_pipeline(args.input, args.output_dir, args.limit, args.format)
