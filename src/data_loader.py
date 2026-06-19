import json
import gzip
import logging
import os
import pandas as pd
from typing import Generator, List, Dict, Any, Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def stream_jsonl(file_path: str, failed_log_path: Optional[str] = None) -> Generator[Dict[str, Any], None, None]:
    """
    Streams JSON lines from a JSONL or JSONL.GZ file, handling malformed lines safely.
    Writes malformed lines to a log if specified.
    """
    is_gzip = file_path.endswith(".gz")
    open_func = gzip.open if is_gzip else open
    mode = "rt" if is_gzip else "r"
    encoding = "utf-8"

    failed_count = 0
    success_count = 0

    # Ensure output folder for failed logs exists
    if failed_log_path:
        failed_dir = os.path.dirname(os.path.abspath(failed_log_path))
        if failed_dir and not os.path.exists(failed_dir):
            os.makedirs(failed_dir, exist_ok=True)
            
        # Clear previous failed log
        if os.path.exists(failed_log_path):
            try:
                os.remove(failed_log_path)
            except Exception as e:
                logger.warning(f"Could not remove existing failed log file: {e}")

    try:
        with open_func(file_path, mode, encoding=encoding) as f:
            for line_idx, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    success_count += 1
                    yield record
                except json.JSONDecodeError as jde:
                    failed_count += 1
                    err_msg = f"Line {line_idx} is malformed: {str(jde)}"
                    logger.debug(err_msg)
                    if failed_log_path:
                        try:
                            with open(failed_log_path, "a", encoding="utf-8") as err_f:
                                err_f.write(json.dumps({"line": line_idx, "error": str(jde), "raw_content": line}) + "\n")
                        except Exception as e:
                            logger.error(f"Failed to write to error log: {e}")
    except FileNotFoundError:
        logger.error(f"Input file not found: {file_path}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error while reading file {file_path}: {e}")
        raise

    logger.info(f"Stream completed. Success: {success_count}, Failed/Malformed: {failed_count}")

def load_candidates_to_df(
    file_path: str,
    failed_log_path: Optional[str] = None,
    limit: Optional[int] = None
) -> pd.DataFrame:
    """
    Loads candidates into a Pandas DataFrame.
    Supports limiting the number of records read for debugging/development.
    """
    records: List[Dict[str, Any]] = []
    
    logger.info(f"Loading candidate records from {file_path}...")
    stream = stream_jsonl(file_path, failed_log_path)
    
    for idx, record in enumerate(stream):
        if limit is not None and idx >= limit:
            logger.info(f"Reached specified limit of {limit} records.")
            break
        records.append(record)
        
    df = pd.DataFrame(records)
    logger.info(f"Loaded {len(df)} candidate records into DataFrame.")
    return df
