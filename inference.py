import json
import time
import openai
import pandas as pd
import logging
from typing import List
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

from prompts import FACT_EXTRACTION_SYSTEM_PROMPT

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def call_model(input_text: str, model_id: str, system_prompt: str = FACT_EXTRACTION_SYSTEM_PROMPT,
               max_retries: int = 3) -> List[dict]:
    """Call an OpenAI model with retry logic for invalid JSON responses."""
    for attempt in range(max_retries):
        try:
            response = openai.ChatCompletion.create(
                model=model_id,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": input_text}
                ],
                response_format={"type": "json_object"},
                temperature=0.1,
                max_tokens=2000
            )
            output = response.choices[0].message["content"]
            data = json.loads(output)
            return data.get('facts', [])

        except json.JSONDecodeError as e:
            logger.warning(f"JSON decode error on attempt {attempt+1}/{max_retries}: {e}")
            if attempt == max_retries - 1:
                return f"JSON Error after {max_retries} attempts: {str(e)}"
            time.sleep(1)

        except Exception as e:
            logger.warning(f"API call error on attempt {attempt+1}/{max_retries}: {e}")
            if attempt == max_retries - 1:
                return f"Error after {max_retries} attempts: {str(e)}"
            time.sleep(1)

    return "Error: Max retries exceeded"


def run_inference(df: pd.DataFrame, model_id: str, openai_api_key: str,
                  max_workers: int = 10, output_path: str = None) -> pd.DataFrame:
    """
    Run parallel inference on all questions in the dataframe.

    Args:
        df: DataFrame with a 'question' column
        model_id: OpenAI model ID (e.g. fine-tuned model ID)
        openai_api_key: OpenAI API key
        max_workers: Number of parallel threads (tune to your rate limits)
        output_path: Optional CSV path to save results
    """
    openai.api_key = openai_api_key

    results = {}
    results_lock = Lock()

    def process_question(index, input_text):
        result = call_model(input_text, model_id)
        with results_lock:
            results[index] = result
        return index, result

    logger.info(f"Processing {len(df)} questions with {max_workers} workers...")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_index = {
            executor.submit(process_question, i, question): i
            for i, question in enumerate(df["question"])
        }

        for future in as_completed(future_to_index):
            index, result = future.result()
            fact_count = len(result) if isinstance(result, list) else 0
            logger.info(f"Question {index+1}/{len(df)}: {fact_count} facts extracted")

    ordered_results = [results[i] for i in range(len(df))]
    df["FT_response"] = ordered_results

    successful = [r for r in ordered_results if isinstance(r, list)]
    errors = [r for r in ordered_results if isinstance(r, str)]

    logger.info(f"\nDone. Successful: {len(successful)}, Errors: {len(errors)}")
    logger.info(f"Total facts extracted: {sum(len(r) for r in successful)}")

    if output_path:
        df.to_csv(output_path, index=False)
        logger.info(f"Results saved to {output_path}")

    return df
