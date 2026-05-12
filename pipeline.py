import json
import numpy as np
import pandas as pd
import time
import threading
import logging
from typing import List, Dict, Tuple
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

from evaluator import EnhancedFactEvaluatorProduction

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ProgressTracker:
    def __init__(self, total_tasks):
        self.total_tasks = total_tasks
        self.completed_tasks = 0
        self.lock = threading.Lock()
        self.start_time = time.time()

    def update(self, increment=1):
        with self.lock:
            self.completed_tasks += increment
            if self.completed_tasks % 10 == 0 or self.completed_tasks == self.total_tasks:
                elapsed_time = time.time() - self.start_time
                rate = self.completed_tasks / elapsed_time if elapsed_time > 0 else 0
                eta = (self.total_tasks - self.completed_tasks) / rate if rate > 0 else 0

                logger.info(f"Progress: {self.completed_tasks}/{self.total_tasks} "
                           f"({self.completed_tasks/self.total_tasks*100:.1f}%) "
                           f"Rate: {rate:.2f} samples/sec "
                           f"ETA: {eta/60:.1f} minutes")


def process_single_row(args):
    """Process a single row - designed for multiprocessing"""
    idx, row_data, openai_api_key = args

    try:
        evaluator = EnhancedFactEvaluatorProduction(openai_api_key)

        predicted_facts = row_data['FT_response']
        golden_facts = row_data['gt_facts']
        question = row_data.get('question', '')

        metrics = evaluator.evaluate_fact_sft_model_enhanced(
            predicted_facts, golden_facts, question, debug=False
        )

        result = {
            'sample_id': idx,
            'question': question,
            'predicted_facts': predicted_facts,
            'golden_facts': golden_facts,
            'json_validity': metrics['json_validity'],
            **metrics['enhanced_matching']
        }

        return result

    except Exception as e:
        logger.error(f"Error processing row {idx}: {e}")
        return {
            'sample_id': idx,
            'question': row_data.get('question', ''),
            'predicted_facts': row_data.get('FT_response', ''),
            'golden_facts': row_data.get('gt_facts', ''),
            'error': str(e)
        }


def integrate_enhanced_evaluation_v2_optimized(df: pd.DataFrame, openai_api_key: str = None,
                                               max_workers: int = 40) -> List[Dict]:
    """Optimized integration with multiprocessing for 40 workers"""
    logger.info(f"Starting optimized enhanced evaluation with {max_workers} workers for {len(df)} samples...")

    args_list = [(idx, row.to_dict(), openai_api_key) for idx, row in df.iterrows()]
    progress_tracker = ProgressTracker(len(args_list))
    results = []

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_idx = {executor.submit(process_single_row, args): args[0] for args in args_list}

        for future in as_completed(future_to_idx):
            try:
                result = future.result()
                results.append(result)
                progress_tracker.update()
            except Exception as e:
                idx = future_to_idx[future]
                logger.error(f"Task {idx} generated an exception: {e}")
                results.append({'sample_id': idx, 'error': str(e)})
                progress_tracker.update()

    results.sort(key=lambda x: x.get('sample_id', 0))
    logger.info(f"Completed processing {len(results)} samples")
    return results


def integrate_enhanced_evaluation_v2_threaded(df: pd.DataFrame, openai_api_key: str = None,
                                              max_workers: int = 40) -> List[Dict]:
    """ThreadPoolExecutor implementation - better for I/O bound API calls"""
    logger.info(f"Starting threaded enhanced evaluation with {max_workers} workers for {len(df)} samples...")

    args_list = [(idx, row.to_dict(), openai_api_key) for idx, row in df.iterrows()]
    progress_tracker = ProgressTracker(len(args_list))
    results = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_idx = {executor.submit(process_single_row, args): args[0] for args in args_list}

        for future in as_completed(future_to_idx):
            try:
                result = future.result()
                results.append(result)
                progress_tracker.update()
            except Exception as e:
                idx = future_to_idx[future]
                logger.error(f"Task {idx} generated an exception: {e}")
                results.append({'sample_id': idx, 'error': str(e)})
                progress_tracker.update()

    results.sort(key=lambda x: x.get('sample_id', 0))
    logger.info(f"Completed processing {len(results)} samples")
    return results


def integrate_enhanced_evaluation_v2_batched(df: pd.DataFrame, openai_api_key: str = None,
                                             max_workers: int = 40, batch_size: int = 100) -> List[Dict]:
    """Batch processing version for very large datasets"""
    logger.info(f"Starting batched enhanced evaluation with {max_workers} workers, "
               f"batch size {batch_size} for {len(df)} samples...")

    all_results = []
    total_batches = (len(df) + batch_size - 1) // batch_size

    for batch_idx in range(0, len(df), batch_size):
        batch_end = min(batch_idx + batch_size, len(df))
        batch_df = df.iloc[batch_idx:batch_end]

        logger.info(f"Processing batch {batch_idx//batch_size + 1}/{total_batches} "
                   f"(rows {batch_idx}-{batch_end-1})")

        batch_results = integrate_enhanced_evaluation_v2_threaded(batch_df, openai_api_key, max_workers)
        all_results.extend(batch_results)

        if batch_idx > 0 and batch_idx % (batch_size * 5) == 0:
            intermediate_filename = f"intermediate_results_batch_{batch_idx//batch_size}.json"
            with open(intermediate_filename, 'w') as f:
                json.dump(all_results, f, indent=2)
            logger.info(f"Saved intermediate results to {intermediate_filename}")

    logger.info(f"Completed processing all {len(all_results)} samples")
    return all_results


def run_enhanced_evaluation_pipeline_optimized(df: pd.DataFrame, openai_api_key: str,
                                               output_filename: str = "enhanced_evaluation_results.csv",
                                               max_workers: int = 40,
                                               use_batching: bool = False,
                                               batch_size: int = 100) -> Tuple[pd.DataFrame, Dict]:
    """Complete optimized pipeline for enhanced evaluation with CSV export"""
    start_time = time.time()

    if use_batching:
        results = integrate_enhanced_evaluation_v2_batched(df, openai_api_key, max_workers, batch_size)
    else:
        results = integrate_enhanced_evaluation_v2_threaded(df, openai_api_key, max_workers)

    evaluator = EnhancedFactEvaluatorProduction(openai_api_key)
    results_df = evaluator.export_results_to_csv(results, output_filename)

    valid_results = [r for r in results if 'error' not in r]
    summary_stats = {
        'total_samples': len(results),
        'successful_samples': len(valid_results),
        'failed_samples': len(results) - len(valid_results),
        'avg_iogt_score': np.mean([r.get('iogt_score', 0) for r in valid_results]) if valid_results else 0,
        'avg_matches': np.mean([r.get('total_matches', 0) for r in valid_results]) if valid_results else 0,
        'avg_contradictions': np.mean([r.get('total_contradictions', 0) for r in valid_results]) if valid_results else 0,
        'avg_relevant_unmatched': np.mean([r.get('total_relevant_unmatched', 0) for r in valid_results]) if valid_results else 0,
        'avg_irrelevant_unmatched': np.mean([r.get('total_irrelevant_unmatched', 0) for r in valid_results]) if valid_results else 0,
        'processing_time_minutes': (time.time() - start_time) / 60,
        'samples_per_minute': len(results) / ((time.time() - start_time) / 60)
    }

    logger.info("\n" + "="*80)
    logger.info("           OPTIMIZED ENHANCED EVALUATION SUMMARY")
    logger.info("="*80)
    logger.info(f"Total Samples: {summary_stats['total_samples']}")
    logger.info(f"Successful Samples: {summary_stats['successful_samples']}")
    logger.info(f"Failed Samples: {summary_stats['failed_samples']}")
    logger.info(f"Average IoGT Score: {summary_stats['avg_iogt_score']:.3f}")
    logger.info(f"Average Matches per Sample: {summary_stats['avg_matches']:.1f}")
    logger.info(f"Average Contradictions per Sample: {summary_stats['avg_contradictions']:.1f}")
    logger.info(f"Average Relevant Unmatched per Sample: {summary_stats['avg_relevant_unmatched']:.1f}")
    logger.info(f"Average Irrelevant Unmatched per Sample: {summary_stats['avg_irrelevant_unmatched']:.1f}")
    logger.info(f"Processing Time: {summary_stats['processing_time_minutes']:.1f} minutes")
    logger.info(f"Processing Rate: {summary_stats['samples_per_minute']:.1f} samples/minute")
    logger.info(f"Workers Used: {max_workers}")
    logger.info(f"\nDetailed results saved to: {output_filename}")

    return results_df, summary_stats
