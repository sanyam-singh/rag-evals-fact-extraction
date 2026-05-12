import pandas as pd
import json
import numpy as np
import re
import openai
from typing import List, Dict, Any, Tuple
import ast
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import multiprocessing as mp
from functools import partial
import time
import threading
from queue import Queue
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EnhancedFactEvaluatorProduction:
    def __init__(self, openai_api_key: str = None):
        self.openai_api_key = openai_api_key
        if openai_api_key:
            import openai
            openai.api_key = openai_api_key

    def check_valid_json(self, predicted_facts):
        """Fixed version of check_valid_json method"""
        try:
            if isinstance(predicted_facts, str):
                try:
                    # First try json.loads
                    parsed = json.loads(predicted_facts)
                except json.JSONDecodeError:
                    try:
                        # Then try ast.literal_eval for Python string representations
                        parsed = ast.literal_eval(predicted_facts)
                    except:
                        return {
                            'is_valid': False,
                            'parsed_data': {"facts": []},
                            'error': f'Failed to parse string: {predicted_facts[:100]}...'
                        }
            elif isinstance(predicted_facts, list):
                parsed = predicted_facts
            elif isinstance(predicted_facts, dict):
                parsed = predicted_facts.get('facts', predicted_facts)
            else:
                return {
                    'is_valid': False,
                    'parsed_data': {"facts": []},
                    'error': f'Unsupported data type: {type(predicted_facts)}'
                }

            if isinstance(parsed, list):
                facts_list = parsed
            elif isinstance(parsed, dict) and 'facts' in parsed:
                facts_list = parsed['facts']
            else:
                facts_list = [parsed] if parsed else []

            return {
                'is_valid': True,
                'parsed_data': {"facts": facts_list},
                'error': None
            }
        except Exception as e:
            return {
                'is_valid': False,
                'parsed_data': {"facts": []},
                'error': str(e)
            }

    def extract_facts_by_category(self, facts_data: Any) -> Dict[str, List[Dict]]:
        """Extract facts grouped by category"""
        category_facts = defaultdict(list)

        if isinstance(facts_data, str):
            try:
                facts_data = json.loads(facts_data)
            except:
                try:
                    facts_data = eval(facts_data)
                except:
                    return category_facts

        if isinstance(facts_data, list):
            facts_list = facts_data
        elif isinstance(facts_data, dict) and 'facts' in facts_data:
            facts_list = facts_data.get('facts', [])
        else:
            return category_facts

        for fact in facts_list:
            if isinstance(fact, dict):
                category = fact.get('category', 'unknown')
                category_facts[category].append(fact)
            elif isinstance(fact, str):
                category_facts['unknown'].append({'fact': fact, 'category': 'unknown'})

        return dict(category_facts)

    def find_best_semantic_match(self, gold_fact: str, pred_facts: List[str],
                                category: str, debug: bool = False) -> Dict:
        """Find the best semantic match using the matching prompt"""

        if not pred_facts:
            return {'best_match': None, 'reason': 'No predicted facts available', 'confidence': 0.0}

        matching_prompt = f"""You are an agricultural fact comparison expert. Compare the reference fact with the candidate facts to find the best semantic match based on agricultural meaning and context.

REFERENCE FACT (Category: {category}):
{gold_fact}

CANDIDATE FACTS:
{json.dumps(pred_facts, indent=2)}

INSTRUCTIONS:
1. Find the candidate fact that conveys the most similar agricultural meaning to the reference fact
2. Prioritize matches that share the same:
   - Crop/plant type
   - Agricultural practice or technique
   - Specific measurements, dosages, or timing
   - Expected outcomes or benefits
3. Consider facts as matching even with different wording if they convey equivalent agricultural advice
4. Focus on semantic similarity and practical agricultural application rather than exact word matching
5. If no candidate fact is semantically similar enough (confidence < 0.7), return null for best_match

MATCHING CRITERIA EXAMPLES:
- Fertilizer application: "Apply NPK fertilizer" ≈ "Use balanced fertilizer with nitrogen, phosphorus, and potassium"
- Timing: "Sow wheat in November" ≈ "Plant wheat during late autumn"
- Pest control: "Control pests with neem oil" ≈ "Use organic neem-based pesticide for pest management"
- Spacing: "Plant single-bud setts at wider spacing for sugarcane" ≈ "For sugarcane, plant single-bud setts at wider spacing to enhance growth"
- Dosage: "Apply 5-10 kg zinc per hectare for sugarcane" ≈ "Apply 5-10 kg of Zinc (Zn) per hectare for sugarcane growth"

RESPOND WITH ONLY JSON:
{{
    "best_match": "exact text of best matching candidate fact or null if no good match",
    "reason": "detailed explanation focusing on specific agricultural elements that align (crop type, practice, measurements, outcomes) or why no adequate match exists",
    "confidence": 0.0-1.0
}}
"""

        try:
            messages = [
                {"role": "system", "content": "You are an expert agricultural fact comparison specialist. Respond ONLY with valid JSON."},
                {"role": "user", "content": matching_prompt}
            ]

            response = openai.ChatCompletion.create(
                model="gpt-4o",
                messages=messages,
                temperature=0.0,
                max_tokens=1000,
                response_format={"type": "json_object"}
            )

            content = response.choices[0].message.content.strip()
            data = json.loads(content)
            best_match = data.get('best_match')
            reason = data.get('reason', 'No reason provided')
            confidence = data.get('confidence', 0.0)

            # Verify the match is in our candidate list
            if best_match and best_match in pred_facts:
                return {'best_match': best_match, 'reason': reason, 'confidence': confidence}
            else:
                return {'best_match': None, 'reason': reason, 'confidence': 0.0}

        except Exception as e:
            if debug:
                logger.error(f"Error in finding best match: {e}")
            return {'best_match': None, 'reason': f'Error during matching: {str(e)}', 'confidence': 0.0}

    def check_contradictions(self, gold_fact: str, pred_facts: List[str],
                           category: str, debug: bool = False) -> List[Dict]:
        """Check for contradictions between gold fact and predicted facts"""

        if not pred_facts:
            return []

        contradiction_prompt = f"""
You are an agricultural contradiction-detection expert. Your task: IDENTIFY ONLY genuine contradictions between a single REFERENCE FACT and a list of CANDIDATE FACTS, and EXPLAIN each finding with a short, structured justification (NOT internal chain-of-thought).

REFERENCE FACT (Category: {category}):
{gold_fact}

CANDIDATE FACTS:
{json.dumps(pred_facts, indent=2)}

--- INSTRUCTIONS & OVERVIEW ---
1) Output: ONLY a single JSON object (see schema below). Do NOT produce any text outside JSON.
2) Do NOT reveal internal chain-of-thought. Instead provide a concise, structured summary of the evaluation steps used for each contradiction (max 2–3 short sentences / bullet-like items).
3) A *genuine contradiction* = two facts that make OPPOSITE or CONFLICTING claims about the SAME agricultural aspect (same subject and same property/attribute). Consider compound statements component-wise (temperature, humidity, timing, quantity, effect, method, scale, nutrient, crop, season, or location).

--- NORMALIZATION & PARSING (apply first) ---
A. Normalize text (lowercase; canonicalize units like °C, kg/ha, %; expand common synonyms when possible).
B. Decompose each fact into structured components:
   - subject/entity (e.g., "sandy soils", "onions")
   - attribute/property (e.g., "lodging risk", "storage temperature", "humidity", "application rate")
   - polarity (increase/decrease/avoid/allow/always/never)
   - numeric_range or numeric_value with units (e.g., 0-5°C; 1-2 kg/ha)
   - timing/season/scale/context qualifiers (e.g., "small-scale", "Rabi", "during flowering")
   - method or intervention (e.g., "hand-pick", "vacuum", "mulch")
C. If candidate and reference do not share the same subject/entity AND property, treat as NOT CONTRADICTION unless the candidate explicitly negates or directly opposes the reference.

--- COMPARISON RULES (apply in order) ---
1) Same subject + opposite polarity on same property => GENUINE CONTRADICTION (High confidence).
   - Example: "sandy soils reduce lodging risk" vs "sandy soils increase lodging risk".
2) Numeric ranges / quantities:
   - Parse ranges as [a,b]. If ranges DO NOT overlap at all => CONTRADICTION (High).
   - If ranges overlap partially:
       • If intersection / union < 0.5 (i.e., small overlap) => CONTRADICTION (Med).
       • If intersection substantial (>= 0.5) => NOT CONTRADICTION (or INCONSISTENT if upper/lower bounds differ markedly); prefer NOT CONTRADICTION but flag as "numeric_inconsistency".
   - For single-value vs range: check if value lies inside the range.
   - Quantities (fertilizer rates, doses): treat >2x difference (or absolute difference judged significant for that unit) as Contradiction (Med-High).
3) Timing/statements of absolutes:
   - "Never/Always" vs "Sometimes/Do in morning" => CONTRADICTION if they directly oppose timing.
4) Opposite recommendations or explicit negations:
   - "Water daily" vs "Avoid daily watering" => CONTRADICTION (High).
5) Methods, nutrients, or scale differences:
   - Different method for same goal (hand-pick vs vacuum) => NOT CONTRADICTION.
   - Different nutrients (Zn vs Fe) => NOT CONTRADICTION.
   - Different scale (small-scale manual vs large-scale mechanical) => NOT CONTRADICTION.
6) Qualitative descriptors (map to defaults; allow override):
   - humidity: low <= 60%, moderate 60–75%, high > 75% (default mapping)
   - temperature qualitative ranges are compared numerically when present.
   - If qualitative vs quantitative and the qualitative interpretation conflicts with numeric value => treat as CONTRADICTION (Med) if clearly opposite.
7) Compound statements:
   - Decompose into components (temp, humidity, storage duration, ventilation). Compare each component independently. If any KEY component is in direct conflict, label the candidate as CONTRADICTION but include which component(s) caused the decision (e.g., humidity conflict).

--- CONFIDENCE GUIDELINES ---
- High: direct, explicit opposites on same subject/property OR numeric ranges with no overlap.
- Med: numeric ranges with small overlap or qualitative vs numeric conflict; clear but some ambiguity.
- Low: potential conflict that is context-dependent or relies on implied context/definitions.

--- OUTPUT JSON SCHEMA (RESPOND WITH ONLY THIS JSON) ---
Return exactly one JSON object matching the schema below. If there are no genuine contradictions, return {{"contradictions": []}}.

{{
  "contradictions": [
    {{
      "contradicting_fact": "exact text of the contradicting candidate fact",
      "reference_fact": "exact text of the reference fact",
      "reason": "short, specific explanation of the direct opposition or conflict (mention component(s) compared)",
      "confidence": "High|Med|Low",
      "components_compared": [
        {{
          "component": "temperature|humidity|effect|quantity|timing|method|nutrient|scale|other",
          "reference_value": "normalized value or text",
          "candidate_value": "normalized value or text",
          "status": "conflict|compatible|ambiguous|different_topic"
        }}
      ],
      "structured_justification": [
        "Step 1: one-line action (e.g., decomposed into components and matched subject)",
        "Step 2: one-line action (e.g., numeric ranges compared and found non-overlapping)",
        "Step 3: concise conclusion (e.g., contradiction due to humidity mismatch)"
      ]
    }}
  ]
}}
"""

        try:
            messages = [
                {"role": "system", "content": "You are an expert agricultural contradiction detection specialist. Respond ONLY with valid JSON."},
                {"role": "user", "content": contradiction_prompt}
            ]

            response = openai.ChatCompletion.create(
                model="gpt-4o",
                messages=messages,
                temperature=0.0,
                max_tokens=2000,
                response_format={"type": "json_object"}
            )

            content = response.choices[0].message.content.strip()
            data = json.loads(content)
            return data.get('contradictions', [])

        except Exception as e:
            if debug:
                logger.error(f"Error in contradiction detection: {e}")
            return []

    def evaluate_unmatched_relevance(self, question: str, ground_facts: List[str],
                                   unmatched_facts: List[str], debug: bool = False) -> Dict:
        """Evaluate unmatched facts for relevance and quality"""

        if not unmatched_facts:
            return {
                'relevant_facts': [],
                'irrelevant_facts': [],
                'analysis_results': []
            }

        unmatched_relevant_prompt = f"""You are an agricultural expert tasked with analyzing the relevance and accuracy of predicted facts in relation to specific agricultural questions and ground truth facts. Your goal is to evaluate how well each predicted fact addresses the given question, aligns with established ground facts, and determine its practical value for farmers.

Task instructions:
- For each predicted fact, provide an evaluation covering:
  1. Relevance - How directly it addresses the question.
  2. Ground Truth Alignment - How well it aligns with or complements the ground facts.
  3. Practical Value - How actionable and useful it is for farmers.
  4. Completeness - Whether it provides sufficient detail for implementation.
  5. Confidence Level - Assess the fact's accuracy and reliability.

- Use the Evaluation Framework to score each predicted fact on a scale from 1 to 10:
  1. Direct Relevance: Does it directly answer the question?
  2. Ground Truth Consistency: Does it align with or complement the ground facts?
  3. Practical Implementation: Can farmers easily apply this advice?
  4. Specificity: Does it provide enough detail for action?
  5. Agricultural Soundness: Is the advice scientifically and practically sound?

- For every predicted fact, compute an overall score (1-10) that summarizes the fact's usefulness. Provide a short explanation, list any gaps or missing details, and give a short 'farmer_applicability' statement about how easy it is for a typical farmer to implement.

- Output MUST be valid JSON following the exact structure below. Do not include any extra top-level keys. Do not add commentary outside the JSON. Use numbers for numeric fields and arrays for lists.

Output JSON schema:
{{
  "question": "string - The agricultural question being analyzed",
  "ground_facts": ["array of ground truth facts"],
  "predicted_facts_analysis": [
    {{
      "predicted_fact": "string - The predicted fact being evaluated",
      "relevance_score": "number",
      "ground_truth_alignment_score": "number",
      "practical_value_score": "number",
      "specificity_score": "number",
      "agricultural_soundness_score": "number",
      "overall_score": "number",
      "explanation": "string - Brief explanation of the evaluation",
      "gaps_identified": ["array of missing information or improvements needed"],
      "farmer_applicability": "string - Assessment of practical implementation ease"
    }}
  ],
  "summary": {{
    "total_predicted_facts": "number",
    "average_overall_score": "number",
    "key_insights": ["array of main findings"],
    "recommendations": ["array of suggestions for improvement"]
  }}
}}

Now analyze the following input and produce the JSON response:

-- INPUT --
QUESTION: {question}
GROUND_FACTS: {json.dumps(ground_facts)}
PREDICTED_FACTS: {json.dumps(unmatched_facts)}

-- END INPUT --

Produce the JSON evaluation now."""

        try:
            messages = [
                {"role": "system", "content": "You are an expert agricultural fact evaluation specialist. Respond ONLY with valid JSON."},
                {"role": "user", "content": unmatched_relevant_prompt}
            ]

            response = openai.ChatCompletion.create(
                model="gpt-4o",
                messages=messages,
                temperature=0.0,
                max_tokens=3000,
                response_format={"type": "json_object"}
            )

            content = response.choices[0].message.content.strip()
            data = json.loads(content)

            relevant_facts = []
            irrelevant_facts = []

            for analysis in data.get('predicted_facts_analysis', []):
                if analysis.get('overall_score', 0) >= 6:
                    relevant_facts.append({
                        'fact': analysis['predicted_fact'],
                        'score': analysis['overall_score'],
                        'reason': analysis['explanation']
                    })
                else:
                    irrelevant_facts.append({
                        'fact': analysis['predicted_fact'],
                        'score': analysis['overall_score'],
                        'reason': analysis['explanation']
                    })

            return {
                'relevant_facts': relevant_facts,
                'irrelevant_facts': irrelevant_facts,
                'analysis_results': data.get('predicted_facts_analysis', []),
                'summary': data.get('summary', {})
            }

        except Exception as e:
            if debug:
                logger.error(f"Error in unmatched relevance evaluation: {e}")
            return {
                'relevant_facts': [],
                'irrelevant_facts': unmatched_facts,
                'analysis_results': [],
                'error': str(e)
            }

    def enhanced_category_wise_matching(self, predicted_facts: Any, golden_facts: Any,
                                      question: str = "", debug: bool = False) -> Dict[str, Any]:
        """
        Enhanced matching with new strategy:
        1. Match facts using existing strategy
        2. Check contradictions for leftover facts
        3. Check relevance for remaining unmatched facts
        4. Store results in structured format for CSV export
        """

        pred_by_category = self.extract_facts_by_category(predicted_facts)
        gold_by_category = self.extract_facts_by_category(golden_facts)

        if debug:
            logger.info(f"Predicted categories: {list(pred_by_category.keys())}")
            logger.info(f"Golden categories: {list(gold_by_category.keys())}")

        all_results = []
        total_matches = 0
        total_gold_facts = 0
        total_contradictions = 0
        total_relevant_unmatched = 0
        total_irrelevant_unmatched = 0

        for category in gold_by_category.keys():
            gold_facts_in_category = gold_by_category[category]
            pred_facts_in_category = pred_by_category.get(category, [])

            total_gold_facts += len(gold_facts_in_category)

            if debug:
                logger.info(f"\nProcessing Category: {category}")
                logger.info(f"Gold facts: {len(gold_facts_in_category)}")
                logger.info(f"Pred facts: {len(pred_facts_in_category)}")

            gold_fact_texts = [self._extract_fact_text(fact) for fact in gold_facts_in_category]
            pred_fact_texts = [self._extract_fact_text(fact) for fact in pred_facts_in_category]

            used_pred_facts = set()
            category_results = []

            for gold_fact in gold_fact_texts:
                available_pred_facts = [f for f in pred_fact_texts if f not in used_pred_facts]

                if not available_pred_facts:
                    category_results.append({
                        'gold_fact': gold_fact,
                        'matched_pred_fact': None,
                        'match_reason': 'No available predicted facts',
                        'match_confidence': 0.0,
                        'status': 'unmatched_gold'
                    })
                    continue

                match_result = self.find_best_semantic_match(
                    gold_fact, available_pred_facts, category, debug=debug
                )

                if match_result['best_match'] and match_result['confidence'] >= 0.7:
                    used_pred_facts.add(match_result['best_match'])
                    total_matches += 1

                    category_results.append({
                        'gold_fact': gold_fact,
                        'matched_pred_fact': match_result['best_match'],
                        'match_reason': match_result['reason'],
                        'match_confidence': match_result['confidence'],
                        'status': 'matched'
                    })
                else:
                    category_results.append({
                        'gold_fact': gold_fact,
                        'matched_pred_fact': None,
                        'match_reason': match_result['reason'],
                        'match_confidence': match_result['confidence'],
                        'status': 'unmatched_gold'
                    })

            leftover_pred_facts = [f for f in pred_fact_texts if f not in used_pred_facts]

            if leftover_pred_facts:
                if debug:
                    logger.info(f"Processing {len(leftover_pred_facts)} leftover predicted facts")

                for pred_fact in leftover_pred_facts:
                    contradictions = self.check_contradictions(
                        pred_fact, gold_fact_texts, category, debug=debug
                    )

                    if contradictions:
                        total_contradictions += len(contradictions)
                        category_results.append({
                            'pred_fact': pred_fact,
                            'status': 'contradictory',
                            'contradictions': contradictions,
                            'contradiction_count': len(contradictions)
                        })
                    else:
                        relevance_result = self.evaluate_unmatched_relevance(
                            question, gold_fact_texts, [pred_fact], debug=debug
                        )

                        if relevance_result['relevant_facts']:
                            total_relevant_unmatched += 1
                            category_results.append({
                                'pred_fact': pred_fact,
                                'status': 'unmatched_relevant',
                                'relevance_analysis': relevance_result['analysis_results'][0] if relevance_result['analysis_results'] else {},
                                'relevance_score': relevance_result['relevant_facts'][0]['score'] if relevance_result['relevant_facts'] else 0
                            })
                        else:
                            total_irrelevant_unmatched += 1
                            category_results.append({
                                'pred_fact': pred_fact,
                                'status': 'unmatched_irrelevant',
                                'relevance_analysis': relevance_result['analysis_results'][0] if relevance_result['analysis_results'] else {},
                                'relevance_score': relevance_result['irrelevant_facts'][0]['score'] if relevance_result['irrelevant_facts'] else 0
                            })

            all_results.extend(category_results)

        iogt_score = total_matches / total_gold_facts if total_gold_facts > 0 else 0

        return {
            'iogt_score': iogt_score,
            'total_matches': total_matches,
            'total_gold_facts': total_gold_facts,
            'total_contradictions': total_contradictions,
            'total_relevant_unmatched': total_relevant_unmatched,
            'total_irrelevant_unmatched': total_irrelevant_unmatched,
            'detailed_results': all_results,
            'summary_stats': {
                'match_rate': iogt_score,
                'contradiction_rate': total_contradictions / len([r for r in all_results if 'pred_fact' in r]) if any('pred_fact' in r for r in all_results) else 0,
                'relevant_unmatched_rate': total_relevant_unmatched / len([r for r in all_results if 'pred_fact' in r]) if any('pred_fact' in r for r in all_results) else 0,
                'irrelevant_rate': total_irrelevant_unmatched / len([r for r in all_results if 'pred_fact' in r]) if any('pred_fact' in r for r in all_results) else 0
            }
        }

    def _extract_fact_text(self, fact_item: Any) -> str:
        """Extract fact text from various formats"""
        if isinstance(fact_item, str):
            return fact_item
        elif isinstance(fact_item, dict):
            return fact_item.get('fact', str(fact_item))
        else:
            return str(fact_item)

    def evaluate_fact_sft_model_enhanced(self, predicted_facts: Any, golden_facts: Any,
                                       question: str = "", debug: bool = False) -> Dict[str, Any]:
        """Enhanced evaluation with new strategy"""

        metrics = {}

        json_result = self.check_valid_json(predicted_facts)
        metrics['json_validity'] = json_result['is_valid']

        pred_data = json_result['parsed_data'] if json_result['is_valid'] else predicted_facts

        metrics['enhanced_matching'] = self.enhanced_category_wise_matching(
            pred_data, golden_facts, question, debug=debug
        )

        return metrics

    def export_results_to_csv(self, results: List[Dict], filename: str = "enhanced_evaluation_results.csv"):
        """Export detailed results to CSV format"""

        csv_rows = []

        for result in results:
            base_info = {
                'question': result.get('question', ''),
                'sample_id': result.get('sample_id', ''),
                'iogt_score': result.get('iogt_score', 0),
                'total_matches': result.get('total_matches', 0),
                'total_gold_facts': result.get('total_gold_facts', 0),
                'total_contradictions': result.get('total_contradictions', 0),
                'total_relevant_unmatched': result.get('total_relevant_unmatched', 0),
                'total_irrelevant_unmatched': result.get('total_irrelevant_unmatched', 0)
            }

            for detail in result.get('detailed_results', []):
                row = base_info.copy()

                if detail.get('status') == 'matched':
                    row.update({
                        'fact_type': 'gold_fact',
                        'fact_text': detail.get('gold_fact', ''),
                        'matched_fact': detail.get('matched_pred_fact', ''),
                        'status': 'matched',
                        'confidence': detail.get('match_confidence', 0),
                        'reason': detail.get('match_reason', ''),
                        'contradiction_count': 0,
                        'relevance_score': 10
                    })
                elif detail.get('status') == 'contradictory':
                    contradictions_json = json.dumps(detail.get('contradictions', []), ensure_ascii=False)
                    row.update({
                        'fact_type': 'pred_fact',
                        'fact_text': detail.get('pred_fact', ''),
                        'matched_fact': '',
                        'status': 'contradictory',
                        'confidence': 0,
                        'reason': f"Contradicts {detail.get('contradiction_count', 0)} gold facts",
                        'contradiction_count': detail.get('contradiction_count', 0),
                        'relevance_score': 0,
                        'contradictions_full': contradictions_json
                    })
                elif detail.get('status') == 'unmatched_relevant':
                    row.update({
                        'fact_type': 'pred_fact',
                        'fact_text': detail.get('pred_fact', ''),
                        'matched_fact': '',
                        'status': 'unmatched_relevant',
                        'confidence': 0,
                        'reason': detail.get('relevance_analysis', {}).get('explanation', ''),
                        'contradiction_count': 0,
                        'relevance_score': detail.get('relevance_score', 0)
                    })
                elif detail.get('status') == 'unmatched_irrelevant':
                    row.update({
                        'fact_type': 'pred_fact',
                        'fact_text': detail.get('pred_fact', ''),
                        'matched_fact': '',
                        'status': 'unmatched_irrelevant',
                        'confidence': 0,
                        'reason': detail.get('relevance_analysis', {}).get('explanation', ''),
                        'contradiction_count': 0,
                        'relevance_score': detail.get('relevance_score', 0)
                    })
                elif detail.get('status') == 'unmatched_gold':
                    row.update({
                        'fact_type': 'gold_fact',
                        'fact_text': detail.get('gold_fact', ''),
                        'matched_fact': '',
                        'status': 'unmatched_gold',
                        'confidence': 0,
                        'reason': detail.get('match_reason', ''),
                        'contradiction_count': 0,
                        'relevance_score': 0
                    })

                csv_rows.append(row)

        df = pd.DataFrame(csv_rows)
        df.to_csv(filename, index=False)
        logger.info(f"Results exported to {filename}")

        return df
