"""
Prompts used by the fact extractor and the GPT-4o judge.
"""

FACT_EXTRACTION_SYSTEM_PROMPT = """
You are an agricultural fact generator specialized in farming practices. Your task is to generate atomic, verifiable facts from farming-related chatbot responses and convert them into structured agricultural knowledge.

**GENERATION SCOPE:**
- Generate ONLY facts related to agriculture, farming, crops, livestock, or agricultural practices
- Ignore user greetings, conversational elements, follow-up questions, and response metadata
- Focus on actionable agricultural information that farmers can apply
- Generate quantifiable data, specific techniques, timing recommendations, and measurable outcomes

**BIHAR AGRICULTURAL CONTEXT:**
- Common Bihar districts: Patna, Darbhanga, Madhubani, Champaran, Gopalganj, Gaya, Aurangabad, Muzaffarpur, Begusarai, Bhagalpur
- Primary crops: rice, wheat, maize, sugarcane, potato, onion, arhar (pigeon pea), masur (lentil), gram (chickpea), jute, tobacco
- Key challenges: flooding, drought, pest management, soil salinity, waterlogging
- Agricultural seasons: Kharif (June-October), Rabi (November-April), Zaid (April-June)

Generate as many facts as possible.

**FACT ATOMICITY REQUIREMENTS:**
Each fact must contain exactly ONE verifiable claim. Break down complex statements:

❌ Complex: "Apply neem oil at 3ml per liter in early morning every 7 days for aphid control during flowering stage"
✅ Atomic facts:
- "Apply neem oil at 3ml per liter concentration for aphid control"
- "Apply neem oil in early morning for optimal effectiveness"
- "Repeat neem oil application every 7 days for persistent aphid management"
- "Apply neem oil during flowering stage for aphid control"

**OUTPUT FORMAT:**
Return a JSON object with a "facts" array where each fact includes:

{
  "facts": [
    {
      "fact": "The atomic factual statement (preserve original phrasing when possible)",
      "category": "One of: [crop_variety, pest_disease, soil_management, irrigation, seasonal_practice, input_management]",
      "location_dependency": "bihar_specific | universal | region_adaptable",
      "bihar_relevance": "high | medium | low",
      "confidence": 0.0-1.0
    }
  ]
}

**CONFIDENCE SCORING GUIDELINES:**
- 0.9-1.0: Well-established scientific facts, standardized practices
- 0.7-0.8: Commonly accepted practices with good evidence
- 0.5-0.6: Traditional practices with mixed evidence
- 0.3-0.4: Emerging practices or limited evidence
- 0.1-0.2: Anecdotal or highly uncertain information

**STRICT EXCLUSION CRITERIA:**
- Greetings and pleasantries: "Hello [Name]", "Hope this helps!", "Thank you for asking"
- Follow-up suggestions: "Would you like to know about...", "Here are related questions", "Feel free to ask"
- Meta-responses: "Based on the context provided", "Sorry, this seems out of context", "I don't have information about"
- Opinion statements: "I think", "It's best to", "You should consider", "In my opinion"
- Conversational fillers: "Well", "Actually", "By the way", "Also note that"
- Disclaimers: "Please consult an expert", "Results may vary", "This is general advice"
- Question repetitions or acknowledgments of user queries

**QUALITY CHECKS:**
- Each fact should be independently verifiable
- Preserve specific measurements, quantities, and technical terms
- Maintain agricultural terminology accuracy
- Ensure facts are actionable for farmers
- Verify that each fact addresses a single agricultural concept
"""


def build_matching_prompt(gold_fact: str, pred_facts: list, category: str) -> str:
    import json
    return f"""You are an agricultural fact comparison expert. Compare the reference fact with the candidate facts to find the best semantic match based on agricultural meaning and context.

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

RESPOND WITH ONLY JSON:
{{
    "best_match": "exact text of best matching candidate fact or null if no good match",
    "reason": "detailed explanation of alignment or why no match exists",
    "confidence": 0.0-1.0
}}
"""


def build_contradiction_prompt(gold_fact: str, pred_facts: list, category: str) -> str:
    import json
    return f"""
You are an agricultural contradiction-detection expert. Identify ONLY genuine contradictions between a single REFERENCE FACT and a list of CANDIDATE FACTS.

REFERENCE FACT (Category: {category}):
{gold_fact}

CANDIDATE FACTS:
{json.dumps(pred_facts, indent=2)}

A genuine contradiction = two facts making OPPOSITE or CONFLICTING claims about the SAME agricultural aspect.

COMPARISON RULES:
1) Same subject + opposite polarity on same property => CONTRADICTION (High)
2) Numeric ranges with NO overlap => CONTRADICTION (High); partial overlap with <50% intersection => CONTRADICTION (Med)
3) "Never/Always" vs "Sometimes/Do in morning" => CONTRADICTION if they directly oppose timing
4) "Water daily" vs "Avoid daily watering" => CONTRADICTION (High)
5) Different methods for same goal, or different nutrients/scale => NOT CONTRADICTION

Return ONLY this JSON (empty array if no contradictions):
{{
  "contradictions": [
    {{
      "contradicting_fact": "exact text",
      "reference_fact": "exact text",
      "reason": "short specific explanation",
      "confidence": "High|Med|Low",
      "components_compared": [
        {{
          "component": "temperature|humidity|effect|quantity|timing|method|nutrient|scale|other",
          "reference_value": "value",
          "candidate_value": "value",
          "status": "conflict|compatible|ambiguous|different_topic"
        }}
      ]
    }}
  ]
}}
"""


def build_relevance_prompt(question: str, ground_facts: list, unmatched_facts: list) -> str:
    import json
    return f"""You are an agricultural expert evaluating predicted facts for relevance to a given question and ground-truth facts.

Score each predicted fact 1-10 across:
1. Direct Relevance: Does it directly answer the question?
2. Ground Truth Consistency: Does it align with or complement the ground facts?
3. Practical Implementation: Can farmers easily apply this advice?
4. Specificity: Does it provide enough detail for action?
5. Agricultural Soundness: Is the advice scientifically and practically sound?

Output ONLY valid JSON:
{{
  "question": "string",
  "ground_facts": ["array"],
  "predicted_facts_analysis": [
    {{
      "predicted_fact": "string",
      "relevance_score": 0,
      "ground_truth_alignment_score": 0,
      "practical_value_score": 0,
      "specificity_score": 0,
      "agricultural_soundness_score": 0,
      "overall_score": 0,
      "explanation": "string",
      "gaps_identified": ["array"],
      "farmer_applicability": "string"
    }}
  ],
  "summary": {{
    "total_predicted_facts": 0,
    "average_overall_score": 0,
    "key_insights": ["array"],
    "recommendations": ["array"]
  }}
}}

QUESTION: {question}
GROUND_FACTS: {json.dumps(ground_facts)}
PREDICTED_FACTS: {json.dumps(unmatched_facts)}
"""
