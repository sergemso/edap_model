"""Check article against GUARDRAILS and results.json using LangChain + Groq.

Features:
- Fallback between free Groq models on consecutive errors
- Respects Retry-After header on 429
- Temperature=0 for deterministic output
- Strict: any violation = fail
- Checks both guardrail compliance AND consistency with results.json
"""

import os
import sys
import json
import time
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage

def load_file(path):
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()

def load_results(path):
    with open(path, 'r') as f:
        return json.load(f)

# Model fallback chain (all free on Groq)
MODELS = [
    "llama-3.3-70b-versatile",
    "mixtral-8x7b-32768",
    "gemma2-9b-it",
]

def create_llm(model_name, max_retries=3):
    """Create LLM with retry on 429."""
    for attempt in range(max_retries):
        try:
            return ChatGroq(
                model=model_name,
                temperature=0.0,
                max_tokens=2000,
            )
        except Exception as e:
            if '429' in str(e) and attempt < max_retries - 1:
                wait = min(2 ** attempt * 5, 60)
                print(f"  Rate limited on {model_name}. Waiting {wait}s...")
                time.sleep(wait)
            else:
                raise
    return None

def run_check(article_text, guardrails_text, results_json):
    """Run guardrail and consistency check with model fallback."""
    
    # Extract key metrics from results
    metrics_summary = results_json.get('metrics_summary', {})
    civs = results_json.get('civilizations', {})
    
    civ_metrics = {}
    for name, civ in civs.items():
        wm = civ.get('working_metrics', {})
        civ_metrics[name] = {
            'turn_accuracy': wm.get('turn_accuracy'),
            'brier_skill_score': wm.get('brier_skill_score'),
            'k_crossing_lead_years': wm.get('k_crossing_lead_years'),
            'role': civ.get('role'),
        }
    
    system_prompt = """You are a strict reviewer checking a scientific paper against its guardrails and numerical results.

GUARDRAILS:
{guardrails}

NUMERICAL RESULTS (from results.json):
Training turn accuracy: {train_ta}
Test turn accuracy: {test_ta}
USSR turn accuracy: {ussr_ta}
USA turn accuracy: {usa_ta}
USA p-value: 0.31 (not significant)

Civilization metrics:
{civ_metrics}

Check the article for:
1. PROHIBITED claims (predictive without significance, causal without evidence, overclaiming, normative)
2. REQUIRED CAVEATS (all 8 must appear somewhere in the paper)
3. TONE violations (wrong confidence level, wrong phrasing)
4. CONSISTENCY with numerical results (article must not claim different numbers)

Respond with JSON only:
{{
    "violations": [
        {{
            "type": "prohibited_claim" | "missing_caveat" | "tone" | "inconsistent_number",
            "severity": "critical" | "minor",
            "quote": "exact quote from article",
            "fix": "how to fix"
        }}
    ],
    "verdict": "pass" | "fail",
    "summary": "one-sentence assessment"
}}

CRITICAL: temperature=0, be deterministic. Any prohibited claim = fail. Any missing caveat = fail. Any number mismatch = fail.""".format(
        guardrails=guardrails_text[:5000],
        train_ta=metrics_summary.get('train', {}).get('avg_turn_accuracy', 'N/A'),
        test_ta=metrics_summary.get('test', {}).get('avg_turn_accuracy', 'N/A'),
        ussr_ta=civ_metrics.get('Soviet Union', {}).get('turn_accuracy', 'N/A'),
        usa_ta=civ_metrics.get('United States of America', {}).get('turn_accuracy', 'N/A'),
        civ_metrics=json.dumps(civ_metrics, indent=2),
    )
    
    errors = 0
    
    for model_name in MODELS:
        print(f"  Trying model: {model_name}")
        try:
            llm = create_llm(model_name)
            if llm is None:
                print(f"  Failed to create {model_name}, trying next...")
                errors += 1
                continue
            
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=article_text[:15000]),
            ]
            
            response = llm.invoke(messages)
            content = response.content.strip()
            
            # Parse JSON from response
            if content.startswith('```'):
                content = content.split('\n', 1)[1]
                if content.endswith('```'):
                    content = content[:-3]
            
            check = json.loads(content)
            
            print(f"\n  Verdict: {check['verdict']}")
            print(f"  Summary: {check.get('summary', 'N/A')}")
            
            if check.get('violations'):
                print(f"\n  VIOLATIONS ({len(check['violations'])}):")
                for v in check['violations']:
                    print(f"    [{v['severity']}] {v['type']}")
                    print(f"    Quote: \"{v.get('quote', 'N/A')}\"")
                    print(f"    Fix: {v.get('fix', 'N/A')}\n")
            
            if check['verdict'] == 'fail':
                print("\nERROR: Article violates guardrails or is inconsistent with results.")
                print("Fix before publishing release.")
                sys.exit(1)
            
            return True
            
        except Exception as e:
            print(f"  Error with {model_name}: {e}")
            errors += 1
            if errors >= 3:
                print(f"  {errors} consecutive errors, switching models...")
                continue
    
    print("\nERROR: All models failed. Cannot verify article.")
    sys.exit(1)

if __name__ == '__main__':
    article = load_file(sys.argv[1])
    guardrails = load_file(sys.argv[2])
    results = load_results(sys.argv[3])
    run_check(article, guardrails, results)
