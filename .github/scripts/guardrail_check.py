"""Check article against GUARDRAILS and results.json using LLM models.

Features:
- GitHub Models Marketplace
- Fallback between models on consecutive errors
- Temperature=0 for deterministic output
- Strict: any violation = fail
"""

import os
import sys
import json
from langchain_openai import ChatOpenAI


def load_file(path):
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()


def load_results(path):
    with open(path, 'r') as f:
        return json.load(f)


# Model fallback chain
MODELS = [
    "Llama-3.3-70B-Instruct",
    "Mistral-large",
    "Phi-4",
]

SYSTEM_TEMPLATE = """You are a strict reviewer checking a scientific paper against its guardrails and numerical results.

GUARDRAILS:
{guardrails}

NUMERICAL RESULTS (from results.json):
Training turn accuracy: {train_ta}
Test turn accuracy: {test_ta}

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
            "type": "prohibited_claim | missing_caveat | tone | inconsistent_number",
            "severity": "critical | minor",
            "quote": "exact quote from article",
            "fix": "how to fix"
        }}
    ],
    "verdict": "pass | fail",
    "summary": "one-sentence assessment"
}}

CRITICAL: temperature=0, be deterministic. Any prohibited claim = fail. Any missing caveat = fail. Any number mismatch = fail."""


def build_system_prompt(guardrails_text, results_json):
    """Build prompt once, reuse for all model attempts."""
    metrics_summary = results_json.get('metrics_summary', {})
    civs = results_json.get('civilizations', {})
    
    civ_metrics = {}
    for name, civ in civs.items():
        wm = civ.get('working_metrics', {})
        civ_metrics[name] = {
            'turn_accuracy': wm.get('turn_accuracy'),
            'brier_skill_score': wm.get('brier_skill_score'),
            'role': civ.get('role'),
        }
    
    return SYSTEM_TEMPLATE.format(
        guardrails=guardrails_text[:8000],
        train_ta=metrics_summary.get('train', {}).get('avg_turn_accuracy', 'N/A'),
        test_ta=metrics_summary.get('test', {}).get('avg_turn_accuracy', 'N/A'),
        civ_metrics=json.dumps(civ_metrics, indent=2),
    )


def create_llm(model_name, github_token):
    """Create LLM via GitHub Models. Token passed explicitly — no env read in loop."""
    return ChatOpenAI(
        base_url="https://models.inference.ai.azure.com",
        api_key=github_token,
        model=model_name,
        temperature=0.0,
        max_tokens=2000,
    )


def parse_response(content):
    """Strip markdown fences from LLM response."""
    if content.startswith('```'):
        content = content.split('\n', 1)[1]
        if content.endswith('```'):
            content = content[:-3]
    return json.loads(content)


def print_violations(violations):
    """Print violations once, reused."""
    print(f"\n  VIOLATIONS ({len(violations)}):")
    for v in violations:
        print(f"    [{v['severity']}] {v['type']}")
        print(f"    Quote: \"{v.get('quote', 'N/A')}\"")
        print(f"    Fix: {v.get('fix', 'N/A')}\n")


def run_check(article_text, guardrails_text, results_json):
    """Run guardrail and consistency check with model fallback."""
    github_token = os.environ.get('GITHUB_TOKEN')
    if not github_token:
        print("ERROR: GITHUB_TOKEN not set.")
        sys.exit(1)
    
    # Build immutable values once
    system_prompt = build_system_prompt(guardrails_text, results_json)
    article_snippet = article_text[:15000]
    messages = [("system", system_prompt), ("human", article_snippet)]
    
    errors = 0
    
    for model_name in MODELS:
        print(f"  Trying model: {model_name}")
        try:
            llm = create_llm(model_name, github_token)
            response = llm.invoke(messages)
            check = parse_response(response.content.strip())
            
            print(f"\n  Verdict: {check['verdict']}")
            print(f"  Summary: {check.get('summary', 'N/A')}")
            
            if check.get('violations'):
                print_violations(check['violations'])
            
            if check['verdict'] == 'fail':
                print("\nERROR: Article violates guardrails or is inconsistent with results.")
                print("Fix before publishing release.")
                sys.exit(1)
            
            return True
            
        except Exception as e:
            print(f"  Error with {model_name}: {e}")
            errors += 1
            if errors >= 3:
                print(f"  {errors} consecutive errors with {model_name}, switching models...")
                continue
    
    print("\nERROR: All models failed. Cannot verify article.")
    sys.exit(1)


if __name__ == '__main__':
    article = load_file(sys.argv[1])
    guardrails = load_file(sys.argv[2])
    results = load_results(sys.argv[3])
    run_check(article, guardrails, results)