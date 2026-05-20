# guardrail_check.py — orchestration only, no embedded data
"""Check article against GUARDRAILS and results.json using GitHub Models.

All configuration in guardrail_models.json and guardrail_system_prompt.txt.
"""

import os
import re
import sys
import json
from pathlib import Path
from langchain_openai import ChatOpenAI

SCRIPT_DIR = Path(__file__).parent


def strip_latex(text):
    """Remove LaTeX markup, leaving plain text for LLM."""
    # Remove math: $...$, $$...$$, \(...\), \[...\]
    text = re.sub(r"\$\$.*?\$\$", "", text, flags=re.DOTALL)
    text = re.sub(r"\$.*?\$", "", text)
    text = re.sub(r"\\\(.*?\\\)", "", text)
    text = re.sub(r"\\\[.*?\\\]", "", text, flags=re.DOTALL)
    # Remove commands: \begin{...}, \end{...}, \cite{...}, \ref{...}, \input{...}
    text = re.sub(r"\\begin\{[^}]*\}", "", text)
    text = re.sub(r"\\end\{[^}]*\}", "", text)
    text = re.sub(r"\\cite\{[^}]*\}", "", text)
    text = re.sub(r"\\ref\{[^}]*\}", "", text)
    text = re.sub(r"\\input\{[^}]*\}", "", text)
    text = re.sub(r"\\label\{[^}]*\}", "", text)
    # Remove \usepackage, \documentclass, \newcommand etc.
    text = re.sub(
        r"\\(usepackage|documentclass|newcommand|renewcommand|title|author|date|maketitle)\{.*\}",
        "",
        text,
    )
    text = re.sub(r"\\\w+", "", text)  # residual commands
    # Collapse whitespace
    text = re.sub(r"\n\s*\n", "\n\n", text)
    return text.strip()


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def load_text(path):
    with open(path, "r") as f:
        return f.read()


def build_system_prompt(guardrails_text, results_json):
    """Build prompt from template + data."""
    template = load_text(SCRIPT_DIR / "guardrail_system.txt")
    metrics_summary = results_json.get("metrics_summary", {})
    civs = results_json.get("civilizations", {})

    civ_metrics = {}
    for name, civ in civs.items():
        wm = civ.get("working_metrics", {})
        civ_metrics[name] = {
            "turn_accuracy": wm.get("turn_accuracy"),
            "brier_skill_score": wm.get("brier_skill_score"),
            "role": civ.get("role"),
        }

    return template.format(
        guardrails=guardrails_text[:8000],
        train_ta=metrics_summary.get("train", {}).get("avg_turn_accuracy", "N/A"),
        test_ta=metrics_summary.get("test", {}).get("avg_turn_accuracy", "N/A"),
        civ_metrics=json.dumps(civ_metrics, indent=2),
    )


def create_llm(model_name, github_token, config):
    return ChatOpenAI(
        base_url="https://models.inference.ai.azure.com",
        api_key=github_token,
        model=model_name,
        temperature=config["temperature"],
        max_tokens=config["max_tokens"],
    )


def parse_response(content):
    if content.startswith("```"):
        content = content.split("\n", 1)[1]
        if content.endswith("```"):
            content = content[:-3]
    return json.loads(content)


def print_violations(violations):
    print(f"\n  VIOLATIONS ({len(violations)}):")
    for v in violations:
        print(f"    [{v['severity']}] {v['type']}")
        print(f"    Quote: \"{v.get('quote', 'N/A')}\"")
        print(f"    Fix: {v.get('fix', 'N/A')}\n")


def run_check(article_text, guardrails_text, results_json):
    github_token = os.environ.get("GITHUB_TOKEN")
    if not github_token:
        print("ERROR: GITHUB_TOKEN not set.")
        sys.exit(1)

    config = load_json(SCRIPT_DIR / "guardrail_models.json")

    if config.get("strip_latex", False):
        article_text = strip_latex(article_text)

    system_prompt = build_system_prompt(guardrails_text, results_json)
    messages = [
        ("system", system_prompt),
        ("human", article_text[: config["article_max_chars"]]),
    ]

    errors = 0

    for model_name in config["models"]:
        print(f"  Trying model: {model_name}")
        try:
            llm = create_llm(model_name, github_token, config)
            response = llm.invoke(messages)
            check = parse_response(response.content.strip())

            print(f"\n  Verdict: {check['verdict']}")
            print(f"  Summary: {check.get('summary', 'N/A')}")

            if check.get("violations"):
                print_violations(check["violations"])

            if check["verdict"] == "fail":
                print(
                    "\nERROR: Article violates guardrails or is inconsistent with results."
                )
                print("Fix before publishing release.")
                sys.exit(1)

            return True

        except Exception as e:
            print(f"  Error with {model_name}: {e}")
            errors += 1
            if errors >= config["max_consecutive_errors"]:
                continue

    print("\nERROR: All models failed. Cannot verify article.")
    sys.exit(1)


if __name__ == "__main__":
    article = load_text(sys.argv[1])
    guardrails = load_text(sys.argv[2])
    results = load_json(sys.argv[3])
    run_check(article, guardrails, results)
