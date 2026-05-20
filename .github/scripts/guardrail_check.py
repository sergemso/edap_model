"""Check article sections against GUARDRAILS using GitHub Models.

Usage: python guardrail_check.py <section_name> <sections_dir> <guardrails_file> <results_json>
Example: python guardrail_check.py abstract paper/sections GUARDRAILS.md output/results.json
"""

import os
import re
import sys
import json
from pathlib import Path
from langchain_openai import ChatOpenAI

SCRIPT_DIR = Path(__file__).parent


LATEX_COMMANDS_TO_EXTRACT = ["textbf", "textit", "emph", "textsc", "texttt"]
LATEX_COMMANDS_TO_REMOVE = [
    "usepackage",
    "documentclass",
    "newcommand",
    "renewcommand",
    "title",
    "author",
    "date",
    "maketitle",
    "newpage",
    "hypersetup",
    "usetikzlibrary",
    "selectlanguage",
    "bibliographystyle",
    "bibliography",
    "addbibresource",
]

EXTRACT_PATTERN = re.compile(
    r"\\(?:" + "|".join(LATEX_COMMANDS_TO_EXTRACT) + r")\{([^}]*)\}"
)
REMOVE_PATTERN = re.compile(
    r"\\(?:" + "|".join(LATEX_COMMANDS_TO_REMOVE) + r")(?:\{[^}]*\}|\[[^]]*\])*"
)


def strip_latex(text):
    text = EXTRACT_PATTERN.sub(r"\1", text)
    text = re.sub(r"\$\$(.+?)\$\$", " [equation] ", text, flags=re.DOTALL)
    text = re.sub(r"\$(.+?)\$", " [var] ", text)
    text = re.sub(r"\\\((.+?)\\\)", " [eq] ", text)
    text = re.sub(r"\\\[(.+?)\\\]", " [equation] ", text, flags=re.DOTALL)
    text = re.sub(r"\_\{([^}]*)\}", r"_\1", text)
    text = re.sub(r"\^\{([^}]*)\}", r"^\1", text)
    text = re.sub(r"\\cite\{([^}]*)\}", "[ref]", text)
    text = re.sub(r"\\ref\{([^}]*)\}", "[ref]", text)
    text = re.sub(r"\\input\{([^}]*)\}", " [data] ", text)
    text = re.sub(r"\\includegraphics(?:\[[^]]*\])?\{[^}]*\}", "[figure]", text)
    text = re.sub(r"\\(?:begin|end)\{[^}]*\}", "", text)
    text = re.sub(r"\\label\{[^}]*\}", "", text)
    text = REMOVE_PATTERN.sub("", text)
    text = re.sub(r"\\\w+", "", text)
    text = re.sub(r"\{\}", "", text)
    text = re.sub(r"^\s*[%].*$", "", text, flags=re.MULTILINE)
    text = re.sub(
        r"^\s*\\(?:hline|toprule|midrule|bottomrule|cline)\s*$",
        "",
        text,
        flags=re.MULTILINE,
    )
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n", text)
    text = "\n".join(line.strip() for line in text.split("\n") if line.strip())
    return text.strip()


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def load_text(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def build_prompt(section_name, section_text, guardrails_text, results_json):
    template = load_text(SCRIPT_DIR / 'guardrail_system_prompt.txt')
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

    return template.format(
        section_name=section_name,
        section_text=section_text,
        guardrails=guardrails_text[:8000],
        train_ta=metrics_summary.get('train', {}).get('avg_turn_accuracy', 'N/A'),
        test_ta=metrics_summary.get('test', {}).get('avg_turn_accuracy', 'N/A'),
        civ_metrics=json.dumps(civ_metrics, indent=2),
    )

def run_check(section_name, section_text, guardrails_text, results_json):
    github_token = os.environ.get("GITHUB_TOKEN")
    if not github_token:
        print("ERROR: GITHUB_TOKEN not set.")
        sys.exit(1)

    config = load_json(SCRIPT_DIR / "guardrail_models.json")

    system_prompt = build_prompt(
        section_name, section_text, guardrails_text, results_json
    )
    messages = [
        ("system", system_prompt),
        ("human", f"Check the {section_name} section."),
    ]

    errors = 0
    for model_name in config["models"]:
        print(f"  [{section_name}] Trying model: {model_name}")
        try:
            llm = ChatOpenAI(
                base_url="https://models.inference.ai.azure.com",
                api_key=github_token,
                model=model_name,
                temperature=config["temperature"],
                max_tokens=config["max_tokens"],
                model_kwargs={"response_format": {"type": "json_object"}},
            )
            response = llm.invoke(messages)
            content = response.content.strip()
            if content.startswith("```"):
                content = content.split("\n", 1)[1]
                if content.endswith("```"):
                    content = content[:-3]
            check = json.loads(content)

            print(f"  [{section_name}] Verdict: {check['verdict']}")
            if check.get("violations"):
                print(f"  [{section_name}] VIOLATIONS ({len(check['violations'])}):")
                for v in check["violations"]:
                    print(
                        f"    [{v['severity']}] {v['type']}: {v.get('quote', 'N/A')[:80]}"
                    )

            if check["verdict"] == "fail":
                print(f"\nERROR: Section {section_name} violates guardrails.")
                sys.exit(1)

            cache_dir = Path(".guardrail-cache")
            cache_dir.mkdir(exist_ok=True)
            (cache_dir / f"{section_name}.json").write_text(json.dumps(check, indent=2))
            return True

        except Exception as e:
            print(f"  [{section_name}] Error with {model_name}: {e}")
            errors += 1
            if errors >= config["max_consecutive_errors"]:
                continue

    print(f"\nERROR: All models failed for section {section_name}.")
    sys.exit(1)


if __name__ == "__main__":
    if len(sys.argv) != 5:
        print(
            "Usage: guardrail_check.py <section_name> <sections_dir> <guardrails_file> <results_json>"
        )
        sys.exit(1)

    section_name = sys.argv[1]
    sections_dir = Path(sys.argv[2])
    guardrails = load_text(sys.argv[3])
    results = load_json(sys.argv[4])

    section_path = sections_dir / f"{section_name}.tex"
    if not section_path.exists():
        print(f"ERROR: Section file not found: {section_path}")
        sys.exit(1)

    section_text = strip_latex(load_text(section_path))
    run_check(section_name, section_text, guardrails, results)
