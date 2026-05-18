"""Compare current results.json with previous release metrics."""

import json
import sys
import subprocess

def load_results(path):
    with open(path, 'r') as f:
        return json.load(f)

def get_previous_results():
    """Get results.json from the previous git tag."""
    try:
        # Get previous tag
        tags = subprocess.check_output(
            ['git', 'tag', '--sort=-version:refname'],
            text=True
        ).strip().split('\n')
        
        if len(tags) < 2:
            print("No previous tag found. Skipping diff.")
            return None
        
        prev_tag = tags[1]  # Second most recent
        print(f"Comparing with previous release: {prev_tag}")
        
        # Extract results.json from previous tag
        results_json = subprocess.check_output(
            ['git', 'show', f'{prev_tag}:output/results.json'],
            text=True
        )
        return json.loads(results_json)
    except Exception as e:
        print(f"Could not load previous results: {e}")
        return None

def diff_metrics(current, previous):
    if previous is None:
        return
    
    current_civs = current.get('civilizations', {})
    previous_civs = previous.get('civilizations', {})
    
    print("\nMetric changes from previous release:")
    print("-" * 60)
    
    changed = False
    for name in current_civs:
        if name not in previous_civs:
            print(f"  {name}: NEW (not in previous release)")
            changed = True
            continue
        
        curr_wm = current_civs[name].get('working_metrics', {})
        prev_wm = previous_civs[name].get('working_metrics', {})
        
        for metric in ['turn_accuracy', 'brier_skill_score']:
            curr_val = curr_wm.get(metric)
            prev_val = prev_wm.get(metric)
            
            if curr_val is not None and prev_val is not None:
                delta = curr_val - prev_val
                if abs(delta) > 0.001:
                    direction = '↑' if delta > 0 else '↓'
                    print(f"  {name} {metric}: {prev_val:.3f} → {curr_val:.3f} ({direction}{abs(delta):.3f})")
                    changed = True
    
    if not changed:
        print("  No significant changes in metrics.")
    
    print("-" * 60)

if __name__ == '__main__':
    current = load_results(sys.argv[1])
    previous = get_previous_results()
    diff_metrics(current, previous)
