from utils import extract_events
from nlp.coref import resolve_coreferences
from visualization.translator import generate_prompts
import json
from pathlib import Path

print(">>> MAVIS: Semantic Event Extraction")

# 1. Load Story
story_path = Path("data/story.txt")
with open(story_path, "r", encoding="utf-8") as f:
    story_raw = f.read()


print(">>> Resolving Coreferences (This checks the whole story context)...")
story_resolved = resolve_coreferences(story_raw)
print(">>> Coreference Resolution Complete.")


memory = {} 
all_events = []


lines = [line.strip() for line in story_resolved.split("\n") if line.strip()]

for scene_id, line in enumerate(lines):
    events = extract_events(line, memory)
    for e in events:
        e["scene_id"] = scene_id
        all_events.append(e)

print("\n>>> EVENT EXTRACTION COMPLETE")
print(f"{'ACTOR':>15}  {'ACTION':>15}  {'TARGET':>20}  {'EMOTION':>10}")
print("-" * 65)

for e in all_events:
    print(f"{e['actor']:>15}  {e['action']:>15}  {e['target']:>20}  {e['emotion']:>10}")

# 4. Generate Visual Prompts
prompts = generate_prompts(all_events)

# Save outputs
Path("outputs").mkdir(exist_ok=True)
with open("outputs/events.json", "w") as f:
    json.dump(all_events, f, indent=2)
with open("outputs/prompts.json", "w") as f:
    json.dump(prompts, f, indent=2)

print(f"\nSUCCESS: {len(all_events)} events + prompts ready")