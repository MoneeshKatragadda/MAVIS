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

# 2. Resolve Coreferences (GLOBAL CONTEXT)
# We do this ONCE for the whole text so the model understands the full narrative
print(">>> Resolving Coreferences (This checks the whole story context)...")
story_resolved = resolve_coreferences(story_raw)
print(">>> Coreference Resolution Complete.")

# 3. Process Events Line-by-Line (on the resolved text)
memory = {}  # Memory is less critical now due to coref, but we keep it for strict parsing
all_events = []

# Split the RESOLVED story into lines
lines = [line.strip() for line in story_resolved.split("\n") if line.strip()]

for scene_id, line in enumerate(lines):
    # Extract semantic events
    events = extract_events(line, memory)
    for e in events:
        e["scene_id"] = scene_id
        all_events.append(e)

print("\n>>> EVENT EXTRACTION COMPLETE")
# Header for clean output
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

print(f"\n✅ SUCCESS: {len(all_events)} events + prompts ready")