from character.discovery import discover_characters
from character.coreference import resolve_coreference
from narrative.event_extraction import extract_events

story = """
Moneesh stood at the edge of the ravine, his horse, Starlight, snorting impatiently at the rising mist. The mist crawled up the cliffs like a silent hunter. Behind them, a young boy watched from the bushes, hoping to remain unseen. This boy was a local thief who knew every inch of the valley.

A vulture circled overhead, screaming a sharp command to the wind. The wind obeyed, pushing the travelers toward the narrow crossing. Nearby, a stray cat sat on a mossy rock. The cat winked at Moneesh and pointed its paw toward the center of the bridge.

The bridge itself looked ancient and grumpy. One stone near the center shifted intentionally, trying to catch Starlight's hoof. "Steady," a hidden voice called out from the shadows. An old woman stepped into the light, leaning on a wooden staff. The staff was gnarled and made of oak, but it stayed perfectly still in her hand.

Suddenly, a fox dashed across the path. The fox was simply looking for its den and ignored the humans entirely. However, the river below was not so indifferent. The river roared a challenge, throwing its spray high into the air to soak the old woman. She laughed, and even the bridge seemed to chuckle under her feet.
"""

print(">>> STEP 1: Discovery...")
raw_chars = discover_characters(story)

print(">>> STEP 2: Coreference...")
merged_chars, timeline = resolve_coreference(story, raw_chars)

print(">>> STEP 3: Event Extraction (MAVIS Engine)...")
events = extract_events(story, timeline)

print(f"\n{'TIME':<5} | {'ACTOR':<15} | {'ACTION':<15} | {'EMOTION':<15} | {'TARGET'}")
print("-" * 75)

for e in events:
    # Lookup actor name
    actor_name = next((c['surface_form'] for c in merged_chars if c['id'] == e['actor_id']), "Unknown")
    
    print(f"{e['sentence_index']:<5} | {actor_name:<15} | {e['action']:<15} | {e['emotion_cue']:<15} | {e['target']}")