from narrative.scene_segmentation import segment_scenes
from personality.personality_mapper import infer_personality_from_text
from personality.emotion_mapper import infer_emotion_from_scene

with open("data/story/story.txt", encoding="utf-8", errors="ignore") as f:
    story = f.read()

scenes = segment_scenes(story)

print("\n=== Character Personality & Scene Emotion (NRC-based) ===")

for i, scene in enumerate(scenes):
    personality = infer_personality_from_text(scene)
    emotion = infer_emotion_from_scene(scene)

    print(f"\nScene {i}")
    print("Text:", scene)
    print("Personality [dom, ext, agr, age]:", personality)
    print("Emotion [valence, arousal]:", emotion)
