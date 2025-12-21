def segment_scenes(story_text: str):
    scenes = [s.strip() for s in story_text.split("\n\n") if s.strip()]
    return scenes
