import re

def extract_dialogues(scene: str):
    """
    Extract quoted dialogues from a scene.
    """
    return re.findall(r'"(.*?)"', scene)
