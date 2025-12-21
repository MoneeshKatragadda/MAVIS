import re
from collections import Counter
from personality.nrc_lexicon import load_nrc_lexicon

NRC = load_nrc_lexicon()

NEGATIVE = {"anger", "fear", "sadness", "disgust"}
POSITIVE = {"joy", "trust"}
HIGH_AROUSAL = {"anger", "fear"}
LOW_AROUSAL = {"sadness"}

def infer_emotion_from_scene(text: str):
    """
    Dataset-driven emotion inference using NRC lexicon.
    Returns: [valence (-1 to 1), arousal (0 to 1)]
    """

    words = re.findall(r"\b\w+\b", text.lower())
    emotion_counts = Counter()

    for w in words:
        if w in NRC:
            for e in NRC[w]:
                emotion_counts[e] += 1

    if not emotion_counts:
        return [0.0, 0.3]  # neutral fallback

    # Valence calculation
    pos = sum(emotion_counts[e] for e in POSITIVE)
    neg = sum(emotion_counts[e] for e in NEGATIVE)
    valence = (pos - neg) / max(pos + neg, 1)

    # Arousal calculation
    high = sum(emotion_counts[e] for e in HIGH_AROUSAL)
    low = sum(emotion_counts[e] for e in LOW_AROUSAL)
    arousal = min(1.0, 0.3 + 0.4 * high - 0.2 * low)

    return [round(valence, 2), round(arousal, 2)]
