def load_nrc_lexicon(path="data/NRC-Emotion-Lexicon.txt"):
    """
    Loads NRC Emotion Lexicon safely.
    Returns: dict[word] -> set(emotions)
    """
    lexicon = {}

    with open(path, encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()

            # Skip empty or comment lines
            if not line or line.startswith("#"):
                continue

            parts = line.split("\t")

            # Skip malformed lines
            if len(parts) != 3:
                continue

            word, emotion, association = parts

            if association == "1":
                lexicon.setdefault(word, set()).add(emotion)

    return lexicon
