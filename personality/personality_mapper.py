def infer_personality_from_text(text: str):
    """
    Infer stable character personality from dialogue style.
    Returns a 4D vector:
    [dominance, extraversion, agreeableness, age_factor]
    """

    text_lower = text.lower()
    length = len(text.split())

    # Dominance: commands, threats
    dominance = 0.5
    if any(w in text_lower for w in ["must", "now", "never", "listen"]):
        dominance += 0.2

    # Extraversion: verbosity, exclamations
    extraversion = min(1.0, length / 20)
    extraversion += min(0.2, text.count("!") * 0.1)

    # Agreeableness: polite words
    agreeableness = 0.5
    if any(w in text_lower for w in ["please", "sorry", "thank"]):
        agreeableness += 0.3

    # Age factor: heuristic (can be improved later)
    age_factor = 0.5
    if any(w in text_lower for w in ["son", "child", "kid"]):
        age_factor = 0.2
    if any(w in text_lower for w in ["old", "father", "sir"]):
        age_factor = 0.8

    return [
        round(min(1.0, dominance), 2),
        round(min(1.0, extraversion), 2),
        round(min(1.0, agreeableness), 2),
        round(min(1.0, age_factor), 2),
    ]
