def generate_prompts(events):
    prompts = []
    for e in events:
        actor = e['actor']
        action = e['action']
        target = e['target']
        emotion = e['emotion']
        
        # VISUAL FIXES
        # If target is 'scene', don't write it.
        # If action is 'be happy', format as 'Happy [Actor]'
        
        base_desc = f"{actor} {action}"
        
        if target != 'scene':
            # Ensure we don't double up prepositions
            if not any(target.startswith(p) for p in ['on', 'in', 'at', 'with']):
                base_desc += f" {target}"
            else:
                base_desc += f" {target}"

        # Atmosphere injection based on emotion
        lighting = "cinematic lighting"
        if emotion == 'positive':
            lighting = "warm golden hour lighting, soft shadows"
        elif emotion == 'negative':
            lighting = "dramatic chiaroscuro lighting, cool tones, misty"

        prompt = f"{base_desc}, {lighting}, highly detailed, 8k, photorealistic, masterpiece"
        prompts.append(prompt)
    return prompts