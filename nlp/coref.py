from fastcoref import FCoref
import logging

_model = None

def get_model():
    global _model
    if _model is None:
        print(">>> Loading Coreference Model (FCoref)...")
        _model = FCoref(device='cpu') 
    return _model

def resolve_coreferences(text):
    model = get_model()
    
    preds = model.predict(texts=[text])
    result = preds[0]
    
    try:
        return result.get_resolved_text()
    except AttributeError:
        print("... using manual coreference resolution fallback ...")
        return manual_resolution(text, result)

def manual_resolution(text, result):
    """
    Manually replaces pronouns using the cluster indices.
    """
    clusters = result.get_clusters(as_strings=False)
    
    replacements = []
    
    for cluster in clusters:
        main_mention_span = cluster[0] 
        main_mention_text = text[main_mention_span[0]:main_mention_span[1]]
        
        for span in cluster[1:]:
            start, end = span
            replacements.append((start, end, main_mention_text))
            
    replacements.sort(key=lambda x: x[0], reverse=True)
    
    resolved_text = list(text)
    
    for start, end, replacement in replacements:
        original = text[start:end]
        if original.lower() == replacement.lower():
            continue

        resolved_text[start:end] = list(replacement)
        
    return "".join(resolved_text)