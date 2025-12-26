from fastcoref import FCoref
import logging

# Initialize model once (Singleton)
_model = None

def get_model():
    global _model
    if _model is None:
        print(">>> Loading Coreference Model (FCoref)...")
        # specific args to ensure CPU compatibility if GPU fails
        _model = FCoref(device='cpu') 
    return _model

def resolve_coreferences(text):
    """
    Resolves pronouns (He, She, It) to their names for the ENTIRE text.
    Includes a fallback manual resolver if the library method is missing.
    """
    model = get_model()
    
    # Predict coreference clusters
    # We pass [text] because the model expects a batch
    preds = model.predict(texts=[text])
    result = preds[0]
    
    # 1. Try the standard library method
    try:
        return result.get_resolved_text()
    except AttributeError:
        # 2. FALLBACK: Manual resolution if the method is missing
        # This fixes the "AttributeError: 'CorefResult'..." crash
        print("... using manual coreference resolution fallback ...")
        return manual_resolution(text, result)

def manual_resolution(text, result):
    """
    Manually replaces pronouns using the cluster indices.
    """
    clusters = result.get_clusters(as_strings=False)
    # We need to replace from the END of the text to the START
    # so that we don't mess up the indices of earlier replacements.
    
    replacements = []
    
    for cluster in clusters:
        # The first mention in a cluster is usually the 'main' entity (e.g., "Silas")
        # fastcoref usually sorts mentions by occurrence, so index 0 is the antecedent
        main_mention_span = cluster[0] 
        main_mention_text = text[main_mention_span[0]:main_mention_span[1]]
        
        # All other mentions (index 1+) are likely pronouns or partials to replace
        for span in cluster[1:]:
            start, end = span
            replacements.append((start, end, main_mention_text))
            
    # Sort replacements by start index in DESCENDING order
    replacements.sort(key=lambda x: x[0], reverse=True)
    
    resolved_text = list(text)
    
    for start, end, replacement in replacements:
        # Careful not to replace proper nouns with themselves if they are identical
        original = text[start:end]
        if original.lower() == replacement.lower():
            continue
            
        # Replace the slice
        resolved_text[start:end] = list(replacement)
        
    return "".join(resolved_text)