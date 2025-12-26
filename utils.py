import spacy
import nltk
from nltk.corpus import wordnet as wn
from transformers import pipeline
import logging
import re

# Quiet transformers logging
logging.getLogger("transformers").setLevel(logging.ERROR)

print(">>> MAVIS: Loading NLP Models...")
try:
    nlp = spacy.load("en_core_web_lg")
except OSError:
    print("⚠ 'en_core_web_lg' not found. Using 'en_core_web_sm' (Lower Accuracy)")
    nlp = spacy.load("en_core_web_sm")

try:
    wn.ensure_loaded()
except LookupError:
    nltk.download('wordnet')
    nltk.download('omw-1.4')

# Sentiment Model
sentiment_pipeline = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

# --- CONFIGURATION ---

# 1. VISUAL CONSISTENCY MAPPING
# If the story says "You", we map it to "Julian" (or whoever the main char is).
# For your new story, we can remove "The Protagonist" and stick to names.
ACTOR_MAPPING = {
    'you': 'Julian',   # Map "You" to the specific character name
    'we': 'The Group',
    'i': 'Silas'       # Assuming 1st person is the narrator/Silas
}

BODY_PARTS = {'hand', 'finger', 'eyes', 'gaze', 'face', 'breath', 'voice', 'expression', 'head', 'heart', 'mouth', 'lips', 'fist', 'glance', 'look', 'arm', 'leg', 'shoulders', 'adam’s apple'}
ATMOSPHERIC_AGENTS = {'rain', 'wind', 'mist', 'fog', 'shadow', 'sun', 'moon', 'river', 'storm', 'silence', 'darkness', 'light', 'smoke'}

INVALID_ACTORS = {
    'it', 'that', 'this', 'what', 'who', 'which', 'there', 'here', 'scene', 
    'moment', 'way', 'idea', 'something', 'anything', 'nothing', 'everything',
    'lot', 'kind', 'sort', 'part', 'side', 'series', 'line', 'door', 'item'
}

SPEECH_VERBS = {'say', 'ask', 'shout', 'whisper', 'mutter', 'tell', 'speak', 'reply', 'command', 'snap', 'hiss', 'call', 'scream', 'cry', 'interrupt', 'warn'}

def get_true_subject(token):
    for child in token.children:
        if child.dep_ == 'poss': return child
    return token

def is_semantic_agent(token):
    word = token.lemma_.lower()
    if token.ent_type_ in ['PERSON', 'ORG', 'NORP', 'ANIMAL']: return True
    synsets = wn.synsets(word, pos=wn.NOUN)
    if not synsets: return False
    target_hypernyms = {wn.synset('person.n.01'), wn.synset('animal.n.01'), wn.synset('creature.n.01'), wn.synset('living_thing.n.01')}
    primary_synset = synsets[0]
    if not set(primary_synset.closure(lambda s: s.hypernyms())).isdisjoint(target_hypernyms): return True
    return False

def is_acting_human(token):
    head_verb = token.head
    human_verbs = SPEECH_VERBS.union({'think', 'know', 'decide', 'hope', 'laugh', 'smile', 'stare', 'look', 'walk', 'run', 'grab', 'hold', 'take', 'reach', 'sit', 'stand', 'swallow', 'roll'})
    return head_verb.lemma_.lower() in human_verbs

def is_valid_actor(token):
    text = token.text.lower()
    if text in INVALID_ACTORS: return False
    if token.pos_ not in ['PROPN', 'NOUN', 'PRON']: return False
    
    if text in ACTOR_MAPPING: return True
    if token.lemma_.lower() in ATMOSPHERIC_AGENTS: return True
    if is_semantic_agent(token): return True
    if is_acting_human(token): return True
    if text in ['he', 'she', 'they', 'you', 'we']: return True

    return False

def extract_dialogue_content(sent):
    """Robust quote extraction."""
    text = sent.text
    # Matches straight quotes "" and curly quotes “”
    matches = re.findall(r'["“](.*?)["”]', text)
    if not matches:
        # Fallback for simple quotes
        matches = re.findall(r'"(.*?)"', text)
    if matches:
        return max(matches, key=len) 
    return None

def extract_events(text, memory):
    events = []
    doc = nlp(text)
    
    # Context Memory: Remembers who acted last to assign "orphaned" dialogue
    last_active_actor = memory.get('last_actor', None)
    
    for sent in doc.sents:
        dialogue_text = extract_dialogue_content(sent)
        sent_events = []
        
        # 1. Extract Actors & Actions
        for token in sent:
            if token.pos_ == "VERB" or (token.pos_ == "AUX" and token.dep_ == "ROOT"):
                subject = None
                for child in token.children:
                    if child.dep_ in ['nsubj', 'nsubjpass']:
                        subject = child
                        break
                if not subject: continue

                real_actor_token = subject
                is_proxy_action = False
                if subject.lemma_.lower() in BODY_PARTS:
                    owner = get_true_subject(subject)
                    if owner != subject:
                        real_actor_token = owner
                        is_proxy_action = True
                
                if not is_valid_actor(real_actor_token): continue

                # Mapping
                final_actor_name = real_actor_token.text
                if final_actor_name.lower() in ACTOR_MAPPING:
                    final_actor_name = ACTOR_MAPPING[final_actor_name.lower()]
                else:
                    final_actor_name = re.sub(r"[^\w\s]", "", final_actor_name).title()
                
                # Update Context Memory
                if final_actor_name not in ATMOSPHERIC_AGENTS:
                    last_active_actor = final_actor_name
                    memory['last_actor'] = final_actor_name

                action = token.lemma_
                if not action.replace("'", "").isalpha(): continue
                
                if token.pos_ == "AUX":
                    for child in token.children:
                        if child.dep_ == "acomp": action = f"be {child.text}"; break
                if is_proxy_action: action = f"{action} with {subject.text}"

                target = "scene"
                for child in token.children:
                    if child.dep_ in ['dobj', 'attr']: target = child.text
                    elif child.dep_ == 'prep': 
                        for pchild in child.children:
                            if pchild.dep_ == 'pobj': target = f"{child.text} {pchild.text}"
                
                try:
                    res = sentiment_pipeline(sent.text[:512])[0]
                    emotion = res['label'].lower() if res['score'] > 0.65 else 'neutral'
                except: emotion = 'neutral'

                sent_events.append({
                    'actor': final_actor_name,
                    'action': action,
                    'target': target,
                    'emotion': emotion,
                    'dialogue': None,
                    'type': 'action',
                    'is_speech': action.split()[0].lower() in SPEECH_VERBS
                })

        # 2. Attach Dialogue (The Fix)
        if dialogue_text:
            attached = False
            # A. Attach to speech verb (e.g. Silas said "...")
            for e in sent_events:
                if e['is_speech']:
                    e['dialogue'] = dialogue_text
                    attached = True
                    break
            
            # B. Attach to any valid actor in sentence (e.g. Silas smiled. "...")
            if not attached:
                for e in sent_events:
                    if e['actor'] not in ATMOSPHERIC_AGENTS:
                        e['dialogue'] = dialogue_text
                        attached = True
                        break
            
            # C. ORPHAN HANDLING: If NO actor found in sentence (e.g. "Is the money ready?"),
            # attach to the LAST known actor from previous lines.
            if not attached and last_active_actor:
                events.append({
                    'actor': last_active_actor,
                    'action': 'speak',
                    'target': 'scene',
                    'emotion': 'neutral',
                    'dialogue': dialogue_text,
                    'type': 'action'
                })
                attached = True

        for e in sent_events:
            if 'is_speech' in e: del e['is_speech']
            events.append(e)

    return events