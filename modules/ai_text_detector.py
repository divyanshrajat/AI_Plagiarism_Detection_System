import re
import torch
import logging
from typing import Dict, Any, List
from transformers import pipeline, AutoTokenizer, GPT2LMHeadModel
from transformers import TextClassificationPipeline
from torch.nn.functional import softmax

logger = logging.getLogger(__name__)

# Superior ensemble: proven high-acc models + perplexity
MODELS = [
    "openai-community/roberta-base-openai-detector",  # 84% acc
    "Hello-SimpleAI/chatgpt-detector-roberta",  # Backup
    "rohitg00/roberta-chatgpt-detector-finetuned"  # Additional
]

# Weights based on reported acc
MODEL_WEIGHTS = [0.5, 0.3, 0.2]

# GPT2 for perplexity (AI text has lower perplexity)
PERPLEX_MODEL = "gpt2-medium"
PERPLEX_WEIGHT = 0.1
PERPLEX_THRESHOLD = 20.0  # Lower = more AI-like

_pipelines = {}
_perplex_tokenizer = None
_perplex_model = None

def load_perplexity_model():
    """Load GPT2 for perplexity once."""
    global _perplex_tokenizer, _perplex_model
    if _perplex_model is None:
        _perplex_tokenizer = AutoTokenizer.from_pretrained(PERPLEX_MODEL)
        _perplex_tokenizer.pad_token = _perplex_tokenizer.eos_token
        _perplex_model = GPT2LMHeadModel.from_pretrained(PERPLEX_MODEL)
        _perplex_model.eval()
    return _perplex_tokenizer, _perplex_model

def calculate_perplexity(text: str, tokenizer, model) -> float:
    """Fast perplexity approximation."""
    encodings = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    seq_len = encodings.input_ids.size(1)
    nlls = []
    for begin_loc in range(0, seq_len, 256):
        end_loc = min(begin_loc + 256, seq_len)
        input_ids = encodings.input_ids[:, begin_loc:end_loc]
        target_ids = input_ids.clone()
        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)
            nlls.append(outputs.loss)
    ppl = torch.exp(torch.stack(nlls).mean())
    return ppl.item()

def get_pipeline(model_name: str):
    """Lazy load classifier."""
    if model_name not in _pipelines:
        try:
            _pipelines[model_name] = pipeline("text-classification", model=model_name, device=0 if torch.cuda.is_available() else -1)
        except Exception as e:
            logger.warning(f"Failed {model_name}: {e}")
            return None
    return _pipelines[model_name]

def detect_ai_assistance(text: str) -> Dict[str, Any]:
    if len(text.strip()) < 100:
        return {"label": "Human", "probability": 0, "explanation": "Text too short.", "model_type": "none"}

    # Chunk for long texts
    chunks = [text[i:i+1024] for i in range(0, len(text), 768)]
    ensemble_probs = []
    
    for chunk in chunks:
        chunk_probs = []
        for i, model_name in enumerate(MODELS):
            pipe = get_pipeline(model_name)
            if pipe is None:
                continue
            try:
                result = pipe(chunk)[0]
                prob_ai = result['score'] if result['label'] == 'Real' or result['label'] == 'LABEL_0' else 1 - result['score']
                chunk_probs.append(prob_ai * MODEL_WEIGHTS[i])
            except Exception:
                continue
        
        if chunk_probs:
            ensemble_probs.append(sum(chunk_probs) / sum(MODEL_WEIGHTS[:len(chunk_probs)]))
    
    if not ensemble_probs:
        return {"label": "Human", "probability": 0, "explanation": "Models unavailable."}

    model_prob = max(ensemble_probs)
    
    # Perplexity boost
    try:
        tokenizer, model = load_perplexity_model()
        ppl = calculate_perplexity(text[:2000], tokenizer, model)  # Truncate for speed
        perplex_factor = max(0, 1 - (ppl / PERPLEX_THRESHOLD)) * PERPLEX_WEIGHT
        final_prob = min(model_prob + perplex_factor, 0.99)
    except Exception:
        final_prob = model_prob
    
    prob_pct = final_prob * 100
    label = "AI" if prob_pct > 55 else "Human"  # Tuned threshold for F1
    
    if prob_pct > 80:
        explanation = "High confidence AI detection (ensemble + perplexity)."
    elif prob_pct > 60:
        explanation = "Likely AI-assisted."
    else:
        explanation = "Human-like characteristics."
    
    return {
        "label": label,
        "probability": round(prob_pct, 2),
        "explanation": explanation,
        "model_type": "improved-ensemble-perplexity"
    }

