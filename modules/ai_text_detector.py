import re
import torch
import logging
from typing import Dict, Any, List
from transformers import pipeline, AutoTokenizer, GPT2LMHeadModel

logger = logging.getLogger(__name__)

# Standard AI detection models
MODELS = [
    "openai-community/roberta-base-openai-detector",  # 0: Fake, 1: Real
    "Hello-SimpleAI/chatgpt-detector-roberta",       # 0: Human, 1: ChatGPT
]

PERPLEX_MODEL = "gpt2" 

_pipelines = {}
_perplex_tokenizer = None
_perplex_model = None

def load_perplexity_model():
    global _perplex_tokenizer, _perplex_model
    if _perplex_model is None:
        try:
            _perplex_tokenizer = AutoTokenizer.from_pretrained(PERPLEX_MODEL)
            _perplex_tokenizer.pad_token = _perplex_tokenizer.eos_token
            _perplex_model = GPT2LMHeadModel.from_pretrained(PERPLEX_MODEL)
            _perplex_model.eval()
        except Exception as e:
            logger.error(f"Failed to load perplexity model: {e}")
    return _perplex_tokenizer, _perplex_model

def calculate_perplexity(text: str) -> float:
    tokenizer, model = load_perplexity_model()
    if not model or not tokenizer:
        return 100.0

    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs, labels=inputs["input_ids"])
    
    loss = outputs.loss
    return torch.exp(loss).item()

def get_pipeline(model_name: str):
    if model_name not in _pipelines:
        try:
            _pipelines[model_name] = pipeline(
                "text-classification", 
                model=model_name, 
                device=0 if torch.cuda.is_available() else -1
            )
        except Exception as e:
            logger.warning(f"Failed to load {model_name}: {e}")
            return None
    return _pipelines[model_name]

def detect_ai_assistance(text: str) -> Dict[str, Any]:
    # 1. Minimum length check
    if not text or len(text.strip()) < 100:
        return {
            "label": "Human Written",
            "probability": 0.0,
            "explanation": "Content is too short for reliable analysis.",
            "model_type": "none"
        }

    # 2. Chunking
    words = text.split()
    chunks = []
    if len(words) > 500:
        chunks.append(" ".join(words[:400]))
        chunks.append(" ".join(words[-400:]))
    else:
        chunks.append(text)

    all_ai_probs = []
    
    for chunk in chunks:
        chunk_model_probs = []
        for model_name in MODELS:
            pipe = get_pipeline(model_name)
            if not pipe: continue
            
            try:
                res = pipe(chunk)[0]
                label = res['label']
                score = res['score']
                
                # prob_ai should be probability of AI
                if label in ['Fake', 'ChatGPT', 'LABEL_0' if 'openai' in model_name else 'LABEL_1']:
                    # Note: roberta-base-openai-detector uses 0: Fake, 1: Real
                    # Hello-SimpleAI uses 0: Human, 1: ChatGPT
                    # So for openai: Fake is LABEL_0. For Hello-SimpleAI: ChatGPT is LABEL_1.
                    ai_prob = score if label != 'Real' and label != 'Human' else 1 - score
                else:
                    ai_prob = 1 - score if label in ['Real', 'Human'] else score
                
                # Simplified robust mapping
                if label == 'Fake' or label == 'ChatGPT':
                    ai_prob = score
                elif label == 'Real' or label == 'Human':
                    ai_prob = 1 - score
                
                chunk_model_probs.append(ai_prob)
                logger.info(f"Model {model_name} -> {label} ({score:.2f}) -> AI Prob: {ai_prob:.2f}")
            except Exception as e:
                logger.error(f"Inference error: {e}")

        if chunk_model_probs:
            all_ai_probs.append(sum(chunk_model_probs) / len(chunk_model_probs))

    if not all_ai_probs:
        return {
            "label": "Human Written",
            "probability": 5.0,
            "explanation": "Analysis engines currently offline.",
            "model_type": "fallback"
        }

    # Base probability
    base_prob = max(all_ai_probs)

    # 3. Perplexity Influence
    try:
        ppl = calculate_perplexity(chunks[0])
        logger.info(f"Perplexity: {ppl:.2f}")
        # AI often has PPL < 20. Human > 40.
        if ppl < 15: # Very predictable
            base_prob = min(base_prob + 0.2, 0.99)
        elif ppl > 60: # High complexity
            base_prob = max(base_prob - 0.2, 0.01)
    except:
        pass

    final_pct = round(base_prob * 100, 2)
    
    if final_pct > 70:
        label = "AI Content Detected"
        explanation = "The content shows high predictability and structural patterns characteristic of AI-generated text."
    elif final_pct > 35:
        label = "Likely AI-Assisted"
        explanation = "Moderate patterns suggestive of AI paraphrasing or generation detected."
    else:
        label = "Human Written"
        explanation = "The text exhibits complex linguistic patterns typical of human authors."

    return {
        "label": label,
        "probability": final_pct,
        "explanation": explanation,
        "model_type": "ensemble_v3_perplexity"
    }
