import sys
from pathlib import Path

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from modules.ai_text_detector import detect_ai_assistance

import logging

# Configure logging to see output
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def test_ai_detection():
    print("Testing AI detection module...")
    
    # Human-like natural text (from a Wikipedia-style excerpt or personal writing)
    human_text = """
    Agriculture has been the backbone of the Indian economy for centuries, providing a livelihood to more than half of its population. 
    However, the sector has faced numerous challenges, ranging from climate change to outdated farming practices. 
    Despite these hurdles, Indian farmers have shown remarkable resilience, adapting to new technologies and diversifying their crops to meet growing demand. 
    The government's role in providing subsidies and better infrastructure remains crucial for the long-term sustainability of the sector. 
    It is essential that we continue to support our farmers to ensure food security for the entire nation.
    """
    
    # AI-like text (GPT-style structured output)
    ai_text = """
    Artificial intelligence (AI) refers to the simulation of human intelligence by machines, especially computer systems. 
    These processes include learning (the acquisition of information and rules for using the information), reasoning 
    (using rules to reach approximate or definite conclusions) and self-correction. Particular applications of AI 
    include expert systems, speech recognition and machine vision. AI is being applied across many industries, 
    including finance and healthcare, to optimize processes and improve decision-making capabilities. 
    The ethical implications of AI deployment are a subject of significant debate in contemporary society.
    """

    print("\n[Human Text Test]")
    res_human = detect_ai_assistance(human_text)
    print(f"Result: {res_human}")

    print("\n[AI Text Test]")
    res_ai = detect_ai_assistance(ai_text)
    print(f"Result: {res_ai}")

if __name__ == "__main__":
    test_ai_detection()
