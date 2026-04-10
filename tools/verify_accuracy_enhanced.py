import sys
from pathlib import Path
import json
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import pandas as pd

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from modules.ai_text_detector import detect_ai_assistance
from modules.semantic_similarity import SemanticSimilarityEngine
from modules.text_extractor import extract_text
from modules.text_preprocessor import clean_text

def verify_accuracy():
    print("=" * 60)
    print("AI Plagiarism Detection - Accuracy Verification (Target 80%+)")
    print("=" * 60)
    
    # Test 1: AI Detection (8 samples)
    print("\n[1] AI/Human Detection...")
    ai_samples = [
        # AI-like (1)
        ("The integration of artificial intelligence into higher education presents both significant opportunities and complex challenges. On one hand, LLMs can serve as personalized tutors, providing students with immediate feedback and alternative explanations of difficult concepts. On the other hand, the ease with which these models generate human-like prose raises concerns about academic integrity and the potential for deskilling in critical writing tasks. Educators must therefore develop new pedagogical strategies that leverage AI as a tool for cognitive augmentation rather than a replacement for independent thought.", 1),
        ("Machine learning algorithms have revolutionized data analysis by enabling predictive modeling at unprecedented scales across various industries including finance and healthcare.", 1),
        ("The principles of quantum computing promise exponential speedups for certain computational problems that are intractable for classical computers.", 1),
        ("Sustainable development requires balancing economic growth with environmental protection and social equity through integrated policy frameworks.", 1),
        # Human-like (0)
        ("Last summer, I visited my grandmother's house in the countryside. We spent most of the afternoons picking wild berries and talking about how much the village has changed since she was a little girl. It was a simple, quiet time that I'll always cherish, far away from the noise of the city and the constant pinging of my smartphone.", 0),
        ("Yesterday I woke up late and missed the bus. My coffee was cold anyway so I made another cup and read the news online before work.", 0),
        ("My dog chased the cat around the yard until it climbed the tree and stayed there meowing for hours.", 0),
        ("I love reading books on rainy days with a hot cup of tea and some biscuits by the window.", 0),
    ]
    
    y_true = [label for _, label in ai_samples]
    y_pred = []
    for i, (text, _) in enumerate(ai_samples):
        result = detect_ai_assistance(text)
        pred = 1 if result['probability'] >= 50 else 0
        y_pred.append(pred)
        print(f"AI {i+1}: True={y_true[i]} Pred={pred} Prob={result['probability']:.1f}%")
    
    acc_ai = accuracy_score(y_true, y_pred)
    p, r, f1_ai, _ = precision_recall_fscore_support(y_true, y_pred, average='binary', zero_division=0)
    ai_pass = f1_ai >= 0.8
    print(f"\nAI Metrics: Acc={acc_ai:.1%} F1={f1_ai:.1%} [ {'PASS (OK)' if ai_pass else 'FAIL (Review)'} ]")
    
    # Test 2: Plagiarism Detection
    print("\n[2] Plagiarism Detection...")
    dataset_base = Path(__file__).resolve().parent.parent / "datasets/benchmark_dataset/pan-plagiarism-corpus-2011-1/external-detection-corpus/source-document/part1"
    engine = SemanticSimilarityEngine(dataset_base.parent.parent)
    
    # Self tests (high similarity)
    test_paths = [
        dataset_base / "source-document00002.txt",
        dataset_base / "source-document00009.txt", 
        dataset_base / "source-document00022.txt",
        dataset_base / "source-document00027.txt",
    ]
    y_true_plag = []
    y_pred_plag = []
    for path in test_paths:
        if path.exists():
            text = clean_text(extract_text(path))
            score, _ = engine.check_plagiarism(text)
            is_plag_pred = 1 if score >= 30 else 0  # Lower threshold since self-match may vary
            y_true_plag.append(1)
            y_pred_plag.append(is_plag_pred)
            print(f"Self {path.name}: {score:.1f}% {'DETECTED' if is_plag_pred else 'MISSED'}")
        else:
            print(f"Skip missing {path.name}")
    
    # Non-plag texts
    non_plag_texts = [
        "This is a completely unrelated short text for testing.",
        "Hello world example sentence not from dataset.",
        "Random words: apple banana cherry dog elephant.",
        "Another original paragraph with no source match.",
    ]
    for text in non_plag_texts:
        score, _ = engine.check_plagiarism(text)
        is_plag_pred = 1 if score >= 30 else 0
        y_true_plag.append(0)
        y_pred_plag.append(is_plag_pred)
        print(f"Non-plag: {score:.1f}% {'FALSE POS' if is_plag_pred else 'CORRECT'}")
    
    acc_plag = accuracy_score(y_true_plag, y_pred_plag)
    print(f"\nPlagiarism Accuracy: {acc_plag:.1%} [ {'PASS (OK)' if acc_plag >= 0.8 else 'FAIL (Review)'} ]")
    
    # Overall
    overall_pass = ai_pass and acc_plag >= 0.8
    results = {
        "timestamp": pd.Timestamp.now().isoformat(),
        "ai_f1": round(f1_ai, 3),
        "ai_accuracy": round(acc_ai, 3),
        "plag_accuracy": round(acc_plag, 3),
        "overall_pass_80": overall_pass
    }
    
    out_path = Path(__file__).resolve().parent.parent / "verification_results.txt"
    out_path.write_text(f"Accuracy Verification Results (Target 80%+)\n{'='*50}\n{json.dumps(results, indent=2)}\nRun at: {results['timestamp']}")
    print(f"\nResults saved to verification_results.txt")
    print(f"Overall: {'PASS (80%+ achieved)' if overall_pass else 'Review for improvements'}")

if __name__ == "__main__":
    verify_accuracy()
