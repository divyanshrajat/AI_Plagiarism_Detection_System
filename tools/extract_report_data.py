import json
from pathlib import Path
import re

def extract_report_info(report_path):
    if not report_path.exists():
        return None
    
    content = report_path.read_text(encoding='utf-8', errors='ignore')
    
    # Extract Plagiarism Score
    plag_match = re.search(r'Overall plagiarism score: ([\d.]+)%', content)
    plag_score = plag_match.group(1) if plag_match else "N/A"
    
    # Extract AI Indicator
    ai_prob_match = re.search(r'Probability: ([\d.]+)%', content)
    ai_label_match = re.search(r'Label: (\w+)', content)
    
    ai_score = ai_prob_match.group(1) if ai_prob_match else "N/A"
    ai_label = ai_label_match.group(1) if ai_label_match else "N/A"
    
    return {
        "plagiarism_score": plag_score,
        "ai_score": ai_score,
        "ai_label": ai_label
    }

def generate_table():
    base_dir = Path(r'c:\Users\Lenovo\Desktop\AI_Plagiarism_Detection_System')
    uploads_file = base_dir / 'uploads' / 'index.json'
    reports_index_file = base_dir / 'reports' / 'index.json'
    reports_dir = base_dir / 'reports'
    
    if not uploads_file.exists():
        print("Uploads index not found.")
        return

    uploads = json.loads(uploads_file.read_text(encoding='utf-8'))
    
    reports_index = {}
    if reports_index_file.exists():
        try:
            reports_data = json.loads(reports_index_file.read_text(encoding='utf-8'))
            for r in reports_data:
                reports_index[r.get('path')] = r
        except:
            pass
    
    table_rows = []
    
    for upload in uploads:
        filename = upload.get('filename', 'N/A')
        uploader = upload.get('uploader', 'N/A')
        timestamp = upload.get('timestamp', 'N/A')
        report_filename = upload.get('report_path')
        
        plag_score = "N/A"
        ai_score = "N/A"
        ai_label = "N/A"
        
        if report_filename:
            report_path = reports_dir / report_filename
            info = extract_report_info(report_path)
            if info:
                plag_score = info['plagiarism_score']
                ai_score = info['ai_score']
                ai_label = info['ai_label']
            
            # Fallback to index if file extraction failed or returned N/A
            if plag_score == "N/A" and report_filename in reports_index:
                plag_score = str(reports_index[report_filename].get('score', "N/A"))
            if ai_score == "N/A" and report_filename in reports_index:
                ai_score = str(reports_index[report_filename].get('ai_score', "N/A"))
        
        # Format timestamp for better readability
        try:
            from datetime import datetime
            dt = datetime.strptime(timestamp, "%Y%m%d_%H%M%S")
            formatted_ts = dt.strftime("%Y-%m-%d %H:%M:%S")
        except:
            formatted_ts = timestamp

        table_rows.append(f"| {filename} | {uploader} | {formatted_ts} | {plag_score}% | {ai_score}% ({ai_label}) |")

    # Header
    print("| Document Name | Uploader | Timestamp | Plagiarism Score | AI Score (Label) |")
    print("| :--- | :--- | :--- | :--- | :--- |")
    for row in table_rows:
        print(row)

if __name__ == "__main__":
    generate_table()
