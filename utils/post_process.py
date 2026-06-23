import re
from typing import List, Dict, Any

def post_process(raw_text: str, words_data: List[Dict[str, Any]], stamp_type: str) -> List[Dict[str, Any]]:
    """
    Extracts dates via Regex from the raw text string.
    
    Classifies date types based on Azure's natural reading order sequence:
    - First date found in an Entry stamp = Entry date.
    - Second date found in an Entry stamp = Until date.
    - Single date or Exit stamp dates = Matches the stamp_type.
    """
    # Keep the text clean without naive character replacements
    text = raw_text.upper()
    pattern = r'\b(\d{1,2})\s+(JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)\s+(\d{4})\b'
    
    matches = list(re.finditer(pattern, text))
    extracted_dates = []

    for index, match in enumerate(matches):
        date_str = match.group(0)
        
        # 1. Match tokens back to Azure words to determine the lowest word confidence
        date_tokens = date_str.split()
        token_confs = []
        
        for token in date_tokens:
            for wd in words_data:
                if token in wd["text"].upper():
                    token_confs.append(wd["conf"])
                    break
        
        # If no matching tokens are found in words_data, default to a neutral confidence flag
        final_conf = min(token_confs) if token_confs else 0.0
        
        # 2. Sequential Reading-Order Logic
        if stamp_type == "Entry":
            if index == 0:
                date_type = "Entry"
            elif index == 1:
                date_type = "Until"
            else:
                # Fallback flag for unexpected anomaly dates within a single region
                date_type = "Until"
        else:
            # If the region is an Exit stamp, all detected dates default to Exit
            date_type = stamp_type
                
        extracted_dates.append({
            "value": date_str,
            "type": date_type,
            "ocr_conf": final_conf
        })

    return extracted_dates