import re
from typing import List, Dict, Any

def clean_ocr_date_string(raw_text: str) -> str:
    """Passes raw OCR text through a 4-stage cleaning funnel."""
    
    # 1. Punctuation Stripping: Replace everything that isn't A-Z or 0-9 with a space
    text = re.sub(r'[^A-Z0-9]', ' ', raw_text.upper())
    
    # 2. De-fragmentation: Remove spaces between isolated digits (e.g., "2 9" -> "29", "2 0 2 3" -> "2023")
    prev_text = ""
    while text != prev_text:
        prev_text = text
        # Looks for a digit, a space, and another digit, then squashes them
        text = re.sub(r'(?<=\b\d)\s+(?=\d\b)', '', text)
        
    # 3. Alphanumeric Correction
    words = text.split()
    valid_months = {"JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"}
    
    clean_words = []
    for word in words:
        if word in valid_months:
            clean_words.append(word)
        else:
            # Fix common OCR character confusions on numbers
            fixed = word.replace('O', '0').replace('I', '1').replace('S', '5').replace('Z', '2').replace('B', '8')
            
            # Strip trailing alphabetical garbage from numbers (e.g., "202MMS" -> "202")
            match = re.match(r'^(\d+)[A-Z]*$', fixed)
            if match:
                fixed = match.group(1)
                
            clean_words.append(fixed)
            
    return " ".join(clean_words)


def post_process(raw_text: str, words_data: List[Dict[str, Any]], stamp_type: str) -> List[Dict[str, Any]]:
    # Clean the raw text through the funnel
    cleaned_text = clean_ocr_date_string(raw_text)
    
    # 4. Forgiving Extraction
    # Day: 1 or 2 digits
    # Month: Exact match
    # Year: Matches EXACTLY 4 digits starting with 19 or 20. 
    # By not putting \b at the end of the year, it ignores trailing garbage digits like "202343"
    pattern = r'\b(\d{1,2})\s+(JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)\s+((?:19|20)\d{2})'
    
    matches = list(re.finditer(pattern, cleaned_text))
    extracted_dates = []

    for index, match in enumerate(matches):
        # Pad the day with a leading zero if necessary (e.g., "4" -> "04")
        day = match.group(1).zfill(2)
        month = match.group(2)
        year = match.group(3)
        
        date_str = f"{day} {month} {year}"
        
        # --- Sanity Check ---
        # If the OCR read a day greater than 31, it is invalid and we skip it
        if not (1 <= int(day) <= 31):
            continue
            
        # The remainder of your mapping and sequential logic remains identical
        date_tokens = date_str.split()
        token_confs = []
        
        for token in date_tokens:
            for wd in words_data:
                if token in wd["text"].upper():
                    token_confs.append(wd["conf"])
                    break
                    
        final_conf = min(token_confs) if token_confs else 0.0
        
        if stamp_type == "Entry":
            if index == 0:
                date_type = "Entry"
            elif index == 1:
                date_type = "Until"
            else:
                date_type = "Until"
        else:
            date_type = stamp_type
                
        extracted_dates.append({
            "value": date_str,
            "type": date_type,
            "ocr_conf": final_conf
        })

    return extracted_dates