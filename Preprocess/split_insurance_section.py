"""
This module provides a utility function to split text into structured sections
based on a specific pattern and returns the structured sections as a JSON-formatted
string. It is intended for processing legal or insurance text documents that contain
labeled sections marked by phrases such as "第...條". The structured output is saved in json.
"""

import re
import json

def split_by_section(text):
    """
    Split a given text into sections by matching the pattern "\n第.{1,7}條".

    Args:
        text (str): The text to be split.

    Returns:
        list[dict]: A list of dictionaries, each representing a section. Each dictionary contains the keys "title", "sequence_number", and "content".
    """
    pattern = r"\n第.{1,7}條"  # Pattern to match sections like \n第...條
    matches = re.split(f"({pattern})", text)  # Split and keep delimiters

    sections = []
    before_match = 0
    section_seq = 1
    
    # Add content before the first match as the first section with a default title
    if matches[0].strip():  # Check if there's non-empty content before the first match
        sections.append({
            "title": "no section name",
            "sequence_number": 1,
            "content": matches[0].strip()
        })
        before_match += 1

    # Loop through each match to create sections
    for seq in range(1, len(matches), 2):  # Start at first matched pattern
        section_title = matches[seq].strip()  # The matched pattern (e.g., \n第...條)
        section_content = matches[seq + 1].strip() if seq + 1 < len(matches) else ""  # Content after the match
        sections.append({
            "title": section_title, 
            "sequence_number": section_seq + before_match,
            "content": section_title + ": " + section_content
        })

        section_seq += 1
    
    return sections


with open("./raw_json/corpus_insurance.json", "r", encoding="utf-8") as f:
    json_content = json.load(f)

    after_split = {}
    
    for doc in json_content:
        split_result = split_by_section(json_content[doc])
        
        after_split[doc] = {
            "metadata": {
                "category": "insurance",
                "source_id": doc
            },
            "sections": split_result
        }

with open("./chunk_json/insurance_chunk.json","w", encoding="utf-8") as output:
    json.dump(after_split, output,ensure_ascii=False, indent=4)

