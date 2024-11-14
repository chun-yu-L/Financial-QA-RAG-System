"""
PDF Text Processor
=================
Features:
---------
- Extracts text from PDF files while preserving proper formatting
- Special handling for Chinese character spacing
- Processes multiple PDFs with progress tracking
- Organizes extracted text into separate insurance and finance corpora
- Saves processed text as structured JSON files

Dependencies:
------------
- pdfplumber: For PDF text extraction
- argparse: For command-line argument parsing

Usage:
------
Run the script from command line with required arguments:
    python script.py --output_path <output_directory> --source_path <source_directory>

The source directory should contain two subdirectories:
- insurance/: Contains insurance-related PDF files
- finance/: Contains finance-related PDF files

Output:
-------
The script generates two JSON files in the specified output directory:
- raw_json/corpus_insurance.json: Contains extracted text from insurance PDFs
- corpus_finance.json: Contains extracted text from finance PDFs

Each JSON file contains a dictionary mapping document names to their extracted text content.

Example:
--------
    python pdf_to_text.py --source_path ./競賽資料集/reference/ --output_path ./
"""

import os
import re
import json
import argparse
from typing import Dict

from tqdm import tqdm
import pdfplumber

def merge_chinese_chars(text: str) -> str:
    """
    Merge individual Chinese characters into proper lexemes/phrases.
    
    Args:
    >>>     text (str): Input text containing Chinese characters potentially separated by spaces
    Returns:
    >>>     str: Text with spaces between Chinese characters removed while preserving other spacing
        
    Example:
    >>> merge_chinese_chars("你 好 世界 hello")
    >>> "你好 世界 hello"
    """
    pattern = r'(?<=[\u4e00-\u9fff])\s+(?=[\u4e00-\u9fff])'
    merged_text = re.sub(pattern, '', text)
    return merged_text

def extract_text_from_page(page) -> str:
    """
    Extracts and processes text from a single PDF page.
    
    Args:
    >>>     page: A pdfplumber page object containing the PDF page content

    Returns:
    >>>     str: Processed text from the page with appropriate spacing and merged Chinese characters

    """
    text = page.extract_text(x_tolerance=8, y_tolerance=8, keep_blank_chars=True)
    if text:
        return merge_chinese_chars(text.strip())
    return ""

def read_pdf(pdf_loc: str) -> str:
    """
    Reads a PDF file and extracts all text into a single string.

    Args:
    >>>     pdf_loc (str): File path to the PDF document

    Returns:
    >>>     str: Concatenated text content of the PDF

    Example:
    >>>     read_pdf("document.pdf")\n\n

    """
    with pdfplumber.open(pdf_loc) as pdf:
        pages = pdf.pages
        
        # Extract and concatenate text from all pages
        full_text = ""
        for page in pages:
            page_text = extract_text_from_page(page)
            if page_text:
                full_text += page_text + " "
        
        return full_text.strip()

def load_data(source_path: str) -> Dict[str, str]:
    """
    Loads and processes all PDF files from a specified directory.
    
    Args:
    >>> source_path (str): Directory path containing PDF files

    Returns:
    >>> Dict[str, str]: Dictionary mapping document names to their text content

    Example:
    >>> load_data("path/to/pdfs")

    """
    pdf_files = [f for f in os.listdir(source_path) if f.endswith('.pdf')]
    corpus_dict = {}
    
    for file in tqdm(pdf_files):
        try:
            doc_name = file.replace('.pdf', '')
            full_path = os.path.join(source_path, file)
            corpus_dict[doc_name] = read_pdf(full_path)
        except Exception as e:
            print(f"Error processing {file}: {str(e)}")
    
    return corpus_dict

def save_corpus_to_json(corpus_dict: Dict[str, str], output_path: str, filename: str):
    """
    Saves the processed corpus to a JSON file.
    
    Args:
    >>> corpus_dict (Dict[str, str]): Dictionary mapping document names to their text content
    >>> output_path (str): Directory path where the JSON file will be saved
    >>> filename (str): Name of the output JSON file

    Example output format:\n
    >>> {
        "doc1": "text content of document 1...",
        "doc2": "text content of document 2..."
    }

    """
    os.makedirs(output_path, exist_ok=True)
    output_file = os.path.join(output_path, filename)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(corpus_dict, f, ensure_ascii=False, indent=4)
    print(f"Saved corpus to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Extract PDF text into single strings')
    parser.add_argument('--output_path', type=str, required=True, help='Path for output files')
    parser.add_argument('--source_path', type=str, required=True, help='Source data path')

    args = parser.parse_args()

    # Process insurance corpus
    source_path_insurance = os.path.join(args.source_path, 'insurance')
    corpus_dict_insurance = load_data(source_path_insurance)
    
    # Process finance corpus
    source_path_finance = os.path.join(args.source_path, 'finance')
    corpus_dict_finance = load_data(source_path_finance)

    # Save corpora
    save_corpus_to_json(corpus_dict_insurance, args.output_path, 'raw_json/corpus_insurance.json')
    save_corpus_to_json(corpus_dict_finance, args.output_path, 'corpus_finance.json')