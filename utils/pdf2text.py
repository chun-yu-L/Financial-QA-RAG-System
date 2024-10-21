import os
import re
import json
import argparse
from typing import Dict, List

from tqdm import tqdm
import pdfplumber

def merge_chinese_chars(text: str) -> str:
    """
    Merge individual Chinese characters into proper lexemes/phrases
    """
    # Pattern to identify clusters of single Chinese characters with potential spaces
    pattern = r'(?<=[\u4e00-\u9fff])\s+(?=[\u4e00-\u9fff])'
    
    # Remove spaces between Chinese characters
    merged_text = re.sub(pattern, '', text)
    
    return merged_text

def extract_lines_from_page(page) -> List[str]:
    """
    Extracts text from a page line by line, with enhanced handling for Chinese text
    """
    # Extract text with original spacing
    text = page.extract_text(x_tolerance=8, y_tolerance=8, keep_blank_chars=True)
    
    # Process each line
    lines = []
    for line in text.split('\n'):
        if line.strip():
            # Merge Chinese characters appropriately
            processed_line = merge_chinese_chars(line.strip())
            lines.append(processed_line)
    
    return lines

def generate_debug_image(page, output_folder: str, page_num: int):
    """
    Generates debug image showing word locations for visual inspection
    """
    img = page.to_image()
    img.reset().draw_rects(page.extract_words()).save(
        f"{output_folder}/page_{page_num}_words.png")

def create_debug_report(lines: List[str], page_num: int) -> str:
    """
    Creates a simple markdown report showing extracted lines
    """
    md = f"# Page {page_num + 1} Extracted Lines\n\n"
    md += "```\n"
    for i, line in enumerate(lines, 1):
        md += f"{i:03d}: {line}\n"
    md += "```\n"
    return md

def analyze_text_positions(page):
    """
    Analyzes the positions of text elements for debugging
    """
    words = page.extract_words(x_tolerance=3, y_tolerance=3)
    return {
        'word_count': len(words),
        'word_positions': [
            {
                'text': w['text'],
                'x0': w['x0'],
                'x1': w['x1'],
                'y0': w['top'],
                'y1': w['bottom'],
                'width': w['x1'] - w['x0']
            }
            for w in words
        ]
    }

def generate_debug_data(page, lines: List[str], page_num: int) -> Dict:
    """
    Generates comprehensive debug data
    """
    text_positions = analyze_text_positions(page)
    
    return {
        'page_number': page_num + 1,
        'extracted_lines': lines,
        'text_analysis': text_positions
    }

def read_pdf(pdf_loc: str, output_folder: str, page_infos: list = None) -> Dict:
    """
    Reads a PDF with enhanced Chinese text handling and debugging
    """
    os.makedirs(output_folder, exist_ok=True)
    
    with pdfplumber.open(pdf_loc) as pdf:
        pages = pdf.pages[page_infos[0]:page_infos[1]] if page_infos else pdf.pages
        file_name = os.path.splitext(os.path.basename(pdf_loc))[0]
        last_directory = os.path.basename(os.path.dirname(pdf_loc))
        
        result = {
            'metadata': {
                'category': last_directory,
                'source': file_name
            },
            'pages': []
        }
        
        for i, page in enumerate(pages):
            # Extract lines with enhanced Chinese handling
            lines = extract_lines_from_page(page)
            
            # Generate debug data
            debug_data = generate_debug_data(page, lines, i)
            
            # Store page data
            page_data = {
                'page_number': i + 1,
                'lines': lines
            }
            result['pages'].append(page_data)
            
            # Save debug data
            debug_file = os.path.join(output_folder, f'page_{i+1}_debug.json')
            with open(debug_file, 'w', encoding='utf-8') as f:
                json.dump(debug_data, f, ensure_ascii=False, indent=2)
            
            # Generate debug image
            img = page.to_image()
            img.reset().draw_rects(page.extract_words(
                x_tolerance=3, y_tolerance=3
            )).save(f"{output_folder}/page_{i+1}_words.png")
        
        return result

def load_data(source_path: str, output_base_path: str) -> Dict:
    masked_file_ls = [f for f in os.listdir(source_path) if f.endswith('.pdf')]
    corpus_dict = {}
    
    for file in tqdm(masked_file_ls):
        try:
            file_id = int(file.replace('.pdf', ''))
            full_path = os.path.join(source_path, file)
            output_folder = os.path.join(output_base_path, f'debug_{file_id}')
            corpus_dict[file_id] = read_pdf(full_path, output_folder)
        except Exception as e:
            print(f"Error processing {file}: {str(e)}")
    
    return corpus_dict

def save_corpus_to_json(corpus_dict: Dict, output_path: str, filename: str):
    output_file = os.path.join(output_path, filename)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(corpus_dict, f, ensure_ascii=False, indent=4)
    print(f"Saved corpus to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Extract PDF text line by line with debugging')
    parser.add_argument('--output_path', type=str, required=True, help='Path for output files')
    parser.add_argument('--source_path', type=str, required=True, help='Source data path')
    parser.add_argument('--debug_path', type=str, required=True, help='Path for debug outputs')

    args = parser.parse_args()

    # Process insurance corpus
    source_path_insurance = os.path.join(args.source_path, 'insurance')
    debug_path_insurance = os.path.join(args.debug_path, 'insurance')
    corpus_dict_insurance = load_data(source_path_insurance, debug_path_insurance)
    
    # Process finance corpus
    source_path_finance = os.path.join(args.source_path, 'finance')
    debug_path_finance = os.path.join(args.debug_path, 'finance')
    corpus_dict_finance = load_data(source_path_finance, debug_path_finance)

    # Save corpora
    save_corpus_to_json(corpus_dict_insurance, args.output_path, 'corpus_insurance.json')
    save_corpus_to_json(corpus_dict_finance, args.output_path, 'corpus_finance.json')