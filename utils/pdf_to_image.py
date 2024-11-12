import fitz
import os

# Path to text file contains only PDF files' name
# note: to list PDF files please see pdf_tools.py 
with open("<WAIT_FOR_CONVERT>.txt",'r') as f:
    for pdf in f:
        # Add full path to the PDF files 
        pdffile = f"<FULL_PATH>/{pdf.strip()}"

        file_name = os.path.splitext(os.path.basename(pdffile))[0]
        doc = fitz.open(pdffile)
        zoom = 2 # output 1191*1684 image
        mat = fitz.Matrix(zoom, zoom)
        count = 0
        # Count variable is to get the number of pages in the pdf
        for i in range(len(doc)):
            val = f"{file_name}_{i+1}.png"
            page = doc.load_page(i)
            pix = page.get_pixmap(matrix=mat)
            pix.save(f"<OUTPUT_PATH>/{val}")
        doc.close()

## Example
## <WAIT_FOR_CONVERT>.txt
## 10.txt
## 101.txt
## 102.txt

## <FULL_PATH>
## ./2024_AI_cup/Financial-QA-RAG-System/

## <OUTPUT_PATH>
## ./finance