import os
from PyPDF2 import PdfReader
import pandas as pd


def count_pdf_pages_in_folder(folder_path):
    # Initialize lists to store data
    pdf_files = []
    page_counts = []
    
    # Get all PDF files in the folder
    files = [f for f in os.listdir(folder_path) if f.endswith('.pdf')]
    
    # Process each PDF file
    for pdf_file in files:
        pdf_path = os.path.join(folder_path, pdf_file)
        try:
            reader = PdfReader(pdf_path)
            pages = len(reader.pages)
            
            # Append to lists
            pdf_files.append(pdf_file)
            page_counts.append(pages)
            
            # print(f"Processed {pdf_file}: {pages} pages")
        except Exception as e:
            print(f"Error reading {pdf_file}: {e}")
    
    # Create DataFrame
    df = pd.DataFrame({
        'PDF_File': pdf_files,
        'Page_Count': page_counts
    })
    
    # Add total row
    total_pages = df['Page_Count'].sum()
    print(f"\nTotal number of pages in all PDFs: {total_pages}")
    
    # Sort by page count in descending order
    df = df.sort_values(by='Page_Count', ascending=False)
    
    return df

def analyze_page_distribution(df):
    # Get value counts of page numbers
    page_distribution = df['Page_Count'].value_counts().sort_index()
    
    # Create a DataFrame with the distribution
    distribution_df = pd.DataFrame({
        'Pages_Per_PDF': page_distribution.index,
        'Number_of_PDFs': page_distribution.values
    })
    
    # Calculate percentage
    total_pdfs = len(df)
    distribution_df['Percentage'] = (distribution_df['Number_of_PDFs'] / total_pdfs * 100).round(2)
    
    # Sort by Pages_Per_PDF
    distribution_df = distribution_df.sort_values('Pages_Per_PDF')
    
    return distribution_df

# Example usage
folder_path = "./reference/finance/"
result_df = count_pdf_pages_in_folder(folder_path)
distribution_df = analyze_page_distribution(result_df)

with open("five_up_page_finance.txt",'w') as f:
    for i in result_df[result_df.Page_Count > 4]['PDF_File'].sort_values():
        f.write(i+'\n')
