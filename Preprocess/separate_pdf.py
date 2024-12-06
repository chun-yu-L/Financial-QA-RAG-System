import os
from PyPDF2 import PdfReader, PdfWriter

def split_pdf(input_path):
    """Split a single PDF into individual pages with custom naming convention.
    
    This function reads a PDF file and creates a separate PDF for each page,
    saving them in a predefined directory with a specific naming format.

    Args:
        input_path (str): Absolute or relative path to the input PDF file.

    Returns:
        list: Absolute file paths of the created PDF pages.

    Raises:
        FileNotFoundError: If the input PDF file does not exist.
        PermissionError: If there are insufficient permissions to read or write files.
    """
    # Ensure the input path is absolute
    input_path = os.path.abspath(input_path)
    
    # Get the directory and filename without extension
    # directory = os.path.dirname(input_path)
    directory = "reference/finance_per_page"
    filename = os.path.splitext(os.path.basename(input_path))[0]
    
    # Open the PDF
    reader = PdfReader(input_path)
    
    # List to store created file paths
    created_files = []
    
    # Iterate through pages
    for page_num in range(len(reader.pages)):
        # Create a new PDF writer
        writer = PdfWriter()
        
        # Add the current page
        writer.add_page(reader.pages[page_num])
        
        # Create output filename 
        output_filename = f"{filename}_{page_num + 1}.pdf"
        output_path = os.path.join(directory, output_filename)
        
        # Write the page to a new PDF
        with open(output_path, 'wb') as output_pdf:
            writer.write(output_pdf)
        
        created_files.append(output_path)
        print(f"Created: {output_filename}")
    
    return created_files

def process_pdf_folder(folder_path):
    """Process all PDF files in a given folder by splitting them into individual pages.

    This function iterates through all PDF files in the specified directory,
    splitting each PDF into separate page-level PDF files.

    Args:
        folder_path (str): Absolute or relative path to the folder containing PDFs.

    Returns:
        list: Absolute file paths of all created PDF pages.

    Raises:
        NotADirectoryError: If the specified path is not a valid directory.
        PermissionError: If there are insufficient permissions to access the directory.
    """
    # Ensure the folder path is absolute
    folder_path = os.path.abspath(folder_path)
    
    # Validate folder exists
    if not os.path.isdir(folder_path):
        print(f"Error: {folder_path} is not a valid directory.")
        return []
    
    # List to store all created files
    all_created_files = []
    
    # Iterate through all files in the folder
    for filename in os.listdir(folder_path):
        # Check if the file is a PDF
        if filename.lower().endswith('.pdf'):
            # Full path to the PDF
            pdf_path = os.path.join(folder_path, filename)
            
            try:
                # Split the PDF and add to total list of created files
                created_files = split_pdf(pdf_path)
                all_created_files.extend(created_files)
            except Exception as e:
                print(f"Error processing {filename}: {e}")
    
    print(f"\nTotal PDFs processed: {len(all_created_files)} pages")
    return all_created_files

# Example usage
if __name__ == "__main__":
    pdf_folder_path = "reference/finance" # Replace with the path to your folder containing PDFs
    process_pdf_folder(pdf_folder_path)