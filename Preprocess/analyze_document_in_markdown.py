## This script is modified from "Samples for Azure Document Intelligence client library for Python"
## Please ref the origin sample code:https://github.com/Azure/azure-sdk-for-python/blob/main/sdk/documentintelligence/azure-ai-documentintelligence/samples/sample_analyze_documents_output_in_markdown.py

import os
import json
import argparse
import logging

logging.basicConfig(level=logging.INFO, filename='log_layout_md.txt',
    format='[%(asctime)s %(levelname)-8s] %(message)s',
    datefmt='%Y%m%d %H:%M:%S',
)

def analyze_documents_output_in_markdown(pdf, output_folder):
    """Analyze a PDF document using Azure Document Intelligence and output in Markdown format.

    This function takes a PDF file and uses Azure's Document Intelligence service 
    to extract its layout and convert it to Markdown. The extracted content is 
    saved as a JSON file in the specified output folder.

    Args:
        pdf (str): Path to the input PDF file to be analyzed.
        output_folder (str): Directory where the extracted Markdown content 
                             will be saved as a JSON file.

    Returns:
        None

    Raises:
        Azure specific exceptions:
        - HttpResponseError: If there are issues with the Azure Document Intelligence service
        - FileNotFoundError: If the input PDF file cannot be found
        - PermissionError: If there are issues with file or folder permissions

    Note:
        - Requires Azure Document Intelligence endpoint and API key to be set 
          as environment variables
        - The output is a JSON file with the filename format: 
          '{original_filename}_layout.json'
    """
    # [START analyze_documents_output_in_markdown]
    from azure.core.credentials import AzureKeyCredential
    from azure.ai.documentintelligence import DocumentIntelligenceClient
    from azure.ai.documentintelligence.models import ContentFormat, AnalyzeResult

    endpoint = os.getenv("DOCUMENTINTELLIGENCE_ENDPOINT")
    key = os.getenv("DOCUMENTINTELLIGENCE_API_KEY")

    document_intelligence_client = DocumentIntelligenceClient(endpoint=endpoint, credential=AzureKeyCredential(key))

    path_to_document = pdf

    filename = os.path.splitext(os.path.basename(path_to_document))[0]

    with open(path_to_document, "rb") as f:
        poller = document_intelligence_client.begin_analyze_document(
            "prebuilt-layout", analyze_request=f, 
            content_type="application/octet-stream",
            output_content_format=ContentFormat.MARKDOWN
        )
    result: AnalyzeResult = poller.result()    

    logging.info(f"Here's the full content in format {result.content_format}:{filename}")
    with open(f"{output_folder}/{filename}_layout.json", "w", encoding="utf-8") as output_file:
        json.dump({f"{filename}":f"{result.content}"}, output_file, ensure_ascii=False, indent=4)
    # [END analyze_documents_output_in_markdown]


if __name__ == "__main__":
    """
    Script entry point for analyzing PDF documents using Azure Document Intelligence.

    Command-line arguments:
    - --pdf_path (required): Path to the input PDF file
    - --output_path (required): Path to the output folder for saving results

    Handles:
    - Loading environment variables
    - Parsing command-line arguments
    - Error handling for Azure Document Intelligence service
    """
    from azure.core.exceptions import HttpResponseError
    from dotenv import find_dotenv, load_dotenv

    parser = argparse.ArgumentParser(description="Use Azure Document Intelligence layout model")
    parser.add_argument('--pdf_path', type=str, required=True, help='Path to pdf file.')
    parser.add_argument('--output_path', type=str, required=True, help='Path to output folder.')
    args = parser.parse_args()

    try:
        load_dotenv(find_dotenv())
        analyze_documents_output_in_markdown(args.pdf_path, args.output_path)
    except HttpResponseError as error:
        # Examples of how to check an HttpResponseError
        # Check by error code:
        if error.error is not None:
            if error.error.code == "InvalidImage":
                print(f"Received an invalid image error: {error.error}")
            if error.error.code == "InvalidRequest":
                print(f"Received an invalid request error: {error.error}")
            # Raise the error again after printing it
            raise
        # If the inner error is None and then it is possible to check the message to get more information:
        if "Invalid request".casefold() in error.message.casefold():
            print(f"Uh-oh! Seems there was an invalid request: {error}")
        # Raise the error again
        raise