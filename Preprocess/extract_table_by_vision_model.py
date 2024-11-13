from typing import List, Optional
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

from langchain_openai import AzureChatOpenAI # langchain-openai == 0.1.25
from langchain_core.prompts import PromptTemplate # langchain == 0.2.16
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, SystemMessage

import mimetypes
import time
import json
import base64
import requests
import fnmatch
import logging
import os

load_dotenv()
logging.basicConfig(level=logging.INFO, filename='log_batch.txt',
	format='[%(asctime)s %(levelname)-8s] %(message)s',
	datefmt='%Y%m%d %H:%M:%S',
	)


def convert_image_to_data_url(image_path: str) -> Optional[str]:
    """
    Convert a local image file to a data URL format.
    Returns None if the file cannot be read or encoded.
    """
    try:
        # Determine the MIME type of the image
        mime_type, _ = mimetypes.guess_type(image_path)
        if mime_type is None:
            mime_type = 'image/png'  # default to png if type cannot be determined
            
        with open(image_path, "rb") as image_file:
            encoded_image = base64.b64encode(image_file.read()).decode('ascii')
            return f"data:{mime_type};base64,{encoded_image}"
    except Exception as e:
        print(f"Error converting image {image_path}: {str(e)}")
        return None

def initialize_chain(
    azure_endpoint: str,
    api_key: str,
    deployment_name: str,
    api_version: str
):
    """
    Initialize LangChain with Azure OpenAI configuration
    """
    if not all([azure_endpoint, api_key, deployment_name, api_version]):
        raise ValueError("All Azure OpenAI configuration parameters are required")
    
    model = AzureChatOpenAI(
        deployment_name=deployment_name,
        model_name="gpt-4o",  # Azure deployment model
        azure_endpoint=azure_endpoint,
        openai_api_version=api_version,
        openai_api_key=api_key,
        openai_api_type="azure",
        temperature=0,
        max_tokens=6000 # TPM < 8k
    )
    
    return model

def process_images_in_batches(
    image_paths: List[str],
    azure_config: dict,
    sys_prompt: str,
    file_number: str,
    batch_size: int = 1,
    delay: int = 15,
    output_dir: Optional[str] = None
) -> List[dict]:
    """
    Process images in batches with proper error handling and output saving
    """
    total_results = {}
    total_tokens = 0
    
    # Create output directory if specified
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    try:
        for i in range(0, len(image_paths), batch_size):
            batch = image_paths[i:i + batch_size]
            model = initialize_chain(**azure_config)
            # user_content = [{"type": "text","text": "\n"}]
            batch_messages = []

            for image_path in batch:
                try:
                    # Convert local image to data URL
                    image_data_url = convert_image_to_data_url(image_path)
                    if not image_data_url:
                        logging.error(f"Skipping {image_path} due to conversion error: {str(e)}")
                        break

                    # Append to user content for LLM input
                    # user_content.append({"type": "image_url", "image_url": {"url": image_data_url}})
                    # user_content.append({"type": "text","text": "\n"})

                    # Create messages with image content
                    messages = [
                        SystemMessage(content=sys_prompt),
                        HumanMessage(
                            content=[
                                {"type": "text","text": "\n"},
                                {"type": "image_url", "image_url": {"url": image_data_url}},
                                {"type": "text","text": "\n"}]
                        )
                    ]
                    
                    batch_messages.append(messages)

                except Exception as e:
                    logging.error(f"Error processing {image_path}: {str(e)}")
                    break
            
            # Check if user content contains all input images
            # expected_length = len(batch) * 2 + 1
            # if len(user_content) != expected_length:
            #     logging.error(f"Error processing {file_number}. User content lenght: {len(user_content)}, expected length {expected_length}")
            #     break

            try:
                # Get response and token usage
                response = model.batch(batch_messages)
                logging.info(f"{file_number} finished batch {i+3}/{len(image_paths)}")

                inner_count = 1
                for ai_message in response:
                    results = {}
                    output = ai_message.content
                    token_usage = ai_message.usage_metadata
                    
                    # Update token counts
                    input_tokens = token_usage.get('input_tokens', 0)
                    output_tokens = token_usage.get('output_tokens', 0)
                    total_tokens_this_call = token_usage.get('total_tokens', 0)
                    total_tokens += total_tokens_this_call
                        
                    print(f"\nToken usage for {file_number}_{i + inner_count}.png:")
                    print(f"Input tokens: {input_tokens}")
                    print(f"Output tokens: {output_tokens}")
                    print(f"Total tokens: {total_tokens_this_call}")
                    print(f"Running total tokens: {total_tokens}\n")
                    
                    # Log token usage
                    logging.info(f"Token usage for {file_number}_{i + inner_count}.png - "
                                f"Prompt: {input_tokens}, "
                                f"Completion: {output_tokens}, "
                                f"Total: {total_tokens_this_call}")

                    # Store result with file name as key
                    results[file_number + '_' + str(i + inner_count)] = output
                    total_results[file_number + '_' + str(i + inner_count)] = output

                    # Save output
                    if output_dir and file_number:
                        output_file = Path(output_dir) / f"{file_number}_{i + inner_count}_gpt_4o.json"

                        with open(output_file, "w", encoding='utf-8') as f:
                            json.dump(results, f, ensure_ascii=False, indent=4)

                        logging.info(f"Outputed {file_number}_{i + inner_count}")
                
                    logging.info(f"Processed file_{file_number}: {i + inner_count}/{len(image_paths)} images")
                    inner_count += 1

            except Exception as e:
                logging.error(f"Error while invoking LLM with batch: {file_number}. {str(e)}")

            # Sleep between batches if there are more images to process
            if i + batch_size < len(image_paths):
                print(f"Sleeping for {delay} seconds before next batch...")
                time.sleep(delay)
    
    except Exception as e:
        logging.error(f"Fatal error in batch processing: {str(e)}")
    
    # logging.info(f"Running total tokens: {total_tokens}\n")
    return total_results

def main():
    # Azure OpenAI Configuration
    azure_config = {
        "azure_endpoint": os.getenv("AZURE_OPENAI_ENDPOINT"),
        "api_key": os.getenv("AZURE_OPENAI_API_KEY"),
        "deployment_name": os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT"),
        "api_version": os.getenv("AZURE_OPENAI_API_VERSION")
    }
    
    # Validate Azure configuration
    missing_configs = [k for k, v in azure_config.items() if not v]
    if missing_configs:
        raise ValueError(f"Missing Azure OpenAI configurations: {', '.join(missing_configs)}")
    
    # Read to do list and save all images dict
    output_dir = "./gpt4o_output/"
    image_paths = {}

    with open("./five_up_page_finance.txt", 'r') as f:
        for line in f:
            file_name = line.strip()
            prefix = file_name[:-4]
            directory = "./finance_image"
            
            paths = []
            for filename in os.listdir(directory):
                if fnmatch.fnmatch(filename, f"{prefix}_*.png"):
                    image_path = os.path.normpath(os.path.join(directory, filename))
                    paths.append(image_path)
            
            if paths:
                image_paths[prefix] = paths

    # Iterate and get LLM responses
    for doc_num in image_paths:
        page_size = len(image_paths[doc_num])

        total_results = process_images_in_batches(
            image_paths=image_paths[doc_num],
            azure_config=azure_config,
            sys_prompt=os.getenv('GPT_4o_SYS_PROMPT'),
            file_number=doc_num,
            # batch_size=page_size,
            batch_size=3, # notice the endpoint rate limit
            delay=70,
            output_dir=output_dir
        )

        output_file = Path(output_dir) / f"total_{doc_num}_gpt_4o.json"
        with open(output_file, "w", encoding='utf-8') as f:
            json.dump(total_results, f, ensure_ascii=False, indent=4)

        logging.info(f"Processed {len(total_results)} docs successfully")
        logging.info(f"Sleeping for 70 seconds before next documents...")
        print(f"Processed {len(total_results)} docs successfully")
        print(f"Sleeping for 70 seconds before next documents...")
        time.sleep(70)


if __name__ == "__main__":
    main()