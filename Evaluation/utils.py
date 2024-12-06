from openai import AzureOpenAI
import logging

def get_prompt():
    """
    Return the prompt and example strings for the evaluation task.
    """
    INSTRUCTIONS = """
    # 任務:
    你是專業的繁體中文助手，你會得到一個問題、一個模型預測和一個真實答案列表，判斷模型預測是否與真實答案列表中的任何答案匹配。按照以下步驟進行判斷。
    1. 如果模型預測與真實答案列表中的任何提供答案匹配，"Accuracy" 為 "True"；否則，"Accuracy" 為 "False"。
    2. 如果模型預測表示它**不知道**，無法回答問題或沒有足夠的信息，"Accuracy" 為 "False"。
    # 輸出:
    只需回應一個包含 "Accuracy" 欄位的 JSON 字符串，該欄位的值為 "True" 或 "False"。
    """

    IN_CONTEXT_EXAMPLES = """
    # 範例:
    Question: 3 分鐘 05 秒是多少秒？
    Ground truth: 185 秒
    Prediction: 3 分鐘 05 秒是 185 秒。
    Accuracy: True

    Question: 請問玉山銀行的總部位於台灣的哪個縣市？
    Ground truth: 台北市
    Prediction: 玉山銀行的總部位於新竹市。
    Accuracy: False

    Question: 請問在2023年第三季的聯電財務報告中，哪些公司被列為聯電的關聯企業及其他關係人？
    Ground truth: 智源科技, 新興電子
    Prediction: 不知道
    Accuracy: False
    """
    return INSTRUCTIONS, IN_CONTEXT_EXAMPLES


class REFEREE_Model():
    def __init__(self, azure_deployment_name, azure_openai_api_key, api_version, azure_endpoint):
        
        """
        Initialize the REFEREE model with the given parameters.

        Parameters:
        azure_deployment_name (str): The name of the Azure OpenAI deployment.
        azure_openai_api_key (str): The API key for Azure OpenAI.
        api_version (str): The version of the Azure OpenAI API.
        azure_endpoint (str): The endpoint URL of the Azure OpenAI API.
        """
        self.gpt_engine = azure_deployment_name
        self.gpt_client = AzureOpenAI(
                api_key=azure_openai_api_key,  
                api_version=api_version,
                azure_endpoint=azure_endpoint
        )


    def create_prompt(self, system_prompt, query, ground_truth, prediction):
        '''
        system_prompt: str
        user_prompt: str
        '''
        messages = [
            {
                "role": "system", 
                "content": system_prompt},
            {
                "role": "user",
                "content": f"Question: {query}\n Ground truth: {ground_truth}\n Prediction: {prediction}\n",
            },
        ]
        return messages


    def generate_response(self, messages, idx):
        """
        依據傳入的系統提示、問題、標準答案與模型預測答案，使用 GPT 產生回應。

        Args:
            messages (list): 系統提示、問題、標準答案與模型預測答案的列表
            idx (int): 問題的 id

        Returns:
            str: GPT 產生的回應
        """
        logging.basicConfig(level=logging.INFO, filename='log_generate_response.txt',
            format='[%(asctime)s %(levelname)-8s] %(message)s',
            datefmt='%Y%m%d %H:%M:%S',
        )

        chat_completion = self.gpt_client.chat.completions.create(messages=messages,
                                                                  model=self.gpt_engine)
        if chat_completion.choices[0].finish_reason == "stop":
            response = chat_completion.choices[0].message.content

            logging.info(f"qid: {idx} \t {response}")
            # print(f"qid: {idx} \t {response}")
            return response
        else:
            logging.error(f"Failed to generate response. \t qid: {idx}")
            response = None
            return response
