"""
This module provides functionality to preprocess financial data queries, primarily by transforming user input into a structured format
for further data processing and retrieval. The module defines a `QueryDict` TypedDict structure to represent key components of a
financial query, including fields for the company name, year, season (quarter), specific information to query, and a list of relevant keywords.

The primary function, `create_finance_query_parser`, initializes a language model with custom prompts tailored to handle financial
terminology, especially in the Chinese language. It returns a query parsing pipeline that interprets user input in a standardized format.

Classes:
    QueryDict: A TypedDict representing the components of a financial query.

Functions:
    create_finance_query_parser: Configures and returns a parser that structures financial queries for streamlined data retrieval.
"""

import json
from typing import List

from langchain_community.chat_models import ChatLlamaCpp
from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate
from tqdm import tqdm
from typing_extensions import Annotated, TypedDict


class QueryDict(TypedDict):
    """A TypedDict representing the components of a financial query, including company name, year, season, scenario, and keywords."""

    company: Annotated[str, ..., "公司名稱"]
    year: Annotated[str, ..., "年份"]
    season: Annotated[str, ..., "季度"]
    scenario: Annotated[str, ..., "目標查詢資訊"]
    keyword: Annotated[List[str], ..., "關鍵字與專有名詞列表"]


def create_finance_query_parser(llm=None):
    """
    Creates a structured query parser for financial data requests, converting user queries into a structured format
    with fields like company name, year, quarter, main query, and relevant keywords. Uses a language model configured
    with example prompts to handle finance-specific terminology and output in Chinese.

    Returns:
        A prompt pipeline that processes financial queries into a structured format for easy data retrieval.
    """
    if not llm:
        llm = ChatLlamaCpp(
            temperature=0.01,
            top_p=0.95,
            model_path="qwen2.5-3b-instruct-q8_0.gguf",
            n_ctx=2048,
            max_token=400,
            n_gpu_layers=-1,
            n_batch=512,
        )

    structured_llm = llm.with_structured_output(QueryDict)

    example_prompt = ChatPromptTemplate(
        [
            ("human", "{input}"),
            ("ai", "{output}"),
        ]
    )

    examples = [
        {
            "input": "聯電在2023年第1季的營業利益是多少？",
            "output": {
                "company": "聯電",
                "year": "2023年",
                "season": "第1季",
                "scenario": "營業利益是多少",
                "keyword": ["營業利益"],
            },
        },
        {
            "input": "光寶科技股份有限公司2023年第1季的合併財務報告中，部分非重要子公司未經會計師核閱的資產總額占合併資產總額的百分比是多少？",
            "output": {
                "company": "光寶科技股份有限公司",
                "year": "2023年",
                "season": "第1季",
                "scenario": "合併財務報告中，部分非重要子公司未經會計師核閱的資產總額占合併資產總額的百分比",
                "keyword": ["合併財務報告", "資產總額"],
            },
        },
        {
            "input": "長榮於2022年第3季的合併權益變動表中，歸屬於母公司業主之本期綜合損益總額為多少新台幣仟元？",
            "output": {
                "company": "長榮",
                "year": "2022年",
                "season": "第3季",
                "scenario": "合併權益變動表中，歸屬於母公司業主之本期綜合損益總額",
                "keyword": ["合併權益變動表", "綜合損益"],
            },
        },
        {
            "input": "2022年第3季，聯電公司及子公司因進口機器設備開立但未使用的信用狀約為多少億元？",
            "output": {
                "company": "聯電公司及子公司",
                "year": "2022年",
                "season": "第3季",
                "scenario": "進口機器設備開立但未使用的信用狀約為多少億元",
                "keyword": ["信用狀"],
            },
        },
        {
            "input": "國巨2022年第三季關係人交易中，與關聯企業相關的銷貨收入是多少？",
            "output": {
                "company": "國巨",
                "year": "2022年",
                "season": "第3季",
                "scenario": "關係人交易中，與關聯企業相關的銷貨收入",
                "keyword": ["銷貨收入"],
            },
        },
        {
            "input": "截至2023年第3季，智邦公司因進出口貨物需要由銀行出具保證函予海關之金額為多少？",
            "output": {
                "company": "智邦公司",
                "year": "2023年",
                "season": "第3季",
                "scenario": "進出口貨物需要由銀行出具保證函予海關之金額",
                "keyword": ["進出口貨物", "保證函", "海關"],
            },
        },
        {
            "input": "在聯發科2022年第1季的合併財務報表中，有哪幾項專利侵權訴訟被提及，涉及哪些專利號碼？",
            "output": {
                "company": "聯發科",
                "year": "2022年",
                "season": "第1季",
                "scenario": "合併財務報表中提及的專利侵權訴訟及涉及的專利號碼",
                "keyword": ["合併財務報表", "專利侵權訴訟", "專利號碼"],
            },
        },
        {
            "input": "在鴻海2023年第1季的文件中，提到哪些部門被辨認為應報導部門？",
            "output": {
                "company": "鴻海",
                "year": "2023年",
                "season": "第1季",
                "scenario": "提到的應報導部門",
                "keyword": ["應報導部門"],
            },
        },
        {
            "input": "在2022年第3季度，瑞昱公司有哪些專利訴訟案件被提起且目前正在處理中？",
            "output": {
                "company": "瑞昱公司",
                "year": "2022年",
                "season": "第3季",
                "scenario": "專利訴訟案件被提起且正在處理中",
                "keyword": ["專利訴訟"],
            },
        },
        {
            "input": "瑞昱在2022年第1季的財報中提到，BANDSPEED, LLC於哪一地區法院控告瑞昱產品侵害其專利權？",
            "output": {
                "company": "瑞昱",
                "year": "2022年",
                "season": "第1季",
                "scenario": "BANDSPEED, LLC於哪一地區法院控告瑞昱侵害專利權",
                "keyword": ["專利侵權"],
            },
        },
        {
            "input": "研華在2023年第3季為興建林口園區第三期工程已簽約但尚未發生之資本支出總計是多少？",
            "output": {
                "company": "研華",
                "year": "2023年",
                "season": "第3季",
                "scenario": "興建林口園區第三期工程已簽約但尚未發生之資本支出總計是多少",
                "keyword": ["林口園區", "第三期工程", "資本支出"],
            },
        },
    ]

    few_shot_prompt = FewShotChatMessagePromptTemplate(
        example_prompt=example_prompt,
        examples=examples,
    )

    prompt = ChatPromptTemplate(
        [
            (
                "system",
                """
                    你是一個專注於財務查詢的語義處理助手，解析用戶查詢並將其轉化為結構化格式，
                    包含「公司名稱」、「年份」、「季度」、「目標查詢資訊」和「關鍵字與專有名詞列表」等欄位，以便於後續的向量檢索處理。\n
                    需被包含在keywords內的重要關鍵字和專有名詞列表包含但不限於：營業利益、淨現金流、合併資產、綜合損益、稅前淨利、
                    權益變動、現金流、營業收入、基本每股盈餘、非控制權益、投資活動、信用狀、綜合損益表、合併權益變動表、合併財務報表等。
                    你的回答必須是全部都是**正體中文**。
                """,
            ),
            few_shot_prompt,
            ("human", "{query}"),
        ]
    )

    return prompt | structured_llm


def query_preprocessor(finance_question_set):
    """
    Preprocess a list of finance questions and parse them into structured format

    Args:
        finance_question_set (list[dict]): A list of dictionaries, each contains a question and its category

    Returns:
        list[dict]: The input list with an additional key "parsed_query" containing the structured output from the language model
    """
    chain = create_finance_query_parser()

    for Q in tqdm(finance_question_set):
        Q["parsed_query"] = chain.invoke({"query": Q["query"]})

    return finance_question_set


if __name__ == "__main__":
    with open("./競賽資料集/dataset/preliminary/questions_example.json", "r") as q:
        question_set = json.load(q)
    finance_question_set = [
        item for item in question_set["questions"] if item["category"] == "finance"
    ]
    finance_question_set = query_preprocessor(finance_question_set)

    with open("./test_parsed_query_v4.json", "w") as output:
        json.dump(finance_question_set, output, ensure_ascii=False, indent=4)
