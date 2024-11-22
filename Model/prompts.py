from langchain_core.prompts import ChatPromptTemplate

faq_ans_prompt = ChatPromptTemplate(
    [
        (
            "human",
            """
            你是一位精通金融問答的專家，回答時請嚴格根據以下<參考資料>內容，避免加入任何未提供的資訊或主觀推測。使用正體中文回答。

            問題：{question}

            <參考資料>
            {context}
            </參考資料>

            精準的專家回答：
            """,
        )
    ]
)

insurance_ans_prompt = ChatPromptTemplate(
    [
        (
            "human",
            """
            你是一位精通保險問答的專家，回答時僅依據以下<保單資訊>內容，避免加入任何未提供的資訊或主觀推測。使用正體中文回答。

            問題：{question}

            <保單資訊>
            {context}
            </保單資訊>

            精準的專家回答：
            """,
        )
    ]
)


# 提供 import 的接口
PROMPTS = {
    "faq_ans": faq_ans_prompt,
    "insurance_ans": insurance_ans_prompt
}
