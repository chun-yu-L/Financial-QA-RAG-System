from langchain_core.prompts import ChatPromptTemplate

faq_ans_prompt = ChatPromptTemplate(
    [
        (
            "human",
            """
            你是一位精通金融問答的專家，請嚴格地根據<參考資料>回答問題，使用正體中文。

            問題：{question}

            <參考資料>
            {context}
            </參考資料>

            專家回答：
            """,
        )
    ]
)


# 提供 import 的接口
PROMPTS = {
    "faq_ans": faq_ans_prompt,
}
