from langgraph.graph import END, StateGraph

from graph.nodes import (
    faq_node,
    finance_generation,
    finance_retrieve,
    insurance_node,
    llm_eval_retrieve,
    route_question,
)
from graph.state import QAState


def build_workflow() -> callable:
    """
    構建 LangGraph 的工作流。

    Returns:
        callable: 構建完成的工作流執行器。
    """
    # 創建一個工作流圖
    workflow = StateGraph(QAState)

    # 添加起始節點
    workflow.add_node("start", lambda state: state)
    # 指定為起始節點
    workflow.set_entry_point("start")

    # 定義從起始點的條件邊
    workflow.add_conditional_edges(
        "start",  # 起始點
        route_question,  # 用於判斷路由的函數
        {
            "process_insurance": "process_insurance",
            "process_finance": "finance_retrieve",
            "process_faq": "process_faq",
        },
    )

    # 添加處理節點
    workflow.add_node("process_insurance", insurance_node)
    workflow.add_node("process_faq", faq_node)

    # 為每個節點設置結束點
    workflow.add_edge("process_insurance", END)
    workflow.add_edge("process_faq", END)

    # finance 流程
    workflow.add_node("finance_retrieve", finance_retrieve)
    workflow.add_node("finance_generation", finance_generation)

    workflow.add_conditional_edges(
        "finance_retrieve",
        llm_eval_retrieve,
        {
            "No": END,
            "Yes": "finance_generation",
        },
    )

    workflow.add_edge("finance_generation", END)

    return workflow.compile()
