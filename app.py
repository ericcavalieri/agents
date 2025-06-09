from promptflow.core import tool
from langchain_core.messages import HumanMessage, AIMessage


# from agent_email import make_agent

# from agent_agenda import make_agent

from agent_graph_tools import make_agent


# The inputs section will change based on the arguments of the tool function, after you save the code
# Adding type to arguments and return value will help the system show the types properly
# Please update the function name/signature per need
@tool
def my_python_tool(chat_history: str, question: str) -> str:
    prompt = {"messages": []}
    for item in chat_history:
        prompt["messages"].append(HumanMessage(content=item["inputs"]["question"]))
        prompt["messages"].append(AIMessage(content=item["outputs"]["answer"]))

    prompt["messages"].append(HumanMessage(content=question))
    app = make_agent()

    return app.invoke(prompt)["messages"][-1].content
