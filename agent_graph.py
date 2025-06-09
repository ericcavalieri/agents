from langchain_cohere import ChatCohere
from langgraph.prebuilt import create_react_agent
from agent_agenda import make_agent as make_agent_agenda
from agent_email import make_agent as make_agent_email
from print_message import pretty_print_messages


from typing import Annotated
from langchain_core.tools import tool, InjectedToolCallId
from langgraph.prebuilt import InjectedState
from langgraph.graph import StateGraph, START, MessagesState, END
from langgraph.types import Command
from langchain_core.messages import AIMessage


def create_handoff_tool(*, agent_name: str, description: str | None = None):
    name = f"transfer_to_{agent_name}"
    description = description or f"Ask {agent_name} for help."

    @tool(name, description=description)
    def handoff_tool(
        state: Annotated[MessagesState, InjectedState],
        tool_call_id: Annotated[str, InjectedToolCallId],
    ) -> Command:
        tool_message = {
            "role": "tool",
            "content": f"Successfully transferred to {agent_name}",
            "name": name,
            "tool_call_id": tool_call_id,
        }
        # highlight-next-line
        return Command(
            # highlight-next-line
            goto=agent_name,  # (1)!
            # highlight-next-line
            update={**state, "messages": state["messages"] + [tool_message]},  # (2)!
            # highlight-next-line
            graph=Command.PARENT,  # (3)!
        )

    return handoff_tool


# Handoffs
assign_to_schedule_agent = create_handoff_tool(
    agent_name="schedule_agent",
    description="Assign task to an agent responsible for scheduling events",
)

assign_to_read_email_agent = create_handoff_tool(
    agent_name="read_email_agent",
    description="Assign task to an experienced email agent.",
)

read_email_agent = make_agent_email([])
schedule_agent = make_agent_agenda([])


supervisor_agent = create_react_agent(
    tools=[assign_to_schedule_agent, assign_to_read_email_agent],
    model=ChatCohere(model="command-a-03-2025"),
    prompt=(
        "Você é um supervidor de dois agents:"
        "- um agente interpretador de emails"
        "- um agente controlador da agenda de eventos"
        "Chame um agent por vez, não chame os agentes em paralelo"
        "Não faça nenhum atividade você mesmo."
    ),
    name="supervisor",
    debug=True,
)

# Define the multi-agent supervisor graph
supervisor = (
    StateGraph(MessagesState)
    # NOTE: `destinations` is only needed for visualization and doesn't affect runtime behavior
    .add_node(
        supervisor_agent, destinations=("schedule_agent", "read_email_agent", END)
    )
    .add_node(read_email_agent)
    .add_node(schedule_agent)
    .add_edge(START, "supervisor")
    # always return back to the supervisor
    .add_edge("schedule_agent", "supervisor")
    .add_edge("read_email_agent", "supervisor")
    .compile()
)

for chunk in supervisor.stream(
    {
        "messages": [
            {
                "role": "user",
                "content": "Gostaria de agendar uma reunião para o dia 09 as 16h",
            }
        ]
    },
):
    pretty_print_messages(chunk, last_message=True)

final_message_history = chunk["supervisor"]["messages"]
