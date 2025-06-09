from langchain_cohere import ChatCohere
from agent_agenda import make_agent as make_agent_agenda
from agent_email import make_agent as make_agent_email
from memory import save_recall_memory, search_recall_memories

from langchain_core.tools import tool
from datetime import datetime

from langgraph.graph import START, StateGraph
from langgraph.prebuilt import tools_condition
from langgraph.prebuilt import ToolNode
from langgraph.graph import MessagesState
from langchain_core.messages import HumanMessage, SystemMessage


@tool
def mail_analysis_agent(query: str):
    """
    Agent responsible for analyzing and summarizing incoming emails.

    This agent reads the email content and extracts important information, such as:
    - Sender
    - Priority level
    - Notes or observations
    - Upcoming events mentioned in the message

    Useful for automating inbox management, prioritizing responses, and extracting useful insights from email traffic.

    For better performance, it would be good to identify if there are upcoming events related to the email.
    """

    agent_email = make_agent_email()
    result = agent_email.invoke({"messages": [HumanMessage(content=query)]})

    return result["messages"][-1].content


@tool
def schedule_agent(query: str):
    """
    Agent responsible for managing and interpreting information related to scheduling.

    This agent identifies and processes events, deadlines and appointments from various sources
    (e.g. emails, messages, calendars). It can detect:
    - Upcoming events
    - Time-limited tasks
    - Scheduling conflicts
    - Relevant metadata, such as date, time and attendees

    Useful for creating calendar entries, sending reminders and ensuring efficient time management.

    For best performance, it is necessary to receive the full date, with day, month and year, the time in 24h format and subject.
    """
    agent_agenda = make_agent_agenda()
    result = agent_agenda.invoke({"messages": [HumanMessage(content=query)]})
    return result["messages"][-1].content


def make_agent():

    tools = [
        mail_analysis_agent,
        schedule_agent,
        save_recall_memory,
        search_recall_memories,
    ]
    llm = ChatCohere(model="command-a-03-2025")
    llm_with_tools = llm.bind_tools(tools)

    # System message
    data_hoje = datetime.now()
    data_formatada = data_hoje.strftime("%d de %B de %Y")
    sys_msg = SystemMessage(
        content=(
            f"""Today: {data_formatada}\n\n
Você é um assistente inteligente especializado em gestão de e-mails e organização de agendas. Seu principal objetivo é analisar mensagens recebidas e identificar informações úteis, como remetente, nível de prioridade, observações importantes e eventos futuros mencionados. Também é sua responsabilidade processar solicitações de agendamento sempre que forem detectadas.

Você possui acesso a duas ferramentas especializadas:
mail_analysis_agent(query: str)
Use esta ferramenta sempre que receber o conteúdo de um e-mail ou mensagem que precise ser analisado. Ela extrai:
Remetente
Nível de prioridade
Observações importantes
Eventos futuros relacionados

schedule_agent(query: str)
Use esta ferramenta sempre que:
A análise identificar um evento futuro com data e hora.
O usuário fizer uma solicitação direta ou implícita de agendamento (como "agende uma reunião", "marque um compromisso", etc.).
Certifique-se de fornecer: data completa (dia, mês e ano), hora (formato 24h) e assunto do evento.

Instruções de uso:
Ao receber o conteúdo de um e-mail ou mensagem, inicie com o uso do mail_analysis_agent para identificar informações importantes e possíveis eventos.
Se houver um evento detectado, ou se o usuário estiver pedindo para agendar algo, formule imediatamente a entrada apropriada para o schedule_agent.
Caso faltem informações essenciais para o agendamento (ex: data ou hora), peça ao usuário que forneça os dados que faltam.
Priorize a clareza, extração de dados objetivos e o uso correto das ferramentas para automatizar ao máximo a análise e organização.
"""
        )
    )

    # Node
    def assistant(state: MessagesState):
        return {"messages": [llm_with_tools.invoke([sys_msg] + state["messages"])]}

    # Graph
    builder = StateGraph(MessagesState)

    # Define nodes: these do the work
    builder.add_node("assistant", assistant)
    builder.add_node("tools", ToolNode(tools))

    # Define edges: these determine how the control flow moves
    builder.add_edge(START, "assistant")
    builder.add_conditional_edges(
        "assistant",
        # If the latest message (result) from assistant is a tool call -> tools_condition routes to tools
        # If the latest message (result) from assistant is a not a tool call -> tools_condition routes to END
        tools_condition,
    )
    builder.add_edge("tools", "assistant")
    return builder.compile()
