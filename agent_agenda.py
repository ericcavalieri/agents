from typing import Any, List
from langgraph.prebuilt import create_react_agent
from langchain_cohere import ChatCohere
from memory import save_recall_memory, search_recall_memories
from schedule_manager import add_event_to_agenda
from datetime import datetime
from util import make_prompt

from dotenv import load_dotenv

load_dotenv()


def make_agent(tools: List[Any] = []):
    data_hoje = datetime.now()
    data_formatada = data_hoje.strftime("%d de %B de %Y")
    system_prompt = f"""
    Hoje é: {data_formatada}
    # Responsabilidade
    Você irá gerenciar a agenda de José Maria, um gerente de uma empresa de tecnologia.
    As solicitações serão sempre de outras pessoas, nunca do José Maria.
    Siga as orientações abaixo:
    - Reuniões só podem ser marcadas no periodo da tarde.
    - Quando houver conflitos, você deverá perguntar ao José Maria via mensagem de texto e responder ao solicitante questionando será deseja pedir um encaixe.
    - Você nunca deverá informar o motivo ou assunto dos horarios ocupados em suas interações.
    """

    llm = ChatCohere(model="command-a-03-2025", temperature=0)
    return create_react_agent(
        llm,
        tools=[save_recall_memory, search_recall_memories, add_event_to_agenda] + tools,
        debug=True,
        name="schedule_agent",
        prompt=make_prompt(system_prompt),
    )
