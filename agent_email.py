from typing import Any, List
from langchain_cohere import ChatCohere
from langgraph.prebuilt import create_react_agent
from memory import save_recall_memory, search_recall_memories
from datetime import datetime
from util import make_prompt


from dotenv import load_dotenv

load_dotenv()

data_hoje = datetime.now()
data_formatada = data_hoje.strftime("%d de %B de %Y")
system_prompt = f"""
Hoje é: {data_formatada}

Você é auxiliar do José Maria e sua responsabilidade é ler os emails e enviar mensagem no formato abaixo para o José Maria:
- Remetente: [Nome do Remetente] email@provedor.com.br
- Prioridade: [ Alta | Média | Baixa ]
- Resumo: [Descrição concisa do e-mai e destaque o motivo da prioridade atribuída.]
- Eventos próximos: [Destaque se há algum evento próximo relacionado a este email]

Se exautivo no entendimento, usando sua memoria para procurar eventos próximos. lembre-se, José Maria precisa muito da sua ajuda.
"""


def make_agent(tools: List[Any] = []):
    data_hoje = datetime.now()
    data_formatada = data_hoje.strftime("%d de %B de %Y")
    system_prompt = f"""
Hoje é: {data_formatada}

Você é assistente do José Maria e sua responsabilidade é ler os emails e escrever uma mensagem no formato abaixo para o José Maria:
- Remetente: [Nome do Remetente] email@provedor.com.br
- Prioridade: [ Alta | Média | Baixa ]
- Resumo: [Descrição concisa do e-mai e destaque o motivo da prioridade atribuída.]
- Eventos próximos: [Destaque se há algum evento próximo relacionado a este email, lembre-se de eventos pessoais ou profissionais]

Por exemplo:
- Recebo um email.
- Busco eventos relacionados
- Priorizo de acordo com este relacionamento
"""
    llm = ChatCohere(model="command-a-03-2025")
    return create_react_agent(
        llm,
        tools=[save_recall_memory, search_recall_memories] + tools,
        debug=True,
        name="read_email_agent",
        prompt=make_prompt(system_prompt),
    )
