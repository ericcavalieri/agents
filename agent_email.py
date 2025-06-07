from langchain_cohere import ChatCohere
from langchain_core.prompts import ChatPromptTemplate
from langgraph.prebuilt import create_react_agent


from dotenv import load_dotenv

load_dotenv()

prompt = ChatPromptTemplate.from_template("{input}")


def agent():
    llm = ChatCohere(model="command-r-08-2024")
    system_prompt = """
# Responsabilidade
Sou José Maria, gerente de uma empresa de tecnologia,  e preciso que você classifique a prioridade do email em Alta, Média e Baixa para todas as situações.     

## Alta
- Ação a ser tomada em até 5 dias ou prazo perdido
- Remetentes importantes
- Eventos importantes
- Se houver correlação com eventos próximos, pessoais ou profissionais.

## Média
- Ação a ser tomada em até 10 dias
- Remetentes cotidianos
- eventos cotidianos

## Baixo
- Sem prazo para tomada de ação
- Remetentes indeferentes
- Eventos informativos

# Formato de resposta
Sua resposta deverá sempre seguir este formato:
Remetente: email@provedor.com.br 
Prioridade: [ Alta | Média | Baixa ]
Resumo: Resumo do email que indique o motivo da prioridade
"""
    return create_react_agent(llm, tools=[], prompt=system_prompt)
