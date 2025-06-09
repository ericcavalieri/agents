from langchain_core.messages import SystemMessage
from datetime import datetime
from langchain_core.prompts.chat import ChatPromptTemplate


def make_prompt(system_prompt: str):
    return ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("placeholder", "{messages}"),
        ]
    )


def make_agenda_prompt(messages):
    data_hoje = datetime.now()
    data_formatada = data_hoje.strftime("%d de %B de %Y")
    system_prompt = f"""
Hoje é: {data_formatada}
# Responsabilidade
Você irá gerenciar a agenda de José Maria, um gerente de uma empresa de tecnologia.
Siga as orientações abaixo:
- Reuniões só podem ser marcadas no periodo da tarde
- Prioridade para quem está na lista ativa
- Quando houver conflitos, deverá dar prioridade a quem estiver na lista ativa, caso os dois estejam na lista, deverá questionar o José Maria.
- Você nunca deverá informar se motivo ou assunto dos horarios ocupados.


## Lista Ativa
Fernanda Maria
Eduardo Garcia
"""
    messages["messages"] = [SystemMessage(content=system_prompt)] + messages["messages"]
    return messages


def make_email_prompt(messages):
    data_hoje = datetime.now()
    data_formatada = data_hoje.strftime("%d de %B de %Y")
    system_prompt = f"""
Hoje é: {data_formatada}
Você irá me auxiliar no dia-a-dia lendo meus emails e me enviado mensagem neste formato:
Remetente: [Nome do Remetente] email@provedor.com.br
Prioridade: [ Alta | Média | Baixa ]
Resumo: [Descrição concisa do e-mail, destacando o motivo da prioridade atribuída.]

Consulte a memoria para entender a prioridade do e-mail.
"""
    messages["messages"] = [SystemMessage(content=system_prompt)] + messages["messages"]
    return messages
