import uuid
from typing import List
import pickle

from langchain_core.runnables import RunnableConfig
from langchain.tools import tool
from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_cohere.embeddings import CohereEmbeddings


def load_memories():
    try:
        return InMemoryVectorStore.load(
            "memory.bin", CohereEmbeddings(model="embed-v4.0")
        )
    except:
        return InMemoryVectorStore(embedding=CohereEmbeddings(model="embed-v4.0"))


def save_memories(memory):
    memory.dump("memory.bin")


@tool
def save_recall_memory(memory: str, config: RunnableConfig) -> str:
    """Save memory to vectorstore for later semantic retrieval."""
    recall_vector_store = load_memories()

    document = Document(page_content=memory, id=str(uuid.uuid4()))
    recall_vector_store.add_documents([document])
    save_memories(recall_vector_store)
    return memory


@tool
def search_recall_memories(query: str, config: RunnableConfig) -> List[str]:
    """Search for relevant memories."""
    recall_vector_store = load_memories()

    documents = recall_vector_store.similarity_search(
        query,
        k=3,
    )
    return [document.page_content for document in documents]
