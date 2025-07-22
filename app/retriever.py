from langchain_community.vectorstores import Qdrant
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.chains import RetrievalQAWithSourcesChain
from langchain_community.llms import OpenAI
import os

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
CONFIDENCE_THRESHOLD = 0.35  # Not yet used

def get_answer(query: str, user_id: str) -> dict:
    try:
        vector_store = Qdrant(
            url=QDRANT_URL,
            collection_name=f"user_{user_id}_docs",
            embeddings=OpenAIEmbeddings()
        )

        retriever = vector_store.as_retriever(search_kwargs={"k": 3})
        qa_chain = RetrievalQAWithSourcesChain.from_chain_type(
            llm=OpenAI(temperature=0),
            retriever=retriever,
            return_source_documents=True
        )

        result = qa_chain({"question": query})
        answer = result["answer"]
        sources = result.get("source_documents", [])

        logs = []
        for doc in sources:
            logs.append({
                "source": doc.metadata.get("source", "unknown"),
                "snippet": doc.page_content[:150] + "...",
            })

        if len(answer.strip()) < 10:
            return {
                "answer": "I'm not confident in the retrieved documents. Please rephrase or upload more context.",
                "fallback": True,
                "logs": logs
            }

        return {
            "answer": answer,
            "fallback": False,
            "logs": logs
        }
    except Exception as e:
        return {
            "answer": "An error occurred while processing your request.",
            "fallback": True,
            "logs": [],
            "error": str(e)
        }
