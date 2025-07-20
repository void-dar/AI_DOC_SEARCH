from fastapi import FastAPI, Header, HTTPException
from .model import QueryInput
from app.retriever import get_answer

app = FastAPI(title="AI Gateway API", version="1.0")

@app.post("/ask-with-context")
async def ask_with_context(data: QueryInput, x_user_id: str = Header(...)):
    try:
        answer = get_answer(data.question, x_user_id)
        return {"answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))