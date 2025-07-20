from langchain.schema import Document
from langchain.document_loaders import PyMuPDFLoader, TextLoader
import os
import email
from bs4 import BeautifulSoup

def parse_file(file_path: str) -> list[Document]:
    ext = os.path.splitext(file_path)[-1].lower()

    if ext == ".pdf":
        loader = PyMuPDFLoader(file_path)
        return loader.load()

    elif ext in [".md", ".txt"]:
        loader = TextLoader(file_path)
        return loader.load()

    elif ext == ".html":
        with open(file_path, "r", encoding="utf-8") as f:
            soup = BeautifulSoup(f.read(), "html.parser")
            text = soup.get_text()
            return [Document(page_content=text)]

    elif ext == ".eml":
        with open(file_path, "r", encoding="utf-8") as f:
            msg = email.message_from_file(f)
            body = ""
            if msg.is_multipart():
                for part in msg.walk():
                    if part.get_content_type() == "text/plain":
                        body += part.get_payload(decode=True).decode(errors="ignore")
            else:
                body = msg.get_payload(decode=True).decode(errors="ignore")
            return [Document(page_content=body)]

    else:
        raise ValueError("Unsupported file type.")
