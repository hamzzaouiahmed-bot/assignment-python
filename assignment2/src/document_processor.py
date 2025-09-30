from pathlib import Path
from typing import List, Optional, Any
from pydantic import BaseModel
import json


class Document(BaseModel):
    id: int
    title: str
    authors: List[str]
    published: bool
    metadata: Optional[dict[str, Any]] = None


def load_documents(file_path: str) -> List[Document]:
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    documents: List[Document] = []
    for doc in data:
        try:
            document = Document(**doc)
            documents.append(document)
        except Exception as e:
            print(f"Skipping invalid document: {doc} ({e})")
    return documents


def display_documents(documents: List[Document]) -> None:
    for doc in documents:
        print(f"ID: {doc.id}")
        print(f"Title: {doc.title}")
        print(f"Pages: {getattr(doc, 'pages', 'N/A')}")
        print(f"Authors: {', '.join(doc.authors)}")
        print(f"Published: {doc.published}")
        if doc.metadata:
            print(f"Metadata: {doc.metadata}")
        print()


if __name__ == "__main__":
    file_path = Path(__file__).parent.parent / "data" / "documents.json"
documents = load_documents(str(file_path))

print(documents)
