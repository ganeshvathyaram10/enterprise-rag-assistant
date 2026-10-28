from pathlib import Path
from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .config import Settings

SUPPORTED = {".txt", ".md", ".pdf"}


def load_documents(docs_dir: str) -> List[Document]:
    docs: List[Document] = []
    for path in sorted(Path(docs_dir).rglob("*")):
        if path.suffix.lower() not in SUPPORTED:
            continue
        if path.suffix.lower() == ".pdf":
            from pypdf import PdfReader

            for i, page in enumerate(PdfReader(str(path)).pages, start=1):
                text = page.extract_text() or ""
                if text.strip():
                    docs.append(Document(page_content=text, metadata={"source": path.name, "page": i}))
        else:
            docs.append(Document(page_content=path.read_text(encoding="utf-8"), metadata={"source": path.name, "page": 1}))
    return docs


def chunk_documents(docs: List[Document], s: Settings) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=s.chunk_size,
        chunk_overlap=s.chunk_overlap,
        separators=["\n\n", "\n", ". ", " "],
    )
    chunks = splitter.split_documents(docs)
    for i, c in enumerate(chunks):
        c.metadata["chunk_id"] = f"{c.metadata['source']}::p{c.metadata['page']}::c{i}"
    return chunks
