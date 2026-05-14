import hashlib
import math
import re
import threading
import uuid
from pathlib import Path

import chromadb
from ollama import chat
from werkzeug.utils import secure_filename
from chromadb.config import Settings


INSUFFICIENT_INFORMATION_RESPONSE = (
    "I do not have enough information in the registration guide to answer that question."
)


class LocalHashEmbeddingFunction:
    def __init__(self, dimensions=384):
        self.dimensions = dimensions

    def __call__(self, input):
        return [self._embed(document) for document in input]

    def _embed(self, text):
        vector = [0.0] * self.dimensions
        words = re.findall(r"\w+", text.lower())

        for word in words:
            digest = hashlib.sha256(word.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "big") % self.dimensions
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[index] += sign

        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0:
            return vector
        return [value / norm for value in vector]


class RAGService:
    def __init__(
        self,
        persist_directory="chroma_store",
        collection_name="campus_documents_local",
        upload_directory="uploads",
        llm_model_name="deepseek-r1:1.5b",
        chunk_size=650,
        chunk_overlap=100,
    ):
        self.upload_directory = Path(upload_directory)
        self.upload_directory.mkdir(parents=True, exist_ok=True)
        self.embedding_function = LocalHashEmbeddingFunction()
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.llm_model_name = llm_model_name

        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(anonymized_telemetry=False),
        )
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.embedding_function,
        )

    def ingest_file(self, file_storage):
        saved_path, original_name = self._save_upload(file_storage)
        return self.ingest_path(saved_path, display_name=original_name)

    def ingest_path(self, document_path, display_name=None):
        path = Path(document_path)
        if not path.exists():
            raise ValueError(f"Document not found: {path}")
        if not path.is_file():
            raise ValueError(f"Document path is not a file: {path}")

        source_path = str(path.resolve())
        original_name = display_name or path.name
        text = self._extract_text(path)
        chunks = self._chunk_text(text)

        if not chunks:
            raise ValueError("The document did not contain readable text.")

        self.collection.delete(where={"source_path": source_path})

        document_id = str(uuid.uuid5(uuid.NAMESPACE_URL, source_path))
        ids = [f"{document_id}-{index}" for index in range(len(chunks))]
        metadatas = [
            {
                "document_id": document_id,
                "filename": original_name,
                "source_path": source_path,
                "chunk_index": index,
            }
            for index in range(len(chunks))
        ]

        self.collection.add(
            ids=ids,
            documents=chunks,
            metadatas=metadatas,
        )

        return {
            "document_id": document_id,
            "filename": original_name,
            "source_path": source_path,
            "chunks_added": len(chunks),
        }

    def retrieve(self, query, top_k=2):
        total_chunks = self.collection.count()
        if not query.strip() or total_chunks == 0:
            return []

        results = self.collection.query(
            query_texts=[query],
            n_results=min(top_k, total_chunks),
            include=["documents", "metadatas", "distances"],
        )

        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        return [
            {
                "content": document,
                "metadata": metadata,
                "distance": distance,
            }
            for document, metadata, distance in zip(documents, metadatas, distances)
        ]

    def answer(self, query, retrieved_chunks=None):
        if retrieved_chunks is None:
            retrieved_chunks = self.retrieve(query)
        context = "\n\n".join(
            f"Source: {chunk['metadata'].get('filename', 'uploaded document')}\n{chunk['content']}"
            for chunk in retrieved_chunks
        )

        return {
            "response": self.generate_llm_answer(
                query=query,
                context=context,
                retrieved_chunks=retrieved_chunks,
            ),
            "sources": [
                {
                    "filename": chunk["metadata"].get("filename"),
                    "chunk_index": chunk["metadata"].get("chunk_index"),
                    "distance": float(chunk["distance"]),
                }
                for chunk in retrieved_chunks
            ],
            "context": context,
        }

    def answer_from_document(self, query, retrieved_chunks=None):
        chunks = retrieved_chunks or self.retrieve(query)
        response = self.extractive_answer(query, chunks)
        sources = []
        if response != INSUFFICIENT_INFORMATION_RESPONSE:
            sources = [
                {
                    "filename": chunk["metadata"].get("filename"),
                    "chunk_index": chunk["metadata"].get("chunk_index"),
                    "distance": float(chunk["distance"]),
                }
                for chunk in chunks
            ]
        return {
            "response": response,
            "sources": sources,
            "context": "\n\n".join(chunk["content"] for chunk in chunks),
        }

    def generate_llm_answer(self, query, context, retrieved_chunks=None):
        if not context:
            return (
                "I do not have matching document context yet. Upload a document first, "
                "then ask a question about it."
            )

        prompt = (
            "Answer only the user's exact question using only the document context. "
            "Do not add extra steps, advice, links, or related information unless the "
            "user asks for them. If the context does not contain the answer, say the "
            "uploaded documents do not provide enough information. Keep the answer "
            "under 3 short sentences.\n\n"
            f"Document context:\n{context}\n\n"
            f"User question: {query}"
        )

        try:
            response = chat(
                model=self.llm_model_name,
                messages=[{"role": "user", "content": prompt}],
                options={
                    "temperature": 0.2,
                    "num_predict": 90,
                },
            )
            generated_answer = response.message.content.strip()
            if generated_answer:
                return generated_answer
            return self.extractive_answer(query, retrieved_chunks or [])
        except Exception as error:
            fallback = self.extractive_answer(query, retrieved_chunks or [])
            return (
                f"{fallback}\n\n"
                "Note: I found this from the document, but I could not reach the Ollama model. "
                f"Make sure Ollama is running and `{self.llm_model_name}` is available.\n\n"
                f"Technical detail: {error}"
            )

    def answer_with_timeout(self, query, retrieved_chunks=None, timeout_seconds=12):
        result = {}

        def worker():
            result["answer"] = self.answer(query=query, retrieved_chunks=retrieved_chunks)

        thread = threading.Thread(target=worker, daemon=True)
        thread.start()
        thread.join(timeout_seconds)

        if "answer" in result:
            return result["answer"]

        chunks = retrieved_chunks or self.retrieve(query)
        return {
            "response": self.extractive_answer(query, chunks),
            "sources": [
                {
                    "filename": chunk["metadata"].get("filename"),
                    "chunk_index": chunk["metadata"].get("chunk_index"),
                    "distance": float(chunk["distance"]),
                }
                for chunk in chunks
            ],
            "context": "\n\n".join(chunk["content"] for chunk in chunks),
        }

    def extractive_answer(self, query, retrieved_chunks):
        if not retrieved_chunks:
            return "I could not find matching information in the uploaded document."

        stop_words = {
            "a",
            "an",
            "and",
            "are",
            "can",
            "center",
            "centre",
            "could",
            "did",
            "do",
            "does",
            "for",
            "from",
            "how",
            "i",
            "in",
            "is",
            "it",
            "me",
            "my",
            "of",
            "on",
            "or",
            "office",
            "place",
            "please",
            "the",
            "them",
            "to",
            "what",
            "when",
            "where",
            "which",
            "who",
            "why",
            "your",
        }
        query_words = self._meaningful_words(query, stop_words)
        if not query_words:
            return INSUFFICIENT_INFORMATION_RESPONSE

        query_text = " ".join(re.findall(r"\w+", query.lower()))
        sentences = []
        for chunk in retrieved_chunks:
            sentences.extend(re.split(r"(?<=[.!?])\s+", chunk["content"]))

        scored_sentences = []
        for sentence in sentences:
            sentence_words = self._meaningful_words(sentence, stop_words)
            sentence_text = " ".join(re.findall(r"\w+", sentence.lower()))
            matched_words = query_words & sentence_words
            score = self._match_score(query_words, matched_words)
            for phrase in ("statement of results", "admission letter", "control number"):
                if phrase in query_text and phrase in sentence_text:
                    score += 10
                    matched_words |= set(phrase.split())
            if score <= 0:
                continue
            if score:
                scored_sentences.append((score, sentence.strip()))

        selected = []
        if scored_sentences:
            selected = [max(scored_sentences, key=lambda item: item[0])[1]]
        if not selected:
            return INSUFFICIENT_INFORMATION_RESPONSE

        return " ".join(selected)

    def _meaningful_words(self, text, stop_words):
        words = set()
        for word in re.findall(r"\w+", text.lower()):
            if word in stop_words or len(word) <= 2:
                continue
            words.add(self._normalize_word(word))
        return words

    def _normalize_word(self, word):
        if word in {"registration", "registering", "registered", "registers"}:
            return "register"
        if word in {"courses", "course"}:
            return "course"
        if word in {"results", "result"}:
            return "result"
        if word in {"obtaining", "obtained", "obtains"}:
            return "obtain"
        return word

    def _match_score(self, query_words, matched_words):
        if not matched_words:
            return 0
        if len(query_words) == 1:
            return 1
        required_matches = min(2, len(query_words))
        if len(matched_words) < required_matches:
            return 0
        return len(matched_words)

    def count(self):
        return self.collection.count()

    def purge_except_source_paths(self, source_paths):
        allowed_paths = {str(Path(source_path).resolve()) for source_path in source_paths}
        stored = self.collection.get(include=["metadatas"])
        ids_to_delete = [
            item_id
            for item_id, metadata in zip(stored.get("ids", []), stored.get("metadatas", []))
            if metadata.get("source_path") not in allowed_paths
        ]

        if ids_to_delete:
            self.collection.delete(ids=ids_to_delete)

        return len(ids_to_delete)

    def _save_upload(self, file_storage):
        original_name = Path(file_storage.filename or "document.txt").name
        safe_name = secure_filename(original_name) or "document.txt"
        saved_path = self.upload_directory / f"{uuid.uuid4()}-{safe_name}"
        file_storage.save(saved_path)
        return saved_path, original_name

    def _extract_text(self, path):
        suffix = path.suffix.lower()
        if suffix in {".txt", ".md", ".csv"}:
            return path.read_text(encoding="utf-8", errors="ignore")
        if suffix == ".pdf":
            return self._extract_pdf_text(path)
        if suffix == ".docx":
            return self._extract_docx_text(path)
        raise ValueError("Unsupported file type. Upload a .txt, .md, .csv, .pdf, or .docx file.")

    def _extract_pdf_text(self, path):
        from pypdf import PdfReader

        reader = PdfReader(str(path))
        return "\n".join(page.extract_text() or "" for page in reader.pages)

    def _extract_docx_text(self, path):
        from docx import Document

        document = Document(str(path))
        return "\n".join(paragraph.text for paragraph in document.paragraphs)

    def _chunk_text(self, text):
        normalized = " ".join(text.split())
        if not normalized:
            return []

        chunks = []
        start = 0
        while start < len(normalized):
            end = min(start + self.chunk_size, len(normalized))
            chunk = normalized[start:end].strip()
            if chunk:
                chunks.append(chunk)
            if end == len(normalized):
                break
            start = max(end - self.chunk_overlap, start + 1)
        return chunks
