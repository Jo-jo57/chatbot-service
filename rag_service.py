import csv
import hashlib
import math
import os
import re
import threading
import uuid
from pathlib import Path
from zipfile import BadZipFile, is_zipfile

import chromadb
from ollama import Client
from werkzeug.utils import secure_filename
from chromadb.config import Settings


INSUFFICIENT_INFORMATION_RESPONSE = (
    "I do not have enough information in the uploaded documents to answer that question."
)

STOP_WORDS = {
    "a",
    "about",
    "an",
    "and",
    "are",
    "building",
    "buildings",
    "can",
    "can i",
    "center",
    "centre",
    "could",
    "did",
    "do",
    "does",
    "explain",
    "give",
    "for",
    "from",
    "help",
    "how",
    "i",
    "in",
    "info",
    "information",
    "is",
    "it",
    "located",
    "location",
    "looking",
    "map",
    "me",
    "my",
    "need",
    "of",
    "on",
    "or",
    "office",
    "place",
    "please",
    "show",
    "tell",
    "the",
    "them",
    "to",
    "want",
    "campus",
    "uni",
    "university",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "your",
}

LOCATION_QUERY_WORDS = {
    "direction",
    "directions",
    "find",
    "get",
    "go",
    "located",
    "location",
    "near",
    "nearby",
    "reach",
    "route",
    "where",
}

NORMALIZED_WORDS = {
    "accommodations": "hostel",
    "accommodation": "hostel",
    "admissions": "admission",
    "admitted": "admission",
    "admit": "admission",
    "admitting": "admission",
    "application": "apply",
    "applications": "apply",
    "applied": "apply",
    "applies": "apply",
    "applying": "apply",
    "certificate": "cert",
    "certificates": "cert",
    "classes": "class",
    "colleges": "college",
    "course": "course",
    "courses": "course",
    "deadline": "deadline",
    "deadlines": "deadline",
    "direction": "direction",
    "directions": "direction",
    "dorm": "hostel",
    "dormitory": "hostel",
    "dormitories": "hostel",
    "exam": "exam",
    "examination": "exam",
    "examinations": "exam",
    "exams": "exam",
    "faculties": "faculty",
    "fees": "pay",
    "fee": "pay",
    "finance": "pay",
    "finances": "pay",
    "getting": "obtain",
    "get": "obtain",
    "graduated": "graduate",
    "graduating": "graduate",
    "hostel": "hostel",
    "hostels": "hostel",
    "housing": "hostel",
    "libraries": "library",
    "loan": "heslb",
    "loans": "heslb",
    "medical": "health",
    "obtained": "obtain",
    "obtaining": "obtain",
    "obtains": "obtain",
    "paid": "pay",
    "paying": "pay",
    "payment": "pay",
    "payments": "pay",
    "pays": "pay",
    "programme": "program",
    "programmes": "program",
    "programs": "program",
    "registered": "register",
    "registering": "register",
    "registration": "register",
    "registers": "register",
    "result": "result",
    "results": "result",
    "room": "hostel",
    "rooms": "hostel",
    "school": "college",
    "schools": "college",
    "semester": "semester",
    "semesters": "semester",
    "transcript": "result",
    "transcripts": "result",
}

ALIASES = {
    "aris": {"academic", "registration", "information", "system", "portal", "student", "sr2"},
    "cafeteria": {"canteen", "dining", "food", "restaurant", "mess"},
    "coict": {"ict", "college", "informatics", "information", "communication", "technologies", "technology"},
    "control": {"bill", "invoice", "number", "payment"},
    "cos": {"college", "science", "sciences"},
    "duce": {"dar", "es", "salaam", "university", "college", "education"},
    "fyp": {"final", "year", "project", "research", "dissertation", "capstone"},
    "heslb": {"loan", "loans", "bodi", "board", "student", "financing"},
    "hostel": {"accommodation", "dorm", "dormitory", "housing", "room", "residence"},
    "ids": {"identity", "id", "card", "cards"},
    "library": {"book", "books", "reading", "study"},
    "mabibo": {"hostel", "accommodation", "dorm", "residence"},
    "nhif": {"health", "insurance", "medical", "card"},
    "nkrumah": {"hall", "auditorium"},
    "pt": {"practical", "training", "field", "attachment", "internship", "industrial"},
    "udsm": {"university", "dar", "salaam", "mwl", "nyerere"},
}

GREETING_WORDS = {
    "hello",
    "hey",
    "hi",
    "hie",
    "howdy",
}


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
        chunk_size=900,
        chunk_overlap=150,
    ):
        self.upload_directory = Path(upload_directory)
        self.upload_directory.mkdir(parents=True, exist_ok=True)
        self.embedding_function = LocalHashEmbeddingFunction()
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.llm_model_name = llm_model_name
        self.ollama_timeout_seconds = float(os.getenv("OLLAMA_REQUEST_TIMEOUT", "7"))
        self.ollama_client = Client(
            host=os.getenv("OLLAMA_HOST"),
            timeout=self.ollama_timeout_seconds,
        )
        self._keyword_index = None
        self._keyword_index_count = None
        self._keyword_index_lock = threading.Lock()
        self._alias_cache = {}

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
        file_stat = path.stat()
        file_size = file_stat.st_size
        file_mtime_ns = file_stat.st_mtime_ns
        stored = self.collection.get(
            where={"source_path": source_path},
            include=["metadatas"],
        )
        stored_metadatas = stored.get("metadatas", [])

        if stored_metadatas:
            metadata = stored_metadatas[0]
            if (
                metadata.get("source_size") == file_size
                and metadata.get("source_mtime_ns") == file_mtime_ns
            ):
                return {
                    "document_id": metadata.get("document_id"),
                    "filename": metadata.get("filename", original_name),
                    "source_path": source_path,
                    "chunks_added": 0,
                    "chunks_total": len(stored_metadatas),
                    "skipped": True,
                }

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
                "source_size": file_size,
                "source_mtime_ns": file_mtime_ns,
                "chunk_index": index,
                "pages": self._chunk_pages(chunk),
                "section": self._chunk_section(chunk),
            }
            for index, chunk in enumerate(chunks)
        ]

        self.collection.add(
            ids=ids,
            documents=chunks,
            metadatas=metadatas,
        )
        self._clear_keyword_index()

        return {
            "document_id": document_id,
            "filename": original_name,
            "source_path": source_path,
            "chunks_added": len(chunks),
            "chunks_total": len(chunks),
            "skipped": False,
        }

    def retrieve(self, query, top_k=6):
        total_chunks = self.collection.count()
        if not query.strip() or total_chunks == 0:
            return []

        vector_results = self.collection.query(
            query_texts=[query],
            n_results=min(top_k * 2, total_chunks),
            include=["documents", "metadatas", "distances"],
        )

        documents = vector_results.get("documents", [[]])[0]
        metadatas = vector_results.get("metadatas", [[]])[0]
        distances = vector_results.get("distances", [[]])[0]

        vector_chunks = [
            {
                "content": document,
                "metadata": metadata,
                "distance": distance,
                "keyword_score": 0,
            }
            for document, metadata, distance in zip(documents, metadatas, distances)
        ]
        keyword_chunks = self._keyword_retrieve(query, top_k=top_k * 2)

        merged_chunks = {}
        for chunk in vector_chunks + keyword_chunks:
            metadata = chunk["metadata"]
            key = (metadata.get("source_path"), metadata.get("chunk_index"))
            existing = merged_chunks.get(key)
            if existing is None:
                merged_chunks[key] = chunk
                continue
            existing["distance"] = min(existing.get("distance", 999), chunk.get("distance", 999))
            existing["keyword_score"] = max(
                existing.get("keyword_score", 0),
                chunk.get("keyword_score", 0),
            )

        return sorted(
            merged_chunks.values(),
            key=lambda chunk: (-chunk.get("keyword_score", 0), chunk.get("distance", 999)),
        )[:top_k]

    def answer(self, query, retrieved_chunks=None):
        if retrieved_chunks is None:
            retrieved_chunks = self.retrieve(query)
        context = "\n\n".join(
            f"Source: {chunk['metadata'].get('filename', 'uploaded document')}\n{chunk['content']}"
            for chunk in retrieved_chunks
        )

        response = self.generate_llm_answer(
            query=query,
            context=context,
            retrieved_chunks=retrieved_chunks,
        )

        return {
            "response": self.format_answer(response, query=query),
            "sources": [
                {
                    "filename": chunk["metadata"].get("filename"),
                    "chunk_index": chunk["metadata"].get("chunk_index"),
                    "pages": chunk["metadata"].get("pages"),
                    "section": chunk["metadata"].get("section"),
                    "distance": float(chunk["distance"]),
                }
                for chunk in retrieved_chunks
            ],
            "context": context,
        }

    def answer_from_document(self, query, retrieved_chunks=None):
        chunks = retrieved_chunks or self.retrieve(query)
        response = self.format_answer(
            self.extractive_answer(query, chunks),
            query=query,
        )
        sources = []
        if response != INSUFFICIENT_INFORMATION_RESPONSE:
            sources = [
                {
                    "filename": chunk["metadata"].get("filename"),
                    "chunk_index": chunk["metadata"].get("chunk_index"),
                    "pages": chunk["metadata"].get("pages"),
                    "section": chunk["metadata"].get("section"),
                    "distance": float(chunk["distance"]),
                }
                for chunk in chunks
            ]
        return {
            "response": response,
            "sources": sources,
            "context": "\n\n".join(chunk["content"] for chunk in chunks),
        }

    def chunks_for_source_path(self, source_path, limit=12):
        stored = self.collection.get(
            where={"source_path": str(Path(source_path).resolve())},
            include=["documents", "metadatas"],
        )
        chunks = [
            {
                "content": document,
                "metadata": metadata,
                "distance": 0,
                "keyword_score": 0,
            }
            for document, metadata in zip(
                stored.get("documents", []),
                stored.get("metadatas", []),
            )
        ]
        return sorted(
            chunks,
            key=lambda chunk: chunk["metadata"].get("chunk_index", 0),
        )[:limit]

    def summarize_chunks(self, chunks):
        if not chunks:
            return "I could not find readable text in the uploaded document."

        combined_text = " ".join(chunk["content"] for chunk in chunks)
        sections = [
            section.strip()
            for section in re.findall(r"\[SECTION\]\s*([^.\[]+)", combined_text)
            if section.strip()
        ]
        qa_pairs = self._qa_pairs(combined_text)

        points = []
        for section in sections[:3]:
            points.append(f"The document includes {section}.")

        for question, answer in qa_pairs[:3]:
            cleaned_question = self._clean_answer_text(question).rstrip(".")
            cleaned_answer = self._clean_answer_text(answer)
            points.append(f"{cleaned_question}: {cleaned_answer}")

        if not points:
            sentences = [
                self._clean_answer_text(sentence)
                for sentence in re.split(r"(?<=[.!?])\s+", combined_text)
                if len(sentence.strip()) > 40
            ]
            points = sentences[:4]

        if not points:
            return "The uploaded document contains readable text, but I could not create a useful summary from it."

        return "\n\n".join(
            f"{index}. {point}"
            for index, point in enumerate(points[:5], start=1)
        )

    def suggest_questions_from_chunks(self, chunks):
        if not chunks:
            return "I could not find readable text in the uploaded document."

        combined_text = " ".join(chunk["content"] for chunk in chunks)
        qa_questions = []
        for question, _ in self._qa_pairs(combined_text):
            cleaned_question = self._clean_answer_text(question).rstrip(".")
            if cleaned_question and cleaned_question not in qa_questions:
                qa_questions.append(cleaned_question)

        if qa_questions:
            questions = qa_questions[:8]
        else:
            sections = [
                section.strip()
                for section in re.findall(r"\[SECTION\]\s*([^.\[]+)", combined_text)
                if section.strip()
            ]
            questions = [
                f"What should I know about {section}?"
                for section in sections[:8]
            ]

        if not questions:
            return (
                "The uploaded document is readable, but I could not identify clear "
                "student questions from it yet."
            )

        return "You can ask questions such as:\n\n" + "\n\n".join(
            f"{index}. {question}"
            for index, question in enumerate(questions, start=1)
        )

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
            "uploaded documents do not provide enough information. Do not include "
            "chain-of-thought, hidden reasoning, or <think> sections. Keep the answer "
            "under 3 short sentences.\n\n"
            f"Document context:\n{context}\n\n"
            f"User question: {query}"
        )

        try:
            response = self.ollama_client.chat(
                model=self.llm_model_name,
                messages=[{"role": "user", "content": prompt}],
                options={
                    "temperature": 0.2,
                    "num_ctx": 2048,
                    "num_predict": 70,
                    "top_p": 0.85,
                },
                keep_alive="5m",
            )
            generated_answer = self._remove_thinking_text(response.message.content)
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

    def format_answer(self, answer, query=""):
        cleaned_answer = self._clean_answer_text(self._remove_thinking_text(answer))
        if cleaned_answer in {
            INSUFFICIENT_INFORMATION_RESPONSE,
            "I could not find matching information in the uploaded document.",
        }:
            return cleaned_answer

        timetable_answer = self._format_timetable_answer(cleaned_answer)
        if timetable_answer:
            return timetable_answer

        if self._should_format_as_points(query, cleaned_answer):
            return self._format_guideline_points(cleaned_answer)

        qa_answer = self._format_question_answer_pairs(cleaned_answer)
        if qa_answer:
            return qa_answer

        return cleaned_answer

    def _remove_thinking_text(self, text):
        cleaned = str(text or "")
        cleaned = re.sub(r"<think>.*?</think>", "", cleaned, flags=re.IGNORECASE | re.DOTALL)
        cleaned = re.sub(r"^\s*(?:thinking|reasoning)\s*:.*?(?=\n\s*\n|$)", "", cleaned, flags=re.IGNORECASE | re.DOTALL)
        return cleaned.strip()

    def _clean_answer_text(self, text):
        cleaned = " ".join(str(text or "").split())
        cleaned = cleaned.strip("[]")
        if not cleaned:
            return cleaned

        replacements = {
            "â€“": "-",
            "â€”": "-",
            "â€˜": "'",
            "â€™": "'",
            "â€œ": '"',
            "â€": '"',
        }
        for old, new in replacements.items():
            cleaned = cleaned.replace(old, new)

        cleaned = re.sub(r"\s+([,.;:!?])", r"\1", cleaned)
        cleaned = re.sub(r"\]+(?=[.!?]?$)", "", cleaned)
        cleaned = re.sub(r"([.!?])(?=[A-Z])", r"\1 ", cleaned)
        cleaned = re.sub(r"\s*;\s*", "; ", cleaned)
        cleaned = re.sub(r"\s*,\s*", ", ", cleaned)
        cleaned = re.sub(
            r"\b(\d+)(St|Nd|Rd|Th)\b",
            lambda match: match.group(1) + match.group(2).lower(),
            cleaned,
        )
        cleaned = re.sub(r"\s{2,}", " ", cleaned).strip()

        if cleaned and cleaned[-1] not in ".!?:":
            cleaned += "."

        return cleaned

    def _should_format_as_points(self, query, answer):
        query_text = " ".join(re.findall(r"\w+", query.lower()))
        point_words = {
            "guideline",
            "guidelines",
            "steps",
            "step",
            "process",
            "procedure",
            "requirements",
            "requirement",
            "criteria",
            "how",
        }
        if set(query_text.split()) & point_words:
            return True
        return bool(re.search(r"(?:^|\n|\s)(?:A\.|B\.|C\.|D\.|\d+[.)]\s+|Step\s+[A-Z]:)", answer))

    def _format_guideline_points(self, answer):
        intro = ""
        body = answer.strip().strip("[]")
        body = re.sub(r"\s*\[SECTION\].*$", "", body, flags=re.IGNORECASE).strip()
        intro_match = re.match(r"^(.{1,120}?(?:steps|guidelines|requirements|process|procedure)\s*:)\s*(.+)$", body, re.IGNORECASE)
        if intro_match:
            intro = intro_match.group(1)
            body = intro_match.group(2)

        sectioned_points = self._format_sectioned_guidelines(body)
        if sectioned_points:
            return "\n\n".join([intro] + sectioned_points if intro else sectioned_points)

        lettered_parts = re.split(r"\s+(?=[A-Z]\.\s+)", body)
        if len(lettered_parts) > 1:
            points = [self._clean_answer_text(part) for part in lettered_parts if part.strip()]
            return "\n\n".join([intro] + points if intro else points)

        labeled_matches = list(re.finditer(r"([A-Z][A-Za-z /()]+):\s*", body))
        if len(labeled_matches) >= 2:
            points = []
            for index, match in enumerate(labeled_matches):
                start = match.end()
                end = labeled_matches[index + 1].start() if index + 1 < len(labeled_matches) else len(body)
                label = match.group(1).strip()
                value = self._clean_answer_text(body[start:end])
                if value:
                    points.append(f"{len(points) + 1}. {label}: {value}")
            if points:
                return "\n\n".join([intro] + points if intro else points)

        sentences = [
            self._clean_answer_text(sentence)
            for sentence in re.split(r"(?<=[.!?])\s+", body)
            if sentence.strip()
        ]
        if len(sentences) >= 2:
            points = [f"{index}. {sentence}" for index, sentence in enumerate(sentences, start=1)]
            return "\n\n".join([intro] + points if intro else points)

        return self._clean_answer_text(answer)

    def _format_question_answer_pairs(self, answer):
        parts = re.split(r"\s+Answer:\s*", answer)
        if len(parts) < 2:
            return ""

        pairs = []
        label = self._clean_answer_text(parts[0]).rstrip(".:")
        for index, part in enumerate(parts[1:]):
            if index < len(parts) - 2:
                value_text, next_label = self._split_answer_from_next_label(part)
            else:
                value_text, next_label = part, ""

            value = self._clean_answer_text(value_text)
            if label and value:
                pairs.append((label, value))
            label = next_label

        if not pairs:
            return ""

        if len(pairs) == 1:
            label, value = pairs[0]
            return f"{label}:\n{value}"

        return "\n\n".join(
            f"{index}. {label}:\n{value}"
            for index, (label, value) in enumerate(pairs, start=1)
        )

    def _split_answer_from_next_label(self, text):
        words = re.findall(r"\S+", text.strip())
        if len(words) < 3:
            return text, ""

        for start in range(len(words) - 1, 0, -1):
            suffix = words[start:]
            if len(suffix) > 8:
                continue
            first_word = re.sub(r"^[^A-Za-z0-9]+|[^A-Za-z0-9]+$", "", suffix[0])
            if not first_word[:1].isupper():
                continue
            label = " ".join(suffix).strip()
            if not self._looks_like_question_label(label):
                continue
            while start > 0:
                previous_word = re.sub(
                    r"^[^A-Za-z0-9]+|[^A-Za-z0-9]+$",
                    "",
                    words[start - 1],
                )
                if (
                    previous_word[:1].isupper()
                    and previous_word.lower() in self._question_label_terms()
                    and len(words[start - 1:]) <= 8
                ):
                    start -= 1
                    label = " ".join(words[start:]).strip()
                    continue
                break
            answer = " ".join(words[:start]).strip()
            if answer:
                return answer, self._clean_answer_text(label).rstrip(".:")

        return text, ""

    def _looks_like_question_label(self, label):
        label_text = " ".join(re.findall(r"\w+", label.lower()))
        label_words = label_text.split()
        if not 2 <= len(label_words) <= 8:
            return False
        question_terms = {
            *self._question_label_terms(),
            }
        return bool(set(label_words) & question_terms)

    def _question_label_terms(self):
        return {
            "activation",
            "admission",
            "application",
            "confirmation",
            "control",
            "email",
            "invalid",
            "link",
            "number",
            "paid",
            "process",
            "received",
            "reflected",
        }

    def _format_sectioned_guidelines(self, text):
        section_pattern = re.compile(
            r"\b(Phases of FYP|Assessment|Rules|Requirements|Guidelines|Procedure|Process|Steps):\s*",
            re.IGNORECASE,
        )
        matches = list(section_pattern.finditer(text))
        if len(matches) < 2:
            return []

        points = []
        for index, match in enumerate(matches):
            start = match.end()
            end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
            label = self._title_label(match.group(1))
            section_text = text[start:end].strip().strip("[]")
            section_text = re.sub(r"\s*\[.*$", "", section_text).strip()
            section_text = re.sub(r"^\s*-\s*", "", section_text)
            items = [
                self._clean_answer_text(item)
                for item in re.split(r"\s+-\s+", section_text)
                if item.strip()
            ]
            if not items:
                continue
            if len(items) == 1:
                detail = items[0]
            else:
                detail = "; ".join(item.rstrip(".") for item in items) + "."
            points.append(f"{len(points) + 1}. {label}: {detail}")

        return points

    def _title_label(self, label):
        known_labels = {
            "phases of fyp": "Phases of FYP",
            "assessment": "Assessment",
            "rules": "Rules",
            "requirements": "Requirements",
            "guidelines": "Guidelines",
            "procedure": "Procedure",
            "process": "Process",
            "steps": "Steps",
        }
        return known_labels.get(label.lower(), label.strip().title())

    def _format_timetable_answer(self, answer):
        date_pattern = (
            r"(Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)\s+"
            r"\d{1,2}\s+[A-Za-z]+\s+\d{4}"
        )
        range_match = re.match(
            rf"^(.+?)\s+begins on\s+({date_pattern})\s+and ends on\s+({date_pattern})\.$",
            answer,
            re.IGNORECASE,
        )
        if range_match:
            event_name = self._format_event_name(range_match.group(1))
            begins_date = range_match.group(2)
            ends_date = range_match.group(4)
            return (
                f"{event_name}:\n\n"
                f"1. Begins: {begins_date}.\n\n"
                f"2. Ends: {ends_date}."
            )

        single_match = re.match(
            rf"^(.+?)\s+(begins|ends) on\s+({date_pattern})\.$",
            answer,
            re.IGNORECASE,
        )
        if single_match:
            event_name = self._format_event_name(single_match.group(1))
            status = single_match.group(2).capitalize()
            date_value = single_match.group(3)
            return f"{event_name}:\n\n1. {status}: {date_value}."

        return ""

    def _format_event_name(self, event_name):
        words = []
        acronyms = {"fyp", "pt", "udsm", "coict", "aris", "nhif"}
        ordinals = {"1st", "2nd", "3rd", "4th"}
        for word in event_name.split():
            lower_word = word.lower()
            if lower_word in acronyms:
                words.append(lower_word.upper())
            elif lower_word in ordinals:
                words.append(lower_word)
            else:
                words.append(word[:1].upper() + word[1:].lower())
        return " ".join(words)

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
        response = self.format_answer(self.extractive_answer(query, chunks), query=query)
        sources = []
        if response != INSUFFICIENT_INFORMATION_RESPONSE:
            sources = [
                {
                    "filename": chunk["metadata"].get("filename"),
                    "chunk_index": chunk["metadata"].get("chunk_index"),
                    "pages": chunk["metadata"].get("pages"),
                    "section": chunk["metadata"].get("section"),
                    "distance": float(chunk["distance"]),
                }
                for chunk in chunks
            ]

        return {
            "response": response,
            "sources": sources,
            "context": "\n\n".join(chunk["content"] for chunk in chunks),
        }

    def extractive_answer(self, query, retrieved_chunks):
        if not retrieved_chunks:
            return "I could not find matching information in the uploaded document."

        stop_words = STOP_WORDS
        query_words = self._meaningful_words(query, stop_words)
        if not query_words:
            return INSUFFICIENT_INFORMATION_RESPONSE

        registration_answer = self._answer_registration_process(query, retrieved_chunks)
        if registration_answer:
            return registration_answer

        schedule_answer = self._answer_from_schedule_dates(query, query_words, retrieved_chunks)
        if schedule_answer:
            return schedule_answer

        qa_answer = self._answer_from_qa_pairs(query, query_words, retrieved_chunks, stop_words)
        if qa_answer:
            return qa_answer

        location_phrase = self._location_query_phrase(query, stop_words)
        location_terms = self._location_query_terms(query, stop_words)
        query_text = " ".join(re.findall(r"\w+", query.lower()))
        sentence_entries = []
        for chunk in retrieved_chunks:
            filename = chunk["metadata"].get("filename", "")
            for sentence in re.split(r"(?<=[.!?])\s+", chunk["content"]):
                cleaned_sentence = sentence.strip()
                if cleaned_sentence:
                    sentence_entries.append((cleaned_sentence, filename))

        scored_sentences = []
        for sentence, filename in sentence_entries:
            sentence_words = self._meaningful_words(sentence, stop_words)
            sentence_text = " ".join(re.findall(r"\w+", sentence.lower()))
            if (
                location_phrase
                and location_phrase not in sentence_text
                and not self._terms_match_words(location_terms, sentence_words)
            ):
                continue
            if location_phrase and len(sentence) > 450:
                continue
            if location_terms:
                sentence_location_words = self._meaningful_words(
                    sentence,
                    stop_words - {"center", "centre", "office", "place"},
                )
                if not self._terms_match_words(location_terms, sentence_location_words):
                    continue
            matched_words = query_words & sentence_words
            score = self._match_score(query_words, matched_words)
            for phrase in ("statement of results", "admission letter", "control number"):
                if phrase in query_text and phrase in sentence_text:
                    score += 10
                    matched_words |= set(phrase.split())
            if score <= 0:
                continue
            if score:
                scored_sentences.append((score, sentence.strip(), filename))

        selected = []
        if scored_sentences:
            scored_sentences.sort(key=lambda item: item[0], reverse=True)
            best_filename = scored_sentences[0][2].lower()
            max_sentences = 1
            if "prospectus" in best_filename or "almanac" in best_filename:
                max_sentences = 3

            seen = set()
            for _, sentence, _ in scored_sentences:
                if sentence in seen:
                    continue
                selected.append(sentence)
                seen.add(sentence)
                if len(selected) == max_sentences:
                    break
        if not selected:
            return INSUFFICIENT_INFORMATION_RESPONSE

        return " ".join(selected)

    def _answer_from_qa_pairs(self, query, query_words, retrieved_chunks, stop_words):
        query_text = " ".join(re.findall(r"\w+", query.lower()))
        best_pair = None

        for chunk in retrieved_chunks:
            for question, answer in self._qa_pairs(chunk["content"]):
                question_text = " ".join(re.findall(r"\w+", question.lower()))
                answer_text = " ".join(re.findall(r"\w+", answer.lower()))
                topic_phrases = ("statement of results", "admission letter", "control number")
                if any(phrase in query_text for phrase in topic_phrases) and not any(
                    phrase in question_text or phrase in answer_text
                    for phrase in topic_phrases
                    if phrase in query_text
                ):
                    continue

                question_words = self._meaningful_words(question, stop_words)
                answer_words = self._meaningful_words(answer, stop_words)
                question_matches = query_words & question_words
                question_matches |= self._alias_matches(query_words, question_words)
                answer_matches = query_words & answer_words
                answer_matches |= self._alias_matches(query_words, answer_words)
                matched_words = question_matches | answer_matches
                score = (len(question_matches) * 4) + len(answer_matches)
                if len(query_words) > 1 and not question_matches:
                    score = 0

                for phrase in topic_phrases:
                    if phrase in query_text and phrase in question_text:
                        score += 10

                if score <= 0:
                    continue
                if best_pair is None or score > best_pair[0]:
                    best_pair = (score, answer.strip())

        if best_pair:
            return best_pair[1]
        return ""

    def _answer_registration_process(self, query, retrieved_chunks):
        query_text = " ".join(re.findall(r"\w+", query.lower()))
        query_words = set(query_text.split())
        if "registration" not in query_words and "register" not in query_words:
            return ""
        if not ({"how", "do", "process", "steps", "procedure"} & query_words):
            return ""

        guide_chunks = [
            chunk
            for chunk in retrieved_chunks
            if chunk["metadata"].get("filename", "").lower() == "registration guide.docx"
        ]
        if not guide_chunks:
            return ""

        guide_text = " ".join(chunk["content"] for chunk in guide_chunks)
        if not all(f"Step {letter}:" in guide_text for letter in ("A", "B", "C", "D")):
            return ""

        verification_place = "CoICT" if "coict" in query_words else "your College/School/Institute"
        prefix = "For CoICT registration" if "coict" in query_words else "For registration"
        return (
            f"{prefix}, follow these steps:\n"
            "A. Confirm acceptance in ARIS, pay NHIF, generate a control number, pay the required fees, then confirm registration.\n"
            "B. Complete the medical examination and have it verified with a UDSM stamp or do it at UDSM Hospital.\n"
            f"C. Present your original CSEE certificate, ACSEE certificate, birth certificate, and admission letter at {verification_place}.\n"
            "D. After verification, log in to ARIS and register your modules/courses."
        )

    def _answer_from_schedule_dates(self, query, query_words, retrieved_chunks):
        if not ({"when", "date", "dates", "start", "starts", "begin", "begins", "end", "ends"} & set(re.findall(r"\w+", query.lower()))):
            return ""

        date_pattern = (
            r"(?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)\s+"
            r"\d{1,2}\s+[A-Za-z]+\s+\d{4}"
        )
        schedule_words = {
            "when",
            "date",
            "dates",
            "start",
            "starts",
            "begin",
            "begins",
            "end",
            "ends",
        }
        ordered_query_terms = [
            self._normalize_word(word)
            for word in re.findall(r"\w+", query.lower())
            if self._normalize_word(word) in query_words
            and self._normalize_word(word) not in schedule_words
        ]
        phrase_pattern = ""
        if len(ordered_query_terms) >= 2:
            phrase_pattern = r"[-\s]+".join(re.escape(term) for term in ordered_query_terms)

        if phrase_pattern:
            for chunk in retrieved_chunks:
                normalized_content = " ".join(chunk["content"].split())
                phrase_match = re.search(phrase_pattern, normalized_content, re.IGNORECASE)
                if not phrase_match:
                    continue

                nearby_text = normalized_content[phrase_match.end():phrase_match.end() + 180]
                begins_match = re.search(rf"Begins?:\s*[–-]?\s*({date_pattern})", nearby_text, re.IGNORECASE)
                ends_match = re.search(rf"Ends?:\s*[–-]?\s*({date_pattern})", nearby_text, re.IGNORECASE)
                if begins_match or ends_match:
                    event_name = phrase_match.group(0).replace("-", " ").title()
                    begins_date = begins_match.group(1) if begins_match else ""
                    ends_date = ends_match.group(1) if ends_match else ""
                    if begins_date and ends_date:
                        return f"{event_name} begins on {begins_date} and ends on {ends_date}."
                    if begins_date:
                        return f"{event_name} begins on {begins_date}."
                    return f"{event_name} ends on {ends_date}."

        best_match = None

        for chunk in retrieved_chunks:
            chunk_words = self._meaningful_words(chunk["content"], STOP_WORDS)
            if len(query_words & chunk_words) < min(2, len(query_words)):
                continue

            normalized_content = " ".join(chunk["content"].split())
            for event_match in re.finditer(r"[A-Z][A-Z0-9 '&/().,-]+", normalized_content):
                event_name = event_match.group(0).strip(" .,-")
                event_words = self._meaningful_words(event_name, STOP_WORDS)
                if len(query_words & event_words) < min(2, len(query_words)):
                    continue

                nearby_text = normalized_content[event_match.end():event_match.end() + 240]
                begins_match = re.search(rf"Begins?:\s*[–-]?\s*({date_pattern})", nearby_text, re.IGNORECASE)
                ends_match = re.search(rf"Ends?:\s*[–-]?\s*({date_pattern})", nearby_text, re.IGNORECASE)
                if not begins_match and not ends_match:
                    continue

                score = len(query_words & event_words)
                if best_match is None or score > best_match[0]:
                    best_match = (
                        score,
                        event_name.title(),
                        begins_match.group(1) if begins_match else "",
                        ends_match.group(1) if ends_match else "",
                    )

        if not best_match:
            return ""

        _, event_name, begins_date, ends_date = best_match
        if begins_date and ends_date:
            return f"{event_name} begins on {begins_date} and ends on {ends_date}."
        if begins_date:
            return f"{event_name} begins on {begins_date}."
        return f"{event_name} ends on {ends_date}."

    def _qa_pairs(self, text):
        pattern = re.compile(
            r"Q:\s*(.*?)\s*A:\s*(.*?)(?=\s+(?:\[[^\]]+\]\s*)?Q:|\s+\[[A-Z0-9_]+\]|\Z)",
            re.IGNORECASE | re.DOTALL,
        )
        return [
            (" ".join(question.split()), " ".join(answer.split()))
            for question, answer in pattern.findall(text)
            if question.strip() and answer.strip()
        ]

    def _meaningful_words(self, text, stop_words):
        words = set()
        for word in re.findall(r"\w+", text.lower()):
            normalized_word = self._normalize_word(word)
            if normalized_word in GREETING_WORDS:
                words.add(normalized_word)
                continue
            if word in stop_words or len(word) <= 2:
                continue
            words.add(normalized_word)
        return words

    def _normalize_word(self, word):
        return NORMALIZED_WORDS.get(word, word)

    def _alias_matches(self, query_words, document_words):
        matched_words = set()
        for query_word in query_words:
            query_aliases = self._expanded_aliases(query_word)
            if query_aliases & document_words:
                matched_words.add(query_word)
                continue

            for document_word in document_words:
                document_aliases = self._expanded_aliases(document_word)
                if query_word in document_aliases or self._expanded_aliases(query_word) & document_aliases:
                    matched_words.add(query_word)
                    break

        return matched_words

    def _expanded_aliases(self, word):
        aliases = set()
        for alias in ALIASES.get(word, set()):
            aliases.add(self._normalize_word(alias))
        for key, values in ALIASES.items():
            normalized_values = {self._normalize_word(value) for value in values}
            if word in normalized_values:
                aliases.add(key)
                aliases |= normalized_values
        return aliases

    def _terms_match_words(self, terms, words):
        if not terms:
            return True
        for term in terms:
            if term in words:
                continue
            if self._expanded_aliases(term) & words:
                continue
            return False
        return True

    def _match_score(self, query_words, matched_words):
        if not matched_words:
            return 0
        if len(query_words) == 1:
            return 1
        required_matches = min(2, len(query_words))
        if len(matched_words) < required_matches:
            return 0
        return len(matched_words)

    def _location_query_terms(self, query, stop_words):
        query_words = set(re.findall(r"\w+", query.lower()))
        if not (LOCATION_QUERY_WORDS & query_words):
            return set()
        location_stop_words = stop_words | LOCATION_QUERY_WORDS | {"obtain"}
        return self._meaningful_words(
            query,
            location_stop_words - {"center", "centre", "office", "place"},
        )

    def _location_query_phrase(self, query, stop_words):
        query_words = set(re.findall(r"\w+", query.lower()))
        if not (LOCATION_QUERY_WORDS & query_words):
            return ""
        location_stop_words = (stop_words | LOCATION_QUERY_WORDS | {"obtain"}) - {
            "center",
            "centre",
            "office",
            "place",
        }
        ordered_terms = []
        for word in re.findall(r"\w+", query.lower()):
            if word in location_stop_words or len(word) <= 2:
                continue
            ordered_terms.append(self._normalize_word(word))
        return " ".join(ordered_terms)

    def _keyword_retrieve(self, query, top_k=12):
        stop_words = STOP_WORDS
        query_words = self._meaningful_words(query, stop_words)
        if not query_words:
            return []

        query_text = " ".join(re.findall(r"\w+", query.lower()))
        scored_chunks = []
        for item in self._get_keyword_index():
            document = item["content"]
            metadata = item["metadata"]
            document_words = item["words"]
            matched_words = query_words & item["searchable_words"]
            for query_word in query_words - matched_words:
                if self._expanded_aliases(query_word) & document_words:
                    matched_words.add(query_word)
            score = len(matched_words)
            document_text = item["text"]
            filename = item["filename"]

            if "register" in query_words and filename == "registration guide.docx":
                score += 8
            if "coict" in query_words and "coict" in document_text:
                score += 3

            for phrase in ("statement of results", "admission letter", "control number"):
                if phrase in query_text and phrase in document_text:
                    score += 10

            if score <= 0:
                continue
            scored_chunks.append(
                {
                    "content": document,
                    "metadata": metadata,
                    "distance": 0,
                    "keyword_score": score,
                }
            )

        return sorted(
            scored_chunks,
            key=lambda chunk: chunk["keyword_score"],
            reverse=True,
        )[:top_k]

    def _get_keyword_index(self):
        current_count = self.collection.count()
        if (
            self._keyword_index is not None
            and self._keyword_index_count == current_count
        ):
            return self._keyword_index

        with self._keyword_index_lock:
            if (
                self._keyword_index is not None
                and self._keyword_index_count == current_count
            ):
                return self._keyword_index

            stored = self.collection.get(include=["documents", "metadatas"])
            keyword_index = []
            for document, metadata in zip(
                stored.get("documents", []),
                stored.get("metadatas", []),
            ):
                document_words = self._meaningful_words(document, STOP_WORDS)
                alias_words = set()
                for word in document_words:
                    alias_words |= self._expanded_aliases(word)
                keyword_index.append(
                    {
                        "content": document,
                        "metadata": metadata,
                        "words": document_words,
                        "searchable_words": document_words | alias_words,
                        "text": " ".join(re.findall(r"\w+", document.lower())),
                        "filename": metadata.get("filename", "").lower(),
                    }
                )
            self._keyword_index = keyword_index
            self._keyword_index_count = current_count
            return self._keyword_index

    def _clear_keyword_index(self):
        with self._keyword_index_lock:
            self._keyword_index = None
            self._keyword_index_count = None

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
            self._clear_keyword_index()

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
            if suffix == ".csv":
                return self._extract_csv_text(path)
            return path.read_text(encoding="utf-8", errors="ignore")
        if suffix == ".pdf":
            return self._extract_pdf_text(path)
        if suffix == ".docx":
            return self._extract_docx_text(path)
        raise ValueError("Unsupported file type. Upload a .txt, .md, .csv, .pdf, or .docx file.")

    def _extract_csv_text(self, path):
        with path.open("r", encoding="utf-8-sig", errors="ignore", newline="") as file:
            sample = file.read(4096)
            file.seek(0)
            try:
                dialect = csv.Sniffer().sniff(sample)
            except csv.Error:
                dialect = csv.excel

            reader = csv.DictReader(file, dialect=dialect)
            if not reader.fieldnames:
                file.seek(0)
                rows = csv.reader(file, dialect=dialect)
                return "\n".join(
                    ". ".join(cell.strip() for cell in row if cell.strip())
                    for row in rows
                    if any(cell.strip() for cell in row)
                )

            lines = []
            for row in reader:
                values = {
                    (key or "").strip().lower(): (value or "").strip()
                    for key, value in row.items()
                    if (value or "").strip()
                }
                if not values:
                    continue

                name = (
                    values.get("name")
                    or values.get("building")
                    or values.get("place")
                    or values.get("location")
                    or values.get("facility")
                    or values.get("office")
                )
                details = [
                    f"{key.replace('_', ' ').title()}: {value}"
                    for key, value in values.items()
                    if value != name
                ]
                if name:
                    lines.append(f"Campus map location: {name}. {'. '.join(details)}.")
                else:
                    lines.append(". ".join(details) + ".")

            return "\n".join(lines)

    def _extract_pdf_text(self, path):
        from pypdf import PdfReader

        reader = PdfReader(str(path))
        pages = []
        for index, page in enumerate(reader.pages, start=1):
            page_text = page.extract_text() or ""
            if page_text.strip():
                pages.append(f"\n[PAGE {index}]\n{page_text}")
        return "\n".join(pages)

    def _extract_docx_text(self, path):
        from docx import Document

        if not is_zipfile(path):
            raise ValueError(
                f"{path.name} is not a readable .docx file. Open it in Word, "
                "use Save As to save it as a Word Document (.docx), close it, "
                "then restart the Flask server."
            )

        try:
            document = Document(str(path))
        except (BadZipFile, OSError) as error:
            raise ValueError(
                f"{path.name} could not be read as a .docx file. Close any app "
                "using it, make sure OneDrive has downloaded it locally, then "
                "restart the Flask server."
            ) from error
        return "\n".join(paragraph.text for paragraph in document.paragraphs)

    def _chunk_text(self, text):
        prepared = self._prepare_text_for_chunking(text)
        normalized = " ".join(prepared.split())
        if not normalized:
            return []

        sentences = re.split(r"(?<=[.!?])\s+|(?=\[PAGE \d+\])", normalized)
        if len(sentences) > 1:
            chunks = []
            current = []
            current_length = 0
            active_page = ""

            for sentence in sentences:
                sentence = sentence.strip()
                if not sentence:
                    continue

                next_length = current_length + len(sentence) + 1
                if current and next_length > self.chunk_size:
                    chunks.append(
                        self._ensure_chunk_page_marker(" ".join(current).strip(), active_page)
                    )
                    overlap_sentences = []
                    overlap_length = 0
                    for previous_sentence in reversed(current):
                        overlap_length += len(previous_sentence) + 1
                        if overlap_length > self.chunk_overlap:
                            break
                        overlap_sentences.insert(0, previous_sentence)
                    current = overlap_sentences
                    current_length = sum(len(item) + 1 for item in current)

                page_match = re.search(r"\[PAGE (\d+)\]", sentence)
                if page_match:
                    active_page = page_match.group(1)

                current.append(sentence)
                current_length += len(sentence) + 1

            if current:
                chunks.append(
                    self._ensure_chunk_page_marker(" ".join(current).strip(), active_page)
                )
            return chunks

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

    def _ensure_chunk_page_marker(self, chunk, active_page):
        if not active_page or re.search(r"\[PAGE \d+\]", chunk):
            return chunk
        return f"[PAGE {active_page}] {chunk}"

    def _prepare_text_for_chunking(self, text):
        lines = []
        current_section = ""

        for raw_line in text.splitlines():
            line = " ".join(raw_line.split())
            if not line:
                continue

            if re.fullmatch(r"\[PAGE \d+\]", line):
                lines.append(line)
                continue

            if self._looks_like_heading(line):
                current_section = line.title()
                lines.append(f"[SECTION] {current_section}.")
                continue

            if current_section and not line.startswith("[SECTION]"):
                lines.append(line)
            else:
                lines.append(line)

        return "\n".join(lines)

    def _looks_like_heading(self, line):
        if len(line) > 90:
            return False
        if re.fullmatch(r"\d+(\.\d+)*\.?\s+[A-Z][A-Za-z0-9,()&/ -]+", line):
            return True
        words = re.findall(r"[A-Za-z]+", line)
        if 2 <= len(words) <= 10 and line.upper() == line and any(len(word) > 3 for word in words):
            return True
        return False

    def _chunk_pages(self, chunk):
        pages = [int(page) for page in re.findall(r"\[PAGE (\d+)\]", chunk)]
        if not pages:
            return ""
        first_page = min(pages)
        last_page = max(pages)
        if first_page == last_page:
            return str(first_page)
        return f"{first_page}-{last_page}"

    def _chunk_section(self, chunk):
        match = re.search(r"\[SECTION\]\s*([^.\n]+)", chunk)
        if not match:
            return ""
        return match.group(1).strip()
