from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from rag_service import RAGService
import logging
import re
from datetime import datetime
from pathlib import Path

# Initialize Flask app
app = Flask(__name__)
app.config["TEMPLATES_AUTO_RELOAD"] = True
CORS(app)

rag_service = RAGService()

# Configure logging
logging.basicConfig(level=logging.INFO)

# Add your local documents here. They will be chunked, embedded, and stored
# in ChromaDB when the Flask app starts.
DOCUMENT_PATHS = [
    "documents/Registration guide.docx",
    "documents/training guide.docx",
    "documents/fyp and pt.docx",
    "documents/FREQUENTLY ASKED QUESTIONS DURING ADMISSION (1).docx",
    "documents/ALMANAC_2025-2026.pdf",
    "documents/UNDERGRADUATE_PROSPECTUS_2025-2026.pdf",
]

DOCUMENT_GLOBS = [
    "documents/*almanac*.pdf",
    "documents/*prospectus*.pdf",
    "documents/*training*.pdf",
]

SUPPORTED_DOCUMENT_SUFFIXES = {".csv", ".docx", ".md", ".pdf", ".txt"}


def configured_document_paths():
    base_directory = Path(__file__).resolve().parent
    paths = []

    for document_path in DOCUMENT_PATHS:
        path = Path(document_path)
        if not path.is_absolute():
            path = base_directory / path
        paths.append(path)

    for document_glob in DOCUMENT_GLOBS:
        for path in base_directory.glob(document_glob):
            paths.append(path)

    seen = set()
    unique_paths = []
    for path in paths:
        resolved_path = str(path.resolve())
        if resolved_path in seen:
            continue
        seen.add(resolved_path)
        unique_paths.append(path)

    return unique_paths


def uploaded_document_display_name(path):
    name = path.name
    if len(name) > 37 and name[36] == "-":
        return name[37:]
    return name


def uploaded_document_paths():
    return sorted(
        (
            path
            for path in rag_service.upload_directory.glob("*")
            if path.is_file() and path.suffix.lower() in SUPPORTED_DOCUMENT_SUFFIXES
        ),
        key=lambda path: path.stat().st_mtime,
    )


def ingest_document_paths(paths, display_name_for_path=None):
    loaded_documents = []
    failed_documents = []

    for path in paths:
        try:
            display_name = display_name_for_path(path) if display_name_for_path else None
            result = rag_service.ingest_path(path, display_name=display_name)
            loaded_documents.append(result)
            if result.get("skipped"):
                logging.info(
                    "RAG document %s is already current with %s chunks",
                    result["filename"],
                    result["chunks_total"],
                )
            else:
                logging.info(
                    "Loaded RAG document %s into %s chunks",
                    result["filename"],
                    result["chunks_added"],
                )
        except Exception as e:
            error_message = str(e)
            failed_documents.append({
                "filename": path.name,
                "source_path": str(path.resolve()),
                "error": error_message,
            })
            logging.error(f"Error loading document {path}: {error_message}")

    return loaded_documents, failed_documents


def load_configured_documents():
    configured_paths = configured_document_paths()
    upload_paths = uploaded_document_paths()
    loaded_documents, failed_documents = ingest_document_paths(configured_paths)
    loaded_uploads, failed_uploads = ingest_document_paths(
        upload_paths,
        display_name_for_path=uploaded_document_display_name,
    )

    source_paths_to_keep = [
        str(path.resolve())
        for path in configured_paths + upload_paths
    ]

    if loaded_documents or loaded_uploads:
        purged_chunks = rag_service.purge_except_source_paths(
            source_paths_to_keep
        )
        if purged_chunks:
            logging.info("Removed %s chunks from unconfigured documents", purged_chunks)

    return loaded_documents, failed_documents, loaded_uploads, failed_uploads


loaded_documents, failed_documents, uploaded_documents, failed_uploaded_documents = (
    load_configured_documents()
)
RAG_DISTANCE_THRESHOLD = 2.8


def get_current_time():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def generate_fallback_response():
    return "I do not have enough information in the uploaded documents to answer that question."


def latest_uploaded_document():
    if not uploaded_documents:
        return None
    return uploaded_documents[-1]


def is_uploaded_document_question(message):
    words = set(re.findall(r"\w+", message.lower()))
    return bool(words & {"file", "document", "upload", "uploaded", "this", "it"})


def is_document_summary_question(message):
    message_text = " ".join(message.lower().split())
    return (
        "about" in message_text
        or "summarize" in message_text
        or "summary" in message_text
        or "overview" in message_text
        or "main points" in message_text
        or "what should i know" in message_text
        or "what do i need to know" in message_text
    )


def is_document_question_suggestion(message):
    message_text = " ".join(message.lower().split())
    return (
        "what questions" in message_text
        or "questions can" in message_text
        or "what can i ask" in message_text
        or "what can students ask" in message_text
    )


@app.route('/')
def home():
    """Serve the main chat interface"""
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    """Handle chat messages"""
    try:
        data = request.get_json(silent=True) or {}
        user_message = data.get('message', '')
        context = data.get('context', {})
        
        # Extract context information
        topic = context.get('topic', None)
        urgency = context.get('urgency', None)
        
        if not user_message.strip():
            return jsonify({
                'error': 'Message cannot be empty'
            }), 400

        uploaded_document = latest_uploaded_document()
        if uploaded_document and is_uploaded_document_question(user_message):
            retrieved_chunks = rag_service.chunks_for_source_path(
                uploaded_document["source_path"]
            )
            if is_document_question_suggestion(user_message):
                response = rag_service.suggest_questions_from_chunks(retrieved_chunks)
                sources = [
                    {
                        "filename": chunk["metadata"].get("filename"),
                        "chunk_index": chunk["metadata"].get("chunk_index"),
                        "pages": chunk["metadata"].get("pages"),
                        "section": chunk["metadata"].get("section"),
                        "distance": float(chunk["distance"]),
                    }
                    for chunk in retrieved_chunks
                ]
                return jsonify({
                    'response': response,
                    'timestamp': get_current_time(),
                    'sources': sources,
                    'context': {
                        'topic': topic,
                        'helpful_links': []
                    }
                })

            if is_document_summary_question(user_message):
                response = rag_service.summarize_chunks(retrieved_chunks)
                sources = [
                    {
                        "filename": chunk["metadata"].get("filename"),
                        "chunk_index": chunk["metadata"].get("chunk_index"),
                        "pages": chunk["metadata"].get("pages"),
                        "section": chunk["metadata"].get("section"),
                        "distance": float(chunk["distance"]),
                    }
                    for chunk in retrieved_chunks
                ]
                return jsonify({
                    'response': response,
                    'timestamp': get_current_time(),
                    'sources': sources,
                    'context': {
                        'topic': topic,
                        'helpful_links': []
                    }
                })

            rag_result = rag_service.answer_from_document(
                user_message,
                retrieved_chunks=retrieved_chunks,
            )
            response = rag_result['response']
            sources = rag_result['sources']
            return jsonify({
                'response': response,
                'timestamp': get_current_time(),
                'sources': sources,
                'context': {
                    'topic': topic,
                    'helpful_links': []
                }
            })

        retrieved_chunks = rag_service.retrieve(user_message) if rag_service.count() > 0 else []
        best_distance = min((chunk["distance"] for chunk in retrieved_chunks), default=None)

        if best_distance is not None and best_distance <= RAG_DISTANCE_THRESHOLD:
            rag_result = rag_service.answer_from_document(
                user_message,
                retrieved_chunks=retrieved_chunks,
            )
            response = rag_result['response']
            sources = rag_result['sources']
        else:
            response = generate_fallback_response()
            sources = []
        
        return jsonify({
            'response': response,
            'timestamp': get_current_time(),
            'sources': sources,
            'context': {
                'topic': topic,
                'helpful_links': []
            }
        })
        
    except Exception as e:
        logging.error(f"Error processing chat request: {str(e)}")
        return jsonify({
            'error': 'Sorry, I encountered an error. Please try again.',
            'response': 'I apologize, but I\'m having trouble processing your request right now. Please try rephrasing your question or contact campus support directly.'
        }), 500

@app.route('/documents/status', methods=['GET'])
def document_status():
    """Return vector database document chunk status."""
    return jsonify({
        'stored_chunks': rag_service.count(),
        'configured_documents': loaded_documents,
        'failed_documents': failed_documents,
        'uploaded_documents': uploaded_documents,
        'failed_uploaded_documents': failed_uploaded_documents,
        'timestamp': get_current_time()
    })

@app.route('/documents/upload', methods=['POST'])
def upload_documents():
    """Upload one or more documents and add their chunks to the vector database."""
    global uploaded_documents, failed_uploaded_documents

    uploaded_files = request.files.getlist('files') or request.files.getlist('file')
    if not uploaded_files:
        return jsonify({'error': 'No files were uploaded. Use the form field "files".'}), 400

    loaded = []
    failed = []
    for uploaded_file in uploaded_files:
        if not uploaded_file or not uploaded_file.filename:
            continue
        try:
            result = rag_service.ingest_file(uploaded_file)
            loaded.append(result)
            uploaded_documents = [
                document
                for document in uploaded_documents
                if document.get('source_path') != result.get('source_path')
            ]
            uploaded_documents.append(result)
        except Exception as e:
            failure = {
                'filename': uploaded_file.filename,
                'error': str(e),
            }
            failed.append(failure)
            failed_uploaded_documents.append(failure)

    if not loaded and not failed:
        return jsonify({'error': 'No valid document files were selected.'}), 400

    return jsonify({
        'loaded_documents': loaded,
        'failed_documents': failed,
        'stored_chunks': rag_service.count(),
        'timestamp': get_current_time(),
    }), 207 if failed and loaded else 400 if failed else 200

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'service': 'Campus Assistant Chatbot',
        'stored_chunks': rag_service.count(),
        'configured_documents': len(loaded_documents),
        'failed_documents': failed_documents,
        'uploaded_documents': len(uploaded_documents),
        'failed_uploaded_documents': failed_uploaded_documents,
        'timestamp': get_current_time()
    })

@app.route('/quick-help', methods=['GET'])
def quick_help():
    """Get quick help topics"""
    try:
        quick_topics = [
            "Registration process",
            "Admission letter",
            "ARIS",
            "Statement of results",
        ]
        return jsonify({
            'topics': quick_topics,
            'timestamp': get_current_time()
        })
    except Exception as e:
        logging.error(f"Error getting quick help: {str(e)}")
        return jsonify({'error': 'Unable to load quick help topics'}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000, use_reloader=False)
