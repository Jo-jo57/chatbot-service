from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from rag_service import RAGService
import logging
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
    loaded_documents, failed_documents = ingest_document_paths(configured_paths)

    source_paths_to_keep = [
        str(path.resolve())
        for path in configured_paths
    ]

    if loaded_documents:
        purged_chunks = rag_service.purge_except_source_paths(
            source_paths_to_keep
        )
        if purged_chunks:
            logging.info("Removed %s chunks from unconfigured documents", purged_chunks)

    return loaded_documents, failed_documents


loaded_documents, failed_documents = load_configured_documents()
RAG_DISTANCE_THRESHOLD = 2.8
OLLAMA_TIMEOUT_SECONDS = 3


def get_current_time():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def generate_fallback_response():
    return "I do not have enough information in the configured documents to answer that question."


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

        retrieved_chunks = rag_service.retrieve(user_message) if rag_service.count() > 0 else []
        retrieved_chunks = [
            {
                **chunk,
                "metadata": chunk.get("metadata") or {},
            }
            for chunk in retrieved_chunks
        ]
        best_distance = min((chunk.get("distance", 999) for chunk in retrieved_chunks), default=None)

        if best_distance is not None and best_distance <= RAG_DISTANCE_THRESHOLD:
            rag_result = rag_service.answer_with_timeout(
                user_message,
                retrieved_chunks=retrieved_chunks,
                timeout_seconds=OLLAMA_TIMEOUT_SECONDS,
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
        logging.exception("Error processing chat request")
        return jsonify({
            'response': generate_fallback_response(),
            'error': f'Internal chat error: {str(e)}',
            'timestamp': get_current_time(),
            'sources': [],
            'context': {
                'topic': None,
                'helpful_links': []
            }
        }), 200

@app.route('/documents/status', methods=['GET'])
def document_status():
    """Return vector database document chunk status."""
    return jsonify({
        'stored_chunks': rag_service.count(),
        'configured_documents': loaded_documents,
        'failed_documents': failed_documents,
        'timestamp': get_current_time()
    })

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'service': 'Campus Assistant Chatbot',
        'stored_chunks': rag_service.count(),
        'configured_documents': len(loaded_documents),
        'failed_documents': failed_documents,
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
