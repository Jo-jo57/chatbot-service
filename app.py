from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from rag_service import RAGService
import logging
from datetime import datetime
from pathlib import Path

# Initialize Flask app
app = Flask(__name__)
CORS(app)

rag_service = RAGService()

# Configure logging
logging.basicConfig(level=logging.INFO)

# Add your local documents here. They will be chunked, embedded, and stored
# in ChromaDB when the Flask app starts.
DOCUMENT_PATHS = [
    "documents/Registration guide.docx",
    "documents/training guide.docx",
]


def load_configured_documents():
    loaded_documents = []
    configured_source_paths = []
    base_directory = Path(__file__).resolve().parent

    for document_path in DOCUMENT_PATHS:
        path = Path(document_path)
        if not path.is_absolute():
            path = base_directory / path
        configured_source_paths.append(str(path.resolve()))

        try:
            result = rag_service.ingest_path(path)
            loaded_documents.append(result)
            logging.info(
                "Loaded RAG document %s into %s chunks",
                result["filename"],
                result["chunks_added"],
            )
        except Exception as e:
            logging.error(f"Error loading configured document {path}: {str(e)}")

    if loaded_documents:
        purged_chunks = rag_service.purge_except_source_paths(
            configured_source_paths
        )
        if purged_chunks:
            logging.info("Removed %s chunks from unconfigured documents", purged_chunks)

    return loaded_documents


loaded_documents = load_configured_documents()
RAG_DISTANCE_THRESHOLD = 2.2


def get_current_time():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def generate_fallback_response():
    return "I do not have enough information in the uploaded documents to answer that question."

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
    app.run(debug=False, host='0.0.0.0', port=5000)
