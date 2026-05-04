from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from chatbot_logic import StudentChatbot
import logging

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Initialize chatbot
chatbot = StudentChatbot()

# Configure logging
logging.basicConfig(level=logging.INFO)

@app.route('/')
def home():
    """Serve the main chat interface"""
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    """Handle chat messages"""
    try:
        data = request.json
        user_message = data.get('message', '')
        context = data.get('context', {})
        
        # Extract context information
        topic = context.get('topic', None)
        urgency = context.get('urgency', None)
        
        if not user_message.strip():
            return jsonify({
                'error': 'Message cannot be empty'
            }), 400
        
        # Generate response using chatbot
        response = chatbot.generate_response(
            question=user_message,
            topic=topic,
            urgency=urgency
        )
        
        return jsonify({
            'response': response,
            'timestamp': chatbot.get_current_time(),
            'context': {
                'topic': topic,
                'helpful_links': chatbot.get_helpful_links(user_message)
            }
        })
        
    except Exception as e:
        logging.error(f"Error processing chat request: {str(e)}")
        return jsonify({
            'error': 'Sorry, I encountered an error. Please try again.',
            'response': 'I apologize, but I\'m having trouble processing your request right now. Please try rephrasing your question or contact campus support directly.'
        }), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'service': 'Campus Assistant Chatbot',
        'timestamp': chatbot.get_current_time()
    })

@app.route('/quick-help', methods=['GET'])
def quick_help():
    """Get quick help topics"""
    try:
        quick_topics = chatbot.get_quick_help_topics()
        return jsonify({
            'topics': quick_topics,
            'timestamp': chatbot.get_current_time()
        })
    except Exception as e:
        logging.error(f"Error getting quick help: {str(e)}")
        return jsonify({'error': 'Unable to load quick help topics'}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
