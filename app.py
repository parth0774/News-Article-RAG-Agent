from flask import Flask, render_template, request, jsonify, send_from_directory
from Agents.Agent1_Orchestrator import process_query
from conversation_manager import conversation_manager
import os
from langsmith import Client
from langchain.callbacks.tracers import LangChainTracer
from langchain.callbacks.manager import CallbackManager
from dotenv import load_dotenv
from datetime import datetime

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__, 
            static_folder='static',
            template_folder='templates')

# Initialize LangSmith client with environment variables
client = Client(
    api_key=os.getenv("LANGSMITH_API_KEY"),
    api_url=os.getenv("LANGSMITH_ENDPOINT")
)

# Configure LangSmith tracing
tracer = LangChainTracer(
    project_name=os.getenv("LANGSMITH_PROJECT")
)

callback_manager = CallbackManager([tracer])

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/static/<path:filename>')
def serve_static(filename):
    return send_from_directory(app.static_folder, filename)

@app.route('/query', methods=['POST'])
def handle_query():
    data = request.get_json()
    query = data.get('query', '')
    
    if not query:
        return jsonify({
            'response': 'Please provide a query.',
            'logs': [{
                'type': 'Error',
                'content': 'Empty query received',
                'timestamp': datetime.now().strftime('%H:%M:%S')
            }]
        })
    
    try:
        # Process the query through the orchestrator with tracing
        result = process_query(query)
        
        # Create detailed logs
        logs = [
            {
                'type': 'System',
                'content': 'Query received and processed',
                'timestamp': datetime.now().strftime('%H:%M:%S')
            },
            {
                'type': 'Agent',
                'content': f'Using {result["agent"]} for processing',
                'timestamp': datetime.now().strftime('%H:%M:%S')
            },
            {
                'type': 'Details',
                'content': result.get('details', 'Processing completed'),
                'timestamp': datetime.now().strftime('%H:%M:%S')
            }
        ]
        
        return jsonify({
            'response': result['response'],
            'logs': logs
        })
    except Exception as e:
        return jsonify({
            'response': f'Error processing query: {str(e)}',
            'logs': [{
                'type': 'Error',
                'content': f'Error occurred: {str(e)}',
                'timestamp': datetime.now().strftime('%H:%M:%S')
            }]
        })

@app.route('/clear_history', methods=['POST'])
def clear_history():
    try:
        # Clear the conversation history
        conversation_manager.initialize()
        return jsonify({'status': 'success'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000) 