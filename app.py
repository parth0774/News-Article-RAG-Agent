from flask import Flask, render_template, request, jsonify
from Agents.Agent1_Orchestrator import process_query
from conversation_manager import conversation_manager
import os
from langsmith import Client
from langchain.callbacks.tracers import LangChainTracer
from langchain.callbacks.manager import CallbackManager

# Initialize LangSmith
os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGSMITH_PROJECT"] = "News-AI-Agent"

# Initialize Flask app
app = Flask(__name__)

# Initialize LangSmith client
client = Client()
tracer = LangChainTracer()
callback_manager = CallbackManager([tracer])

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/query', methods=['POST'])
def handle_query():
    data = request.get_json()
    query = data.get('query', '')
    
    if not query:
        return jsonify({'response': 'Please provide a query.'})
    
    try:
        # Process the query through the orchestrator
        result = process_query(query)
        return jsonify({'response': result['response']})
    except Exception as e:
        return jsonify({'response': f'Error processing query: {str(e)}'})

@app.route('/clear_history', methods=['POST'])
def clear_history():
    try:
        # Clear the conversation history
        conversation_manager.initialize()
        return jsonify({'status': 'success'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

if __name__ == '__main__':
    app.run(debug=True) 