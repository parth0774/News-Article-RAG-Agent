# News AI Agent

A conversational AI agent that can retrieve news information and generate LinkedIn posts.

## Features

- News information retrieval using RAG
- LinkedIn post generation
- Conversation history management
- LangSmith tracing for monitoring and debugging
- Modern web interface

## Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd News-Article-RAG-Agent
```

2. Create a virtual environment and activate it:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
Create a `.env` file in the root directory with the following content:
```
LANGSMITH_API_KEY="your-langsmith-api-key"
OPENAI_API_KEY="your-openai-api-key"
```

5. Run the application:
```bash
python app.py
```

6. Open your browser and navigate to `http://localhost:5000`

## Usage

1. Enter your query in the text input field
2. Press Enter or click the Send button
3. The AI agent will process your query and respond
4. Use the Clear History button to reset the conversation

## Project Structure

- `app.py`: Flask application and API endpoints
- `templates/index.html`: Web interface
- `static/style.css`: Styling for the web interface
- `Agents/`: Contains the AI agents
  - `Agent1_Orchestrator.py`: Main orchestrator
  - `RAG_Agent2.py`: RAG system for news retrieval
  - `LinkedIn_Agent3.py`: LinkedIn post generation

## Monitoring

The application uses LangSmith for tracing and monitoring. You can view the traces and performance metrics in your LangSmith dashboard.

## License

MIT License