# 🎙️ SHL Voice Bot with RAG

A voice-enabled chatbot application that uses Retrieval Augmented Generation (RAG) to answer questions about SHL's talent assessment products and services.

## 📋 Overview

This application combines speech recognition with large language models and a knowledge base built from SHL's website to create an intelligent assistant that can answer spoken or typed questions about SHL's assessment solutions. The bot can:

- Convert speech to text for hands-free interaction
- Query LLMs through the Groq API
- Use RAG to provide grounded answers based on SHL's content
- Switch between different language models
- Toggle between pure LLM responses and knowledge-enhanced responses

## 🚀 Features

- **Voice Recognition**: Speak directly to the bot using your microphone
- **Multiple LLM Options**: Choose between Llama3, Mixtral, and Gemma models
- **RAG Integration**: Get responses grounded in SHL's product documentation
- **Source Attribution**: View the sources used to generate answers
- **Chat History**: Maintain conversation context throughout your session
- **Responsive UI**: Clean Streamlit interface for easy interaction

## 💻 Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/shl_chatbot.git
cd shl-voice-bot
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

4. Create a `.env` file and add your Groq API key:
```
GROQ_API_KEY=your_groq_api_key_here
```

## 📚 Dependencies

- streamlit
- groq
- streamlit-mic-recorder
- langchain
- langchain-groq
- langchain-community
- sentence-transformers
- faiss-cpu

## 🔧 Usage

1. Run the Streamlit application:
```bash
streamlit run shl.py
```

2. Open your browser and navigate to the provided URL (typically http://localhost:8501)

3. Use the interface to:
   - Click the microphone button and speak your question
   - Or type your question in the text input
   - Select your preferred LLM model
   - Toggle RAG functionality on/off based on your needs

## 🔍 How It Works

### Knowledge Base
The application loads content from 30+ SHL website URLs covering their product catalog, assessment types, and solutions. This content is:
1. Loaded using UnstructuredURLLoader
2. Split into chunks using RecursiveCharacterTextSplitter
3. Converted to embeddings using SentenceTransformers
4. Stored in a FAISS vector database for efficient retrieval

### Query Processing
When a user asks a question:
1. If RAG is enabled, the system checks if the query contains SHL-related keywords
2. If relevant, it retrieves the most similar chunks from the knowledge base
3. The chunks and query are sent to the selected LLM via the Groq API
4. The response is displayed with source information when RAG is used

## 🛠️ Configuration

You can modify the following parameters in the code:
- `chunk_size` and `chunk_overlap` in the text splitter
- The list of URLs in the knowledge base
- The list of keyword triggers for RAG
- The system prompt for the LLM

## 🔒 Security Notes

- The application stores your Groq API key in the code. For production use, move this to environment variables.
- Content is fetched from public URLs, so ensure you have permission to use the content.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgements

- SHL for their comprehensive product documentation
- The Langchain team for their RAG implementation tools
- Streamlit for the simple but powerful UI framework
- Groq for their fast LLM API
