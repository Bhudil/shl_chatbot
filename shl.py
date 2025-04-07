import streamlit as st
from groq import Groq
from streamlit_mic_recorder import speech_to_text
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq

st.set_page_config(page_title="🎙️ Voice Bot with RAG", layout="wide")
st.title("🎙️ Speech ChatBot")
st.sidebar.title("Speak with LLMs")

@st.cache_resource
def initializing_knowledge_base():
    url = [
    "https://www.shl.com/solutions/products/",
    "https://www.shl.com/solutions/products/assessments/",
    "https://www.shl.com/solutions/products/assessments/assessment-development-centers/",
    "https://www.shl.com/solutions/products/assessments/behavioral-assessments/",
    "https://www.shl.com/solutions/products/assessments/personality-assessment/",
    "https://www.shl.com/solutions/products/assessments/cognitive-assessments/",
    "https://www.shl.com/solutions/products/assessments/skills-and-simulations/",
    "https://www.shl.com/solutions/products/assessments/job-focused-assessments/",
    "https://www.shl.com/solutions/products/360/",
    "https://www.shl.com/solutions/products/hackathons/",
    "https://www.shl.com/solutions/products/video-feedback/",
    "https://www.shl.com/solutions/products/video-interviews/",
    "https://www.shl.com/solutions/products/video-interviews/smart-interview-on-demand/",
    "https://www.shl.com/solutions/products/video-interviews/smart-interview-live-coding/",
    "https://www.shl.com/solutions/products/video-interviews/smart-interview-professional/",
    "https://www.shl.com/solutions/products/product-catalog/",
    "https://www.shl.com/solutions/products/product-catalog/ability-tests/",
    "https://www.shl.com/solutions/products/product-catalog/personality-questionnaires/",
    "https://www.shl.com/solutions/products/product-catalog/skills-tests/",
    "https://www.shl.com/solutions/products/product-catalog/behavioral-assessments/",
    "https://www.shl.com/solutions/products/product-catalog/job-focused-assessments/",
    "https://www.shl.com/solutions/products/product-catalog/360-degree-feedback/",
    "https://www.shl.com/solutions/products/product-catalog/video-interviews/",
    "https://www.shl.com/solutions/products/product-catalog/assessment-and-development-centers/",
    "https://www.shl.com/solutions/products/product-catalog/hackathons/",
    "https://www.shl.com/solutions/products/product-catalog/video-feedback/",
    "https://www.shl.com/solutions/products/product-catalog/skills-simulations/",
    "https://www.shl.com/solutions/products/product-catalog/talent-management-tools/",
    "https://www.shl.com/solutions/products/product-catalog/talent-acquisition-tools/",
    "https://www.shl.com/solutions/products/product-catalog/platform-integration/",
    "https://www.shl.com/solutions/products/product-catalog/leadership-development/"
]

    
    with st.spinner("Loading knowledge base... This may take a minute"):
        loader = UnstructuredURLLoader(urls=url)
        data = loader.load()
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        docs = text_splitter.split_documents(data)
        
        embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
        vectorstore = FAISS.from_documents(docs, embeddings)

        llm = ChatGroq(
            api_key="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",
            model_name="llama3-8b-8192"
        )
        
        retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 3}
        )
        
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
        )
        
        return qa_chain

def llm_selector():
    groq_models = ['llama3-8b-8192', 'mixtral-8x7b-32768', 'gemma-7b-it']
    
    with st.sidebar:
        model = st.selectbox("LLM", groq_models)
        use_rag = st.checkbox("Enable RAG (Hotel Knowledge Base)", value=True)
        return model, use_rag

def generate_groq_response(text, model, use_rag=False, qa_chain=None):
    if use_rag and qa_chain:
        try:
            hotel_keywords = ["assessment", "recommendation", "talent", "SHL", "skills", "behavior", "cognitive",
    "catalogue", "personality", "product", "test", "solution", "recruitment", "hiring",
    "leadership", "psychometric", "workforce", "development", "tool", "platform"]
            
            if any(keyword in text.lower() for keyword in hotel_keywords):
                result = qa_chain.invoke({"query": text})
                answer = result["result"]

                if "source_documents" in result and len(result["source_documents"]) > 0:
                    sources = set([doc.metadata.get('source', 'Unknown source') for doc in result["source_documents"]])
                    answer += "\n\nSources: " + ", ".join(sources)
                
                return answer
        except Exception as e:
            st.sidebar.error(f"RAG Error: {e}. Falling back to standard response.")

    client = Groq(api_key="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")
    
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant for SHL, a global leader in talent assessment solutions. Your job is to provide clear, concise, and context-specific responses using SHL's product catalogue. Help users identify the right assessment tools for hiring, talent development, and workforce planning."
            },
            {
                "role": "user",
                "content": text,
            }
        ],
        model=model,
    )
    return chat_completion.choices[0].message.content

def print_txt(text):
    st.markdown(text, unsafe_allow_html=True)

def print_chat_message(message):
    text = message["content"]
    if message["role"] == "user":
        with st.chat_message("user", avatar="🎙️"):
            print_txt(text)
    else:
        with st.chat_message("assistant", avatar="🤖"):
            print_txt(text)

def record_voice(language="en"):
    state = st.session_state
    if "text_received" not in state:
        state.text_received = []
    text = speech_to_text(
        start_prompt="🎤 Click and speak to ask question",
        stop_prompt="⚠️Stop recording",
        language=language,
        use_container_width=True,
        just_once=True,
    )
    if text:
        state.text_received.append(text)
    result = ""
    for text in state.text_received:
        result += text
    state.text_received = []
    return result if result else None

def main():
    qa_chain = initializing_knowledge_base()

    model, use_rag = llm_selector()
    
    with st.sidebar:
        voice_question = record_voice(language="en")
        typed_question = st.text_input("Or type your question here:")

        question = voice_question or typed_question

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = {}
    if model not in st.session_state.chat_history:
        st.session_state.chat_history[model] = []
    chat_history = st.session_state.chat_history[model]
    
    for message in chat_history: 
        print_chat_message(message)
    
    if question:
        user_message = {"role": "user", "content": question}
        print_chat_message(user_message)
        chat_history.append(user_message)
        
        answer = generate_groq_response(question, model, use_rag, qa_chain)
        
        ai_message = {"role": "assistant", "content": answer}
        print_chat_message(ai_message)
        chat_history.append(ai_message)

        if len(chat_history) > 20:
            chat_history = chat_history[-20:]

        st.session_state.chat_history[model] = chat_history

if __name__ == "__main__":
    main()
