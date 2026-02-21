import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain_text_splitters import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
import os
import glob

# Load environment variables
load_dotenv()

# Page config
st.set_page_config(
    page_title="TechDocs AI Bot",
    page_icon="🎥",
    layout="centered"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding-top: 2rem;
    }
    </style>
    """, unsafe_allow_html=True)


def get_pdf_text_from_folder(folder_path="/pdfs"):
    """Load all PDFs from the specified folder"""
    text = ""
    pdf_files = glob.glob(os.path.join(folder_path, "*.pdf"))
    
    if not pdf_files:
        st.error(f"No PDF files found in {folder_path}")
        return None
    
    for pdf_path in pdf_files:
        try:
            pdf_reader = PdfReader(pdf_path)
            for page in pdf_reader.pages:
                text += page.extract_text()
        except Exception as e:
            st.warning(f"Could not read {pdf_path}: {str(e)}")
    
    return text


def get_text_chunks(text):
    """Split text into chunks"""
    text_splitter = CharacterTextSplitter(
        separator="\n", 
        chunk_size=1000, 
        chunk_overlap=200, 
        length_function=len
    )
    return text_splitter.split_text(text)


def get_vectorstore(text_chunks):
    """Create vector store from text chunks"""
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.from_texts(texts=text_chunks, embedding=embeddings)


def get_conversation_chain(vectorstore):
    """Create conversation chain with retrieval"""
    llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.7)
    retriever = vectorstore.as_retriever()
    
    prompt = PromptTemplate.from_template("""
You are a helpful assistant. Answer using ONLY the context below. and strictly follow the instructions given in the Precautions section. If you don't know the answer, say you don't know. Do not try to make up an answer. Always use the context to answer. If the question is not related to the context, say "I don't have information about that in my knowledge base. For better Assistance please fill the form https://forms.gle/JzAncKxLungHQzWe7".
Precautions: If the answer is not in the context, say "I don't have information about that in my knowledge base. For better Assistance please fill the form https://forms.gle/JzAncKxLungHQzWe7"   and send the reply in a systematic way arrange the answer in a way that it looks like a human is replying and also if the user is having the conversation in different laungauge then reply in only that language and also if the user is asking about the movie or web series then reply in a way that it looks like a human is replying then translate ur reply and then answer in that language and also if the user is asking about the movie or web series then reply in a way that it looks like a human is replying"
also dont forget to more refine the contxt and increase ur accuracy with the question  only to to user in one language no need to give translations seperately  either anlayze the language the user is speaking and reply only in that language  understant the basic ethics and bheve like a human nomrmal reply to hi  helly  and if the question is genuine  else ur strictly bound to only ans from the given data
                                          
Context: {context}
Previous Conversation: {chat_history}
Question: {input}
Answer:""")

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    def format_history(history):
        if not history:
            return "None"
        lines = []
        for msg in history[-6:]:
            role = "User" if msg["role"] == "user" else "Assistant"
            lines.append(f"{role}: {msg['content']}")
        return "\n".join(lines)

    chain = (
        {
            "context": RunnableLambda(lambda x: format_docs(retriever.invoke(x["input"]))),
            "chat_history": RunnableLambda(lambda x: format_history(x.get("chat_history", []))),
            "input": RunnableLambda(lambda x: x["input"]),
        }
        | prompt | llm | StrOutputParser()
    )
    return chain


def initialize_bot():
    """Initialize the bot by loading PDFs from backend folder"""
    if "conversation" not in st.session_state:
        with st.spinner("🔄 Wait, I'm retrieving the data"):
            # Load PDFs from the pdfs folder
            raw_text = get_pdf_text_from_folder("./pdf")
            
            if raw_text:
                # Process the text
                text_chunks = get_text_chunks(raw_text)
                vectorstore = get_vectorstore(text_chunks)
                st.session_state.conversation = get_conversation_chain(vectorstore)
                st.session_state.chat_history = []
                st.success("So,How can I help you?")
            else:
                st.session_state.conversation = None


def main():
    # Title
    st.title("🤖 Having problems in watching stuff , I Am Here to Help")
    st.markdown("Hi! I'm here to help you. Ask me anything!")
    
    # Initialize chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    # Initialize the bot with backend PDFs
    initialize_bot()
    
    # Display chat messages
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Type your message here..."):
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Add to history
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        
        # Get bot response
        with st.chat_message("Chilbox ai"):
            with st.spinner("Thinking..."):
                if st.session_state.conversation:
                    response = st.session_state.conversation.invoke({
                        "input": prompt,
                        "chat_history": st.session_state.chat_history
                    })
                else:
                    response = "Sorry, I'm not initialized yet. Please make sure PDFs are in the ./pdfs folder."
                
                st.markdown(response)
        
        # Add to history
        st.session_state.chat_history.append({"role": "assistant", "content": response})
    
    # Sidebar
    with st.sidebar:
        st.header("ℹ️ About")
        st.markdown("""
        **AI Chat Assistant**
        
        I'm trained on documents from the backend.
        Ask me anything related to the knowledge base!
        """)
        
        # Clear chat button
        if st.button("🗑️ Clear Chat History"):
            st.session_state.chat_history = []
            st.rerun()


if __name__ == "__main__":
    main()
