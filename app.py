import streamlit as st
import os
from jinaai import JinaAI
from pinecone import Pinecone, ServerlessSpec
import google.generativeai as genai
from dotenv import load_dotenv
from llama_parse import LlamaParse
from langchain_text_splitters import RecursiveCharacterTextSplitter
import tempfile

load_dotenv()

st.set_page_config(
    page_title="RAG study tool",
    layout="wide"
)

@st.cache_resource
def init_apis():
    try:
        PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]
        PINECONE_ENVIRONMENT = st.secrets["PINECONE_ENVIRONMENT"]
        GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
        LLAMAPARSE_API_KEY = st.secrets["LLAMAPARSE_API_KEY"]
    except:
        PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
        PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT")
        GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
        LLAMAPARSE_API_KEY = os.getenv("LLAMAPARSE_API_KEY")
    
    pc = Pinecone(api_key=PINECONE_API_KEY, environment=PINECONE_ENVIRONMENT)
    jina_ai = JinaAI()
    
    genai.configure(api_key=GOOGLE_API_KEY)
    gemini_model = genai.GenerativeModel('gemini-2.0-flash')
    
    parser = LlamaParse(api_key=LLAMAPARSE_API_KEY, result_type="markdown")
    
    return pc, jina_ai, gemini_model, parser

def ensure_index_exists(pc, index_name="gdg_rag_index", embedding_dimension=1024):
    if index_name not in pc.list_indexes():
        st.info(f"Creating Pinecone index: {index_name}")
        pc.create_index(
            name=index_name, 
            dimension=embedding_dimension, 
            metric="cosine", 
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
        st.success(f"Created index: {index_name}")
    return pc.Index(index_name)

def process_and_store_document(file_content, filename, pc, jina_ai, parser):
    index_name = "gdg_rag_index"
    
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
        tmp_file.write(file_content)
        tmp_file_path = tmp_file.name
    
    try:
        st.info(f"Parsing {filename}...")
        parsed = parser.load_data([tmp_file_path])
        
        full_content = ""
        for doc in parsed:
            full_content += doc.page_content + "\n\n"
        
        chunker = RecursiveCharacterTextSplitter(
            chunk_size=1000, 
            chunk_overlap=200, 
            length_function=len, 
            is_separator_regex=False
        )
        chunks = chunker.create_documents([full_content])
        
        st.info(f"Created {len(chunks)} chunks from {filename}")
        
        all_chunks = []
        for i, chunk in enumerate(chunks):
            chunk_id = f"{filename.replace('.pdf', '')}_chunk_{i}"
            all_chunks.append({
                "id": chunk_id,
                "text": chunk.page_content,
                "metadata": {
                    "source": filename,
                    "chunk_index": i,
                    "text": chunk.page_content
                }
            })
        
        index = ensure_index_exists(pc)
        
        batch_size = 100
        total_uploaded = 0
        
        progress_bar = st.progress(0)
        for i in range(0, len(all_chunks), batch_size):
            batch = all_chunks[i:i + batch_size]
            
            texts = [item["text"] for item in batch]
            embeddings_response = jina_ai.embed(texts=texts)
            
            vectors = []
            for j, item in enumerate(batch):
                embedding = embeddings_response['data'][j]['embedding']
                vectors.append({
                    "id": item["id"],
                    "values": embedding,
                    "metadata": item["metadata"]
                })
            
            index.upsert(vectors=vectors)
            total_uploaded += len(batch)
            
            progress_bar.progress(total_uploaded / len(all_chunks))
        
        st.success(f"Successfully processed and stored {filename} ({len(all_chunks)} chunks)")
        return True
        
    except Exception as e:
        st.error(f"Error processing {filename}: {str(e)}")
        return False
    finally:
        if os.path.exists(tmp_file_path):
            os.unlink(tmp_file_path)

def query_rag(query, pc, jina_ai, gemini_model, top_k=3):
    PINECONE_INDEX_NAME = "gdg_rag_index"
    
    query_embed = jina_ai.embed_text(query)
    
    index = pc.Index(PINECONE_INDEX_NAME)
    search_results = index.query(vector=query_embed, top_k=top_k, include_metadata=True)
    
    context = []
    if search_results.matches:
        for match in search_results.matches:
            chunk_content = match.metadata.get('text', 'No content found for this chunk.')
            source_file = match.metadata.get('source', 'Unknown source')
            chunk_index = match.metadata.get('chunk_index', 'N/A')
            context.append(f"--- Document: {source_file}, Chunk: {chunk_index} ---\n{chunk_content}")
        context_str = "\n\n".join(context)
    else:
        context_str = "No relevant chunks found."
    
    prompt = f"""You are a helpful assistant. Use the following context to answer the user's query.
    {context_str}

    User's query: {query}
    Answer the query based on the context provided."""
    
    response = gemini_model.generate_content(prompt)
    return response.text, search_results.matches

def generate_related_links(topic, context, gemini_model):
    prompt = f"""Based on the topic "{topic}" and the following document context, generate 10 educational links that would be useful for studying this topic. Include a mix of educational websites, courses, tutorials, and resources.

    Document context:
    {context[:2000]}

    Please provide 10 links in the following format:
    1. [Link Title](https://example.com) - Brief description
    2. [Link Title](https://example.com) - Brief description
    ...and so on.

    Focus on reputable educational sources like Khan Academy, Coursera, edX, MIT OpenCourseWare, Wikipedia, academic institutions, and other reliable learning platforms.
    """
    
    response = gemini_model.generate_content(prompt)
    return response.text

def main():
    st.title("RAG Study Tool")
    st.markdown("Upload documents and get answers to your questions!")
    
    try:
        pc, jina_ai, gemini_model, parser = init_apis()
    except Exception as e:
        st.error(f"Failed to initialize APIs: {str(e)}")
        st.stop()
    
    with st.sidebar:
        st.header("Settings")
        top_k = st.slider("Number of chunks to retrieve", min_value=1, max_value=10, value=3)
        
        st.header("Index Stats")
        try:
            index = pc.Index("gdg_rag_index")
            stats = index.describe_index_stats()
            st.metric("Total Vectors", stats.total_vector_count)
        except Exception as e:
            st.error(f"Error loading index stats: {str(e)}")
    
    tab1, tab2, tab3 = st.tabs(["Chat", "Upload Documents", "Study Links"])
    
    with tab2:
        st.header("Upload PDF Documents")
        st.markdown("Upload PDF files")
        
        uploaded_files = st.file_uploader(
            "Choose PDF files",
            type=['pdf'],
            accept_multiple_files=True,
            help="Select one or more PDF files to upload and process"
        )
        
        if uploaded_files:
            if st.button("Process Documents", type="primary"):
                success_count = 0
                
                for uploaded_file in uploaded_files:
                    file_content = uploaded_file.read()
                    filename = uploaded_file.name
                    
                    if process_and_store_document(file_content, filename, pc, jina_ai, parser):
                        success_count += 1
                
                if success_count > 0:
                    st.success(f"Successfully processed {success_count} out of {len(uploaded_files)} documents!")
                    st.balloons()
                else:
                    st.error("Failed to process any documents. Please check your files and try again.")
    
    with tab3:
        st.header("Generate Study Links")
        st.markdown("Get related educational links based on your study topic")
        
        topic = st.text_input("What topic would you like to study?", 
                             placeholder="e.g., Machine Learning, Physics, History, etc.")
        
        if st.button("Generate Links", type="primary") and topic:
            with st.spinner("Generating related study links..."):
                try:
                    index = pc.Index("gdg_rag_index")
                    topic_embed = jina_ai.embed_text(topic)
                    search_results = index.query(vector=topic_embed, top_k=5, include_metadata=True)
                    
                    context = ""
                    if search_results.matches:
                        for match in search_results.matches:
                            chunk_content = match.metadata.get('text', '')
                            context += chunk_content + "\n\n"
                    
                    links = generate_related_links(topic, context, gemini_model)
                    
                    st.subheader(f"Study Links for: {topic}")
                    st.markdown(links)
                    
                except Exception as e:
                    st.error(f"Error generating links: {str(e)}")
    
    with tab1:
        st.header("Ask Questions")
        
        if "messages" not in st.session_state:
            st.session_state.messages = []
        
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                if message["role"] == "AI" and "sources" in message:
                    with st.expander("View Sources"):
                        for i, source in enumerate(message["sources"]):
                            st.write(f"**Source {i+1}** (Score: {source.score:.3f})")
                            st.write(f"Document: {source.metadata.get('source', 'Unknown')}")
                            st.write(f"Chunk: {source.metadata.get('chunk_index', 'N/A')}")
                            st.write(f"Content: {source.metadata.get('text', 'No content')[:200]}...")
                            st.divider()
        
        if prompt := st.chat_input("What would you like to know?"):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            
            with st.chat_message("AI"):
                with st.spinner("Generating response..."):
                    try:
                        response, matches = query_rag(prompt, pc, jina_ai, gemini_model, top_k)
                        st.markdown(response)
                        
                        st.session_state.messages.append({
                            "role": "AI", 
                            "content": response,
                            "sources": matches
                        })
                        
                    except Exception as e:
                        error_msg = f"Error: {str(e)}"
                        st.error(error_msg)
                        st.session_state.messages.append({"role": "AI", "content": error_msg})
        
        col1, col2 = st.columns([1, 4])
        with col1:
            if st.button("Clear Chat"):
                st.session_state.messages = []
                st.rerun()

if __name__ == "__main__":
    main()