import streamlit as st
import os
from pinecone import Pinecone, ServerlessSpec
import google.generativeai as genai
from dotenv import load_dotenv
from llama_parse import LlamaParse
from langchain_text_splitters import RecursiveCharacterTextSplitter
import tempfile
import time
from sentence_transformers import SentenceTransformer

load_dotenv()

st.set_page_config(
    page_title="RAG Study Tool",
    layout="wide"
)

LOCAL_EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_DIMENSION = 384 # all-MiniLM-L6-v2 produces 384-dimensional embeddings

@st.cache_resource
def init_apis():
    """Initializes all the necessary APIs and clients."""
    try:
        PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]
        GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
        LLAMAPARSE_API_KEY = st.secrets["LLAMAPARSE_API_KEY"]
    except Exception:
        PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
        GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
        LLAMAPARSE_API_KEY = os.getenv("LLAMAPARSE_API_KEY")

    pc = Pinecone(api_key=PINECONE_API_KEY)
    
    local_embedding_model = SentenceTransformer(LOCAL_EMBEDDING_MODEL_NAME)
    
    genai.configure(api_key=GOOGLE_API_KEY)
    gemini_model = genai.GenerativeModel('gemini-2.0-flash')
    
    parser = LlamaParse(api_key=LLAMAPARSE_API_KEY, result_type="markdown")
    
    return pc, local_embedding_model, gemini_model, parser

def ensure_index_exists(pc, index_name="gdg-rag-index", embedding_dimension=EMBEDDING_DIMENSION): # Use the defined dimension
    """Checks if a Pinecone index exists, and creates it if it doesn't."""
    try:
        existing_indexes = [index.name for index in pc.list_indexes()]
        
        if index_name not in existing_indexes:
            st.info(f"Creating Pinecone index: {index_name}")
            pc.create_index(
                name=index_name, 
                dimension=embedding_dimension, 
                metric="cosine", 
                spec=ServerlessSpec(cloud="aws", region="us-east-1")
            )
            st.success(f"Created index: {index_name}. Waiting for it to be ready...")
            time.sleep(10) # Wait for index to be ready
        
        return pc.Index(index_name)
        
    except Exception as e:
        st.error(f"Error with Pinecone index: {str(e)}")
        return None

def process_and_store_document(file_content, filename, pc, local_embedding_model, parser):
    """Parses, chunks, and stores a document in Pinecone."""
    index_name = "gdg-rag-index"
    
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
        tmp_file.write(file_content)
        tmp_file_path = tmp_file.name
    
    try:
        st.info(f"Parsing {filename}...")
        parsed_docs = parser.load_data([tmp_file_path])
        
        full_content = "".join([doc.text + "\n\n" for doc in parsed_docs])
        
        chunker = RecursiveCharacterTextSplitter(
            chunk_size=1000, 
            chunk_overlap=200, 
            length_function=len, 
            is_separator_regex=False
        )
        chunks = chunker.create_documents([full_content])
        
        st.info(f"Created {len(chunks)} chunks from {filename}")
        
        all_chunks_to_upsert = []
        for i, chunk in enumerate(chunks):
            safe_filename = "".join(c for c in filename if c.isalnum() or c in ('-', '_')).rstrip()
            chunk_id = f"{safe_filename}_chunk_{i}"
            all_chunks_to_upsert.append({
                "id": chunk_id,
                "text": chunk.page_content,
                "metadata": {
                    "source": filename,
                    "chunk_index": i,
                    "text": chunk.page_content # Store full text in metadata for easy retrieval
                }
            })
        
        index = ensure_index_exists(pc)
        if index is None:
            st.error("Failed to create or access Pinecone index.")
            return False
        
        batch_size = 100
        total_uploaded = 0
        
        progress_bar = st.progress(0, text=f"Embedding and upserting {len(all_chunks_to_upsert)} chunks...")
        for i in range(0, len(all_chunks_to_upsert), batch_size):
            batch = all_chunks_to_upsert[i:i + batch_size]
            
            texts = [item["text"] for item in batch]
            
            embeddings = local_embedding_model.encode(texts, convert_to_numpy=True).tolist() # Ensure list format for Pinecone
            
            vectors = []
            for j, item in enumerate(batch):
                embedding = embeddings[j]
                vectors.append({
                    "id": item["id"],
                    "values": embedding,
                    "metadata": item["metadata"]
                })
            
            index.upsert(vectors=vectors)
            total_uploaded += len(batch)
            
            progress_bar.progress(total_uploaded / len(all_chunks_to_upsert), text=f"Upserted {total_uploaded}/{len(all_chunks_to_upsert)} chunks")
        
        st.success(f"Successfully processed and stored {filename} ({len(all_chunks_to_upsert)} chunks)")
        return True
        
    except Exception as e:
        st.error(f"Error processing {filename}: {str(e)}")
        return False
    finally:
        if os.path.exists(tmp_file_path):
            os.unlink(tmp_file_path)

def query_rag(query, pc, local_embedding_model, gemini_model, top_k=3):
    """Queries the RAG system to get an answer for a given query."""
    PINECONE_INDEX_NAME = "gdg-rag-index"
    
    try:
        query_embed = local_embedding_model.encode(query, convert_to_numpy=True).tolist()
        
        index = pc.Index(PINECONE_INDEX_NAME)
        search_results = index.query(vector=query_embed, top_k=top_k, include_metadata=True)
        
        context_parts = []
        if search_results.matches:
            for match in search_results.matches:
                chunk_content = match.metadata.get('text', 'No content found.')
                source_file = match.metadata.get('source', 'Unknown source')
                context_parts.append(f"--- From: {source_file} (Score: {match.score:.2f}) ---\n{chunk_content}") # Added score to context
            context_str = "\n\n".join(context_parts)
        else:
            context_str = "No relevant information found in the documents."
        
        prompt = f"""You are a helpful assistant. Use the following context to answer the user's query.
        If the context does not contain the answer, state that you could not find the information in the provided documents.

        Context:
        {context_str}

        User's query: {query}
        Answer:"""
        
        response = gemini_model.generate_content(prompt)
        return response.text, search_results.matches
    
    except Exception as e:
        st.error(f"Error during query: {str(e)}")
        return "Sorry, I encountered an error while processing your query.", []

def generate_related_links(topic, context, gemini_model):
    """Generates educational links based on a topic and context."""
    prompt = f"""Based on the topic "{topic}" and the provided document context, generate up to 5 relevant educational links.

    Document context:
    {context[:2000]}

    Provide links in the format:
    1. [Link Title](https://example.com) - Brief description
    """
    
    try:
        response = gemini_model.generate_content(prompt)
        return response.text
    except Exception as e:
        st.error(f"Error generating links: {str(e)}")
        return "Sorry, I couldn't generate study links."

def main():
    """Main function to run the Streamlit app."""
    st.title("RAG Study Tool (Local Embeddings)") # Updated title
    st.markdown("Upload documents and get answers to your questions!")
    
    try:
        pc, local_embedding_model, gemini_model, parser = init_apis()
    except Exception as e:
        st.error(f"Failed to initialize APIs. Please check your environment variables. Error: {str(e)}")
        st.stop()
    
    with st.sidebar:
        st.header("Settings")
        top_k = st.slider("Number of chunks to retrieve", min_value=1, max_value=10, value=3)
        st.info(f"Using local embedding model: `{LOCAL_EMBEDDING_MODEL_NAME}` (Dimension: {EMBEDDING_DIMENSION})") # Info about local model
        
        st.header("Index Stats")
        try:
            index_name = "gdg-rag-index"
            existing_indexes = [index.name for index in pc.list_indexes()]
            if index_name in existing_indexes:
                index = pc.Index(index_name)
                stats = index.describe_index_stats()
                st.metric("Total Vectors", stats.total_vector_count)
            else:
                st.info("Index not created yet.")
        except Exception as e:
            st.error(f"Could not load index stats: {str(e)}")
    
    tab1, tab2, tab3 = st.tabs(["Chat", "Upload Documents", "Study Links"])
    
    with tab2:
        st.header("Upload PDF Documents")
        uploaded_files = st.file_uploader(
            "Choose PDF files",
            type=['pdf'],
            accept_multiple_files=True,
            help="Select one or more PDF files to upload and process."
        )
        
        if uploaded_files:
            if st.button("Process Documents", type="primary"):
                with st.spinner("Processing documents... This may take a while."):
                    success_count = 0
                    for uploaded_file in uploaded_files:
                        file_content = uploaded_file.read()
                        if process_and_store_document(file_content, uploaded_file.name, pc, local_embedding_model, parser):
                            success_count += 1
                    
                    if success_count > 0:
                        st.success(f"Successfully processed {success_count}/{len(uploaded_files)} documents!")
                        st.balloons()
                    else:
                        st.error("Failed to process any documents.")
    
    with tab3:
        st.header("Generate Study Links")
        topic = st.text_input("Enter a topic to get related study links:", placeholder="e.g., Machine Learning")
        
        if st.button("Generate Links", type="primary") and topic:
            with st.spinner("Generating links..."):
                try:
                    index_name = "gdg-rag-index"
                    if index_name not in [index.name for index in pc.list_indexes()]:
                        st.warning("Please upload documents first to provide context for link generation.")
                    else:
                        index = pc.Index(index_name)
                        topic_embed = local_embedding_model.encode(topic, convert_to_numpy=True).tolist()
                        search_results = index.query(vector=topic_embed, top_k=5, include_metadata=True)
                        
                        context = "".join([match.metadata.get('text', '') + "\n\n" for match in search_results.matches])
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
                if message["role"] == "AI" and "sources" in message and message["sources"]:
                    with st.expander("View Sources"):
                        for i, source in enumerate(message["sources"]):
                            st.info(f"**Source {i+1}** (Score: {source.score:.2f}) - *{source.metadata.get('source', 'N/A')}*")
                            st.markdown(f"> {source.metadata.get('text', 'No content')[:250]}...")
        
        if prompt := st.chat_input("Ask about your documents..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            
            with st.chat_message("AI"):
                with st.spinner("Thinking..."):
                    try:
                        index_name = "gdg-rag-index"
                        if index_name not in [index.name for index in pc.list_indexes()]:
                            error_msg = "No documents uploaded. Please upload documents in the 'Upload' tab first."
                            st.error(error_msg)
                            st.session_state.messages.append({"role": "AI", "content": error_msg})
                        else:
                            response, matches = query_rag(prompt, pc, local_embedding_model, gemini_model, top_k)
                            st.markdown(response)
                            st.session_state.messages.append({"role": "AI", "content": response, "sources": matches})
                    except Exception as e:
                        error_msg = f"An error occurred: {str(e)}"
                        st.error(error_msg)
                        st.session_state.messages.append({"role": "AI", "content": error_msg})
        
        if len(st.session_state.messages) > 0:
            if st.button("Clear Chat History"):
                st.session_state.messages = []
                st.rerun()

if __name__ == "__main__":
    main()