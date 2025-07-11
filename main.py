import os
from jinaai import JinaAI
from pinecone import Pinecone
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

pc = Pinecone(api_key=PINECONE_API_KEY, environment=PINECONE_ENVIRONMENT)
jina_ai = JinaAI()

genai.configure(api_key=GOOGLE_API_KEY)
gemini_model = genai.GenerativeModel('gemini-2.0-flash')

PINECONE_INDEX_NAME = "gdg_rag_index"
TOP_K_CHUNKS = 3

query = input("Enter your query: ")

query_embed = jina_ai.embed_text(query)
index = pc.Index(PINECONE_INDEX_NAME)
search_results = index.query(vector=query_embed, top_k=TOP_K_CHUNKS, include_metadata=True)

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

response = gemini_model.generate_text(prompt)
print("Response from Gemini:")

            