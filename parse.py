import os
from llama_parse import LlamaParse
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
from jinaai import JinaAI
from pinecone import Pinecone, ServerlessSpec

load_dotenv()

LLAMAPARSE_API_KEY = os.getenv("LLAMAPARSE_API_KEY")

SOURCE_DIR = 'data/sources'
PROCESSED_DIR = 'data/processed_source'

os.makedirs(SOURCE_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)

all_chunks=[]

parser=LlamaParse(api_key=LLAMAPARSE_API_KEY, result_type="markdown")

chunker = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, length_function=len, is_separator_regex=False)
total_chunks = 0
for file in os.listdir(SOURCE_DIR):
    if file.endswith('.pdf'):
        pdf_path = os.path.join(SOURCE_DIR, file)
        print(f"Processing {pdf_path}...")
        try:
            parsed=parser.load_data([pdf_path])
            
            full_content = ""
            for doc in parsed:
                full_content += doc.page_content + "\n\n"
            
            output = os.path.join(PROCESSED_DIR, file.replace('.pdf', '.md'))
            with open(output, 'w', encoding='utf-8') as f:
                f.write(full_content)
            print(f"Processed {file} and saved to {output}")

            chunks = chunker.create_documents([full_content])

            for i, chunk in enumerate(chunks):
                chunkid=f"{file.replace('.pdf','')}_chunk_{i}"
                all_chunks.append({"id":chunkid, "content": chunk.page_content})
            
            print(f"Made {len(chunks)} chunks from {file}")
            total_chunks += len(chunks)
        except:
            print(f"Failed to process {pdf_path}. Skipping...")


PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT")

pc = Pinecone(api_key=PINECONE_API_KEY, environment=PINECONE_ENVIRONMENT)
index_name = "gdg_rag_index"
embedding_dimension = 1024

if index_name not in pc.list_indexes():
    print(f"Now creating pinecone index with name {index_name}")
    pc.create_index(index_name=index_name, dimension=embedding_dimension, metric="cosine", serverless_spec=ServerlessSpec(cloud="aws",region="us-east-1"))
else:
    print(f"Pinecone index with name {index_name} already exists")

index = pc.Index(index_name)
batch_size = 100
for i in range(0, total_chunks, batch_size):
    batch = all_chunks[i:i + batch_size]

    text_embed = [item["text"] for item in batch]
    embeddings_response = JinaAI.embed(texts=text_embed)
    vec_upsert = []
    for j, item in enumerate(batch):
        embedding = embeddings_response['data'][j]['embedding']
        vec_upsert.append({"id": item["id"], "values": embedding, "metadata": item["metadata"]})
    index.upsert(vectors=vec_upsert)

print("Vectors are upserted.")