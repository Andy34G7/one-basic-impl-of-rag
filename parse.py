import os
from llama_parse import LlamaParse
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv

load_dotenv()

LLAMAPARSE_API_KEY = os.getenv("LLAMAPARSE_API_KEY")

SOURCE_DIR = 'data/sources'
PROCESSED_DIR = 'data/processed_source'

os.makedirs(SOURCE_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)

all_chunks=[]

parser=LlamaParse(api_key=LLAMAPARSE_API_KEY, result_type="markdown")

chunker = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, length_function=len, is_separator_regex=False)

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
        except:
            print(f"Failed to process {pdf_path}. Skipping...")