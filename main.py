import streamlit as st
import os
import tempfile
from pathlib import Path
from pypdf import PdfReader
import asyncio

import google.generativeai as genai
from pinecone import Pinecone, Index
from langchain_text_splitters import RecursiveCharacterTextSplitter

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.environ.get("PINECONE_ENVIRONMENT")
PINECONE_INDEX_NAME = os.environ.get("PINECONE_INDEX_NAME")
