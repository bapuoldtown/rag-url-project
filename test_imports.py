from llama_index.readers.wikipedia import WikipediaReader
from llama_index.core import VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.program.openai import OpenAIPydanticProgram
from utils import ApiKeyHandler
from pydantic import BaseModel
import openai

print("✅ All imports successful")
print(f"✅ API Key loaded: {ApiKeyHandler.get_apikey()[:20]}...")