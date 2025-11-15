from pydantic import BaseModel
from abc import ABC
from llama_index.readers.wikipedia import WikipediaReader
from llama_index.core import SummaryIndex  # Changed!
from llama_index.core.node_parser import SentenceSplitter
from typing import List


class WikiPageList(BaseModel):
    pages: List[str]


class WikiPagesListAbstract(ABC):
    @staticmethod
    def get_wiki_page_list(query: str) -> WikiPageList:
        if "index:" in query.lower():
            pages_text = query.lower().split("index:")[1].strip()
            pages = [page.strip() for page in pages_text.split(',')]
        else:
            pages = [query.strip()]
        
        print(f"Extracted pages: {pages}")
        return WikiPageList(pages=pages)


class WikiReaderAbstract(ABC):
    @staticmethod
    def create_wikidocs(wikipage_requests: WikiPageList):
        reader = WikipediaReader()
        documents = reader.load_data(pages=wikipage_requests.pages)
        return documents


class CreateIndexAbstract(ABC):
    @staticmethod
    def create_index(query: str):
        wikipage_requests = WikiPagesListAbstract.get_wiki_page_list(query)
        documents = WikiReaderAbstract.create_wikidocs(wikipage_requests)
        print(f"Loaded {len(documents)} documents")
        
        text_splits = SentenceSplitter(chunk_size=512, chunk_overlap=50)
        nodes = text_splits.get_nodes_from_documents(documents)
        print(f"Created {len(nodes)} chunks")
        
        # Use SummaryIndex - NO EMBEDDINGS NEEDED!
        index = SummaryIndex(nodes)
        print("Index created!")
        return index