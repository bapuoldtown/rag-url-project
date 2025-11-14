from pydantic import BaseModel
from abc import ABC, abstractmethod
from llama_index.readers.wikipedia import WikipediaReader
from llama_index.core import VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.program.openai import OpenAIPydanticProgram
from utils import ApiKeyHandler
from pydantic import BaseModel
import openai
from typing import List


class WikiPageList(BaseModel):
    "Data model for WikiPageList"
    pages: List[str]


class WikiPagesListAbstract(ABC):
    @staticmethod
    def get_wiki_page_list(query):
        openai.api_key = ApiKeyHandler.get_apikey()

        prompt_template_str = """
            Given the input {query}, 
            extract the Wikipedia pages mentioned after 
            "please index:" and return them as a list.
            If only one page is mentioned, return a single
            element list.
            """
       
        program = OpenAIPydanticProgram.from_defaults(
                output_cls=WikiPageList,
                prompt_template_str=prompt_template_str,
                verbose=True,
        )
        wikipage_requests = program(query=query)
        return wikipage_requests
      
        return openai.api_key


class WikiReaderAbstract(ABC):
    @staticmethod
    def create_wikidocs(wikipage_requests):
        reader = WikipediaReader()
        documents = reader.load_data(pages=wikipage_requests.pages)
        return documents


class CreateIndexAbstract(ABC):
    @staticmethod
    def create_index(query):
        global Index
        wikipage_requests = WikiPagesListAbstract.get_wiki_page_list(query)
        documents = WikiReaderAbstract.create_wikidocs(wikipage_requests)
        text_splits = SentenceSplitter(chunk_size=150,chunk_overlap=45)
        nodes = text_splits.get_nodes_from_documents(documents) 
        index = VectorStoreIndex(nodes)
        return index
    

if __name__ == "__main__":
    print(WikiPagesListAbstract.get_wiki_page_list())
