# langchain_agent.py
from index_wikipages import CreateIndexAbstract
from langchain_ollama import ChatOllama
from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from llama_index.core import Settings
from ollama_llm import OllamaLLM

def print_section(title):
    print("\n" + "="*70)
    print(f"  {title}")
    print("="*70 + "\n")

def langchain_rag_test():
    print_section("LANGCHAIN 1.0 AGENTIC RAG WITH OLLAMA")
    
    # Step 1: Build Wikipedia Index
    print_section("STEP 1: Building Wikipedia Index")
    query = "please index: X12_Document_List"
    
    try:
        llm_index = OllamaLLM(model="llama3.2", temperature=0)
        Settings.llm = llm_index
        
        index = CreateIndexAbstract.create_index(query)
        query_engine = index.as_query_engine()
        print("Index built successfully!\n")
    except Exception as e:
        print(f"Failed to build index: {e}")
        return
    
    # Step 2: Create LangChain Agent
    print_section("STEP 2: Creating LangChain Agent")
    
    try:
        # Initialize Ollama
        llm = ChatOllama(
            model="llama3.2",
            temperature=0
        )
        
        # Test connection
        print("Testing Ollama connection...")
        test_response = llm.invoke("Say 'Hello!'")
        print(f"Ollama connected! Test: {test_response.content}\n")
        
        # Create tool
        @tool
        def search_x12_wikipedia(query: str) -> str:
            """Search Wikipedia for information about X12 Document List, X12 transaction sets, and EDI standards.
            
            Use this tool when you need information about X12 or EDI.
            
            Args:
                query: A question about X12 or EDI standards
                
            Returns:
                Information from Wikipedia about X12
            """
            print(f"\n TOOL CALLED: search_x12_wikipedia")
            print(f" INPUT: {query}")
            response = query_engine.query(query)
            result = str(response)
            print(f" OUTPUT: {result[:200]}...\n")
            return result
        
        tools = [search_x12_wikipedia]
        
        # Create agent - FIXED: Use 'model' parameter!
        agent = create_agent(
            model=llm,  # CHANGED from llm= to model=
            tools=tools,
            system_prompt="You are a helpful assistant that answers questions about X12 EDI standards. Always use the search tool to find accurate information."
        )
        
        print("LangChain Agent created!\n")
        
    except Exception as e:
        print(f"Failed to create agent: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Step 3: Test Agentic Reasoning
    print_section("STEP 3: Testing LangChain Agentic Reasoning")
    
    questions = [
        "What is X12 Document List?",
        "List 5 common X12 transaction sets with their codes",
        "What is the 850 transaction set used for?"
    ]
    
    for i, question in enumerate(questions, 1):
        print(f"\n{'#'*70}")
        print(f"# QUESTION {i}/{len(questions)}: {question}")
        print(f"{'#'*70}\n")
        
        try:
            # Invoke agent
            messages = [HumanMessage(content=question)]
            
            print("Agent is thinking...\n")
            
            result = agent.invoke({"messages": messages})
            
            # Extract final answer
            if 'messages' in result:
                final_answer = result['messages'][-1].content
            else:
                final_answer = str(result)
            
            print(f"\n{'='*70}")
            print("FINAL ANSWER:")
            print('='*70)
            print(final_answer)
            print('='*70 + '\n')
            
        except Exception as e:
            print(f"\nError: {e}\n")
            import traceback
            traceback.print_exc()
    
    print_section("LANGCHAIN TEST COMPLETE!")

if __name__ == "__main__":
    print("\nStarting LangChain 1.0 Agentic RAG with Ollama...\n")
    langchain_rag_test()