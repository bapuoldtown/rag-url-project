# langgraph_agent.py
from index_wikipages import CreateIndexAbstract
from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.prebuilt import create_react_agent
from llama_index.core import Settings
from ollama_llm import OllamaLLM

def print_section(title):
    print("\n" + "="*70)
    print(f"  {title}")
    print("="*70 + "\n")

def langgraph_rag_test():
    print_section("LANGGRAPH AGENTIC RAG WITH OLLAMA")
    
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
    
    # Step 2: Create LangGraph Agent
    print_section("STEP 2: Creating LangGraph ReAct Agent")
    
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
        
        # Create tool using @tool decorator
        @tool
        def search_x12_wikipedia(query: str) -> str:
            """Search Wikipedia for information about X12 Document List, X12 transaction sets, and EDI standards.
            
            Use this when you need information about:
            - X12 Document List
            - X12 transaction sets
            - EDI standards
            - Electronic data interchange
            
            Args:
                query: A clear question about X12 or EDI
                
            Returns:
                Detailed information from Wikipedia
            """
            print(f"\nTOOL CALLED: search_x12_wikipedia")
            print(f"  INPUT: {query}")
            response = query_engine.query(query)
            result = str(response)
            print(f" OUTPUT: {result[:200]}...\n")
            return result
        
        tools = [search_x12_wikipedia]
        
        # Create ReAct agent using LangGraph!
        agent_executor = create_react_agent(llm, tools)
        
        print("LangGraph ReAct Agent created!\n")
        
    except Exception as e:
        print(f"Failed to create agent: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Step 3: Test Agentic Reasoning
    print_section("STEP 3: Testing LangGraph Agentic Reasoning")
    
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
            # Stream agent execution to see reasoning!
            messages = [HumanMessage(content=question)]
            
            print("Agent is thinking...\n")
            
            for chunk in agent_executor.stream({"messages": messages}):
                print(chunk)
                print("---")
            
            # Get final result
            result = agent_executor.invoke({"messages": messages})
            final_answer = result['messages'][-1].content
            
            print(f"\n{'='*70}")
            print("FINAL ANSWER:")
            print('='*70)
            print(final_answer)
            print('='*70 + '\n')
            
        except Exception as e:
            print(f"\nError: {e}\n")
            import traceback
            traceback.print_exc()
    
    print_section("LANGGRAPH TEST COMPLETE!")

if __name__ == "__main__":
    print("\nStarting LangGraph Agentic RAG with Ollama...\n")
    langgraph_rag_test()