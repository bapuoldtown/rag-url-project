# test_rag.py
from index_wikipages import CreateIndexAbstract
from llama_index.core.agent import ReActAgent
from llama_index.core.tools import QueryEngineTool
from llama_index.llms.openai import OpenAI
from utils import ApiKeyHandler
import openai

def print_section(title):
    """Print formatted section header"""
    print("\n" + "="*70)
    print(f"  {title}")
    print("="*70 + "\n")

def test_indexing_and_query():
    """Test complete RAG pipeline with X12_Document_List"""
    
    print_section("🧪 WIKIPEDIA RAG TEST - X12 Document List")
    
    # Test 1: Build Index
    print_section("TEST 1: Building Index")
    query = "please index: X12_Document_List"
    print(f"📚 Query: {query}\n")
    
    try:
        index = CreateIndexAbstract.create_index(query)
        print("✅ Index built successfully!\n")
    except Exception as e:
        print(f"❌ Failed to build index: {e}")
        return
    
    # Test 2: Create Agent
    print_section("TEST 2: Creating Agentic RAG System")
    
    try:
        query_engine = index.as_query_engine(similarity_top_k=3)
        
        tool = QueryEngineTool.from_defaults(
            query_engine=query_engine,
            name="x12_knowledge_base",
            description="Search X12 Document List Wikipedia page for information about X12 transaction sets and EDI standards"
        )
        
        openai.api_key = ApiKeyHandler.get_apikey()
        llm = OpenAI(model="gpt-4-turbo", temperature=0)
        agent = ReActAgent.from_tools([tool], llm=llm, verbose=True)
        
        print("✅ Agent created successfully!\n")
    except Exception as e:
        print(f"❌ Failed to create agent: {e}")
        return
    
    # Test 3: Query Testing
    print_section("TEST 3: Testing Queries")
    
    test_questions = [
        {
            "question": "What is X12 Document List?",
            "description": "Basic factual question"
        },
        {
            "question": "List 5 common X12 transaction sets with their codes and purposes",
            "description": "Structured information extraction"
        },
        {
            "question": "What is the 850 transaction set used for?",
            "description": "Specific detail query"
        },
        {
            "question": "Explain the difference between inbound and outbound transaction sets",
            "description": "Conceptual understanding"
        },
        {
            "question": "What are the most frequently used X12 transaction sets in healthcare?",
            "description": "Domain-specific query"
        }
    ]
    
    results = []
    
    for i, test in enumerate(test_questions, 1):
        print(f"\n📝 Question {i}/{len(test_questions)}")
        print(f"Type: {test['description']}")
        print(f"Question: {test['question']}")
        print("-" * 70)
        
        try:
            response = agent.chat(test['question'])
            answer = str(response)
            
            print(f"\n💡 Answer:")
            print(answer)
            print("\n" + "="*70)
            
            results.append({
                "question": test['question'],
                "answer": answer,
                "status": "✅ Success"
            })
            
        except Exception as e:
            print(f"\n❌ Error: {e}\n")
            results.append({
                "question": test['question'],
                "answer": None,
                "status": f"❌ Failed: {e}"
            })
    
    # Test Summary
    print_section("📊 TEST SUMMARY")
    
    success_count = sum(1 for r in results if "Success" in r['status'])
    total_count = len(results)
    
    print(f"Total Tests: {total_count}")
    print(f"Passed: {success_count}")
    print(f"Failed: {total_count - success_count}")
    print(f"Success Rate: {(success_count/total_count)*100:.1f}%\n")
    
    print("Detailed Results:")
    for i, result in enumerate(results, 1):
        print(f"\n{i}. {result['status']}")
        print(f"   Q: {result['question']}")
        if result['answer']:
            preview = result['answer'][:100] + "..." if len(result['answer']) > 100 else result['answer']
            print(f"   A: {preview}")
    
    print("\n" + "="*70)
    print("🎉 TEST COMPLETE!")
    print("="*70 + "\n")

if __name__ == "__main__":
    print("\n🚀 Starting RAG System Test...\n")
    test_indexing_and_query()