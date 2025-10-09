#!/usr/bin/env python3
"""
Test script for the Modern Question Answering System
Run this to verify everything is working correctly
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

def test_imports():
    """Test if all required modules can be imported"""
    print("🧪 Testing imports...")
    
    try:
        import torch
        print(f"✅ PyTorch {torch.__version__}")
    except ImportError as e:
        print(f"❌ PyTorch import failed: {e}")
        return False
    
    try:
        import transformers
        print(f"✅ Transformers {transformers.__version__}")
    except ImportError as e:
        print(f"❌ Transformers import failed: {e}")
        return False
    
    try:
        import streamlit
        print(f"✅ Streamlit {streamlit.__version__}")
    except ImportError as e:
        print(f"❌ Streamlit import failed: {e}")
        return False
    
    try:
        import sentence_transformers
        print(f"✅ Sentence Transformers {sentence_transformers.__version__}")
    except ImportError as e:
        print(f"❌ Sentence Transformers import failed: {e}")
        return False
    
    return True

def test_qa_system():
    """Test the core QA system"""
    print("\n🧠 Testing QA System...")
    
    try:
        from qa_system import ModernQASystem
        from data.mock_database import MockDatabase
        
        # Initialize systems
        qa_system = ModernQASystem(model_name="distilbert")
        database = MockDatabase()
        
        # Load documents
        documents = database.get_content_texts()[:3]  # Use first 3 documents
        qa_system.load_documents(documents)
        
        # Test question
        test_question = "What is artificial intelligence?"
        results = qa_system.answer_with_context_ranking(test_question, top_k=2)
        
        if results:
            print(f"✅ QA System working! Found {len(results)} answers")
            print(f"   Question: {test_question}")
            print(f"   Best Answer: {results[0].answer}")
            print(f"   Confidence: {results[0].confidence:.3f}")
            return True
        else:
            print("❌ QA System returned no results")
            return False
            
    except Exception as e:
        print(f"❌ QA System test failed: {e}")
        return False

def test_database():
    """Test the mock database"""
    print("\n📚 Testing Mock Database...")
    
    try:
        from data.mock_database import MockDatabase
        
        db = MockDatabase()
        stats = db.get_statistics()
        
        print(f"✅ Database loaded successfully!")
        print(f"   Documents: {stats['total_documents']}")
        print(f"   Categories: {len(stats['categories'])}")
        print(f"   Tags: {stats['unique_tags']}")
        
        # Test search
        search_results = db.search_documents("AI")
        print(f"   Search results for 'AI': {len(search_results)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Database test failed: {e}")
        return False

def test_utilities():
    """Test utility functions"""
    print("\n🔧 Testing Utilities...")
    
    try:
        from utils.helpers import clean_text, extract_keywords, validate_question
        
        # Test text cleaning
        dirty_text = "  This   is   a   test!!!  "
        clean = clean_text(dirty_text)
        assert clean == "This is a test!!!"
        print("✅ Text cleaning works")
        
        # Test keyword extraction
        text = "Artificial intelligence is a field of computer science."
        keywords = extract_keywords(text, 3)
        print(f"✅ Keyword extraction works: {keywords}")
        
        # Test question validation
        validation = validate_question("What is AI?")
        assert validation["is_valid"] == True
        print("✅ Question validation works")
        
        return True
        
    except Exception as e:
        print(f"❌ Utilities test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 Modern Question Answering System - Test Suite")
    print("=" * 60)
    
    tests = [
        ("Import Test", test_imports),
        ("Database Test", test_database),
        ("Utilities Test", test_utilities),
        ("QA System Test", test_qa_system),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        if test_func():
            passed += 1
        else:
            print(f"❌ {test_name} FAILED")
    
    print(f"\n{'='*60}")
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The system is ready to use.")
        print("\n🚀 To run the web interface:")
        print("   streamlit run app.py")
        print("\n🚀 To run the command-line interface:")
        print("   python qa_system.py")
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
        print("\n💡 Make sure to install all dependencies:")
        print("   pip install -r requirements.txt")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
