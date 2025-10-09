#!/usr/bin/env python3
"""
Project 175: Modern Question Answering System
A state-of-the-art QA system with multiple models, web UI, and advanced features.

This is the original simple implementation. For the full modern system, see:
- qa_system.py: Core QA functionality with CLI
- app.py: Streamlit web interface
- test_system.py: Test suite

Run the modern system:
    streamlit run app.py          # Web interface
    python qa_system.py           # Command line
    python test_system.py         # Run tests
"""

from transformers import pipeline

def main():
    """Original simple QA implementation"""
    print("🧠 Project 175: Question Answering System")
    print("=" * 50)
    print("This is the original simple implementation.")
    print("For the modern system with advanced features, run:")
    print("  streamlit run app.py")
    print("=" * 50)
    
    # Load Hugging Face QA pipeline
    qa_pipeline = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")
    
    # Context paragraph
    context = """
    Artificial Intelligence (AI) refers to the simulation of human intelligence in machines.
    These machines are programmed to think like humans and mimic their actions. 
    AI is being used in various fields including healthcare, finance, and transportation.
    """
    
    # Sample questions
    questions = [
        "What does AI refer to?",
        "Where is AI being used?",
        "What do machines do in AI?"
    ]
    
    print("🧠 QA Results:\n")
    for question in questions:
        result = qa_pipeline(question=question, context=context)
        print(f"❓ Question: {question}")
        print(f"✅ Answer: {result['answer']}\n")

if __name__ == "__main__":
    main()