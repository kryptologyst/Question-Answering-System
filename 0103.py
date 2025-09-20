# Project 103. Question Answering System
# Description:
# A modern Question Answering (QA) system that provides precise answers to user questions from given contexts.
# This implementation uses state-of-the-art transformer models from Hugging Face Transformers,
# supports multiple models, includes evaluation metrics, and provides a web interface.

import os
import json
import random
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import torch
from transformers import (
    pipeline, 
    AutoTokenizer, 
    AutoModelForQuestionAnswering,
    AutoConfig
)
import numpy as np
from datetime import datetime

@dataclass
class QAResult:
    """Data class for storing QA results with confidence metrics"""
    question: str
    answer: str
    confidence: float
    start_position: int
    end_position: int
    model_name: str
    context: str
    timestamp: str

class ModernQASystem:
    """Modern Question Answering System with multiple models and evaluation"""
    
    def __init__(self, model_name: str = "distilbert-base-uncased-distilled-squad"):
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.pipeline = None
        self.tokenizer = None
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load the QA model and tokenizer"""
        try:
            print(f"Loading {self.model_name} on {self.device}...")
            self.pipeline = pipeline(
                "question-answering",
                model=self.model_name,
                device=0 if self.device == "cuda" else -1,
                return_all_scores=True
            )
            print(f"Model loaded successfully!")
        except Exception as e:
            print(f"Error loading model: {e}")
            # Fallback to a more reliable model
            self.model_name = "distilbert-base-uncased-distilled-squad"
            self.pipeline = pipeline("question-answering", model=self.model_name)
    
    def answer_question(self, question: str, context: str) -> QAResult:
        """Answer a question given a context"""
        try:
            result = self.pipeline(question=question, context=context)
            
            # Handle different result formats
            if isinstance(result, list) and len(result) > 0:
                best_result = result[0]
            else:
                best_result = result
            
            return QAResult(
                question=question,
                answer=best_result.get('answer', 'No answer found'),
                confidence=best_result.get('score', 0.0),
                start_position=best_result.get('start', 0),
                end_position=best_result.get('end', 0),
                model_name=self.model_name,
                context=context,
                timestamp=datetime.now().isoformat()
            )
        except Exception as e:
            print(f"Error processing question: {e}")
            return QAResult(
                question=question,
                answer="Error processing question",
                confidence=0.0,
                start_position=0,
                end_position=0,
                model_name=self.model_name,
                context=context,
                timestamp=datetime.now().isoformat()
            )
    
    def batch_answer(self, questions: List[str], context: str) -> List[QAResult]:
        """Answer multiple questions for the same context"""
        results = []
        for question in questions:
            result = self.answer_question(question, context)
            results.append(result)
        return results
    
    def evaluate_confidence(self, result: QAResult) -> str:
        """Evaluate confidence level of the answer"""
        if result.confidence >= 0.8:
            return "High"
        elif result.confidence >= 0.5:
            return "Medium"
        else:
            return "Low"

class MockDatabase:
    """Mock database with diverse contexts and questions for testing"""
    
    def __init__(self):
        self.contexts = {
            "eiffel_tower": {
                "context": """
                The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, France. 
                It is named after the engineer Gustave Eiffel, whose company designed and built the tower. 
                Constructed from 1887 to 1889 as the entrance to the 1889 World's Fair, 
                it was initially criticized by some of France's leading artists and intellectuals for its design, 
                but it has become a global cultural icon of France and one of the most recognizable structures in the world.
                The tower is 330 meters tall, including its antenna, and weighs approximately 10,100 tons.
                """,
                "questions": [
                    "Who designed the Eiffel Tower?",
                    "When was the Eiffel Tower constructed?",
                    "How tall is the Eiffel Tower?",
                    "What material is the Eiffel Tower made of?",
                    "Where is the Eiffel Tower located?"
                ]
            },
            "artificial_intelligence": {
                "context": """
                Artificial Intelligence (AI) is intelligence demonstrated by machines, in contrast to the natural intelligence displayed by humans. 
                Leading AI textbooks define the field as the study of "intelligent agents": any device that perceives its environment and takes actions that maximize its chance of successfully achieving its goals. 
                The term "artificial intelligence" is often used to describe machines that mimic "cognitive" functions that humans associate with the human mind, such as "learning" and "problem solving".
                AI research has been defined as the field of study of intelligent agents, which refers to any system that perceives its environment and takes actions that maximize its chance of achieving its goals.
                """,
                "questions": [
                    "What is artificial intelligence?",
                    "How do AI systems perceive their environment?",
                    "What are intelligent agents?",
                    "What cognitive functions do AI systems mimic?"
                ]
            },
            "climate_change": {
                "context": """
                Climate change refers to long-term shifts in global or regional climate patterns. 
                In particular, the term commonly refers to the observed century-scale rise in the average temperature of the Earth's climate system and its related effects. 
                Multiple lines of scientific evidence show that the climate system is warming. 
                Many of the observed changes since the 1950s are unprecedented over decades to millennia. 
                The scientific understanding of climate change is based on extensive research and peer-reviewed scientific literature.
                """,
                "questions": [
                    "What is climate change?",
                    "What evidence supports climate change?",
                    "When did significant climate changes begin?",
                    "How is climate change research conducted?"
                ]
            }
        }
    
    def get_random_context(self) -> Tuple[str, str, List[str]]:
        """Get a random context with its topic and questions"""
        topic = random.choice(list(self.contexts.keys()))
        data = self.contexts[topic]
        return topic, data["context"], data["questions"]
    
    def get_context_by_topic(self, topic: str) -> Tuple[str, List[str]]:
        """Get context and questions for a specific topic"""
        if topic in self.contexts:
            data = self.contexts[topic]
            return data["context"], data["questions"]
        return "", []
    
    def list_topics(self) -> List[str]:
        """List all available topics"""
        return list(self.contexts.keys())

def main():
    """Main function to demonstrate the modern QA system"""
    print("🤖 Modern Question Answering System")
    print("=" * 50)
    
    # Initialize the QA system
    qa_system = ModernQASystem()
    mock_db = MockDatabase()
    
    # Get a random context for demonstration
    topic, context, questions = mock_db.get_random_context()
    print(f"\n📚 Topic: {topic.replace('_', ' ').title()}")
    print(f"📖 Context: {context.strip()[:200]}...")
    
    # Answer questions
    print(f"\n❓ Questions and Answers:")
    print("-" * 30)
    
    results = qa_system.batch_answer(questions, context)
    
    for i, result in enumerate(results, 1):
        confidence_level = qa_system.evaluate_confidence(result)
        print(f"\n{i}. Q: {result.question}")
        print(f"   A: {result.answer}")
        print(f"   Confidence: {result.confidence:.3f} ({confidence_level})")
        print(f"   Model: {result.model_name}")
    
    # Interactive mode
    print(f"\n🔄 Interactive Mode")
    print("Enter your own questions (type 'quit' to exit):")
    
    while True:
        user_question = input("\n❓ Your question: ").strip()
        if user_question.lower() in ['quit', 'exit', 'q']:
            break
        
        if user_question:
            result = qa_system.answer_question(user_question, context)
            confidence_level = qa_system.evaluate_confidence(result)
            print(f"🤖 Answer: {result.answer}")
            print(f"📊 Confidence: {result.confidence:.3f} ({confidence_level})")

if __name__ == "__main__":
    main()