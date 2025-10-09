"""
Modern Question Answering System
Core QA functionality with multiple model support and advanced features
"""

import torch
from transformers import (
    pipeline, 
    AutoTokenizer, 
    AutoModelForQuestionAnswering,
    AutoModel
)
from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Dict, Tuple, Optional
import json
import logging
from dataclasses import dataclass
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class QAResult:
    """Structured result for question answering"""
    question: str
    answer: str
    confidence: float
    context: str
    model_name: str
    start_position: int
    end_position: int

class ModelManager:
    """Manages different QA models and their configurations"""
    
    AVAILABLE_MODELS = {
        "distilbert": "distilbert-base-cased-distilled-squad",
        "bert-large": "bert-large-uncased-whole-word-masking-finetuned-squad", 
        "roberta": "roberta-base-squad2",
        "albert": "albert-base-v2-squad2",
        "deberta": "microsoft/deberta-base-squad2"
    }
    
    def __init__(self, model_name: str = "distilbert", device: str = "auto"):
        self.model_name = model_name
        self.device = self._get_device(device)
        self.model_id = self.AVAILABLE_MODELS[model_name]
        self.pipeline = None
        self.tokenizer = None
        self.model = None
        self._load_model()
    
    def _get_device(self, device: str) -> str:
        """Determine the best device to use"""
        if device == "auto":
            if torch.cuda.is_available():
                return "cuda"
            elif torch.backends.mps.is_available():
                return "mps"
            else:
                return "cpu"
        return device
    
    def _load_model(self):
        """Load the selected model and tokenizer"""
        try:
            logger.info(f"Loading model: {self.model_id} on {self.device}")
            
            # Load pipeline for easy QA
            self.pipeline = pipeline(
                "question-answering",
                model=self.model_id,
                device=0 if self.device == "cuda" else -1,
                return_all_scores=True
            )
            
            # Load tokenizer and model for advanced features
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
            self.model = AutoModelForQuestionAnswering.from_pretrained(self.model_id)
            
            if self.device != "cpu":
                self.model = self.model.to(self.device)
                
            logger.info(f"Model loaded successfully on {self.device}")
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def answer_question(self, question: str, context: str) -> QAResult:
        """Answer a question given a context"""
        try:
            result = self.pipeline(question=question, context=context)
            
            # Handle different result formats
            if isinstance(result, list) and len(result) > 0:
                result = result[0]
            
            return QAResult(
                question=question,
                answer=result['answer'],
                confidence=result['score'],
                context=context,
                model_name=self.model_id,
                start_position=result.get('start', 0),
                end_position=result.get('end', 0)
            )
            
        except Exception as e:
            logger.error(f"Error answering question: {e}")
            return QAResult(
                question=question,
                answer="Error processing question",
                confidence=0.0,
                context=context,
                model_name=self.model_id,
                start_position=0,
                end_position=0
            )

class ContextRanker:
    """Ranks and retrieves relevant contexts for questions"""
    
    def __init__(self):
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
    
    def rank_contexts(self, question: str, contexts: List[str], top_k: int = 3) -> List[Tuple[str, float]]:
        """Rank contexts by relevance to the question"""
        try:
            # Encode question and contexts
            question_embedding = self.embedder.encode([question])
            context_embeddings = self.embedder.encode(contexts)
            
            # Calculate similarities
            similarities = np.dot(context_embeddings, question_embedding.T).flatten()
            
            # Rank by similarity
            ranked_indices = np.argsort(similarities)[::-1]
            
            # Return top-k contexts with scores
            ranked_contexts = []
            for i in ranked_indices[:top_k]:
                ranked_contexts.append((contexts[i], float(similarities[i])))
            
            return ranked_contexts
            
        except Exception as e:
            logger.error(f"Error ranking contexts: {e}")
            return [(context, 0.0) for context in contexts[:top_k]]

class ModernQASystem:
    """Main QA system with advanced features"""
    
    def __init__(self, model_name: str = "distilbert", device: str = "auto"):
        self.model_manager = ModelManager(model_name, device)
        self.context_ranker = ContextRanker()
        self.documents = []
    
    def load_documents(self, documents: List[str]):
        """Load documents for context retrieval"""
        self.documents = documents
        logger.info(f"Loaded {len(documents)} documents")
    
    def answer_with_context_ranking(self, question: str, top_k: int = 3) -> List[QAResult]:
        """Answer question using context ranking"""
        if not self.documents:
            logger.warning("No documents loaded")
            return []
        
        # Rank contexts
        ranked_contexts = self.context_ranker.rank_contexts(question, self.documents, top_k)
        
        # Answer question for each top context
        results = []
        for context, relevance_score in ranked_contexts:
            qa_result = self.model_manager.answer_question(question, context)
            qa_result.confidence *= relevance_score  # Weight by relevance
            results.append(qa_result)
        
        return results
    
    def batch_answer(self, questions: List[str], context: str) -> List[QAResult]:
        """Answer multiple questions efficiently"""
        results = []
        for question in questions:
            result = self.model_manager.answer_question(question, context)
            results.append(result)
        return results
    
    def get_model_info(self) -> Dict:
        """Get information about the current model"""
        return {
            "model_name": self.model_manager.model_name,
            "model_id": self.model_manager.model_id,
            "device": self.model_manager.device,
            "available_models": list(ModelManager.AVAILABLE_MODELS.keys())
        }

def main():
    """Command-line interface for the QA system"""
    print("🧠 Modern Question Answering System")
    print("=" * 50)
    
    # Initialize system
    qa_system = ModernQASystem(model_name="distilbert")
    
    # Sample documents
    sample_documents = [
        """
        Artificial Intelligence (AI) refers to the simulation of human intelligence in machines.
        These machines are programmed to think like humans and mimic their actions. 
        AI is being used in various fields including healthcare, finance, and transportation.
        Machine learning is a subset of AI that focuses on algorithms that can learn from data.
        """,
        """
        Natural Language Processing (NLP) is a branch of AI that helps computers understand,
        interpret and manipulate human language. It bridges the gap between human communication
        and computer understanding. Common NLP tasks include text classification, sentiment analysis,
        and machine translation.
        """,
        """
        Deep learning is a subset of machine learning that uses neural networks with multiple layers
        to model and understand complex patterns in data. It has revolutionized fields like computer
        vision, speech recognition, and natural language processing.
        """
    ]
    
    qa_system.load_documents(sample_documents)
    
    # Sample questions
    questions = [
        "What does AI refer to?",
        "What is machine learning?",
        "What are common NLP tasks?",
        "How has deep learning revolutionized AI?"
    ]
    
    print(f"\n📊 Model Info: {qa_system.get_model_info()}")
    print(f"\n📚 Loaded {len(sample_documents)} documents")
    
    print("\n🔍 QA Results with Context Ranking:")
    print("-" * 50)
    
    for question in questions:
        print(f"\n❓ Question: {question}")
        results = qa_system.answer_with_context_ranking(question, top_k=2)
        
        for i, result in enumerate(results, 1):
            print(f"  📝 Answer {i}: {result.answer}")
            print(f"  🎯 Confidence: {result.confidence:.3f}")
            print(f"  📄 Context snippet: {result.context[:100]}...")
            print()

if __name__ == "__main__":
    main()
