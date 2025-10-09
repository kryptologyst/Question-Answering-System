"""
Utility functions for the Question Answering System
"""

import re
import string
from typing import List, Dict, Any
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def clean_text(text: str) -> str:
    """Clean and normalize text"""
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    # Remove special characters but keep punctuation
    text = re.sub(r'[^\w\s.,!?;:]', '', text)
    return text.strip()

def extract_keywords(text: str, max_keywords: int = 10) -> List[str]:
    """Extract keywords from text using TF-IDF"""
    try:
        vectorizer = TfidfVectorizer(
            max_features=max_keywords,
            stop_words='english',
            ngram_range=(1, 2)
        )
        tfidf_matrix = vectorizer.fit_transform([text])
        feature_names = vectorizer.get_feature_names_out()
        scores = tfidf_matrix.toarray()[0]
        
        # Get top keywords
        keyword_scores = list(zip(feature_names, scores))
        keyword_scores.sort(key=lambda x: x[1], reverse=True)
        
        return [keyword for keyword, score in keyword_scores if score > 0]
    except:
        return []

def calculate_text_similarity(text1: str, text2: str) -> float:
    """Calculate similarity between two texts using TF-IDF and cosine similarity"""
    try:
        vectorizer = TfidfVectorizer(stop_words='english')
        tfidf_matrix = vectorizer.fit_transform([text1, text2])
        similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        return float(similarity)
    except:
        return 0.0

def format_confidence_score(confidence: float) -> str:
    """Format confidence score with appropriate styling"""
    if confidence >= 0.8:
        return f"🟢 High ({confidence:.3f})"
    elif confidence >= 0.5:
        return f"🟡 Medium ({confidence:.3f})"
    else:
        return f"🔴 Low ({confidence:.3f})"

def truncate_text(text: str, max_length: int = 200) -> str:
    """Truncate text to specified length with ellipsis"""
    if len(text) <= max_length:
        return text
    return text[:max_length-3] + "..."

def highlight_answer_in_context(answer: str, context: str) -> str:
    """Highlight the answer within the context"""
    # Simple highlighting - in a real implementation, you'd use more sophisticated matching
    highlighted = context.replace(answer, f"**{answer}**")
    return highlighted

def validate_question(question: str) -> Dict[str, Any]:
    """Validate and analyze a question"""
    validation_result = {
        "is_valid": True,
        "length": len(question),
        "word_count": len(question.split()),
        "has_question_mark": question.endswith('?'),
        "issues": []
    }
    
    if len(question.strip()) == 0:
        validation_result["is_valid"] = False
        validation_result["issues"].append("Question is empty")
    
    if len(question.split()) < 3:
        validation_result["is_valid"] = False
        validation_result["issues"].append("Question too short")
    
    if len(question) > 500:
        validation_result["issues"].append("Question very long")
    
    return validation_result

def generate_question_suggestions(context: str, num_suggestions: int = 5) -> List[str]:
    """Generate suggested questions based on context"""
    # Simple implementation - extract key phrases and create questions
    sentences = context.split('.')
    suggestions = []
    
    for sentence in sentences[:num_suggestions]:
        sentence = sentence.strip()
        if len(sentence) > 20:
            # Extract first few words and create a question
            words = sentence.split()[:5]
            if len(words) >= 3:
                question = f"What is {' '.join(words)}?"
                suggestions.append(question)
    
    return suggestions[:num_suggestions]

def calculate_answer_quality(answer: str, question: str, context: str) -> Dict[str, float]:
    """Calculate various quality metrics for an answer"""
    metrics = {}
    
    # Answer length
    metrics["answer_length"] = len(answer.split())
    
    # Answer completeness (how much of the question is addressed)
    question_words = set(question.lower().split())
    answer_words = set(answer.lower().split())
    metrics["question_coverage"] = len(question_words.intersection(answer_words)) / len(question_words) if question_words else 0
    
    # Context utilization (how much of the answer comes from context)
    context_words = set(context.lower().split())
    metrics["context_utilization"] = len(answer_words.intersection(context_words)) / len(answer_words) if answer_words else 0
    
    # Answer specificity (avoid generic answers)
    generic_words = {"the", "a", "an", "is", "are", "was", "were", "be", "been", "being"}
    specific_words = answer_words - generic_words
    metrics["specificity"] = len(specific_words) / len(answer_words) if answer_words else 0
    
    return metrics
