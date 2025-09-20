# Streamlit Web UI for Question Answering System
# Modern, interactive web interface for the QA system

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import json
import time
import random
from typing import List, Dict

# Import our QA system
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import from the main module
import importlib.util
spec = importlib.util.spec_from_file_location("qa_system", "0103.py")
qa_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(qa_module)

ModernQASystem = qa_module.ModernQASystem
MockDatabase = qa_module.MockDatabase
QAResult = qa_module.QAResult

# Page configuration
st.set_page_config(
    page_title="🤖 Modern Question Answering System",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #667eea;
    }
    .answer-box {
        background-color: #e8f4fd;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin: 1rem 0;
    }
    .confidence-high { color: #28a745; }
    .confidence-medium { color: #ffc107; }
    .confidence-low { color: #dc3545; }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_qa_system():
    """Load the QA system with caching"""
    return ModernQASystem()

@st.cache_resource
def load_mock_database():
    """Load the mock database with caching"""
    return MockDatabase()

def display_answer(result: QAResult):
    """Display answer with confidence styling"""
    confidence_class = "confidence-high" if result.confidence >= 0.8 else "confidence-medium" if result.confidence >= 0.5 else "confidence-low"
    
    st.markdown(f"""
    <div class="answer-box">
        <h4>🤖 Answer:</h4>
        <p style="font-size: 1.2rem; margin: 0.5rem 0;">{result.answer}</p>
        <p class="{confidence_class}" style="font-weight: bold;">
            Confidence: {result.confidence:.3f} ({'High' if result.confidence >= 0.8 else 'Medium' if result.confidence >= 0.5 else 'Low'})
        </p>
        <small>Model: {result.model_name} | Timestamp: {result.timestamp}</small>
    </div>
    """, unsafe_allow_html=True)

def create_confidence_chart(results: List[QAResult]):
    """Create a confidence distribution chart"""
    confidences = [r.confidence for r in results]
    
    fig = go.Figure(data=[
        go.Bar(
            x=[f"Q{i+1}" for i in range(len(results))],
            y=confidences,
            marker_color=['#28a745' if c >= 0.8 else '#ffc107' if c >= 0.5 else '#dc3545' for c in confidences],
            text=[f"{c:.3f}" for c in confidences],
            textposition='auto',
        )
    ])
    
    fig.update_layout(
        title="Confidence Scores by Question",
        xaxis_title="Questions",
        yaxis_title="Confidence Score",
        yaxis=dict(range=[0, 1]),
        height=400
    )
    
    return fig

def main():
    """Main Streamlit application"""
    
    # Header
    st.markdown('<h1 class="main-header">🤖 Modern Question Answering System</h1>', unsafe_allow_html=True)
    
    # Initialize systems
    qa_system = load_qa_system()
    mock_db = load_mock_database()
    
    # Sidebar
    st.sidebar.title("⚙️ Configuration")
    
    # Model selection
    st.sidebar.subheader("🤖 Model Selection")
    model_options = {
        "DistilBERT (Fast)": "distilbert-base-uncased-distilled-squad",
        "BERT Base": "bert-base-uncased",
        "RoBERTa": "roberta-base-squad2"
    }
    
    selected_model = st.sidebar.selectbox(
        "Choose Model:",
        options=list(model_options.keys()),
        index=0
    )
    
    # Update model if changed
    if qa_system.model_name != model_options[selected_model]:
        with st.spinner(f"Loading {selected_model}..."):
            qa_system = ModernQASystem(model_options[selected_model])
            st.success(f"✅ {selected_model} loaded successfully!")
    
    # Topic selection
    st.sidebar.subheader("📚 Topic Selection")
    topics = mock_db.list_topics()
    selected_topic = st.sidebar.selectbox(
        "Choose Topic:",
        options=topics,
        format_func=lambda x: x.replace('_', ' ').title()
    )
    
    # Get context and questions
    context, questions = mock_db.get_context_by_topic(selected_topic)
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📖 Context")
        st.text_area("", value=context.strip(), height=200, disabled=True)
        
        st.subheader("❓ Ask Questions")
        
        # Question input
        user_question = st.text_input(
            "Enter your question:",
            placeholder="e.g., Who designed the Eiffel Tower?"
        )
        
        # Predefined questions
        st.write("**Or select a predefined question:**")
        if st.button("🎲 Random Question"):
            user_question = random.choice(questions)
            st.session_state.random_question = user_question
        
        if 'random_question' in st.session_state:
            user_question = st.session_state.random_question
        
        # Answer button
        if st.button("🔍 Get Answer", type="primary"):
            if user_question:
                with st.spinner("Processing your question..."):
                    result = qa_system.answer_question(user_question, context)
                    display_answer(result)
                    
                    # Store result in session state
                    if 'qa_history' not in st.session_state:
                        st.session_state.qa_history = []
                    st.session_state.qa_history.append(result)
            else:
                st.warning("Please enter a question first!")
    
    with col2:
        st.subheader("📊 System Info")
        
        # Metrics
        col2_1, col2_2 = st.columns(2)
        with col2_1:
            st.metric("Model", selected_model)
            st.metric("Device", "GPU" if qa_system.device == "cuda" else "CPU")
        
        with col2_2:
            st.metric("Topic", selected_topic.replace('_', ' ').title())
            st.metric("Questions Available", len(questions))
        
        # Confidence distribution
        if 'qa_history' in st.session_state and st.session_state.qa_history:
            st.subheader("📈 Confidence Distribution")
            fig = create_confidence_chart(st.session_state.qa_history)
            st.plotly_chart(fig, use_container_width=True)
        
        # Recent questions
        if 'qa_history' in st.session_state and st.session_state.qa_history:
            st.subheader("🕒 Recent Questions")
            recent_questions = st.session_state.qa_history[-5:]  # Last 5 questions
            for i, result in enumerate(reversed(recent_questions)):
                with st.expander(f"Q{i+1}: {result.question[:50]}..."):
                    st.write(f"**Answer:** {result.answer}")
                    st.write(f"**Confidence:** {result.confidence:.3f}")
    
    # Batch processing section
    st.subheader("🔄 Batch Processing")
    
    col3, col4 = st.columns([1, 1])
    
    with col3:
        if st.button("📝 Answer All Predefined Questions"):
            with st.spinner("Processing all questions..."):
                results = qa_system.batch_answer(questions, context)
                
                # Display results
                for i, result in enumerate(results, 1):
                    st.write(f"**Q{i}:** {result.question}")
                    display_answer(result)
                    st.divider()
    
    with col4:
        # Export results
        if 'qa_history' in st.session_state and st.session_state.qa_history:
            st.subheader("💾 Export Results")
            
            # Convert to DataFrame
            df_data = []
            for result in st.session_state.qa_history:
                df_data.append({
                    'Question': result.question,
                    'Answer': result.answer,
                    'Confidence': result.confidence,
                    'Model': result.model_name,
                    'Timestamp': result.timestamp
                })
            
            df = pd.DataFrame(df_data)
            
            # Download buttons
            csv = df.to_csv(index=False)
            st.download_button(
                label="📥 Download CSV",
                data=csv,
                file_name=f"qa_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
            
            json_data = json.dumps([result.__dict__ for result in st.session_state.qa_history], indent=2)
            st.download_button(
                label="📥 Download JSON",
                data=json_data,
                file_name=f"qa_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #666;'>
            <p>🤖 Modern Question Answering System | Built with Streamlit & Hugging Face Transformers</p>
            <p>Supports multiple models: DistilBERT, BERT, RoBERTa | Real-time confidence scoring</p>
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
