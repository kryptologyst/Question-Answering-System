"""
Modern Question Answering System - Streamlit Web Interface
Interactive web UI for the QA system with advanced features
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import sys
import json

# Add the project root to the path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from qa_system import ModernQASystem, QAResult
from data.mock_database import MockDatabase

# Page configuration
st.set_page_config(
    page_title="🧠 Modern QA System",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
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
        background-color: #e8f5e8;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #28a745;
    }
    .confidence-high { color: #28a745; font-weight: bold; }
    .confidence-medium { color: #ffc107; font-weight: bold; }
    .confidence-low { color: #dc3545; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_qa_system(model_name: str = "distilbert"):
    """Load the QA system with caching"""
    return ModernQASystem(model_name=model_name)

@st.cache_resource
def load_database():
    """Load the mock database with caching"""
    return MockDatabase()

def get_confidence_color_class(confidence: float) -> str:
    """Get CSS class based on confidence score"""
    if confidence >= 0.7:
        return "confidence-high"
    elif confidence >= 0.4:
        return "confidence-medium"
    else:
        return "confidence-low"

def display_answer_result(result: QAResult, show_context: bool = True):
    """Display a single QA result"""
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.markdown(f"""
        <div class="answer-box">
            <strong>Answer:</strong> {result.answer}
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        confidence_class = get_confidence_color_class(result.confidence)
        st.markdown(f"""
        <div class="metric-card">
            <div class="{confidence_class}">
                Confidence: {result.confidence:.3f}
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    if show_context:
        with st.expander("📄 View Context"):
            st.text(result.context)
    
    st.markdown("---")

def main():
    """Main Streamlit application"""
    
    # Header
    st.markdown('<h1 class="main-header">🧠 Modern Question Answering System</h1>', unsafe_allow_html=True)
    
    # Load systems
    with st.spinner("Loading QA system and database..."):
        qa_system = load_qa_system()
        database = load_database()
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Model selection
        model_options = list(qa_system.model_manager.AVAILABLE_MODELS.keys())
        selected_model = st.selectbox(
            "Select Model",
            model_options,
            index=0,
            help="Choose the QA model to use"
        )
        
        # Reload system if model changed
        if selected_model != qa_system.model_manager.model_name:
            qa_system = load_qa_system(selected_model)
        
        st.markdown("---")
        
        # Database statistics
        st.header("📊 Database Stats")
        stats = database.get_statistics()
        st.metric("Total Documents", stats["total_documents"])
        st.metric("Categories", len(stats["categories"]))
        st.metric("Unique Tags", stats["unique_tags"])
        
        # Category filter
        st.markdown("---")
        st.header("🔍 Filter Documents")
        categories = list(stats["categories"].keys())
        selected_category = st.selectbox("Category", ["All"] + categories)
        
        # Load documents based on filter
        if selected_category == "All":
            documents = database.get_all_documents()
        else:
            documents = database.get_documents_by_category(selected_category)
        
        qa_system.load_documents([doc["content"] for doc in documents])
    
    # Main content area
    tab1, tab2, tab3, tab4 = st.tabs(["🔍 Single Question", "📝 Batch Questions", "📚 Document Explorer", "📊 Analytics"])
    
    with tab1:
        st.header("Ask a Single Question")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            question = st.text_input(
                "Enter your question:",
                placeholder="e.g., What is artificial intelligence?",
                help="Ask any question about the loaded documents"
            )
        
        with col2:
            top_k = st.slider("Number of contexts", 1, 5, 3)
        
        if st.button("🔍 Get Answer", type="primary") and question:
            with st.spinner("Processing your question..."):
                results = qa_system.answer_with_context_ranking(question, top_k)
            
            if results:
                st.success(f"Found {len(results)} answer(s)")
                
                for i, result in enumerate(results, 1):
                    st.subheader(f"Answer {i}")
                    display_answer_result(result)
            else:
                st.warning("No answers found. Try a different question or check your documents.")
    
    with tab2:
        st.header("Batch Question Processing")
        
        # Sample questions
        sample_questions = [
            "What is artificial intelligence?",
            "How does machine learning work?",
            "What are the effects of climate change?",
            "What is quantum computing?",
            "How does blockchain work?"
        ]
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            questions_text = st.text_area(
                "Enter questions (one per line):",
                value="\n".join(sample_questions),
                height=200,
                help="Enter multiple questions, one per line"
            )
        
        with col2:
            st.markdown("**Sample Questions:**")
            for i, q in enumerate(sample_questions, 1):
                st.text(f"{i}. {q}")
        
        if st.button("🚀 Process All Questions", type="primary"):
            questions = [q.strip() for q in questions_text.split('\n') if q.strip()]
            
            if questions:
                progress_bar = st.progress(0)
                results_container = st.container()
                
                all_results = []
                
                for i, question in enumerate(questions):
                    with st.spinner(f"Processing question {i+1}/{len(questions)}..."):
                        question_results = qa_system.answer_with_context_ranking(question, 1)
                        if question_results:
                            all_results.extend(question_results)
                    
                    progress_bar.progress((i + 1) / len(questions))
                
                # Display results
                with results_container:
                    st.success(f"Processed {len(questions)} questions")
                    
                    for i, question in enumerate(questions):
                        question_results = [r for r in all_results if r.question == question]
                        if question_results:
                            st.subheader(f"Q{i+1}: {question}")
                            display_answer_result(question_results[0], show_context=False)
            else:
                st.warning("Please enter at least one question.")
    
    with tab3:
        st.header("Document Explorer")
        
        # Search functionality
        search_query = st.text_input("Search documents:", placeholder="Enter keywords...")
        
        if search_query:
            search_results = database.search_documents(search_query)
            st.info(f"Found {len(search_results)} documents matching '{search_query}'")
        else:
            search_results = documents
        
        # Display documents
        for doc in search_results:
            with st.expander(f"📄 {doc['title']} ({doc['category']})"):
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    st.write(doc['content'])
                
                with col2:
                    st.markdown("**Tags:**")
                    for tag in doc['tags']:
                        st.markdown(f"- {tag}")
    
    with tab4:
        st.header("Analytics Dashboard")
        
        # Model information
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Current Model", qa_system.model_manager.model_name)
        
        with col2:
            st.metric("Device", qa_system.model_manager.device)
        
        with col3:
            st.metric("Available Models", len(qa_system.model_manager.AVAILABLE_MODELS))
        
        # Document distribution
        st.subheader("📊 Document Distribution")
        
        # Category distribution
        category_data = pd.DataFrame(list(stats["categories"].items()), columns=["Category", "Count"])
        fig_cat = px.pie(category_data, values="Count", names="Category", title="Documents by Category")
        st.plotly_chart(fig_cat, use_container_width=True)
        
        # Tag frequency
        st.subheader("🏷️ Most Common Tags")
        all_tags = []
        for doc in database.get_all_documents():
            all_tags.extend(doc["tags"])
        
        tag_counts = pd.Series(all_tags).value_counts().head(10)
        fig_tags = px.bar(x=tag_counts.values, y=tag_counts.index, orientation='h', 
                         title="Top 10 Most Common Tags")
        fig_tags.update_layout(yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(fig_tags, use_container_width=True)
        
        # Model comparison (placeholder)
        st.subheader("🔬 Model Performance Comparison")
        st.info("Model comparison feature coming soon! This would show performance metrics across different models.")

if __name__ == "__main__":
    main()
