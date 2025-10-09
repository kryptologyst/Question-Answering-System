# Question Answering System

A state-of-the-art question answering system built with the latest transformer models and modern techniques. This project demonstrates advanced NLP capabilities including context ranking, multiple model support, confidence scoring, and an interactive web interface.

## Features

- **Multiple Model Support**: Choose from various pre-trained QA models (DistilBERT, BERT, RoBERTa, ALBERT, DeBERTa)
- **Advanced Context Ranking**: Intelligent document retrieval and ranking using sentence transformers
- **Confidence Scoring**: Get confidence scores for answer reliability assessment
- **Interactive Web UI**: Modern Streamlit-based interface with real-time processing
- **Mock Database**: Built-in diverse document collection across multiple topics
- **Batch Processing**: Process multiple questions efficiently
- **Document Explorer**: Search and browse through the knowledge base
- **Analytics Dashboard**: Visualize model performance and document statistics
- **Extensible Architecture**: Easy to add new models and features

## Quick Start

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/kryptologyst/Question-Answering-System.git
cd Question-Answering-System

# Install dependencies
pip install -r requirements.txt
```

### 2. Run the System

**Web Interface (Recommended):**
```bash
streamlit run app.py
```

**Command Line Interface:**
```bash
python qa_system.py
```

**Run Tests:**
```bash
python test_system.py
```

**Original Simple Implementation:**
```bash
python 0175.py
```

## Project Structure

```
├── app.py                    # 🌐 Streamlit web interface
├── qa_system.py             # 🧠 Core QA system with CLI
├── 0175.py                  # 📜 Original simple implementation
├── test_system.py           # 🧪 Test suite
├── requirements.txt         # 📦 Dependencies
├── README.md               # 📖 This file
├── .gitignore              # 🚫 Git ignore rules
├── models/                 # 🤖 Model management
│   ├── model_manager.py    # Model loading and management
│   └── context_ranker.py   # Document ranking and retrieval
├── data/                   # 📚 Data and database
│   ├── mock_database.py    # Mock document database
│   └── documents.json      # Sample documents (generated)
└── utils/                  # 🔧 Utility functions
    └── helpers.py          # Text processing and validation
```

## Available Models

| Model | Description | Speed | Accuracy | Size |
|-------|-------------|-------|----------|------|
| **DistilBERT** | Fast and efficient | ⚡⚡⚡ | ⭐⭐⭐ | Small |
| **BERT Large** | High accuracy | ⚡ | ⭐⭐⭐⭐⭐ | Large |
| **RoBERTa** | Robust performance | ⚡⚡ | ⭐⭐⭐⭐ | Medium |
| **ALBERT** | Lightweight alternative | ⚡⚡⚡ | ⭐⭐⭐ | Small |
| **DeBERTa** | Advanced architecture | ⚡⚡ | ⭐⭐⭐⭐⭐ | Medium |

## Knowledge Base

The system includes a diverse mock database with documents covering:

- **Technology**: AI, Machine Learning, Quantum Computing, Blockchain
- **Science**: Neuroscience, Genetics, Space Exploration
- **Environment**: Climate Change, Renewable Energy, Sustainable Agriculture

## Usage Examples

### Web Interface

1. **Single Question**: Ask individual questions and get ranked answers
2. **Batch Processing**: Process multiple questions at once
3. **Document Explorer**: Browse and search through the knowledge base
4. **Analytics**: View model performance and document statistics

### Command Line

```python
from qa_system import ModernQASystem
from data.mock_database import MockDatabase

# Initialize system
qa_system = ModernQASystem(model_name="distilbert")
database = MockDatabase()

# Load documents
qa_system.load_documents(database.get_content_texts())

# Ask questions
results = qa_system.answer_with_context_ranking("What is AI?", top_k=3)
for result in results:
    print(f"Answer: {result.answer}")
    print(f"Confidence: {result.confidence:.3f}")
```

## 🔧 Advanced Features

### Context Ranking
The system uses sentence transformers to rank documents by relevance to the question, ensuring the most relevant context is used for answering.

### Confidence Scoring
Each answer includes a confidence score that combines:
- Model confidence
- Context relevance
- Answer completeness

### Model Comparison
Switch between different models to compare performance and choose the best one for your use case.

### Batch Processing
Process multiple questions efficiently with progress tracking and result aggregation.

## Testing

Run the comprehensive test suite:

```bash
python test_system.py
```

Tests include:
- ✅ Import validation
- ✅ Database functionality
- ✅ Utility functions
- ✅ QA system core functionality

## Performance

The system is optimized for:
- **Speed**: Efficient model loading and caching
- **Memory**: Smart device selection (CUDA/MPS/CPU)
- **Accuracy**: Multiple model options for different accuracy/speed tradeoffs
- **Scalability**: Batch processing and context ranking

## 🛠️ Customization

### Adding New Models

```python
# In qa_system.py, add to AVAILABLE_MODELS
AVAILABLE_MODELS = {
    "your_model": "huggingface/model-name",
    # ... existing models
}
```

### Adding Documents

```python
# In data/mock_database.py, add to _load_sample_documents()
{
    "id": "your_doc",
    "title": "Your Document Title",
    "content": "Your document content...",
    "category": "your_category",
    "tags": ["tag1", "tag2"]
}
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

This project is part of the "1000 AI Projects" series and is available for educational and research purposes.

## Acknowledgments

- **Hugging Face** for the transformers library and pre-trained models
- **Streamlit** for the web interface framework
- **Sentence Transformers** for semantic similarity
- **PyTorch** for the deep learning framework

## Future Enhancements

- [ ] Support for generative QA models
- [ ] Real-time document ingestion
- [ ] Multi-language support
- [ ] API endpoints
- [ ] Model fine-tuning capabilities
- [ ] Advanced visualization features
- [ ] Integration with external knowledge bases


