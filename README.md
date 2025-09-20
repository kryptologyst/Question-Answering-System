# Question Answering System

A state-of-the-art Question Answering (QA) system built with modern transformer models from Hugging Face. This project demonstrates advanced NLP techniques for extracting precise answers from unstructured text using attention mechanisms.

## Features

- **Multiple Model Support**: DistilBERT, BERT, RoBERTa, and more
- **Real-time Confidence Scoring**: High/Medium/Low confidence levels
- **Interactive Web Interface**: Modern Streamlit-based UI
- **Batch Processing**: Answer multiple questions simultaneously
- **Mock Database**: Diverse contexts and questions for testing
- **Export Capabilities**: CSV and JSON export of results
- **GPU Acceleration**: Automatic CUDA support when available
- **Evaluation Metrics**: Comprehensive confidence analysis

## 🛠️ Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd 0103_Question_answering_system
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**:
   ```bash
   # Command line interface
   python 0103.py
   
   # Web interface
   streamlit run app.py
   ```

## Usage

### Command Line Interface

```python
from qa_system import ModernQASystem, MockDatabase

# Initialize the system
qa_system = ModernQASystem()
mock_db = MockDatabase()

# Get a context and questions
topic, context, questions = mock_db.get_random_context()

# Answer a question
result = qa_system.answer_question("Who designed the Eiffel Tower?", context)
print(f"Answer: {result.answer}")
print(f"Confidence: {result.confidence}")
```

### Web Interface

1. Start the Streamlit app:
   ```bash
   streamlit run app.py
   ```

2. Open your browser to `http://localhost:8501`

3. Select a topic and model from the sidebar

4. Enter questions or use predefined ones

5. View results with confidence scores and export data

## Architecture

### Core Components

- **`ModernQASystem`**: Main QA engine with model management
- **`MockDatabase`**: Sample contexts and questions for testing
- **`QAResult`**: Data structure for storing results with metadata
- **`app.py`**: Streamlit web interface

### Supported Models

| Model | Speed | Accuracy | Use Case |
|-------|-------|----------|----------|
| DistilBERT | Fast | Good | Quick responses |
| BERT Base | Medium | High | Balanced performance |
| RoBERTa | Slower | Highest | Best accuracy |

## Performance Metrics

The system provides confidence scoring based on model predictions:

- **High Confidence (≥0.8)**: Very reliable answers
- **Medium Confidence (0.5-0.8)**: Generally reliable
- **Low Confidence (<0.5)**: May need verification

## 🔧 Configuration

### Model Selection

```python
# Available models
models = {
    "distilbert-base-uncased-distilled-squad": "Fast, efficient",
    "bert-base-uncased": "Balanced performance", 
    "roberta-base-squad2": "Highest accuracy"
}

qa_system = ModernQASystem(model_name="bert-base-uncased")
```

### Custom Contexts

```python
# Add your own contexts
custom_context = """
Your custom text here...
"""

result = qa_system.answer_question("Your question?", custom_context)
```

## Evaluation

The system includes built-in evaluation capabilities:

- **Confidence Distribution**: Visual analysis of answer quality
- **Batch Processing**: Test multiple questions at once
- **Export Results**: Save results for further analysis

## Advanced Features

### Batch Processing

```python
questions = ["Question 1?", "Question 2?", "Question 3?"]
results = qa_system.batch_answer(questions, context)

for result in results:
    print(f"Q: {result.question}")
    print(f"A: {result.answer} (Confidence: {result.confidence})")
```

### Custom Evaluation

```python
def evaluate_answers(results):
    high_conf = sum(1 for r in results if r.confidence >= 0.8)
    total = len(results)
    return f"High confidence rate: {high_conf/total:.2%}"
```

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**: Use CPU mode or smaller models
2. **Model Loading Errors**: Check internet connection for downloads
3. **Low Confidence**: Try different models or rephrase questions

### Performance Tips

- Use DistilBERT for faster responses
- Batch multiple questions together
- Ensure sufficient RAM (4GB+ recommended)

## Technical Details

### Model Architecture

The system uses transformer-based models fine-tuned on SQuAD (Stanford Question Answering Dataset):

- **Attention Mechanisms**: Focus on relevant context parts
- **Token Classification**: Start/end position prediction
- **Confidence Scoring**: Probability-based answer quality

### Data Flow

1. **Input**: Question + Context text
2. **Tokenization**: Convert to model-readable format
3. **Inference**: Generate answer spans and confidence
4. **Post-processing**: Format and evaluate results

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is open source and available under the MIT License.

## Acknowledgments

- Hugging Face for transformer models
- Stanford for SQuAD dataset
- Streamlit for web interface
- PyTorch for deep learning framework

## Support

For questions or issues:
- Open an issue on GitHub
- Check the troubleshooting section
- Review the documentation


# Question-Answering-System
