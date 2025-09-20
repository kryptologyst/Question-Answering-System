#!/usr/bin/env python3
"""
Test suite for Modern Question Answering System
Comprehensive tests for all components
"""

import unittest
import sys
import os
from unittest.mock import patch, MagicMock

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import the main module
import importlib.util
spec = importlib.util.spec_from_file_location("qa_system", "0103.py")
qa_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(qa_module)

ModernQASystem = qa_module.ModernQASystem
MockDatabase = qa_module.MockDatabase
QAResult = qa_module.QAResult

class TestQAResult(unittest.TestCase):
    """Test QAResult data class"""
    
    def test_qa_result_creation(self):
        """Test QAResult object creation"""
        result = QAResult(
            question="Test question?",
            answer="Test answer",
            confidence=0.95,
            start_position=10,
            end_position=20,
            model_name="test-model",
            context="Test context",
            timestamp="2023-01-01T00:00:00"
        )
        
        self.assertEqual(result.question, "Test question?")
        self.assertEqual(result.answer, "Test answer")
        self.assertEqual(result.confidence, 0.95)
        self.assertEqual(result.model_name, "test-model")

class TestMockDatabase(unittest.TestCase):
    """Test MockDatabase functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.mock_db = MockDatabase()
    
    def test_database_initialization(self):
        """Test database initialization"""
        self.assertIsInstance(self.mock_db.contexts, dict)
        self.assertGreater(len(self.mock_db.contexts), 0)
    
    def test_get_random_context(self):
        """Test getting random context"""
        topic, context, questions = self.mock_db.get_random_context()
        
        self.assertIsInstance(topic, str)
        self.assertIsInstance(context, str)
        self.assertIsInstance(questions, list)
        self.assertGreater(len(context), 0)
        self.assertGreater(len(questions), 0)
    
    def test_get_context_by_topic(self):
        """Test getting context by specific topic"""
        topics = self.mock_db.list_topics()
        if topics:
            topic = topics[0]
            context, questions = self.mock_db.get_context_by_topic(topic)
            
            self.assertIsInstance(context, str)
            self.assertIsInstance(questions, list)
    
    def test_list_topics(self):
        """Test listing available topics"""
        topics = self.mock_db.list_topics()
        self.assertIsInstance(topics, list)
        self.assertGreater(len(topics), 0)

class TestModernQASystem(unittest.TestCase):
    """Test ModernQASystem functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Mock the pipeline to avoid loading actual models during tests
        with patch('transformers.pipeline') as mock_pipeline:
            mock_pipeline.return_value = MagicMock()
            self.qa_system = ModernQASystem()
    
    def test_system_initialization(self):
        """Test system initialization"""
        self.assertIsNotNone(self.qa_system.model_name)
        self.assertIsNotNone(self.qa_system.device)
    
    def test_evaluate_confidence(self):
        """Test confidence evaluation"""
        # Test high confidence
        result = QAResult(
            question="Test", answer="Test", confidence=0.9,
            start_position=0, end_position=0, model_name="test",
            context="test", timestamp="2023-01-01"
        )
        self.assertEqual(self.qa_system.evaluate_confidence(result), "High")
        
        # Test medium confidence
        result.confidence = 0.6
        self.assertEqual(self.qa_system.evaluate_confidence(result), "Medium")
        
        # Test low confidence
        result.confidence = 0.3
        self.assertEqual(self.qa_system.evaluate_confidence(result), "Low")
    
    @patch('transformers.pipeline')
    def test_answer_question(self, mock_pipeline):
        """Test answering a question"""
        # Mock the pipeline response
        mock_result = {
            'answer': 'Test answer',
            'score': 0.95,
            'start': 10,
            'end': 20
        }
        mock_pipeline.return_value.return_value = mock_result
        
        # Create a new system with mocked pipeline
        qa_system = ModernQASystem()
        
        result = qa_system.answer_question("Test question?", "Test context")
        
        self.assertIsInstance(result, QAResult)
        self.assertEqual(result.question, "Test question?")
        self.assertEqual(result.answer, "Test answer")
        self.assertEqual(result.confidence, 0.95)
    
    def test_batch_answer(self):
        """Test batch answering"""
        questions = ["Question 1?", "Question 2?"]
        context = "Test context"
        
        # Mock the answer_question method
        with patch.object(self.qa_system, 'answer_question') as mock_answer:
            mock_answer.return_value = QAResult(
                question="Test", answer="Test", confidence=0.8,
                start_position=0, end_position=0, model_name="test",
                context="test", timestamp="2023-01-01"
            )
            
            results = self.qa_system.batch_answer(questions, context)
            
            self.assertEqual(len(results), 2)
            self.assertIsInstance(results[0], QAResult)

class TestIntegration(unittest.TestCase):
    """Integration tests"""
    
    def test_full_workflow(self):
        """Test complete workflow"""
        mock_db = MockDatabase()
        topic, context, questions = mock_db.get_random_context()
        
        # Mock the QA system to avoid loading actual models
        with patch('transformers.pipeline') as mock_pipeline:
            mock_result = {
                'answer': 'Mock answer',
                'score': 0.85,
                'start': 0,
                'end': 10
            }
            mock_pipeline.return_value.return_value = mock_result
            
            qa_system = ModernQASystem()
            result = qa_system.answer_question(questions[0], context)
            
            self.assertIsInstance(result, QAResult)
            self.assertEqual(result.question, questions[0])

def run_performance_tests():
    """Run performance tests (optional)"""
    print("🚀 Running performance tests...")
    
    # This would run actual model tests if models are available
    # For now, just print a message
    print("✅ Performance tests completed (mocked)")

def main():
    """Main test runner"""
    print("🧪 Running Modern Question Answering System Tests")
    print("=" * 50)
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test cases
    suite.addTests(loader.loadTestsFromTestCase(TestQAResult))
    suite.addTests(loader.loadTestsFromTestCase(TestMockDatabase))
    suite.addTests(loader.loadTestsFromTestCase(TestModernQASystem))
    suite.addTests(loader.loadTestsFromTestCase(TestIntegration))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "=" * 50)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.failures:
        print("\n❌ Failures:")
        for test, traceback in result.failures:
            print(f"- {test}: {traceback}")
    
    if result.errors:
        print("\n❌ Errors:")
        for test, traceback in result.errors:
            print(f"- {test}: {traceback}")
    
    if result.wasSuccessful():
        print("\n🎉 All tests passed!")
        return 0
    else:
        print("\n❌ Some tests failed!")
        return 1

if __name__ == "__main__":
    sys.exit(main())
