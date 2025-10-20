# Consolidated test script that includes unit tests for Router, RAG Components(retriever, augmenter, generator), and Safety validator
import unittest
from unittest.mock import Mock, patch
from rag.agents.router import MedicalRAGRouter, QueryType
from rag.retrieval.retriever import MedicalRetriever
from rag.augmentation.augmenter import MedicalAugmenter
from rag.generation.generator import MedicalGenerator
from rag.ingestion.document_schema import MedicalDocument
from safety_layer.validator import MedicalResponseValidator, ValidationResult, ValidationLevel

class TestMedicalRAGRouter(unittest.TestCase):
    def setUp(self):
        self.router = MedicalRAGRouter()

    def test_classify_structured_query(self):
        result = self.router.classify_query("aspirin and warfarin", intent="ask_medication")
        self.assertEqual(result["query_type"], QueryType.STRUCTURED)
        self.assertFalse(result["emergency_flag"])
        self.assertEqual(result["metadata_filter"], {"category": "medication"})

    def test_classify_emergency_query(self):
        result = self.router.classify_query("chest pain")
        self.assertEqual(result["query_type"], QueryType.EMERGENCY)
        self.assertTrue(result["emergency_flag"])
        self.assertEqual(result["metadata_filter"], {"category": "emergency"})

    def test_classify_complex_query(self):
        result = self.router.classify_query("diabetes management")
        self.assertEqual(result["query_type"], QueryType.COMPLEX)
        self.assertFalse(result["emergency_flag"])
        self.assertEqual(result["metadata_filter"], {"category": "treatment", "last_updated": {"$gte": "2023-01-01"}})

class TestMedicalRetriever(unittest.TestCase):
    @patch("rag.retrieval.retriever.PineconeVectorStore")
    @patch("rag.retrieval.retriever.OpenAIEmbeddings")
    def setUp(self, mock_embeddings, mock_vectorstore):
        self.mock_vectorstore = mock_vectorstore.return_value
        self.retriever = MedicalRetriever(index_name="test_index")
        self.mock_retriever = Mock()
        self.mock_vectorstore.as_retriever.return_value = self.mock_retriever

    def test_naive_retrieval(self):
        self.mock_retriever.get_relevant_documents.return_value = [
            {"page_content": "Content 1", "metadata": {"source": "test", "confidence": 0.9}}
        ]
        docs = self.retriever.naive_retrieval("test query")
        self.assertEqual(len(docs), 1)
        self.assertIsInstance(docs[0], MedicalDocument)
        self.assertEqual(docs[0].content, "Content 1")

    def test_mmr_retrieval(self):
        self.mock_retriever.get_relevant_documents.return_value = [
            {"page_content": "Content 2", "metadata": {"source": "test", "confidence": 0.8}}
        ]
        docs = self.retriever.mmr_retrieval("test query")
        self.assertEqual(len(docs), 1)
        self.assertEqual(docs[0].content, "Content 2")

    def test_filtered_retrieval(self):
        filter = {"category": "treatment"}
        self.mock_retriever.get_relevant_documents.return_value = [
            {"page_content": "Content 3", "metadata": {"category": "treatment", "confidence": 0.7}}
        ]
        docs = self.retriever.filtered_retrieval("test query", filter)
        self.assertEqual(len(docs), 1)
        self.assertEqual(docs[0].content, "Content 3")

class TestMedicalAugmenter(unittest.TestCase):
    def setUp(self):
        self.augmenter = MedicalAugmenter()
        self.docs = [
            MedicalDocument(content="Diabetes tip 1", source="WHO", confidence=0.9),
            MedicalDocument(content="Diabetes tip 2", source="CDC", confidence=0.8)
        ]

    def test_augment_with_documents(self):
        augmented = self.augmenter.augment("diabetes tips", self.docs)
        self.assertIn("Based on the following context", augmented["prompt"])
        self.assertIn("Diabetes tip 1", augmented["prompt"])
        self.assertIn("Diabetes tip 2", augmented["prompt"])
        self.assertIn("WHO", augmented["metadata"]["sources"][0])
        self.assertIn("CDC", augmented["metadata"]["sources"][1])

    def test_augment_without_documents(self):
        augmented = self.augmenter.augment("no data query", [])
        self.assertIn("No relevant data found", augmented["prompt"])
        self.assertEqual(len(augmented["metadata"]), 0)

class TestMedicalGenerator(unittest.TestCase):
    @patch("rag.generation.generator.ChatDeepSeek")
    def setUp(self, mock_chat):
        self.mock_llm = Mock()
        mock_chat.return_value = self.mock_llm
        self.generator = MedicalGenerator()
        self.augmented_input = {"prompt": "Test prompt", "metadata": {"sources": [{"source": "test", "confidence": 0.9}]}}

    def test_generate_response(self):
        self.mock_llm.invoke.return_value = Mock(content="Test response")
        result = self.generator.generate(self.augmented_input)
        self.assertEqual(result["response"], "Test response")
        self.assertGreater(result["confidence"], 0.5)

    def test_combine_kb_and_rag(self):
        self.mock_llm.invoke.return_value = Mock(content="Combined response")
        kb_result = {"response": "KB data", "source": "internal_kb", "confidence": 0.9}
        rag_result = {"response": "RAG data", "metadata": {"sources": []}, "confidence": 0.8}
        combined = self.generator.combine_kb_and_rag(kb_result, rag_result)
        self.assertEqual(combined["response"], "Combined response")
        self.assertEqual(combined["confidence"], 0.8)

class TestMedicalResponseValidator(unittest.TestCase):
    def setUp(self):
        self.validator = MedicalResponseValidator()

    def test_validate_unsafe_response(self):
        result = self.validator.validate_response("Stop taking medication.")
        self.assertFalse(result.is_safe)
        self.assertEqual(result.level, ValidationLevel.UNSAFE)
        self.assertIn("consult", result.modified_response.lower())

    def test_validate_emergency_response(self):
        result = self.validator.validate_response("Call 911 for chest pain.")
        self.assertTrue(result.is_safe)
        self.assertEqual(result.level, ValidationLevel.EMERGENCY)
        self.assertIn("911", result.modified_response)

    def test_validate_with_sources(self):
        sources = [{"source": "WHO", "confidence": 0.9}, {"source": "CDC", "confidence": 0.8}]
        result = self.validator.validate_response("Safe advice.", sources)
        self.assertTrue(result.is_safe)
        self.assertTrue(result.sources_verified)
        self.assertGreater(result.confidence_score, 0.7)

if __name__ == "__main__":
    unittest.main()