"""Automated test suite for RAG evaluation metrics."""

import sys
import os

# Ensure the project root is in the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from app.eval.evaluator import evaluate_response


def run_tests():
    print("=== Running RAG Evaluation Module Tests ===")

    # Test Case 1: High quality RAG response (well supported, relevant, complete, valid citations)
    query_1 = "Does BRCA1 play a role in DNA repair?"
    answer_1 = "Yes, BRCA1 is highly involved in DNA repair. According to scientific research, the BRCA1 gene is critical for repairing double-strand DNA breaks via homologous recombination."
    contexts_1 = [
        {
            "id": "chunk_01",
            "document": "The BRCA1 gene is a tumor suppressor that plays a crucial role in maintaining genomic stability. It is directly involved in repairing double-strand DNA breaks through homologous recombination pathway."
        }
    ]
    citations_1 = [
        {
            "chunk_id": "chunk_01",
            "text": "crucial role in maintaining genomic stability. It is directly involved in repairing double-strand DNA breaks"
        }
    ]

    print("\nExecuting Test Case 1: High Quality Response...")
    eval_result_1 = evaluate_response(query_1, answer_1, contexts_1, citations_1)
    
    metrics_1 = eval_result_1["metrics"]
    print(f"  - Relevance Score: {metrics_1['relevance_score']}")
    print(f"  - Hallucination Risk: {metrics_1['hallucination_risk']}")
    print(f"  - Completeness Score: {metrics_1['completeness_score']}")
    print(f"  - Citation Validation Score: {metrics_1['citation_validation_score']}")
    
    # Assertions
    assert metrics_1["relevance_score"] >= 0.7, "Relevance score should be high"
    assert metrics_1["hallucination_risk"] <= 0.4, "Hallucination risk should be low"
    assert metrics_1["completeness_score"] >= 0.7, "Completeness score should be high"
    assert metrics_1["citation_validation_score"] == 1.0, "Citations should be 100% valid"

    # Test Case 2: Hallucinated / Unsupported response with invalid citations
    query_2 = "What chromosome is BRCA1 located on?"
    answer_2 = "The BRCA1 gene is located on chromosome 21 and encodes a protein that synthesizes glucose."
    contexts_2 = [
        {
            "id": "chunk_02",
            "document": "BRCA1 is located on the long arm of chromosome 17 (17q21.31) and plays a key role in double-strand DNA repair."
        }
    ]
    citations_2 = [
        {
            "chunk_id": "chunk_02",
            "text": "chromosome 21 and synthesizes glucose"  # Does not match context
        },
        {
            "chunk_id": "chunk_9999",  # Invalid ID
            "text": "invalid chunk ID"
        }
    ]

    print("\nExecuting Test Case 2: Hallucinated / Unsupported Response...")
    eval_result_2 = evaluate_response(query_2, answer_2, contexts_2, citations_2)
    
    metrics_2 = eval_result_2["metrics"]
    print(f"  - Relevance Score: {metrics_2['relevance_score']}")
    print(f"  - Hallucination Risk: {metrics_2['hallucination_risk']}")
    print(f"  - Completeness Score: {metrics_2['completeness_score']}")
    print(f"  - Citation Validation Score: {metrics_2['citation_validation_score']}")

    # Assertions
    assert metrics_2["hallucination_risk"] >= 0.5, "Hallucination risk should be flagged high"
    assert metrics_2["citation_validation_score"] < 0.5, "Citation validation should be low/zero"

    print("\n=== ALL EVALUATION TESTS PASSED SUCCESSFULLY! ===")


if __name__ == "__main__":
    run_tests()
