#!/usr/bin/env python3
"""
Batch Evaluation Script for NASA RAG Chat

Runs the full RAG pipeline (retrieval -> generation -> evaluation) across
multiple test questions and reports per-question and aggregate metrics.
"""

import json
import os
from pathlib import Path
from typing import Dict, List

# Load environment variables from .env file
def load_env_file():
    env_path = Path(__file__).parent / ".env"
    if env_path.exists():
        with open(env_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    value = value.strip().strip("'\"")
                    os.environ[key] = value

load_env_file()

import rag_client
import llm_client
import ragas_evaluator


def load_evaluation_dataset(filepath: str = "evaluation_dataset.json") -> List[Dict]:
    """Load test questions from the evaluation dataset file"""
    dataset_path = Path(__file__).parent / filepath
    with open(dataset_path, 'r') as f:
        data = json.load(f)
    return data.get("test_questions", [])


def run_batch_evaluation(n_docs: int = 3, model: str = "gpt-3.5-turbo"):
    """Run evaluation across all test questions"""

    print("=" * 60)
    print("NASA RAG Chat - Batch Evaluation")
    print("=" * 60)

    # Load test questions
    questions = load_evaluation_dataset()
    print(f"\nLoaded {len(questions)} test questions\n")

    # Initialize RAG system
    print("Initializing RAG system...")
    backends = rag_client.discover_chroma_backends()
    if not backends:
        print("ERROR: No ChromaDB backends found!")
        return

    # Use the first available backend
    backend_key = list(backends.keys())[0]
    backend = backends[backend_key]
    print(f"Using backend: {backend['display_name']}")

    collection, success, error = rag_client.initialize_rag_system(
        backend["directory"],
        backend["collection_name"]
    )

    if not success:
        print(f"ERROR: Failed to initialize RAG system: {error}")
        return

    # Get API key
    openai_key = os.environ.get("OPENAI_API_KEY", "")
    if not openai_key:
        print("ERROR: OPENAI_API_KEY not set")
        return

    print(f"Model: {model}")
    print(f"Documents per query: {n_docs}")
    print("\n" + "=" * 60)

    # Store results for aggregation
    all_results = []

    # Process each question
    for q in questions:
        print(f"\n--- Question {q['id']} ({q['category']}) ---")
        print(f"Q: {q['question']}")

        # Step 1: Retrieve documents
        docs_result = rag_client.retrieve_documents(collection, q['question'], n_docs)

        if not docs_result or not docs_result.get("documents"):
            print("  No documents retrieved")
            continue

        documents = docs_result["documents"][0]
        metadatas = docs_result["metadatas"][0]
        distances = docs_result.get("distances", [[]])[0] if docs_result.get("distances") else None

        # Step 2: Format context
        context = rag_client.format_context(documents, metadatas, distances)

        # Step 3: Generate response
        response = llm_client.generate_response(
            openai_key,
            q['question'],
            context,
            [],  # No conversation history for batch eval
            model
        )
        print(f"A: {response[:200]}..." if len(response) > 200 else f"A: {response}")

        # Step 4: Evaluate with RAGAS
        scores = ragas_evaluator.evaluate_response_quality(
            q['question'],
            response,
            documents
        )

        # Display per-question metrics
        print(f"\nMetrics:")
        for metric, score in scores.items():
            if isinstance(score, float):
                print(f"  {metric}: {score:.3f}")
            else:
                print(f"  {metric}: {score}")

        # Store for aggregation
        all_results.append({
            "id": q['id'],
            "category": q['category'],
            "question": q['question'],
            "response": response,
            "scores": scores
        })

    # Calculate and display aggregate statistics
    print("\n" + "=" * 60)
    print("AGGREGATE RESULTS")
    print("=" * 60)

    # Collect numeric scores per metric
    metric_scores = {}
    for result in all_results:
        for metric, score in result['scores'].items():
            if isinstance(score, float):
                if metric not in metric_scores:
                    metric_scores[metric] = []
                metric_scores[metric].append(score)

    # Calculate and print means
    print(f"\nTotal questions evaluated: {len(all_results)}")
    print("\nMean scores per metric:")
    for metric, scores in metric_scores.items():
        if scores:
            mean_score = sum(scores) / len(scores)
            min_score = min(scores)
            max_score = max(scores)
            print(f"  {metric}:")
            print(f"    Mean: {mean_score:.3f}")
            print(f"    Min:  {min_score:.3f}")
            print(f"    Max:  {max_score:.3f}")

    # Save results to file
    output_path = Path(__file__).parent / "batch_eval_results.json"
    with open(output_path, 'w') as f:
        json.dump({
            "config": {"n_docs": n_docs, "model": model},
            "results": all_results,
            "aggregate": {
                metric: {
                    "mean": sum(scores) / len(scores) if scores else 0,
                    "min": min(scores) if scores else 0,
                    "max": max(scores) if scores else 0
                }
                for metric, scores in metric_scores.items()
            }
        }, f, indent=2)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    run_batch_evaluation()
