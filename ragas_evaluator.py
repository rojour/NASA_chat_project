from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from typing import Dict, List, Optional

# RAGAS imports
try:
    from ragas import SingleTurnSample
    from ragas.metrics import BleuScore, NonLLMContextPrecisionWithReference, ResponseRelevancy, Faithfulness, RougeScore
    from ragas import evaluate
    RAGAS_AVAILABLE = True
except ImportError:
    RAGAS_AVAILABLE = False

def evaluate_response_quality(question: str, answer: str, contexts: List[str]) -> Dict[str, float]:
    """Evaluate response quality using RAGAS metrics

    Args:
        question: The user's question
        answer: The generated answer from the LLM
        contexts: List of retrieved context documents

    Returns:
        Dictionary with metric names and scores (0-1 scale)
    """
    if not RAGAS_AVAILABLE:
        return {"error": "RAGAS not available"}

    try:
        # Create evaluator LLM with model gpt-3.5-turbo (using Vocareum proxy)
        evaluator_llm = LangchainLLMWrapper(ChatOpenAI(
            model="gpt-3.5-turbo",
            base_url="https://openai.vocareum.com/v1"
        ))

        # Create evaluator embeddings with model text-embedding-3-small (using Vocareum proxy)
        evaluator_embeddings = LangchainEmbeddingsWrapper(
            OpenAIEmbeddings(
                model="text-embedding-3-small",
                base_url="https://openai.vocareum.com/v1"
            )
        )

        # Define an instance for each metric to evaluate
        # ResponseRelevancy - measures if answer is relevant to question
        response_relevancy = ResponseRelevancy(llm=evaluator_llm, embeddings=evaluator_embeddings)

        # Faithfulness - measures if answer is grounded in the context
        faithfulness = Faithfulness(llm=evaluator_llm)

        # RougeScore - measures text overlap between response and contexts (non-LLM metric)
        rouge_score = RougeScore()

        # Create a sample for evaluation
        sample = SingleTurnSample(
            user_input=question,
            response=answer,
            retrieved_contexts=contexts
        )

        # Evaluate the response using the metrics
        results = {}

        # Evaluate each metric individually with error handling
        try:
            relevancy_result = response_relevancy.single_turn_score(sample)
            results['response_relevancy'] = float(relevancy_result)
        except Exception as e:
            results['response_relevancy'] = f"Error: {str(e)[:50]}"

        try:
            faithfulness_result = faithfulness.single_turn_score(sample)
            results['faithfulness'] = float(faithfulness_result)
        except Exception as e:
            results['faithfulness'] = f"Error: {str(e)[:50]}"

        try:
            rouge_result = rouge_score.single_turn_score(sample)
            results['rouge_score'] = float(rouge_result)
        except Exception as e:
            results['rouge_score'] = f"Error: {str(e)[:50]}"

        # Return the evaluation results
        return results

    except Exception as e:
        return {"error": f"Evaluation failed: {str(e)}"}
