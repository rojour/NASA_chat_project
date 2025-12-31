from typing import Dict, List
from openai import OpenAI

def generate_response(openai_key: str, user_message: str, context: str,
                     conversation_history: List[Dict], model: str = "gpt-3.5-turbo") -> str:
    """Generate response using OpenAI with context

    Args:
        openai_key: OpenAI API key
        user_message: The user's question
        context: Retrieved document context from RAG system
        conversation_history: List of previous messages in the conversation
        model: OpenAI model to use

    Returns:
        Generated response string
    """

    # Define system prompt for NASA expertise
    system_prompt = """You are an expert NASA historian and space mission specialist.
Your role is to provide accurate, informative answers about NASA space missions,
particularly Apollo 11, Apollo 13, and the Challenger missions.

When answering questions:
- Use the provided context from mission documents to support your answers
- Cite specific details from the documents when available
- If the context doesn't contain relevant information, say so clearly
- Maintain a professional yet engaging tone
- Explain technical terms when necessary
- Be precise about dates, names, and mission details"""

    # Build messages list
    messages = [{"role": "system", "content": system_prompt}]

    # Add context if available
    if context:
        context_message = f"""Use the following context from NASA mission documents to answer the user's question:

{context}

Please base your answer on this context when relevant."""
        messages.append({"role": "system", "content": context_message})

    # Add conversation history (limit to last 10 exchanges to manage token usage)
    history_limit = 20  # 10 user + 10 assistant messages
    recent_history = conversation_history[-history_limit:] if len(conversation_history) > history_limit else conversation_history

    for msg in recent_history:
        messages.append({"role": msg["role"], "content": msg["content"]})

    # Add current user message
    messages.append({"role": "user", "content": user_message})

    # Create OpenAI client (using Vocareum proxy for educational access)
    client = OpenAI(
        api_key=openai_key,
        base_url="https://openai.vocareum.com/v1"
    )

    # Send request to OpenAI
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.7,
        max_tokens=1024
    )

    # Return response content
    return response.choices[0].message.content