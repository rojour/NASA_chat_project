import chromadb
from chromadb.config import Settings
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from typing import Dict, List, Optional
from pathlib import Path
import os

def discover_chroma_backends() -> Dict[str, Dict[str, str]]:
    """Discover available ChromaDB backends in the project directory

    Returns:
        Dictionary mapping backend keys to their configuration info
    """
    backends = {}
    current_dir = Path(".")

    # Create list of directories that match specific criteria (directory type and name pattern)
    chroma_dirs = [d for d in current_dir.iterdir()
                   if d.is_dir() and 'chroma' in d.name.lower()]

    # Loop through each discovered directory
    for chroma_dir in chroma_dirs:
        # Wrap connection attempt in try-except block for error handling
        try:
            # Initialize database client with directory path and configuration settings
            client = chromadb.PersistentClient(path=str(chroma_dir))

            # Retrieve list of available collections from the database
            collections = client.list_collections()

            # Loop through each collection found
            for collection in collections:
                # Create unique identifier key combining directory and collection names
                key = f"{chroma_dir.name}:{collection.name}"

                # Get document count with fallback for unsupported operations
                try:
                    doc_count = collection.count()
                except Exception:
                    doc_count = "Unknown"

                # Build information dictionary containing all required fields
                backends[key] = {
                    "directory": str(chroma_dir),
                    "collection_name": collection.name,
                    "display_name": f"{chroma_dir.name} - {doc_count} docs",
                    "doc_count": doc_count
                }

        # Handle connection or access errors gracefully
        except Exception as e:
            # Create fallback entry for inaccessible directories
            error_msg = str(e)[:50]  # Truncate error message
            backends[f"{chroma_dir.name}:error"] = {
                "directory": str(chroma_dir),
                "collection_name": "error",
                "display_name": f"{chroma_dir.name} (Error: {error_msg})",
                "doc_count": 0
            }

    # Return complete backends dictionary with all discovered collections
    return backends

def initialize_rag_system(chroma_dir: str, collection_name: str):
    """Initialize the RAG system with specified backend

    Args:
        chroma_dir: Path to ChromaDB directory
        collection_name: Name of the collection to use

    Returns:
        Tuple of (collection, success_bool, error_message)
    """
    try:
        # Create a ChromaDB persistent client
        client = chromadb.PersistentClient(path=chroma_dir)

        # Create embedding function with Vocareum base URL
        api_key = os.environ.get("OPENAI_API_KEY", "")
        embedding_function = OpenAIEmbeddingFunction(
            api_key=api_key,
            model_name="text-embedding-3-small",
            api_base="https://openai.vocareum.com/v1"
        )

        # Return the collection with the collection_name and embedding function
        collection = client.get_collection(
            name=collection_name,
            embedding_function=embedding_function
        )

        return collection, True, ""

    except Exception as e:
        return None, False, str(e)

def retrieve_documents(collection, query: str, n_results: int = 3,
                      mission_filter: Optional[str] = None,
                      data_type_filter: Optional[str] = None,
                      category_filter: Optional[str] = None) -> Optional[Dict]:
    """Retrieve relevant documents from ChromaDB with optional filtering

    Args:
        collection: ChromaDB collection to query
        query: Search query text
        n_results: Number of results to return
        mission_filter: Optional mission name to filter by (e.g., 'apollo_11')
        data_type_filter: Optional data type to filter by (e.g., 'transcript')
        category_filter: Optional document category to filter by (e.g., 'technical')

    Returns:
        Query results dictionary with documents, metadatas, distances
    """
    # Build list of active filters
    filter_conditions = []

    # Check each filter and add to conditions if valid
    if mission_filter and mission_filter.lower() not in ["all", "none", ""]:
        filter_conditions.append({"mission": mission_filter})

    if data_type_filter and data_type_filter.lower() not in ["all", "none", ""]:
        filter_conditions.append({"data_type": data_type_filter})

    if category_filter and category_filter.lower() not in ["all", "none", ""]:
        filter_conditions.append({"document_category": category_filter})

    # Build where_filter based on number of conditions
    where_filter = None
    if len(filter_conditions) == 1:
        where_filter = filter_conditions[0]
    elif len(filter_conditions) > 1:
        # Use $and operator for multiple filters
        where_filter = {"$and": filter_conditions}

    # Execute database query with the required parameters
    results = collection.query(
        query_texts=[query],      # Pass search query in required format
        n_results=n_results,       # Set maximum number of results
        where=where_filter         # Apply conditional filter
    )

    # Return query results to caller
    return results

def format_context(documents: List[str], metadatas: List[Dict],
                   distances: Optional[List[float]] = None) -> str:
    """Format retrieved documents into context for LLM consumption

    Args:
        documents: List of document texts
        metadatas: List of metadata dictionaries
        distances: Optional list of distance scores (lower = more relevant)

    Returns:
        Formatted context string ready for LLM
    """
    if not documents:
        return ""

    # Step 1: Combine documents with metadatas and distances
    if distances:
        combined = list(zip(documents, metadatas, distances))
        # Sort by distance (lower = more relevant)
        combined.sort(key=lambda x: x[2])
    else:
        combined = [(doc, meta, 0) for doc, meta in zip(documents, metadatas)]

    # Step 2: Deduplicate - remove near-identical documents
    unique_docs = []
    for doc, metadata, distance in combined:
        # Check if this document is too similar to one we already have
        is_duplicate = False
        doc_normalized = doc.lower().strip()[:500]

        for existing_doc, _, _ in unique_docs:
            existing_normalized = existing_doc.lower().strip()[:500]
            # Compare first 500 chars - if 90% match, it's a duplicate
            if len(doc_normalized) > 0 and len(existing_normalized) > 0:
                shorter_len = min(len(doc_normalized), len(existing_normalized))
                matches = sum(1 for a, b in zip(doc_normalized, existing_normalized) if a == b)
                if matches / shorter_len >= 0.9:
                    is_duplicate = True
                    break

        if not is_duplicate:
            unique_docs.append((doc, metadata, distance))

    # Step 3: Format the unique, sorted documents
    context_parts = ["=== Retrieved NASA Mission Documents ===\n"]

    for i, (doc, metadata, distance) in enumerate(unique_docs, 1):
        # Extract mission information from metadata with fallback value
        mission = metadata.get("mission", "Unknown Mission")
        mission = mission.replace("_", " ").title()

        # Extract source information from metadata with fallback value
        source = metadata.get("source", "Unknown Source")

        # Extract category information from metadata with fallback value
        category = metadata.get("document_category", "General")
        category = category.replace("_", " ").title()

        # Create formatted source header
        source_header = f"--- Source {i}: {mission} | {category} | {source} ---"
        context_parts.append(source_header)

        # Truncate if necessary (max 2000 chars per doc)
        max_doc_length = 2000
        if len(doc) > max_doc_length:
            context_parts.append(doc[:max_doc_length] + "... [truncated]")
        else:
            context_parts.append(doc)

        context_parts.append("")  # Add blank line between documents

    return "\n".join(context_parts)