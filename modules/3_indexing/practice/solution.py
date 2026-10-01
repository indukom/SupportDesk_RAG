"""
================================================================================
MODULE 3: Indexing Strategies for RAG
================================================================================
"""

import json
import os
import httpx
from dotenv import load_dotenv
from llama_index import (
        VectorStoreIndex,
        SummaryIndex,
        TreeIndex,
        KeywordTableIndex,
        Document
    )


from llama_index.service_context import ServiceContext
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI

load_dotenv()

os.environ["HTTPS_TIMEOUT"] = "300"

# =============================================================================
# CONFIGURE LLAMAINDEX SETTINGS
# =============================================================================
embed_model = OpenAIEmbedding(
    model=os.getenv('OPENAI_EMBEDDING_MODEL', 'text-embedding-3-small'),
    api_key=os.getenv('OPENAI_API_KEY'),
    timeout=120,
    max_retries=5
)

llm = OpenAI(
    model=os.getenv('OPENAI_CHAT_MODEL', 'gpt-4'),
    api_key=os.getenv('OPENAI_API_KEY'),
    timeout=300,
    max_retries=5
)

service_context = ServiceContext.from_defaults(
    embed_model=embed_model,
    llm=llm
)

# =============================================================================
# INTRODUCTION
# =============================================================================
print("="*80)
print("MODULE 3: INDEXING STRATEGIES FOR RAG")
print("="*80)

print("+"*80)
print("Loading Ticket data")
print("+"*80)

with open("../../../data/synthetic_tickets.json", "r") as f:
    tickets = json.load(f)

print(f"✓ Loaded {len(tickets)} tickets")

# -----------------------------------------------------------------------------
# Convert to LlamaIndex Documents
# ---------------------------------------
documents = []
for ticket in tickets:
    content = f"""
    Title: {ticket['title']}
    Description: {ticket["description"]}
    Resolution: {ticket["resolution"]}
    Category: {ticket["category"]}
    Priority: {ticket["priority"]}
"""
    doc = Document(
        text=content,
        metadata={
            'ticket_id': ticket['ticket_id'],
            'category': ticket['category'],
            'priority': ticket['priority'],
            'title': ticket['title']
        }
    )
    documents.append(doc)

print(f"✓ Loaded {len(documents)} LlamaIndex documents (support tickets)")

# Test query - we'll use this across all index types
query = "How do I fix authentication issues after password reset?"
print(f"\nTest Query: '{query}'")
# ============================================================================
# PART 1: Vector Index (Flat Index)
# ============================================================================
print("\n" + "="*80)
print("PART 1: Vector Index (Flat Index)")
print("="*80)

vector_index = VectorStoreIndex.from_documents(
    documents,
    service_context=service_context
)

vector_query_engine = vector_index.as_query_engine(similarity_top_k=3)

print(" Created vector index")
print(f"Query - {query} \n")
vector_response = vector_query_engine.query(query)

print(f"Vector Index Response: {vector_response}\n\n")

for i, node in enumerate(vector_response.source_nodes, 1):
    print(f"Source Node {i}:")
    print(f"  Ticket ID: {node.node.metadata['ticket_id']}")
    print(f"  Category: {node.node.metadata['category']}")
    print(f"  Priority: {node.node.metadata['priority']}")
    print(f"  Title: {node.node.metadata['title']}")
    print(f"  Content: {node.node.get_content()}\n")
