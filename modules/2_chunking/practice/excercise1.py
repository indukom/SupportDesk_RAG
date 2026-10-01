import json
import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    CharacterTextSplitter,
    MarkdownHeaderTextSplitter,
    HTMLHeaderTextSplitter
)

from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document

print("="*80)
print("Chunking and vector stores")
print("="*80)

# Load the documents
with open("../../../data/synthetic_tickets.json", "r") as f:
    tickets = json.load(f)
print(f"Loaded {len(tickets)} tickets")

#Chunking statergies
print("\n"+ "="*80)
print("Chunking Statergies")
print("="*80)

# -----------------------------------------------------------------------------
# First, convert our ticket data into LangChain Document objects
# Document = page_content (text) + metadata (structured info for filtering)
# -----------------------------------------------------------------------------
documents = []
for ticket in tickets:
    # print(f"Ticket: {ticket}.\n")
    full_text = f"""
Ticket ID: {ticket['ticket_id']}
Title: {ticket['title']}
Category: {ticket['category']}
Priority: {ticket['priority']}
Description: {ticket['description']}
""".strip()
    
    # print(f"Full text: {full_text}\n")

    doc = Document(
        page_content=full_text,
        metadata={
            'ticket_id': ticket['ticket_id'],
            'category': ticket['category'],
            'priority': ticket['priority']
        }
    )

    documents.append(doc)

print(f"Created {len(documents)} documents")
print(f"\nSample document length: {len(documents[0].page_content)} characters")

# =============================================================================
# STRATEGY 1: Fixed-Size Chunking
# =============================================================================
print("\n --- Stegry 1: Fixed-Size Chunking ")

fixed_splitter = CharacterTextSplitter(
    chunk_size = 500,
    chunk_overlap=50,
    separator="\n"
)
fixed_chunks = fixed_splitter.split_documents(documents)

print(f" Created {len(fixed_chunks)} chunks")
print(f" Sample chunk: {fixed_chunks[0].page_content[:100]}....")

# for f in fixed_chunks:
#     print(f"Chunk: {f} \n\n")


# =============================================================================
# STRATEGY 2: Recursive Character Splitting (RECOMMENDED DEFAULT)
# =============================================================================
print("\n--- Strategy 2: Recursive Character Splitting ---")

recursive_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=50,
    separators=[
        "\n\n",
        "\n",
        ".",
        " "
    ]
)

recursive_chunks = recursive_splitter.split_documents(documents)

print(f" Created {len(recursive_chunks)} chunks")
print(f" Sample chunk: {recursive_chunks[0].page_content[:100]}....")

# =============================================================================
# STRATEGY 3: Semantic Chunking (Embedding-Based)
# =============================================================================
#
print("\n -- Stategy 3: Semantic Chunking ---")

# embeddings_model = OpenAIEmbeddings(
#     model=os.getenv('OPENAI_EMBEDDING_MODEL', 'text-embedding-3-small')
# )

# # Demo with a paragraph that has CLEAR topic shifts
# # This makes it obvious where semantic chunking will split
# demo_paragraph = """
# Database performance is critical for application speed. Slow queries can cause timeouts and frustrated users. Adding proper indexes to frequently queried columns dramatically improves response times. Query optimization should be a top priority for any development team.

# The weather forecast shows rain expected throughout the weekend. Temperatures will drop to the mid-40s by Sunday evening. Residents should prepare for possible flooding in low-lying areas. Don't forget to bring an umbrella if you're heading out.

# Authentication security requires multiple layers of protection. Passwords should be hashed using bcrypt or Argon2. Two-factor authentication adds an essential second layer of defense. Session tokens must be rotated regularly to prevent hijacking. API keys should never be exposed in client-side code.
# """

# semantic_splitter = SemanticChunker(
#     embeddings=embeddings_model,
#     breakpoint_threshold_type="percentile"
# )

# demo_doc = Document(page_content=demo_paragraph.strip())
# semantic_chunks = semantic_splitter.split_documents([demo_doc])
# print(f"\n Created {len(semantic_chunks)} chunks")

# print("\n. Resulting Semantic Chunks:")
# for i, chunk in enumerate(semantic_chunks):
#     for line in chunk.page_content.strip().split('\n'):
#         print(f"  {line.strip()}")
#     print(" "+ "~"*60)


# =============================================================================
# STRATEGY 4: Markdown Structure-Aware Splitting
# =============================================================================
#
print("\n --- Strategy 4: Markdown Structure-Aware Splitting ---")

markdown_doc = """
# Database Troubleshooting Guide

## Connection Issues

### Timeout Errors
If you encounter timeout errors, check the connection string and ensure the database server is reachable.
Increase the connection timeout value in your configuration.

### Authentication Failures
Verify your credentials are correct. Check for expired passwords or locked accounts.
Ensure the user has proper permissions on the database.

## Performance Problems

### Slow Queries
Analyze query execution plans using EXPLAIN.
Consider adding indexes on frequently queried columns.
Review and optimize JOIN operations.

### High CPU Usage
Monitor long-running queries.
Check for missing indexes causing table scans.
"""

# Headers to split on
headers_to_split_on = [
    ("#", "Header 1"),
    ("##", "Header 2"),
    ("###", "Header 3")
]

markdown_splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=headers_to_split_on,
    strip_headers=False
)

md_chunks = markdown_splitter.split_text(markdown_doc)

print(f" Created {len(md_chunks)} chunks from markdown")

if md_chunks:
    print(f". Sample chunk with metadata:")
    print("--"*60)
    print(f" Content: {md_chunks[0].page_content}...")
    print("--"*60)
    print(f" Metadata: {md_chunks[0].metadata}")

# =============================================================================
# STRATEGY 5: HTML Structure-Aware Splitting
# =============================================================================
#
print("\n --- Strategy 5: HTML Structure-Aware Splitting ---")

# Sample HTML documentation (simulating a scraped help page)
html_doc = """
<!DOCTYPE html>
<html>
<body>
    <h1>Email Configuration Guide</h1>
    
    <h2>SMTP Settings</h2>
    <p>Configure your SMTP server settings in the admin panel. Use port 587 for TLS or port 465 for SSL.</p>
    
    <h3>Common SMTP Servers</h3>
    <p>Gmail: smtp.gmail.com, Outlook: smtp.office365.com, Yahoo: smtp.mail.yahoo.com</p>
    
    <h2>IMAP Configuration</h2>
    <p>Set up IMAP to sync your emails across devices. Use port 993 for secure connections.</p>
    
    <h3>Folder Mapping</h3>
    <p>Map your email folders to the appropriate IMAP folders for proper synchronization.</p>
</body>
</html>
"""

headers_to_split_on_html = [
    ("h1", "Header 1"),
    ("h2", "Header 2"),
    ("h3", "Header 3")
]

html_splitter = HTMLHeaderTextSplitter(
    headers_to_split_on=headers_to_split_on_html,
)
html_chunks = html_splitter.split_text(html_doc)

print(f". Created {len(html_chunks)} chunks from HTML")
for html_chunk in html_chunks:
    print(f"  Chunk: {html_chunk.page_content}...")
# if html_chunks:
    # print(f". Sample chunk with metadata:")
    # print(f" Content: {html_chunks[0].page_content[:80]}...")
    # print(f" Metadata: {html_chunks[0].metadata}")


# ============================================================================
# PART 2: Chroma Vector Store
# ============================================================================
#

print("=="*80)
print("Chroma Vector Store")
print("=="*80)

embeddings_model = OpenAIEmbeddings(
    model=os.getenv('OPENAI_EMBEDDING_MODEL', 'text-embedding-3-small')
)

query = "Authentication problems after password reset"

print("\n Building Chrmoa vector store...")

# Check if Chroma store already exists
chroma_db_path = "./chroma_db"
if os.path.exists(chroma_db_path):
    print(f"Loading existing Chroma store from '{chroma_db_path}'...")
    chroma_store = Chroma(
        collection_name="support_tickets",
        embedding_function=embeddings_model,
        persist_directory=chroma_db_path
    )
    print("✓ Chroma store loaded from disk")
else:
    print(f"Creating new Chroma store...")
    chroma_store = Chroma.from_documents(
        documents=documents,
        embedding=embeddings_model,
        collection_name="support_tickets",
        persist_directory=chroma_db_path
    )
    print("✓ Chroma store created and persisted")

# Basic similarity Search
print(f"\nSearching in Chroma: '{query}'")
chroma_results = chroma_store.similarity_search(query, k=3)

print("\nTop {len(chroma_results)} results:")

for i, doc in enumerate(chroma_results, 1):
    print(f"\n#{i}")
    print(f"{doc}\n")
    # print(f"Ticket: {doc.metadata['ticket_id']}")
    # print(f"Category: {doc.metadata['category']}")

# -----------------------------------------------------------------------------
# MMR Search (Maximal Marginal Relevance)
# -----------------------------------------------------------------------------
# PROBLEM: Similarity search can return very similar documents
# SOLUTION: MMR balances relevance AND diversity

print("\n -- Using MMR for Diverse Results --")
mmr_results = chroma_store.max_marginal_relevance_search(query, k=3)

print(f"\n MMR Results (more diverse):")
for i, doc in enumerate(mmr_results, 1):
    print(f"\n#{i}")
    print(f"{doc}\n")
    # print(f"Ticket: {doc.metadata['ticket_id']}")
    # print(f"Category: {doc.metadata['category']}")


# ============================================================================
# PART 3: Metadata Filtering
# ============================================================================
#
print("\n" + "="*80)
print("PART 3: Metadata Filtering")
print("="*80)

print("\nSearching only in 'Authentication' category:")

filtered_results = chroma_store.similarity_search(
    query,
    k=3,
    filter={"category": "Authentication"}
)

print(f"\nFiltered Results (category='Authentication'):")

for i, doc in enumerate(filtered_results, 1):
    print(f"\n#{i}")
    print(f"{doc}\n")
    # print(f"Ticket: {doc.metadata['ticket_id']}")
    # print(f"Category: {doc.metadata['category']}")

# -----------------------------------------------------------------------------
# Example 2: Filter by priority
# -----------------------------------------------------------------------------
# Scenario: Looking for high-priority issues to prioritize
# Combine semantic search with priority filter
# --------------------------------------------------------
print("\n\nSearching only 'High' priority tickets:")
high_priority_results = chroma_store.similarity_search(
    "Database performance issues",
    k=3,
    filter={"priority": "High"}
)

print(f"\nFiltered Results (priority='High'):")
for i, doc in enumerate(high_priority_results, 1):
    print(f"\n#{i}")
    print(f"{doc}\n")
    # print(f"Ticket: {doc.metadata['ticket_id']}")
    # print(f"Priority: {doc.metadata['priority']}")