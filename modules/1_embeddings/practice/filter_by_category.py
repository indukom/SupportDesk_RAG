import json
import os
import numpy as np
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
model = "text-embedding-3-small"

with open("../../data/synthetic_tickets.json", "r") as f:
    tickets = json.load(f)

texts = [f"{t['title']}. {t['description']}" for t in tickets]
response = client.embeddings.create(input=texts, model=model)
embeddings = np.array(
    [data.embedding for data in response.data]
)

# search by category
def search_with_category(query, category_filter=None, top_k=5):
    """
    Search for tickets similar to the query, optionally filtering by category.
    """
    # Get query embeddings
    response = client.embeddings.create(
        input=query,
        model=model
    )
    query_emb = np.array(
        [response.data[0].embedding]
    )

    similarities = cosine_similarity(query_emb, embeddings)[0]

    print(similarities)
    # Filter by category if provided
    results = []
    for idx in np.argsort(similarities)[::-1]:
        ticket = tickets[idx]

        if category_filter and ticket['category'] != category_filter:
            continue

        results.append((ticket, similarities[idx]))
        if len(results) >= top_k:
            break
    return results


print("\nOnly 'Authentication' category:")
results = search_with_category("Users can't login after changing password", category_filter="Authentication", top_k=5)
for result in results:
    print(f"Ticket ID: {result[0]['ticket_id']}, Similarity: {result[1]:.4f}, Category: {result[0]['category']}, Title: {result[0]['title']}")
#print(results)

print("All categories:")
results = search_with_category("Users can't login after changing password", category_filter=None, top_k=5)
for result in results:
    print(f"Ticket ID: {result[0]['ticket_id']}, Similarity: {result[1]:.4f}, Category: {result[0]['category']}, Title: {result[0]['title']}")

