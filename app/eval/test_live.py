import httpx
import json

url = "https://research-copilot-1044444025734.us-central1.run.app/chat"
query = (
    "Review major literature and research from 2021–2026 regarding the cGAS–STING pathway. "
    "Summarize how these major discoveries have changed our understanding of "
    "(1) innate immune signaling, (2) cancer immunotherapy response, "
    "(3) cellular senescence and aging, and (4) autoimmune disease."
)

payload = {
    "messages": [
        {"role": "user", "content": query}
    ],
    "top_k": 5
}

print(f"Sending query to live Cloud Run agent: {url}")
try:
    # Bypass verification for local execution environment test
    response = httpx.post(url, json=payload, timeout=45.0, verify=False)
    response.raise_for_status()
    data = response.json()
    
    print("\n" + "="*50)
    print("LIVE AGENT RESPONSE:")
    print("="*50)
    print(data.get("response"))
    print("="*50)
    
    citations = data.get("citations", [])
    if citations:
        print(f"\nRetrieved Citations ({len(citations)}):")
        for i, cit in enumerate(citations, 1):
            print(f"[{i}] Chunk ID: {cit.get('chunk_id')}")
            print(f"    Text snippet: {cit.get('text')[:120]}...")
    else:
        print("\nNo citations were used/retrieved.")

except Exception as e:
    print(f"Error testing live agent: {e}")
