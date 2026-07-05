import asyncio
import sys
import logging

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import os
import certifi
os.environ["GRPC_DEFAULT_SSL_ROOTS_FILE_PATH"] = certifi.where()

import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Monkeypatch urllib3 pool connections
from urllib3.connectionpool import HTTPSConnectionPool
original_https_init = HTTPSConnectionPool.__init__
def patched_https_init(self, *args, **kwargs):
    kwargs['cert_reqs'] = 'CERT_NONE'
    original_https_init(self, *args, **kwargs)
HTTPSConnectionPool.__init__ = patched_https_init

# Monkeypatch httpx
import httpx
original_httpx_init = httpx.AsyncClient.__init__
def patched_httpx_init(self, *args, **kwargs):
    kwargs['verify'] = False
    original_httpx_init(self, *args, **kwargs)
httpx.AsyncClient.__init__ = patched_httpx_init

original_httpx_client_init = httpx.Client.__init__
def patched_httpx_client_init(self, *args, **kwargs):
    kwargs['verify'] = False
    original_httpx_client_init(self, *args, **kwargs)
httpx.Client.__init__ = patched_httpx_client_init

# Monkeypatch requests
import requests
original_requests_request = requests.Session.request
def patched_requests_request(self, *args, **kwargs):
    kwargs['verify'] = False
    return original_requests_request(self, *args, **kwargs)
requests.Session.request = patched_requests_request

logging.basicConfig(level=logging.INFO)

sys.path.append("c:/Users/Account1/Documents/kaggleproject/research-copilot")

from app.graph.agent import run_agent

async def main():
    query = (
        "Review major literature and research from 2021–2026 regarding the cGAS–STING pathway. "
        "Summarize how these major discoveries have changed our understanding of "
        "(1) innate immune signaling, (2) cancer immunotherapy response, "
        "(3) cellular senescence and aging, and (4) autoimmune disease."
    )
    
    print("Running local agent query...")
    result = await run_agent(query)
    
    print("\n" + "="*50)
    print("LOCAL AGENT FINAL ANSWER:")
    print("="*50)
    print(result.get("final_answer"))
    print("="*50)
    
    citations = result.get("citations", [])
    print(f"\nCitations retrieved: {len(citations)}")
    for i, c in enumerate(citations, 1):
        print(f"[{i}] PMID: {c.get('pmid')} - {c.get('title')}")

if __name__ == "__main__":
    asyncio.run(main())
