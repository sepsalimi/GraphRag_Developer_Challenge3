import os, sys, importlib, time
from dotenv import load_dotenv

# Resolve directories relative to this file (robust to any working directory)
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..'))
env_path = os.path.join(project_root, '.env')

# Load .env file
load_dotenv(env_path, override=True)

# Get timezone from .env with default fallback
timezone = os.getenv('TZ', 'America/New_York')
os.environ["TZ"] = timezone
if hasattr(time, "tzset"):
    time.tzset()

# Get additional directories
rag_dir = os.path.join(project_root, 'RAG')
kg_dir = os.path.join(project_root, 'KnowledgeGraph')
print(f"Script directory: {script_dir}")
print(f"Project root: {project_root}")
print(f"Project root exists: {os.path.exists(project_root)}")
print(f"RAG dir exists: {os.path.exists(rag_dir)}")
print(f"KnowledgeGraph exists: {os.path.exists(kg_dir)}")

# Add paths to sys.path BEFORE importing modules
if project_root not in sys.path:
    sys.path.insert(0, project_root)
if rag_dir not in sys.path:
    sys.path.insert(0, rag_dir)
    
# Import modules after paths are set up
from KnowledgeGraph.config import load_neo4j_graph
import RAG.UseCatalog as UseCatalog
import RAG.VectorRAG as VectorRAG
import RAG.GraphRAG as GraphRAG
import RAG.batch_query as batch_query

# make sure AB Testing is on sys.path
test_dir = os.path.join(project_root, "AB Testing")
if test_dir not in sys.path:
    sys.path.insert(0, test_dir)

# Reload other modules to pick up changes
importlib.reload(VectorRAG)
importlib.reload(UseCatalog)
importlib.reload(GraphRAG)
importlib.reload(batch_query)

# Not found; ANS = "No. Both notices state that alternative offers are not permitted. [1][2]\n\nReferences\n1. Kuwait Official Gazette 2025-04-27, Lines 1895–1895.\n2. Kuwait Official Gazette 2025-05-11, Lines 13261–13261.",

question_1 = "Across both the late-April Amiri Diwan tender and Practice A/M/804, are alternative offers allowed?"
# Uses catalog; ANS = 
# Closing date: 2025-05-04. 〔2025_04_06_en.md L5948〕
# Closing time: 1:00 PM (“one o’clock in the afternoon”). 〔L5966–L5969〕
# Bid validity: 90 days (initial bond must be valid for 90 days → bid validity period). 〔L5950–L5951〕
# Tender document fee: KD 3,500. 〔L5949〕
# Alternative offers allowed? No (“Alternate offers will not be accepted.”). 〔L5973〕
# Indivisible? Yes (“The tender shall not be divided.”). 〔L5972〕
# Source anchor (tender label in the notice): “The Tender(2021/2020/06)”. 〔L5946〕
question_2 = "For tender 2021/2020/06, what are the closing date, closing time, bid validity, tender document fee, whether alternative offers are allowed, and whether the tender is indivisible?"

choice = question_1

def main():
    # Initialize graph and clients
    graph, openAI_api, openAI_endpoint, openAI_model = load_neo4j_graph()
    # Ask and print answer
    answer = GraphRAG.query_graph_rag(choice)
    print(answer)

    # Debug: show retrieved chunks
    rag = GraphRAG._make_rag(neighbor_window=2)
    resp = rag.search(query_text=choice, retriever_config={"top_k": 30}, return_context=True)
    for i, item in enumerate(resp.retriever_result.items, 1):
        m = item.metadata or {}
        print(f"{i:02d} score={m.get('score')}")
        print((item.content or "")[:200], "\n" + "-"*80)

    # Optional: raw Neo4j records (shows document_key/chunk_index)
    raw = rag.retriever.get_search_results(query_text=choice, top_k=30)
    for i, rec in enumerate(raw.records, 1):
        node = rec["node"]
        doc = node.get("document_key") if hasattr(node, "get") else (node["document_key"] if "document_key" in node else None)
        idx = node.get("chunk_index") if hasattr(node, "get") else (node["chunk_index"] if "chunk_index" in node else None)
        print(f"{i:02d} score={rec['score']} doc={doc} idx={idx}")


if __name__ == "__main__":
    main()