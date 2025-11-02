from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor


# 1. Add main nodes without creating relationships
def create_nodes(graph, data: dict, node_label: str, node_name: str):
    """
    Creates a main node and section nodes without creating relationships.

    Args:
        graph: A knowledge graph client or connection object that has a `query` method.
        data: A dictionary containing section names and their corresponding content.
        node_label: Label for the main node (e.g., "Document").
        node_name: The document's file name (used as parent_name for its sections).
    """

    # Create the main node (Document)
    main_node_query = f"""
    MERGE (main:{node_label} {{name: $name}})
    ON CREATE SET main += $data
    ON MATCH SET main += $data
    """
    graph.query(main_node_query, params={"name": node_name, "data": data})

    # Create Section nodes (children of this document)
    for section, content in data.items():
        if isinstance(content, str) and section not in ['name', 'text'] and len(content) > 100:
            query = """
            MERGE (s:Section {type: $type, parent_name: $parent})
            ON CREATE SET s.text = $text
            ON MATCH SET s.text = $text
            """
            params = {
                "type": section,
                "parent": node_name,  # ✅ ensures section.parent_name matches document
                "text": content
            }
            graph.query(query, params=params)



# 2. Add Chunks
def ingest_Chunks(graph, chunks, node_name, node_label):
    """
    Ingests file chunk data into the knowledge graph by merging chunk nodes.

    Args:
        graph: A knowledge graph client or connection object that has a `query` method.
        chunks: A list of dictionaries, each representing a file chunk with keys:
                     'chunkId', 'text', 'source', 'formItem', and 'chunkSeqId'.
        node_name: A string used to tag the chunk nodes.
        node_label: The dynamic label for the chunk nodes.
    """
    
    merge_chunk_node_query = f"""
    MERGE (mergedChunk:{node_label} {{chunkId: $chunkParam.chunkId}})
        ON CREATE SET
            mergedChunk.text = $chunkParam.text, 
            mergedChunk.source = $chunkParam.source,
            mergedChunk.formItem = $chunkParam.formItem, 
            mergedChunk.chunkSeqId = $chunkParam.chunkSeqId,
            mergedChunk.node_name = $node_name
    RETURN mergedChunk
    """

    node_count = 0
    for chunk in chunks:
        graph.query(merge_chunk_node_query, params={'chunkParam': chunk, 'node_name': node_name})
        node_count += 1


# 3. Create Relationships

def create_relationship(graph, query: str):
    """
    Executes the provided Cypher query on the given graph.
    
    Parameters:
        graph: An instance of your Neo4j connection.
        query: A string containing a valid Cypher query.
    """
    graph.query(query)


def create_vector_index(graph, index_name, dims=None):
    """
    Create a Neo4j vector index for the given label/property.

    Args:
        graph: Neo4j connection
        index_name: Label to index (and index name)
        dims: Embedding dimensions. If None, uses OPENAI_EMBED_DIM env (default 1536).
    """
    import os
    if dims is None:
        try:
            dims = int(os.getenv("OPENAI_EMBED_DIM", "1536"))
        except ValueError:
            dims = 1536

    vector_index_query = f"""
    CREATE VECTOR INDEX `{index_name}` IF NOT EXISTS
    FOR (n:{index_name}) ON (n.textEmbeddingOpenAI)
    OPTIONS {{ indexConfig: {{
        `vector.dimensions`: {dims},
        `vector.similarity_function`: 'cosine'
    }}}}
    """
    graph.query(vector_index_query)



def embed_text(graph, OPENAI_API_KEY, OPENAI_ENDPOINT, node_name, model_name='text-embedding-3-small', max_workers=None):
    """
    Creates embeddings for nodes with a dynamic label using the OpenAI endpoint,
    and displays a single-line progress bar using tqdm.
    
    Args:
        graph: A knowledge graph client/connection object that has a `query` method.
        OPENAI_API_KEY: The API key for the OpenAI service.
        OPENAI_ENDPOINT: The OpenAI endpoint URL.
        node_name: The label of nodes to process.
        model_name: The OpenAI embedding model to use.
        max_workers: Optional level of parallelism for embedding updates. When >1,
            updates are executed concurrently.
    """
    print("Starting embedding update...")

    # Fetch nodes without embeddings using elementId to avoid deprecated id() warnings
    fetch_nodes_query = f"""
    MATCH (n:{node_name})
    WHERE n.textEmbeddingOpenAI IS NULL
    RETURN elementId(n) AS node_id, n.text AS text
    """
    nodes = list(graph.query(fetch_nodes_query))
    total_nodes = len(nodes)
    print(f"Found {total_nodes} nodes without embeddings.")

    # Prepare the update query once
    update_query = f"""
    MATCH (n:{node_name})
    WHERE elementId(n) = $node_id
    WITH n, genai.vector.encode(
      n.text, 
      "OpenAI", 
      {{
        token: $openAiApiKey, 
        endpoint: $openAiEndpoint,
        model: $modelName
      }}
    ) AS vector
    CALL db.create.setNodeVectorProperty(n, "textEmbeddingOpenAI", vector)
    """

    def _update_node(record):
        node_id = record["node_id"]
        graph.query(update_query, params={
            "node_id": node_id,
            "openAiApiKey": OPENAI_API_KEY,
            "openAiEndpoint": OPENAI_ENDPOINT,
            "modelName": model_name
        })

    # Use a single-line progress bar for node updates
    with tqdm(total=total_nodes, desc="Embedding nodes", ncols=100, leave=True) as pbar:
        if max_workers and int(max_workers) > 1:
            with ThreadPoolExecutor(max_workers=int(max_workers)) as executor:
                for _ in executor.map(_update_node, nodes):
                    pbar.update(1)
        else:
            for record in nodes:
                _update_node(record)
                pbar.update(1)

    print("Finished embedding update.")
