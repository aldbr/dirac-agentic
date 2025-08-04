from pymilvus import MilvusClient, FieldSchema, CollectionSchema, DataType
from pymilvus.milvus_client import IndexParams


def store_embeddings_in_milvus(
    embeddings: list[list[float]],
    texts: list[str],
    source: str = "generic",
    collection_name: str = "doc_embeddings",
    db_path: str = "./milvus_demo.db"
):
    """
    Store embeddings and texts into Milvus with indexing for retrieval.

    Parameters:
        embeddings: List[List[float]] - The embeddings to store.
        texts: List[str] - Corresponding texts.
        source: str - Source label ("paper", "doc", "issue", etc.) for tracking.
        collection_name: str - Milvus collection name.
        db_path: str - Local Milvus DB storage path.
    """

    # Initialize Milvus client
    client = MilvusClient(db_path)
    
    # 处理空列表情况
    if not embeddings or len(embeddings) == 0:
        return 0

    # Create collection schema if not exists
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=len(embeddings[0])),
        FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=2048),
        FieldSchema(name="source", dtype=DataType.VARCHAR, max_length=128),
    ]
    schema = CollectionSchema(fields, description="Document embeddings with raw text and source info")

    if not client.has_collection(collection_name):
        client.create_collection(collection_name=collection_name, schema=schema)

    # Prepare data for insertion
    insert_data = [
        {"embedding": emb, "text": txt, "source": source}
        for emb, txt in zip(embeddings, texts)
    ]

    # Insert and flush
    client.insert(collection_name=collection_name, data=insert_data)
    client.flush(collection_name=collection_name)

    # Create index if not already created
    index_params = IndexParams()
    index_params.add_index(
        field_name="embedding",
        index_type="IVF_FLAT",
        index_name="ivf_index",
        metric_type="COSINE",
        nlist=128
    )
    client.create_index(collection_name=collection_name, index_params=index_params)
    client.flush(collection_name=collection_name)
    return len(insert_data)