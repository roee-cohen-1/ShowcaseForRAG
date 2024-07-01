from pinecone import Pinecone, ServerlessSpec, Vector
from tqdm import tqdm


def _create_index(pc, index_name, dimension, metric):
    index_exists = any(index['name'] == index_name for index in pc.list_indexes())
    if not index_exists:
        pc.create_index(
            name=index_name,
            dimension=dimension,
            metric=metric,
            spec=ServerlessSpec(cloud='aws', region='us-east-1')
        )


def index_vectors(vectors, metadata, api_key, index_name='stock-news', metric='cosine', batch_size=128):
    pc = Pinecone(api_key=api_key)
    num_vectors, dimension = vectors.shape
    _create_index(pc, index_name, dimension, metric)

    index = pc.Index(index_name)
    stats = index.describe_index_stats()
    if stats['total_vector_count'] == num_vectors:
        return index

    vectors = [
        Vector(id=str(i), values=vector.tolist(), metadata={'text': text})
        for i, (vector, text) in enumerate(zip(vectors, metadata))
    ]

    for i in tqdm(range(0, num_vectors, batch_size), desc='Upserting'):
        index.upsert(vectors=vectors[i:i+batch_size])
    return index
