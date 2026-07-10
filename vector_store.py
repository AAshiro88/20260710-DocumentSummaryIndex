import uuid
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct


class QdrantStore:

    def __init__(self, collection_name, vector_size=768, path=None, host=None, port=6333):
        if path:
            self.client = QdrantClient(path=path)
        else:
            self.client = QdrantClient(host=host, port=port)
        self.collection_name = collection_name
        self.vector_size = vector_size

    def _to_uuid(self, sid):
        return str(uuid.uuid5(uuid.NAMESPACE_DNS, sid))

    def create_collection(self, force=False):
        if force:
            self.client.recreate_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=self.vector_size, distance=Distance.COSINE),
            )
        else:
            try:
                self.client.get_collection(self.collection_name)
            except Exception:
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(size=self.vector_size, distance=Distance.COSINE),
                )

    def add(self, documents, embeddings, metadatas, ids):
        points = [
            PointStruct(id=self._to_uuid(id_), vector=emb, payload={**meta, "document": doc})
            for id_, emb, doc, meta in zip(ids, embeddings, documents, metadatas)
        ]
        self.client.upsert(collection_name=self.collection_name, points=points)

    def search(self, query_embedding, top_k):
        result = self.client.query_points(
            collection_name=self.collection_name,
            query=query_embedding,
            limit=top_k,
            with_payload=True,
        )
        return result.points

    def count(self):
        return self.client.count(collection_name=self.collection_name).count
