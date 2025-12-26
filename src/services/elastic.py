from __future__ import annotations

import time
from typing import Any, Dict, Iterable, List

from elasticsearch import ApiError, Elasticsearch

from src.core.config import settings
from src.core.logger import logger


class ElasticClient:
    """Elasticsearch Cloud client."""
    
    def __init__(self) -> None:
        # Validate required settings
        if not settings.es_cloud_id:
            raise ValueError("ES_CLOUD_ID is required. Get it from cloud.elastic.co")
        if not settings.es_api_key:
            raise ValueError("ES_API_KEY is required. Create one in Elasticsearch Cloud")
        
        logger.info("Connecting to Elasticsearch Cloud...")
        
        self.client = Elasticsearch(
            cloud_id=settings.es_cloud_id,
            api_key=settings.es_api_key,
            request_timeout=60,
            retry_on_timeout=True,
            max_retries=3,
        )
        
        # Verify connection
        try:
            info = self.client.info()
            logger.info(f"✅ Connected to Elasticsearch Cloud v{info['version']['number']}")
        except Exception as e:
            logger.error(f"❌ Failed to connect to Elasticsearch Cloud: {e}")
            raise

    def ensure_index(self, index_name: str, dims: int) -> None:
        if self.client.indices.exists(index=index_name):
            return
        logger.info("Creating index %s with dimension %s", index_name, dims)
        mappings = {
            "properties": {
                "content": {"type": "text", "analyzer": "standard"},
                "content_vector": {
                    "type": "dense_vector",
                    "dims": dims,
                    "index": True,
                    "similarity": "cosine",
                },
                "keywords": {"type": "keyword"},
                "metadata": {"type": "object", "enabled": True},
                "language": {"type": "keyword"},
                "source": {"type": "keyword"},
                "page": {"type": "integer"},
                "chunk_id": {"type": "keyword"},
                "document_id": {"type": "keyword"},
                "sibling_chunk_ids": {"type": "keyword"},
                "adjacent_chunk_ids": {"type": "keyword"},
                "content_type": {"type": "keyword"},
                "has_table_on_page": {"type": "boolean"},
                "has_image_on_page": {"type": "boolean"},
                "created_at": {"type": "date"},
            }
        }
        # Elasticsearch Cloud supports knn natively
        index_settings = {
            "number_of_shards": 1,
            "number_of_replicas": 1,
        }
        try:
            self.client.indices.create(
                index=index_name, 
                mappings=mappings, 
                settings=index_settings
            )
            logger.info(f"✅ Created index: {index_name}")
        except ApiError as exc:
            if "resource_already_exists_exception" in str(exc):
                logger.info(f"Index {index_name} already exists")
            else:
                raise

    def bulk_index(self, index_name: str, docs: Iterable[Dict[str, Any]]) -> None:
        operations: List[Dict[str, Any]] = []
        for doc in docs:
            operations.append({"index": {"_index": index_name}})
            operations.append(doc)
        if not operations:
            return
        response = self.client.bulk(operations=operations, refresh=True)
        if response.get("errors"):
            error_items = [item for item in response.get("items", []) if "error" in item.get("index", {})]
            logger.error("Bulk index encountered errors: %s", error_items[:3])
        else:
            logger.info("Indexed %s documents into %s", len(operations) // 2, index_name)

    def search(self, index_name: str, body: Dict[str, Any]) -> Dict[str, Any]:
        return self.client.search(index=index_name, body=body)

    def count(self, index_name: str, query: Dict[str, Any]) -> int:
        try:
            response = self.client.count(index=index_name, body={"query": query})
            return int(response.get("count", 0))
        except ApiError as exc:
            logger.warning("Failed to count documents in %s: %s", index_name, exc)
            return 0

    def delete_by_source(self, source: str) -> int:
        """Delete all documents from a source."""
        total_deleted = 0
        query = {"term": {"source": source}}
        for index_name in [settings.es_index_text, settings.es_index_tables, settings.es_index_images]:
            try:
                response = self.client.delete_by_query(
                    index=index_name,
                    body={"query": query},
                    refresh=True,
                )
                deleted = response.get("deleted", 0)
                total_deleted += deleted
                if deleted > 0:
                    logger.info(f"Deleted {deleted} documents from {index_name}")
            except ApiError:
                pass
        return total_deleted


class ElasticIndexer:
    def __init__(self, embedding_dim: int) -> None:
        self.elastic = ElasticClient()
        self.embedding_dim = embedding_dim
        self.elastic.ensure_index(settings.es_index_text, embedding_dim)
        self.elastic.ensure_index(settings.es_index_tables, embedding_dim)
        self.elastic.ensure_index(settings.es_index_images, embedding_dim)
        self.index_names = [
            settings.es_index_text,
            settings.es_index_tables,
            settings.es_index_images,
        ]

    def index_text_documents(self, docs: List[Dict[str, Any]]) -> None:
        self.elastic.bulk_index(settings.es_index_text, docs)

    def index_table_documents(self, docs: List[Dict[str, Any]]) -> None:
        self.elastic.bulk_index(settings.es_index_tables, docs)

    def index_image_documents(self, docs: List[Dict[str, Any]]) -> None:
        self.elastic.bulk_index(settings.es_index_images, docs)

    def source_document_count(self, source: str) -> int:
        total = 0
        query = {"term": {"source": source}}
        for index_name in self.index_names:
            total += self.elastic.count(index_name, query)
        return total

    def delete_source(self, source: str) -> int:
        return self.elastic.delete_by_source(source)


__all__ = ["ElasticIndexer", "ElasticClient"]
