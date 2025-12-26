from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Set
from collections import defaultdict

from langchain_core.documents import Document
from elasticsearch import ApiError

from src.core.config import settings
from src.core.logger import logger
from src.services.elastic import ElasticClient
from src.services.embedding import EmbeddingService


@dataclass
class RetrievedDocument:
    document: Document
    score: float
    source_index: str
    rerank_score: float = 0.0  # Score after re-ranking


class HybridRetriever:
    """
    Advanced hybrid retriever with:
    - BM25 + Vector search fusion
    - Cross-reference expansion
    - Re-ranking based on relevance signals
    - Diversity-aware result selection
    """
    
    def __init__(self, alpha: float | None = None, top_k: int | None = None) -> None:
        self.alpha = alpha or settings.hybrid_alpha
        self.top_k = top_k or settings.retriever_top_k
        self.rerank_top_k = settings.rerank_top_k
        self.client = ElasticClient()
        self.embedding = EmbeddingService()
        self.indexes = [
            settings.es_index_text,
            settings.es_index_tables,
            settings.es_index_images,
        ]

    def retrieve(self, query: str) -> List[RetrievedDocument]:
        """Main retrieval method with all enhancements."""
        query_vector = self.embedding.embed_query(query)
        
        # Phase 1: Initial retrieval from all indexes
        combined: List[RetrievedDocument] = []
        for index in self.indexes:
            bm25_hits = self._bm25_search(index, query)
            knn_hits = self._knn_search(index, query_vector)
            combined.extend(self._fuse(index, bm25_hits, knn_hits))
        
        # Phase 2: Expand with cross-references
        if settings.enable_cross_references:
            combined = self._expand_with_cross_references(combined, query_vector)
        
        # Phase 3: Re-rank results
        if settings.rerank_enabled:
            combined = self._rerank(combined, query, query_vector)
        
        # Phase 4: Diversity-aware selection
        combined = self._diversify_results(combined)
        
        # Sort by final score and return top results
        combined.sort(key=lambda item: item.rerank_score or item.score, reverse=True)
        return combined[: self.rerank_top_k]

    def _bm25_search(self, index: str, query: str) -> List[Dict]:
        """BM25 text search with boosted fields."""
        body = {
            "size": self.top_k * 3,
            "query": {
                "bool": {
                    "should": [
                        {
                            "multi_match": {
                                "query": query,
                                "fields": [
                                    "content^3",
                                    "keywords^2",
                                    "metadata.table_summary",
                                    "metadata.source",
                                ],
                                "fuzziness": "AUTO",
                                "operator": "or",
                            }
                        },
                        {
                            "match_phrase": {
                                "content": {
                                    "query": query,
                                    "boost": 2.0,
                                    "slop": 2,
                                }
                            }
                        },
                    ],
                    "minimum_should_match": 1,
                }
            },
        }
        try:
            response = self.client.search(index, body)
            return response.get("hits", {}).get("hits", [])
        except ApiError as exc:
            logger.warning("BM25 search failed on index %s: %s", index, exc)
            return []

    def _knn_search(self, index: str, vector: List[float]) -> List[Dict]:
        """KNN vector search."""
        body = {
            "size": self.top_k * 3,
            "knn": {
                "field": "content_vector",
                "query_vector": vector,
                "k": self.top_k * 3,
                "num_candidates": self.top_k * 6,
            },
        }
        try:
            response = self.client.search(index, body)
            return response.get("hits", {}).get("hits", [])
        except ApiError as exc:
            logger.warning("kNN search failed on index %s: %s", index, exc)
            return []

    def _fuse(self, index: str, bm25_hits: List[Dict], knn_hits: List[Dict]) -> List[RetrievedDocument]:
        """Reciprocal Rank Fusion (RRF) for combining BM25 and KNN results."""
        k = 60  # RRF constant
        scores: Dict[str, Dict] = {}
        
        # RRF scoring for BM25
        for rank, hit in enumerate(bm25_hits, start=1):
            doc_id = hit.get("_id")
            rrf_score = 1.0 / (k + rank)
            if doc_id not in scores:
                scores[doc_id] = {"bm25_rrf": 0, "knn_rrf": 0, "hit": hit}
            scores[doc_id]["bm25_rrf"] = rrf_score
        
        # RRF scoring for KNN
        for rank, hit in enumerate(knn_hits, start=1):
            doc_id = hit.get("_id")
            rrf_score = 1.0 / (k + rank)
            if doc_id not in scores:
                scores[doc_id] = {"bm25_rrf": 0, "knn_rrf": 0, "hit": hit}
            scores[doc_id]["knn_rrf"] = rrf_score
            if not scores[doc_id].get("hit"):
                scores[doc_id]["hit"] = hit

        fused: List[RetrievedDocument] = []
        for doc_id, data in scores.items():
            # Combined RRF score
            combined_score = (
                self.alpha * data["bm25_rrf"] + 
                (1 - self.alpha) * data["knn_rrf"]
            )
            
            hit = data["hit"]
            source = hit.get("_source", {})
            
            document = Document(
                page_content=source.get("content", ""),
                metadata=source.get("metadata", {}),
            )
            
            # Enrich metadata
            document.metadata.update({
                "language": source.get("language"),
                "keywords": source.get("keywords", []),
                "source": source.get("source"),
                "page": source.get("page"),
                "chunk_id": source.get("chunk_id"),
                "document_id": source.get("document_id"),
                "sibling_chunk_ids": source.get("sibling_chunk_ids", []),
                "adjacent_chunk_ids": source.get("adjacent_chunk_ids", []),
                "content_type": source.get("content_type"),
                "has_table_on_page": source.get("has_table_on_page", False),
                "has_image_on_page": source.get("has_image_on_page", False),
                "score_breakdown": {
                    "bm25_rrf": data["bm25_rrf"],
                    "knn_rrf": data["knn_rrf"],
                },
            })
            
            fused.append(RetrievedDocument(
                document=document,
                score=combined_score,
                source_index=index,
            ))
        
        return fused

    def _expand_with_cross_references(
        self, 
        results: List[RetrievedDocument],
        query_vector: List[float],
    ) -> List[RetrievedDocument]:
        """Expand results by fetching related chunks from cross-references."""
        if not results:
            return results
        
        # Collect all cross-reference IDs
        sibling_ids: Set[str] = set()
        adjacent_ids: Set[str] = set()
        existing_ids: Set[str] = set()
        
        for doc in results:
            chunk_id = doc.document.metadata.get("chunk_id")
            if chunk_id:
                existing_ids.add(chunk_id)
            sibling_ids.update(doc.document.metadata.get("sibling_chunk_ids", []))
            adjacent_ids.update(doc.document.metadata.get("adjacent_chunk_ids", []))
        
        # Remove already retrieved chunks
        sibling_ids -= existing_ids
        adjacent_ids -= existing_ids
        
        # Fetch related chunks (limit to avoid too many)
        related_ids = list(sibling_ids)[:5] + list(adjacent_ids)[:3]
        
        if related_ids:
            related_docs = self._fetch_by_chunk_ids(related_ids)
            # Add with lower score (as they're context, not direct matches)
            for rdoc in related_docs:
                rdoc.score *= 0.7  # Discount related docs
            results.extend(related_docs)
        
        return results

    def _fetch_by_chunk_ids(self, chunk_ids: List[str]) -> List[RetrievedDocument]:
        """Fetch documents by their chunk IDs."""
        if not chunk_ids:
            return []
        
        results: List[RetrievedDocument] = []
        body = {
            "size": len(chunk_ids),
            "query": {
                "terms": {
                    "chunk_id": chunk_ids
                }
            }
        }
        
        for index in self.indexes:
            try:
                response = self.client.search(index, body)
                hits = response.get("hits", {}).get("hits", [])
                for hit in hits:
                    source = hit.get("_source", {})
                    document = Document(
                        page_content=source.get("content", ""),
                        metadata=source.get("metadata", {}),
                    )
                    document.metadata.update({
                        "source": source.get("source"),
                        "page": source.get("page"),
                        "chunk_id": source.get("chunk_id"),
                        "content_type": source.get("content_type"),
                        "is_cross_reference": True,
                    })
                    results.append(RetrievedDocument(
                        document=document,
                        score=0.5,  # Base score for cross-references
                        source_index=index,
                    ))
            except ApiError:
                continue
        
        return results

    def _rerank(
        self, 
        results: List[RetrievedDocument], 
        query: str,
        query_vector: List[float],
    ) -> List[RetrievedDocument]:
        """Re-rank results using multiple relevance signals."""
        if not results:
            return results
        
        query_terms = set(query.lower().split())
        
        for doc in results:
            base_score = doc.score
            boost = 1.0
            
            content = doc.document.page_content.lower()
            keywords = doc.document.metadata.get("keywords", [])
            
            # Boost 1: Keyword overlap
            keyword_overlap = len(query_terms & set(kw.lower() for kw in keywords))
            if keyword_overlap > 0:
                boost += 0.1 * keyword_overlap
            
            # Boost 2: Content type preference (text > tables > images for most queries)
            content_type = doc.document.metadata.get("content_type", "")
            if content_type == "pdf_text":
                boost += 0.05
            elif content_type == "pdf_table":
                # Boost tables if query contains table-related terms
                if any(term in query.lower() for term in ["table", "data", "numbers", "statistics"]):
                    boost += 0.15
            
            # Boost 3: Cross-reference context
            if doc.document.metadata.get("has_table_on_page"):
                boost += 0.03
            if doc.document.metadata.get("has_image_on_page"):
                boost += 0.02
            
            # Boost 4: Exact phrase match
            if query.lower() in content:
                boost += 0.2
            
            # Boost 5: Position in document (earlier pages often more relevant)
            page = doc.document.metadata.get("page", 1)
            if page and page <= 5:
                boost += 0.05 * (1 - page / 10)
            
            doc.rerank_score = base_score * boost
        
        return results

    def _diversify_results(self, results: List[RetrievedDocument]) -> List[RetrievedDocument]:
        """Ensure diversity in results (different pages, sources, content types)."""
        if len(results) <= self.rerank_top_k:
            return results
        
        # Group by source and page
        seen_pages: Dict[str, Set[int]] = defaultdict(set)
        diversified: List[RetrievedDocument] = []
        remaining: List[RetrievedDocument] = []
        
        # Sort by score first
        results.sort(key=lambda x: x.rerank_score or x.score, reverse=True)
        
        for doc in results:
            source = doc.document.metadata.get("source", "")
            page = doc.document.metadata.get("page", 0)
            
            # Allow max 2 chunks per page
            if page not in seen_pages[source] or len([p for p in seen_pages[source] if p == page]) < 2:
                diversified.append(doc)
                seen_pages[source].add(page)
            else:
                remaining.append(doc)
            
            if len(diversified) >= self.rerank_top_k:
                break
        
        # Fill remaining slots if needed
        while len(diversified) < self.rerank_top_k and remaining:
            diversified.append(remaining.pop(0))
        
        return diversified


__all__ = ["HybridRetriever", "RetrievedDocument"]
