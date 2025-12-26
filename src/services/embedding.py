from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import List, Sequence, Optional

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from sklearn.feature_extraction.text import TfidfVectorizer

from src.core.config import settings
from src.core.logger import logger


@dataclass
class ChunkedDocument:
    document: Document
    keywords: List[str]
    chunk_id: str = ""
    parent_id: str = ""  # For hierarchical chunks


class TextChunker:
    """Advanced text chunker with parent-child relationships."""
    
    def __init__(
        self,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
        min_chunk_size: int | None = None,
    ) -> None:
        self.chunk_size = chunk_size or settings.chunk_size
        self.chunk_overlap = chunk_overlap or settings.chunk_overlap
        self.min_chunk_size = min_chunk_size or settings.min_chunk_size
        
        # Primary splitter for main chunks
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", "। ", ". ", "? ", "! ", "; ", ", ", " "],
            length_function=len,
        )
        
        # Large chunk splitter for parent context
        self.parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size * 3,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n"],
        )

    def split(self, text: str, metadata: dict | None = None) -> List[Document]:
        """Split text into chunks with IDs for linkage."""
        if not text or len(text.strip()) < self.min_chunk_size:
            return []
        
        base_metadata = metadata or {}
        docs = self.splitter.create_documents([text], metadatas=[base_metadata])
        
        # Generate unique IDs for each chunk
        source = base_metadata.get("source", "unknown")
        page = base_metadata.get("page", 0)
        
        for idx, doc in enumerate(docs):
            chunk_id = self._generate_chunk_id(source, page, idx, doc.page_content)
            doc.metadata["chunk_id"] = chunk_id
            doc.metadata["chunk_index"] = idx
            doc.metadata["total_chunks"] = len(docs)
        
        return docs
    
    def split_with_context(
        self, 
        text: str, 
        metadata: dict | None = None,
        preceding_context: str = "",
        following_context: str = "",
    ) -> List[Document]:
        """Split text with surrounding context for better retrieval."""
        docs = self.split(text, metadata)
        
        for doc in docs:
            # Add context hints to metadata for retrieval boosting
            if preceding_context:
                doc.metadata["preceding_summary"] = preceding_context[:200]
            if following_context:
                doc.metadata["following_summary"] = following_context[:200]
        
        return docs

    def _generate_chunk_id(self, source: str, page: int, idx: int, content: str) -> str:
        """Generate a unique chunk ID."""
        content_hash = hashlib.md5(content[:100].encode()).hexdigest()[:8]
        return f"{source}:p{page}:c{idx}:{content_hash}"


class KeywordExtractor:
    """Extract keywords using TF-IDF with multilingual support."""
    
    def __init__(self, top_k: int | None = None) -> None:
        self.top_k = top_k or settings.keyword_top_k

    def extract(self, texts: Sequence[str]) -> List[List[str]]:
        if not texts:
            return [[]]
        
        # Handle multilingual - don't use stopwords for Devanagari
        try:
            vectorizer = TfidfVectorizer(
                stop_words="english",
                max_features=self.top_k * 2,
                ngram_range=(1, 2),  # Include bigrams
                min_df=1,
                max_df=0.95,
            )
            tfidf_matrix = vectorizer.fit_transform(texts)
            feature_names = vectorizer.get_feature_names_out()
            
            keywords_per_doc: List[List[str]] = []
            for doc_index in range(tfidf_matrix.shape[0]):
                row = tfidf_matrix.getrow(doc_index)
                sorted_indices = row.toarray().flatten().argsort()[::-1]
                keywords = [feature_names[idx] for idx in sorted_indices if row[0, idx] > 0]
                keywords_per_doc.append(keywords[: self.top_k])
            return keywords_per_doc
        except Exception as e:
            logger.warning(f"Keyword extraction failed: {e}")
            return [[] for _ in texts]


class EmbeddingService:
    """OpenAI-based embedding service."""
    
    def __init__(self) -> None:
        if not settings.openai_api_key:
            raise ValueError("OPENAI_API_KEY is required for embeddings")
        
        self.model = settings.openai_embedding_model
        logger.info(f"Initializing OpenAI Embeddings: {self.model}")
        
        self.embedder = OpenAIEmbeddings(
            model=self.model,
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url,
        )
        
        # Cache for frequently used embeddings
        self._cache: dict[str, List[float]] = {}

    def embed_documents(self, documents: Sequence[Document]) -> List[List[float]]:
        """Embed multiple documents with batching."""
        texts = [doc.page_content for doc in documents]
        
        # Check cache
        uncached_texts = []
        uncached_indices = []
        results: List[Optional[List[float]]] = [None] * len(texts)
        
        for idx, text in enumerate(texts):
            cache_key = hashlib.md5(text[:500].encode()).hexdigest()
            if cache_key in self._cache:
                results[idx] = self._cache[cache_key]
            else:
                uncached_texts.append(text)
                uncached_indices.append(idx)
        
        # Embed uncached texts
        if uncached_texts:
            embeddings = self.embedder.embed_documents(uncached_texts)
            for idx, emb in zip(uncached_indices, embeddings):
                results[idx] = emb
                cache_key = hashlib.md5(texts[idx][:500].encode()).hexdigest()
                self._cache[cache_key] = emb
        
        return results  # type: ignore

    def embed_query(self, query: str) -> List[float]:
        """Embed a single query."""
        cache_key = hashlib.md5(query.encode()).hexdigest()
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        embedding = self.embedder.embed_query(query)
        self._cache[cache_key] = embedding
        return embedding


__all__ = ["TextChunker", "KeywordExtractor", "EmbeddingService", "ChunkedDocument"]

