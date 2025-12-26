from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Optional

from langchain_core.documents import Document

from src.core.config import settings
from src.core.logger import logger
from src.services.embedding import EmbeddingService, KeywordExtractor, TextChunker
from src.services.elastic import ElasticIndexer
from src.services.ocr import OCRService
from src.utils.file_utils import DataFileManager
from src.utils.language import SUPPORTED_LANGS, detect_language
from src.utils.pdf_parser import PdfExtractionResult, extract_from_pdf


@dataclass
class IngestionSummary:
    source: Path
    text_chunks: int
    table_chunks: int
    image_chunks: int
    document_id: str = ""  # Unique document identifier
    total_pages: int = 0
    cross_references: int = 0  # Number of cross-references created
    skipped_reason: str | None = None
    reindexed: bool = True


@dataclass
class PageContent:
    """Holds all content from a single page for cross-referencing."""
    page_number: int
    text_chunks: List[Document] = field(default_factory=list)
    table_chunks: List[Document] = field(default_factory=list)
    image_chunks: List[Document] = field(default_factory=list)
    
    @property
    def all_chunks(self) -> List[Document]:
        return self.text_chunks + self.table_chunks + self.image_chunks
    
    @property
    def has_content(self) -> bool:
        return bool(self.text_chunks or self.table_chunks or self.image_chunks)


class IngestionPipeline:
    """Advanced ingestion pipeline with document linkage and cross-references."""
    
    def __init__(self) -> None:
        self.data_manager = DataFileManager()
        self.chunker = TextChunker()
        self.keyword_extractor = KeywordExtractor()
        self.embedding = EmbeddingService()
        probe_vector = self.embedding.embed_query("dimension probe text")
        self.indexer = ElasticIndexer(len(probe_vector))
        self.ocr = OCRService()

    def _normalize_language_hint(self, language: str | None) -> str | None:
        if not language:
            return None
        normalized = language.strip().lower()
        if normalized in {"", "auto", "none"}:
            return None
        if normalized in SUPPORTED_LANGS:
            return normalized
        logger.warning("Unsupported image language hint '%s'; defaulting to auto-detect.", language)
        return None

    def _generate_document_id(self, path: Path) -> str:
        """Generate a unique document ID."""
        return f"doc_{path.stem}_{uuid.uuid4().hex[:8]}"

    def ingest_upload(self, upload_file, force: bool = False, image_language: str | None = None) -> IngestionSummary:
        stored_path = self.data_manager.store_upload(upload_file)
        language_hint = self._normalize_language_hint(image_language)
        return self.ingest_path(stored_path, force=force, image_language=language_hint)

    def ingest_path(self, path: Path, force: bool = False, image_language: str | None = None) -> IngestionSummary:
        language_hint = self._normalize_language_hint(image_language)
        existing_docs = self.indexer.source_document_count(str(path))
        needs_reindex = self.data_manager.file_needs_reindex(path)
        
        if not force and not needs_reindex and existing_docs > 0:
            logger.info("No changes detected for %s; skipping reindex.", path)
            return IngestionSummary(
                source=path,
                text_chunks=0,
                table_chunks=0,
                image_chunks=0,
                skipped_reason="no_changes_detected",
                reindexed=False,
            )

        suffix = path.suffix.lower()
        if suffix == ".pdf":
            summary = self._ingest_pdf(path, language_hint=language_hint)
        elif suffix in {".png", ".jpg", ".jpeg", ".tif", ".tiff"}:
            summary = self._ingest_image(path, language_hint=language_hint)
        else:
            logger.warning("Unsupported file extension for %s", path)
            summary = IngestionSummary(
                source=path,
                text_chunks=0,
                table_chunks=0,
                image_chunks=0,
                skipped_reason="unsupported_extension",
                reindexed=False,
            )
        
        if summary.reindexed:
            self.data_manager.mark_indexed(path)
        return summary

    def _create_cross_references(
        self, 
        page_contents: Dict[int, PageContent],
        document_id: str,
        source: str,
    ) -> int:
        """Create cross-references between chunks on the same and adjacent pages."""
        if not settings.enable_cross_references:
            return 0
        
        cross_ref_count = 0
        pages = sorted(page_contents.keys())
        
        for page_num in pages:
            page = page_contents[page_num]
            
            # Get adjacent pages for context
            adjacent_chunk_ids = []
            for offset in range(-settings.page_context_window, settings.page_context_window + 1):
                adj_page = page_num + offset
                if adj_page in page_contents and adj_page != page_num:
                    for chunk in page_contents[adj_page].all_chunks:
                        if "chunk_id" in chunk.metadata:
                            adjacent_chunk_ids.append(chunk.metadata["chunk_id"])
            
            # Link all chunks on this page together
            page_chunk_ids = [
                c.metadata.get("chunk_id") 
                for c in page.all_chunks 
                if "chunk_id" in c.metadata
            ]
            
            # Add cross-references to each chunk
            for chunk in page.all_chunks:
                # Same-page references (siblings)
                sibling_ids = [cid for cid in page_chunk_ids if cid != chunk.metadata.get("chunk_id")]
                chunk.metadata["sibling_chunk_ids"] = sibling_ids[:10]  # Limit
                
                # Adjacent page references
                chunk.metadata["adjacent_chunk_ids"] = adjacent_chunk_ids[:10]
                
                # Document-level metadata
                chunk.metadata["document_id"] = document_id
                
                # Content type linkage
                chunk.metadata["has_table_on_page"] = bool(page.table_chunks)
                chunk.metadata["has_image_on_page"] = bool(page.image_chunks)
                chunk.metadata["page_chunk_count"] = len(page.all_chunks)
                
                cross_ref_count += len(sibling_ids) + len(adjacent_chunk_ids)
        
        return cross_ref_count

    def _prepare_documents(
        self, 
        documents: List[Document], 
        metadata_overrides: dict | None = None
    ) -> List[Dict]:
        """Prepare documents for indexing with embeddings and keywords."""
        if not documents:
            return []
        
        texts = [doc.page_content for doc in documents]
        keywords_all = self.keyword_extractor.extract(texts)
        embeddings = self.embedding.embed_documents(documents)
        
        prepared = []
        for doc, keywords, vector in zip(documents, keywords_all, embeddings):
            metadata = {**doc.metadata}
            if metadata_overrides:
                metadata.update(metadata_overrides)
            
            prepared.append({
                "content": doc.page_content,
                "content_vector": vector,
                "keywords": keywords,
                "metadata": metadata,
                "language": metadata.get("language", detect_language(doc.page_content)),
                "source": metadata.get("source"),
                "page": metadata.get("page"),
                "chunk_id": metadata.get("chunk_id"),
                "document_id": metadata.get("document_id"),
                "sibling_chunk_ids": metadata.get("sibling_chunk_ids", []),
                "adjacent_chunk_ids": metadata.get("adjacent_chunk_ids", []),
                "content_type": metadata.get("type"),
                "has_table_on_page": metadata.get("has_table_on_page", False),
                "has_image_on_page": metadata.get("has_image_on_page", False),
                "created_at": int(time.time() * 1000),
            })
        return prepared

    def _ingest_pdf(self, path: Path, language_hint: str | None = None) -> IngestionSummary:
        logger.info("Parsing PDF %s", path)
        extraction: PdfExtractionResult = extract_from_pdf(path)
        
        document_id = self._generate_document_id(path)
        page_contents: Dict[int, PageContent] = {}
        
        # Phase 1: Extract and chunk all content by page
        logger.info("Phase 1: Extracting text blocks...")
        for block in extraction.text_blocks:
            if block.page not in page_contents:
                page_contents[block.page] = PageContent(page_number=block.page)
            
            chunk_docs = self.chunker.split(
                block.content,
                metadata={
                    "source": str(path),
                    "page": block.page,
                    "language": block.language,
                    "type": "pdf_text",
                },
            )
            page_contents[block.page].text_chunks.extend(chunk_docs)
        
        logger.info("Phase 1: Extracting tables...")
        for table in extraction.tables:
            if table.page not in page_contents:
                page_contents[table.page] = PageContent(page_number=table.page)
            
            table_markdown = table.dataframe.to_markdown(index=False)
            # For tables, add structured representation
            table_summary = f"Table with {len(table.dataframe)} rows and columns: {', '.join(table.dataframe.columns.tolist()[:5])}"
            
            chunk_docs = self.chunker.split(
                table_markdown,
                metadata={
                    "source": str(path),
                    "page": table.page,
                    "language": table.language,
                    "type": "pdf_table",
                    "table_summary": table_summary,
                    "table_columns": list(table.dataframe.columns)[:20],
                    "table_row_count": len(table.dataframe),
                },
            )
            page_contents[table.page].table_chunks.extend(chunk_docs)
        
        logger.info("Phase 1: Processing images with OCR...")
        for pdf_img in extraction.images:
            ocr_result = self.ocr.read(pdf_img.image, language_hint=language_hint)
            if not ocr_result.text.strip():
                continue
            
            if pdf_img.page not in page_contents:
                page_contents[pdf_img.page] = PageContent(page_number=pdf_img.page)
            
            chunk_docs = self.chunker.split(
                ocr_result.text,
                metadata={
                    "source": str(path),
                    "page": pdf_img.page,
                    "language": ocr_result.language,
                    "type": "pdf_image",
                    "image_label": pdf_img.label,
                    "ocr_confidence": ocr_result.confidence,
                },
            )
            page_contents[pdf_img.page].image_chunks.extend(chunk_docs)
        
        # Phase 2: Create cross-references
        logger.info("Phase 2: Creating cross-references...")
        cross_ref_count = self._create_cross_references(page_contents, document_id, str(path))
        
        # Phase 3: Prepare and index documents
        logger.info("Phase 3: Embedding and indexing...")
        text_chunks_total = 0
        table_chunks_total = 0
        image_chunks_total = 0
        
        for page_num, page in page_contents.items():
            if page.text_chunks:
                prepared = self._prepare_documents(page.text_chunks)
                self.indexer.index_text_documents(prepared)
                text_chunks_total += len(prepared)
            
            if page.table_chunks:
                prepared = self._prepare_documents(page.table_chunks)
                self.indexer.index_table_documents(prepared)
                table_chunks_total += len(prepared)
            
            if page.image_chunks:
                prepared = self._prepare_documents(page.image_chunks)
                self.indexer.index_image_documents(prepared)
                image_chunks_total += len(prepared)
        
        total_chunks = text_chunks_total + table_chunks_total + image_chunks_total
        logger.info(
            f"Ingested {total_chunks} chunks from {path}: "
            f"{text_chunks_total} text, {table_chunks_total} tables, {image_chunks_total} images, "
            f"{cross_ref_count} cross-references"
        )
        
        return IngestionSummary(
            source=path,
            text_chunks=text_chunks_total,
            table_chunks=table_chunks_total,
            image_chunks=image_chunks_total,
            document_id=document_id,
            total_pages=len(page_contents),
            cross_references=cross_ref_count,
        )

    def _ingest_image(self, path: Path, language_hint: str | None = None) -> IngestionSummary:
        logger.info("Processing image %s", path)
        import cv2

        image = cv2.imread(str(path))
        if image is None:
            logger.error("Failed to load image %s", path)
            return IngestionSummary(
                source=path,
                text_chunks=0,
                table_chunks=0,
                image_chunks=0,
                skipped_reason="image_load_error",
                reindexed=False,
            )
        
        document_id = self._generate_document_id(path)
        ocr_result = self.ocr.read(image, language_hint=language_hint)
        
        if not ocr_result.text.strip():
            logger.warning("No text detected in image %s", path)
            return IngestionSummary(
                source=path,
                text_chunks=0,
                table_chunks=0,
                image_chunks=0,
                skipped_reason="no_text_detected",
                reindexed=False,
            )
        
        docs = self.chunker.split(
            ocr_result.text,
            metadata={
                "source": str(path),
                "language": ocr_result.language,
                "type": "image",
                "document_id": document_id,
                "ocr_confidence": ocr_result.confidence,
            },
        )
        
        prepared = self._prepare_documents(docs)
        self.indexer.index_image_documents(prepared)
        
        return IngestionSummary(
            source=path,
            text_chunks=0,
            table_chunks=0,
            image_chunks=len(prepared),
            document_id=document_id,
        )


__all__ = ["IngestionPipeline", "IngestionSummary"]
