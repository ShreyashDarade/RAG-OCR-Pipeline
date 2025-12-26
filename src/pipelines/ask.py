from __future__ import annotations

from dataclasses import dataclass
from typing import List

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from src.core.config import settings
from src.core.logger import logger
from src.pipelines.retrieval import RetrievalPipeline, RetrievalResult


@dataclass
class AskResponse:
    query: str
    expanded_queries: List[str]
    answer: str
    context: List[dict]


ASK_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a helpful assistant for enterprise knowledge retrieval. 
            
Instructions:
- Use ONLY the provided context to answer questions
- Cite sources using [Source: filename, Page: X] format
- If the context contains tables, reference the table data specifically
- If information comes from an image/OCR, mention that
- Support Hindi (हिंदी) and Marathi (मराठी) - respond in the same language as the question
- If you cannot find the answer in the context, say so clearly
- Be concise but thorough""",
        ),
        (
            "user",
            """Question: {query}

Related queries considered: {expanded}

Retrieved Context:
{context}

Please provide a comprehensive answer based on the context above.""",
        ),
    ]
)


def get_openai_llm() -> ChatOpenAI:
    """Initialize OpenAI LLM."""
    if not settings.openai_api_key:
        raise ValueError("OPENAI_API_KEY is required. Set it in your .env file.")
    
    logger.info(f"Using OpenAI LLM: {settings.openai_model}")
    
    return ChatOpenAI(
        model=settings.openai_model,
        api_key=settings.openai_api_key,
        base_url=settings.openai_base_url,
        temperature=0.1,  # Low temperature for factual responses
        max_tokens=2048,
    )


class AskPipeline:
    """Q&A pipeline using OpenAI for generation."""
    
    def __init__(self) -> None:
        self.retrieval = RetrievalPipeline()
        self.chat = get_openai_llm()

    def ask(self, query: str) -> AskResponse:
        """Process a question and generate an answer."""
        
        # Retrieve relevant documents
        retrieval: RetrievalResult = self.retrieval.retrieve(query)
        documents = retrieval.documents[: settings.rerank_top_k]
        
        # Build context with rich metadata
        context_blocks: List[str] = []
        structured: List[dict] = []
        
        for idx, hit in enumerate(documents, start=1):
            meta = hit.document.metadata
            snippet = hit.document.page_content.strip()
            
            # Build informative context block
            content_type = meta.get("content_type", meta.get("type", "text"))
            source_info = f"Source: {meta.get('source', 'Unknown')}"
            page_info = f"Page: {meta.get('page', 'N/A')}"
            type_info = f"Type: {content_type}"
            
            # Add table metadata if available
            extra_info = ""
            if meta.get("table_summary"):
                extra_info = f"\nTable Info: {meta.get('table_summary')}"
            if meta.get("ocr_confidence"):
                extra_info += f"\nOCR Confidence: {meta.get('ocr_confidence'):.2f}"
            
            block = f"[{idx}] {source_info} | {page_info} | {type_info}{extra_info}\n{snippet}"
            context_blocks.append(block)
            
            # Structured output for API response
            structured.append({
                "rank": idx,
                "score": hit.rerank_score or hit.score,
                "source": meta.get("source"),
                "page": meta.get("page"),
                "keywords": meta.get("keywords", []),
                "type": content_type,
                "content": hit.document.page_content,
                "document_id": meta.get("document_id"),
                "has_cross_references": bool(meta.get("sibling_chunk_ids")),
            })
        
        # Generate answer using LLM
        messages = ASK_PROMPT.format_messages(
            query=query,
            expanded=", ".join(retrieval.expanded_queries) if retrieval.expanded_queries else "None",
            context="\n\n".join(context_blocks) if context_blocks else "No relevant context found.",
        )
        
        try:
            result = self.chat.invoke(messages)
            answer = result.content
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            answer = f"Error generating answer: {str(e)}"
        
        return AskResponse(
            query=query,
            expanded_queries=retrieval.expanded_queries,
            answer=answer,
            context=structured,
        )


__all__ = ["AskPipeline", "AskResponse"]
