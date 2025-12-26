from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import easyocr
import numpy as np
from PIL import Image
import cv2
import torch
import re

from src.core.logger import logger
from src.utils.image_ops import preprocess_image
from src.utils.language import detect_language
from src.utils.nlp_processing import clean_ocr_text

_MODEL_STORAGE = Path(__file__).resolve().parents[2] / "models"
_MODEL_STORAGE.mkdir(parents=True, exist_ok=True)


@dataclass
class OcrResult:
    text: str
    language: str
    confidence: float


class BaseOCREngine:
    def read(self, image: Image.Image | np.ndarray) -> OcrResult:
        raise NotImplementedError


class EasyEnglishOCREngine(BaseOCREngine):
    def __init__(self, gpu: bool = False) -> None:
        self.reader = easyocr.Reader(
            ["en"],
            gpu=gpu,
            verbose=False,
            model_storage_directory=str(_MODEL_STORAGE),
        )

    def read(self, image: Image.Image | np.ndarray) -> OcrResult:
        result = self.reader.readtext(image, detail=1, paragraph=True)
        text = "\n".join(chunk[1] for chunk in result if len(chunk) >= 2)
        confidence = _compute_confidence(result)
        # Apply language-aware NLP cleaning
        cleaned_text = clean_ocr_text(text, language="en")
        return OcrResult(text=cleaned_text, language="en", confidence=confidence)


class DevanagariOCREngine(BaseOCREngine):
    def __init__(self, gpu: bool = False) -> None:
        logger.info("Initializing Devanagari (Marathi/Hindi) OCR with EasyOCR")
        self.reader = easyocr.Reader(
            ["mr", "hi"],
            gpu=gpu,
            verbose=False,
            model_storage_directory=str(_MODEL_STORAGE),
        )

    def read(self, image: Image.Image | np.ndarray) -> OcrResult:
        result = self.reader.readtext(image, detail=1, paragraph=True)
        text = "\n".join(chunk[1] for chunk in result if len(chunk) >= 2)
        confidence = _compute_confidence(result)
        # Apply language-aware NLP cleaning for Devanagari
        cleaned_text = clean_ocr_text(text, language="mr")
        return OcrResult(text=cleaned_text, language="mr", confidence=confidence)


def _compute_confidence(chunks: Sequence) -> float:
    scores = [chunk[2] for chunk in chunks if isinstance(chunk, (list, tuple)) and len(chunk) >= 3]
    return float(np.mean(scores)) if scores else 0.0


def _devanagari_ratio(text: str) -> float:
    letters = [ch for ch in text if ch.isalpha()]
    if not letters:
        return 0.0
    devanagari_letters = [ch for ch in letters if 0x0900 <= ord(ch) <= 0x097F]
    return len(devanagari_letters) / len(letters)


def _latin_ratio(text: str) -> float:
    letters = [ch for ch in text if ch.isalpha()]
    if not letters:
        return 0.0
    latin_letters = [ch for ch in letters if ch.isascii()]
    return len(latin_letters) / len(letters)


def _to_bgr_array(image: Image.Image | np.ndarray) -> np.ndarray:
    if isinstance(image, Image.Image):
        rgb = image.convert("RGB")
        return cv2.cvtColor(np.array(rgb), cv2.COLOR_RGB2BGR)
    arr = np.asarray(image)
    if arr.ndim == 2:
        return cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
    return arr


class OCRService:
    def __init__(self) -> None:
        gpu = torch.cuda.is_available()
        logger.info(f"OCR Service GPU Available: {gpu}")
        self.eng_engine = EasyEnglishOCREngine(gpu=gpu)
        self.dev_engine = DevanagariOCREngine(gpu=gpu)
        self.eng_code = "en"
        self.mr_code = "mr"
        self.hi_code = "hi"

    def _best_result(self, engine: BaseOCREngine, candidates: Sequence[np.ndarray]) -> OcrResult:
        best: OcrResult | None = None
        for candidate in candidates:
            result = engine.read(candidate)
            if best is None or result.confidence > best.confidence:
                best = result
        return best or OcrResult(text="", language="unknown", confidence=0.0)

    def read(self, image: Image.Image | np.ndarray, language_hint: str | None = None) -> OcrResult:

        base = _to_bgr_array(image)
        # Preprocess now includes deskewing via image_ops
        processed = preprocess_image(base.copy())
        candidates = [base, processed]

        if language_hint in [self.mr_code, self.hi_code]:
            chosen = self._best_result(self.dev_engine, candidates)
            chosen.language = language_hint 
            return chosen
        if language_hint == self.eng_code:
            chosen = self._best_result(self.eng_engine, candidates)
            return chosen

        devanagari = self._best_result(self.dev_engine, candidates)
        english = self._best_result(self.eng_engine, candidates)

        # Devanagari ratio check
        dev_ratio = _devanagari_ratio(devanagari.text)
        eng_ratio = _latin_ratio(english.text)
        
        # Simple heuristic: if significant Devanagari, prefer it
        if dev_ratio > 0.1 and dev_ratio > eng_ratio:
            # Try to distinguish Hindi vs Marathi if needed, 
            # for now assume Marathi as per original or detect
            lang_guess = detect_language(devanagari.text)
            devanagari.language = lang_guess if lang_guess in [self.mr_code, self.hi_code] else self.mr_code
            return devanagari

        if eng_ratio > dev_ratio:
            return english

        # Fallback to confidence
        if devanagari.confidence > english.confidence:
            lang_guess = detect_language(devanagari.text)
            devanagari.language = lang_guess if lang_guess in [self.mr_code, self.hi_code] else self.mr_code
            return devanagari
            
        return english


__all__ = ["OCRService", "OcrResult"]
