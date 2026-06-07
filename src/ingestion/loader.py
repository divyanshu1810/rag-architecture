"""
Document Loader Module

Load data from PDFs, CSVs, websites, and other file formats.
"""

import csv
import io
import json
import logging
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)


class DocumentLoader:
    """Load documents from various file formats."""

    SUPPORTED_EXTENSIONS = {".pdf", ".csv", ".txt", ".md", ".json", ".html"}

    def __init__(self, source_dir: Optional[str] = None):
        self.source_dir = Path(source_dir) if source_dir else None

    def load_file(self, file_path: str) -> str:
        """Load a single file and return its content as text."""
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        ext = path.suffix.lower()
        if ext not in self.SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported file type: {ext}. "
                f"Supported: {self.SUPPORTED_EXTENSIONS}"
            )

        logger.info(f"Loading file: {file_path}")

        if ext == ".pdf":
            return self._load_pdf(path)
        elif ext == ".csv":
            return self._load_csv(path)
        elif ext in {".txt", ".md"}:
            return self._load_text(path)
        elif ext == ".json":
            return self._load_json(path)
        elif ext == ".html":
            return self._load_html(path)
        else:
            return self._load_text(path)

    def load_directory(self, directory: Optional[str] = None) -> List[str]:
        """Load all supported files from a directory."""
        dir_path = Path(directory) if directory else self.source_dir
        if not dir_path or not dir_path.exists():
            raise FileNotFoundError(f"Directory not found: {dir_path}")

        documents = []
        for file_path in sorted(dir_path.rglob("*")):
            if file_path.suffix.lower() in self.SUPPORTED_EXTENSIONS:
                try:
                    content = self.load_file(str(file_path))
                    documents.append(content)
                    logger.info(f"Loaded: {file_path.name}")
                except Exception as e:
                    logger.warning(f"Failed to load {file_path}: {e}")

        logger.info(f"Loaded {len(documents)} documents from {dir_path}")
        return documents

    def load_from_url(self, url: str) -> str:
        """Load content from a web URL."""
        try:
            import requests
        except ImportError:
            raise ImportError(
                "requests package required for URL loading. "
                "Install with: uv add requests"
            )

        logger.info(f"Loading content from URL: {url}")

        headers = {
            "User-Agent": (
                "Mozilla/5.0 (compatible; RAGBot/1.0; "
                "+https://github.com/divyanshu1810/rag-architecture)"
            )
        }
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()

        content_type = response.headers.get("Content-Type", "")

        if "text/html" in content_type:
            return self._extract_text_from_html(response.text)
        elif "application/pdf" in content_type:
            return self._extract_text_from_pdf_bytes(response.content)
        else:
            # Plain text, JSON, etc.
            return response.text

    # -- Private loaders --

    def _load_text(self, path: Path) -> str:
        return path.read_text(encoding="utf-8")

    def _load_pdf(self, path: Path) -> str:
        """Load PDF using PyPDF2 with pdfplumber fallback."""
        # Try PyPDF2 first
        try:
            from PyPDF2 import PdfReader

            reader = PdfReader(str(path))
            pages = []
            for page in reader.pages:
                text = page.extract_text()
                if text:
                    pages.append(text)

            content = "\n\n".join(pages)
            if content.strip():
                logger.info(
                    f"Loaded PDF with PyPDF2: {path.name} "
                    f"({len(reader.pages)} pages)"
                )
                return content
        except ImportError:
            pass
        except Exception as e:
            logger.warning(f"PyPDF2 failed for {path.name}: {e}")

        # Fallback to pdfplumber
        try:
            import pdfplumber

            pages = []
            with pdfplumber.open(str(path)) as pdf:
                for page in pdf.pages:
                    text = page.extract_text()
                    if text:
                        pages.append(text)

            content = "\n\n".join(pages)
            logger.info(
                f"Loaded PDF with pdfplumber: {path.name} "
                f"({len(pages)} pages)"
            )
            return content
        except ImportError:
            raise ImportError(
                "PDF loading requires PyPDF2 or pdfplumber. "
                "Install with: uv add PyPDF2  or  uv add pdfplumber"
            )

    def _load_csv(self, path: Path) -> str:
        """Load CSV and convert rows to readable text."""
        text_parts = []
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader):
                row_text = " | ".join(
                    f"{k}: {v}" for k, v in row.items() if v
                )
                text_parts.append(row_text)

        content = "\n".join(text_parts)
        logger.info(f"Loaded CSV: {path.name} ({len(text_parts)} rows)")
        return content

    def _load_json(self, path: Path) -> str:
        """Load JSON and pretty-print as text."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, list):
            # If it's a list of objects, join them
            parts = []
            for item in data:
                if isinstance(item, dict):
                    parts.append(
                        " | ".join(f"{k}: {v}" for k, v in item.items())
                    )
                else:
                    parts.append(str(item))
            return "\n".join(parts)
        elif isinstance(data, dict):
            return json.dumps(data, indent=2, ensure_ascii=False)
        else:
            return str(data)

    def _load_html(self, path: Path) -> str:
        """Load HTML file and extract text content."""
        raw_html = path.read_text(encoding="utf-8")
        return self._extract_text_from_html(raw_html)

    # -- Shared helpers --

    def _extract_text_from_html(self, html: str) -> str:
        """Extract readable text from HTML using BeautifulSoup."""
        try:
            from bs4 import BeautifulSoup
        except ImportError:
            raise ImportError(
                "beautifulsoup4 package required for HTML parsing. "
                "Install with: uv add beautifulsoup4"
            )

        soup = BeautifulSoup(html, "html.parser")

        # Remove script and style elements
        for element in soup(["script", "style", "nav", "footer", "header"]):
            element.decompose()

        text = soup.get_text(separator="\n", strip=True)

        # Collapse multiple blank lines
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        return "\n".join(lines)

    def _extract_text_from_pdf_bytes(self, pdf_bytes: bytes) -> str:
        """Extract text from raw PDF bytes (e.g. from a URL download)."""
        try:
            from PyPDF2 import PdfReader

            reader = PdfReader(io.BytesIO(pdf_bytes))
            pages = []
            for page in reader.pages:
                text = page.extract_text()
                if text:
                    pages.append(text)
            return "\n\n".join(pages)
        except ImportError:
            pass

        try:
            import pdfplumber

            pages = []
            with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
                for page in pdf.pages:
                    text = page.extract_text()
                    if text:
                        pages.append(text)
            return "\n\n".join(pages)
        except ImportError:
            raise ImportError(
                "PDF loading requires PyPDF2 or pdfplumber. "
                "Install with: uv add PyPDF2  or  uv add pdfplumber"
            )
