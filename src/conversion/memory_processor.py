"""Memory-based file processing with 512MB buffer and fallback strategies."""

from __future__ import annotations

import io
import logging
import tempfile
from pathlib import Path
from typing import Optional, Union, BinaryIO
from dataclasses import dataclass, field
from contextlib import asynccontextmanager
from uuid import uuid4

logger = logging.getLogger(__name__)

# Memory configuration
MAX_MEMORY_BUFFER_MB = 512
MAX_MEMORY_BUFFER_BYTES = MAX_MEMORY_BUFFER_MB * 1024 * 1024


@dataclass 
class MemoryStats:
    """Memory usage statistics."""
    current_usage: int = 0
    peak_usage: int = 0
    files_in_memory: int = 0
    fallbacks_to_disk: int = 0


@dataclass
class MemoryHandle:
    """Represents a blob stored in RAM under MemoryProcessor control."""

    processor: "MemoryProcessor"
    data: bytes
    original_name: str
    id: str = field(default_factory=lambda: uuid4().hex)
    released: bool = False
    size: int = field(init=False)

    def __post_init__(self) -> None:  # noqa: D401
        self.size = len(self.data)

    def release(self) -> None:
        """Release memory usage associated with this handle."""
        if self.released:
            return
        self.processor._release_memory(self.size)
        self.released = True
        self.data = b""

    def get_bytes_io(self) -> io.BytesIO:
        """Return a BytesIO wrapper over the payload for downstream consumers."""
        return io.BytesIO(self.data)

    def get_bytes(self) -> bytes:
        """Return raw bytes for consumers that need direct access."""
        return self.data


class MemoryProcessor:
    """Manages file processing in memory with automatic fallback to disk."""
    
    def __init__(self, max_buffer_bytes: int = MAX_MEMORY_BUFFER_BYTES):
        self.max_buffer = max_buffer_bytes
        self.stats = MemoryStats()
        logger.info("MemoryProcessor initialized with %d MB buffer", max_buffer_bytes // 1024 // 1024)

    def store_bytes(self, payload: bytes, original_name: str) -> MemoryHandle:
        """Persist payload in RAM and return a managed handle."""
        size = len(payload)
        if size == 0:
            raise ValueError("Cannot store empty payload in memory")
        if not self.can_fit_in_memory(size):
            raise MemoryError(
                f"Payload '{original_name}' size {size} exceeds available memory budget"
            )

        self.stats.current_usage += size
        self.stats.files_in_memory += 1
        self.stats.peak_usage = max(self.stats.peak_usage, self.stats.current_usage)
        logger.debug(
            "Stored %s in memory (%d bytes). Current usage: %d MB",
            original_name,
            size,
            self.stats.current_usage // 1024 // 1024,
        )
        return MemoryHandle(processor=self, data=payload, original_name=original_name)
    
    def can_fit_in_memory(self, file_size: int) -> bool:
        """Check if file can fit in current memory budget."""
        return (self.stats.current_usage + file_size) <= self.max_buffer

    def _release_memory(self, size: int) -> None:
        self.stats.current_usage = max(0, self.stats.current_usage - size)
        self.stats.files_in_memory = max(0, self.stats.files_in_memory - 1)
        logger.debug(
            "Released %d bytes from memory. Current usage: %d MB",
            size,
            self.stats.current_usage // 1024 // 1024,
        )
    
    @asynccontextmanager
    async def load_file_to_memory(self, file_path: Path):
        """Load file to memory if possible, fallback to disk access."""
        file_size = file_path.stat().st_size
        
        if self.can_fit_in_memory(file_size):
            # Load to memory
            try:
                with file_path.open('rb') as f:
                    file_data = f.read()
                
                self.stats.current_usage += file_size
                self.stats.files_in_memory += 1
                self.stats.peak_usage = max(self.stats.peak_usage, self.stats.current_usage)
                
                logger.debug("Loaded %s to memory (%d bytes), total usage: %d MB", 
                           file_path.name, file_size, self.stats.current_usage // 1024 // 1024)
                
                try:
                    yield io.BytesIO(file_data)
                finally:
                    # Clean up memory
                    self.stats.current_usage -= file_size
                    self.stats.files_in_memory -= 1
                    del file_data  # Explicit cleanup
                    
            except (OSError, MemoryError) as e:
                logger.warning("Failed to load %s to memory: %s, falling back to disk", file_path.name, e)
                # Fallback to disk
                self.stats.fallbacks_to_disk += 1
                with file_path.open('rb') as f:
                    yield f
        else:
            # Direct disk access
            logger.debug("File %s (%d bytes) too large for memory buffer, using disk", 
                        file_path.name, file_size)
            self.stats.fallbacks_to_disk += 1
            with file_path.open('rb') as f:
                yield f
    
    @asynccontextmanager
    async def create_memory_file(self, max_size_hint: Optional[int] = None):
        """Create a file-like object in memory if possible."""
        use_memory = True
        if max_size_hint and not self.can_fit_in_memory(max_size_hint):
            use_memory = False
        
        if use_memory:
            # Use BytesIO for in-memory operations
            buffer = io.BytesIO()
            try:
                yield buffer
                # Track actual size after writing
                actual_size = buffer.tell()
                if actual_size > 0:
                    self.stats.current_usage += actual_size
                    self.stats.peak_usage = max(self.stats.peak_usage, self.stats.current_usage)
                    logger.debug("Created memory file: %d bytes, total usage: %d MB",
                               actual_size, self.stats.current_usage // 1024 // 1024)
            finally:
                # Clean up if size was tracked
                actual_size = buffer.tell()
                if actual_size > 0:
                    self.stats.current_usage = max(0, self.stats.current_usage - actual_size)
        else:
            # Use temporary file on disk
            logger.debug("Creating temporary file on disk due to memory constraints")
            self.stats.fallbacks_to_disk += 1
            with tempfile.NamedTemporaryFile() as tmp_file:
                yield tmp_file
    
    def save_bytes_to_path(self, data: bytes, output_path: Path) -> None:
        """Save bytes data to file path."""
        with output_path.open('wb') as f:
            f.write(data)
        logger.debug("Saved %d bytes to %s", len(data), output_path)
    
    def get_memory_stats(self) -> MemoryStats:
        """Get current memory usage statistics."""
        return self.stats
    
    def clear_memory(self) -> None:
        """Force memory cleanup and reset stats."""
        self.stats.current_usage = 0
        self.stats.files_in_memory = 0
        logger.debug("Memory cleared, peak usage was: %d MB", self.stats.peak_usage // 1024 // 1024)
    
    def log_stats(self) -> None:
        """Log current memory statistics."""
        logger.info(
            "Memory stats - Current: %d MB, Peak: %d MB, Files in memory: %d, Disk fallbacks: %d",
            self.stats.current_usage // 1024 // 1024,
            self.stats.peak_usage // 1024 // 1024,
            self.stats.files_in_memory,
            self.stats.fallbacks_to_disk
        )


# Global memory processor instance
memory_processor = MemoryProcessor()


def convert_file_in_memory(
    file_data: Union[bytes, BinaryIO], 
    file_type: str,
    original_name: str
) -> Optional[bytes]:
    """Convert file data in memory without touching disk.
    
    Args:
        file_data: File content as bytes or file-like object
        file_type: 'doc', 'docx', 'pdf', or 'image'
        original_name: Original filename for logging
        
    Returns:
        Converted DOCX file as bytes, or None if conversion failed
    """
    try:
        if file_type == 'pdf':
            return _convert_pdf_in_memory(file_data, original_name)
        elif file_type == 'doc':
            return _convert_doc_in_memory(file_data, original_name)
        elif file_type == 'docx':
            return _convert_docx_in_memory(file_data)
        elif file_type == 'image':
            return _convert_image_in_memory(file_data, original_name)
        else:
            logger.error("Unsupported file type: %s", file_type)
            return None
            
    except Exception as e:
        logger.error("Memory conversion failed for %s: %s", original_name, e)
        return None


def _convert_pdf_in_memory(file_data: Union[bytes, BinaryIO], original_name: str) -> Optional[bytes]:
    """Convert PDF to DOCX in memory using reliability-first converter pipeline."""
    try:
        from ..conversion.converter import convert_pdf_to_docx
        
        # Ensure we have bytes
        if hasattr(file_data, 'read'):
            pdf_bytes = file_data.read()
        else:
            pdf_bytes = file_data

        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp_input:
            tmp_input.write(pdf_bytes)
            tmp_input_path = Path(tmp_input.name)

        try:
            with tempfile.TemporaryDirectory() as tmp_dir:
                converted_path = convert_pdf_to_docx(tmp_input_path, Path(tmp_dir))
                if converted_path and converted_path.exists():
                    with converted_path.open('rb') as f:
                        return f.read()
                return None
        finally:
            if tmp_input_path.exists():
                tmp_input_path.unlink()
    except Exception as e:
        logger.error("PDF memory conversion failed for %s: %s", original_name, e)
        return None


def _convert_docx_in_memory(file_data: Union[bytes, BinaryIO]) -> Optional[bytes]:
    """Pass DOCX through without conversion (used for transliteration workflow)."""
    if hasattr(file_data, 'read'):
        return file_data.read()
    return file_data


def _convert_doc_in_memory(file_data: Union[bytes, BinaryIO], original_name: str) -> Optional[bytes]:
    """Convert DOC to DOCX in memory.
    
    Note: LibreOffice requires files on disk, so this will use temporary files
    but clean them up immediately.
    """
    try:
        import tempfile
        from ..conversion.converter import convert_doc_to_docx
        from ..config.settings import settings
        
        # Ensure we have bytes  
        if hasattr(file_data, 'read'):
            doc_bytes = file_data.read()
        else:
            doc_bytes = file_data
        
        # LibreOffice requires disk files, so use minimal temporary storage
        with tempfile.NamedTemporaryFile(suffix='.doc', delete=False) as tmp_input:
            tmp_input.write(doc_bytes)
            tmp_input_path = Path(tmp_input.name)
        
        try:
            # Convert using LibreOffice
            with tempfile.TemporaryDirectory() as tmp_dir:
                converted_path = convert_doc_to_docx(
                    tmp_input_path, 
                    Path(tmp_dir),
                    settings.libreoffice_path
                )
                
                # Read result back to memory
                if converted_path and converted_path.exists():
                    with converted_path.open('rb') as f:
                        docx_bytes = f.read()
                    return docx_bytes
                else:
                    return None
                    
        finally:
            # Clean up input file
            if tmp_input_path.exists():
                tmp_input_path.unlink()
                
    except Exception as e:
        logger.error("DOC memory conversion failed for %s: %s", original_name, e)
        return None


def _convert_image_in_memory(file_data: Union[bytes, BinaryIO], original_name: str) -> Optional[bytes]:
    """Convert image bytes to DOCX using OCR pipeline through temporary files."""
    try:
        from ..conversion.converter import convert_image_to_docx

        if hasattr(file_data, 'read'):
            image_bytes = file_data.read()
        else:
            image_bytes = file_data

        suffix = Path(original_name).suffix.lower() or ".jpg"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp_input:
            tmp_input.write(image_bytes)
            tmp_input_path = Path(tmp_input.name)

        try:
            with tempfile.TemporaryDirectory() as tmp_dir:
                converted_path = convert_image_to_docx(tmp_input_path, Path(tmp_dir))
                if converted_path and converted_path.exists():
                    with converted_path.open('rb') as f:
                        return f.read()
                return None
        finally:
            if tmp_input_path.exists():
                tmp_input_path.unlink()

    except Exception as e:
        logger.error("Image memory conversion failed for %s: %s", original_name, e)
        return None
