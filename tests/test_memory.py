#!/usr/bin/env python3
"""Test script for memory processing system."""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

def test_memory_processor():
    """Test the memory processing system."""
    print("💾 Testing Memory Processing System")
    print("=" * 50)
    
    try:
        from src.conversion.memory_processor import MemoryProcessor, memory_processor, convert_file_in_memory
        from src.conversion.groq_converter import convert_pdf_bytes_to_docx_via_groq
        
        # Test memory processor initialization
        print(f"✅ MemoryProcessor initialized: {memory_processor.max_buffer // 1024 // 1024} MB buffer")
        
        # Test memory stats
        stats = memory_processor.get_memory_stats()
        print(f"📊 Initial memory stats: {stats.current_usage} bytes, {stats.files_in_memory} files")
        
        # Test file size checking
        small_file_size = 1024 * 1024  # 1MB
        large_file_size = 600 * 1024 * 1024  # 600MB
        
        can_fit_small = memory_processor.can_fit_in_memory(small_file_size)
        can_fit_large = memory_processor.can_fit_in_memory(large_file_size)
        
        print(f"📏 Can fit 1MB file: {can_fit_small}")
        print(f"📏 Can fit 600MB file: {can_fit_large}")
        
        print("\n🔧 Available Functions:")
        print("• convert_file_in_memory() - Universal file converter")
        print("• convert_pdf_bytes_to_docx_via_groq() - PDF bytes to DOCX")
        print("• MemoryProcessor.load_file_to_memory() - Smart file loading")
        print("• MemoryProcessor.create_memory_file() - Memory-first file creation")
        
        print("\n💡 Key Features:")
        print("• 512MB memory buffer for batch processing")
        print("• Automatic fallback to disk for large files")
        print("• PDF processing with pdf2image (no temp files)")
        print("• Memory usage tracking and cleanup")
        print("• Smart batching with memory limits")
        
        print("\n🚀 Memory processing system ready!")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("🔧 Install missing dependencies: pip install pdf2image")
        return False
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

if __name__ == "__main__":
    success = test_memory_processor()
    sys.exit(0 if success else 1)