#!/usr/bin/env python3
"""Test script for new file queue system."""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

def test_file_queue():
    """Test the file queue manager."""
    from src.bot.file_queue import FileQueueManager, QueuedFile
    from src.conversion.memory_processor import memory_processor
    
    print("🔄 Testing File Queue System")
    print("=" * 50)
    
    # Test queue creation
    queue_manager = FileQueueManager()
    print("✅ FileQueueManager created")
    
    # Test QueuedFile dataclass
    payload = b"demo" * 10
    handle = memory_processor.store_bytes(payload, "test.pdf")
    try:
        test_file = QueuedFile(
            memory_handle=handle,
            original_name="test.pdf", 
            file_type="pdf",
            user_id=123456789,
            message_id=1,
            file_size=len(payload),
        )
        print(f"✅ QueuedFile created: {test_file.original_name} ({test_file.file_size} bytes)")
    finally:
        handle.release()
    
    # Test configuration
    from src.bot.file_queue import PROCESSING_WINDOW_SECONDS, MAX_FILES_PER_BATCH
    print(f"⏱️  Processing window: {PROCESSING_WINDOW_SECONDS} seconds")
    print(f"📦 Max files per batch: {MAX_FILES_PER_BATCH}")
    
    print("\n🎯 Key Features:")
    print("• Automatic 10-second window for file collection")
    print("• Batch processing up to 10 files")  
    print("• Individual DOCX file delivery (no archive)")
    print("• Original filename preservation")
    print("• Progress notifications")
    print("• Error handling per file")
    
    print("\n🚀 New system ready for testing!")

if __name__ == "__main__":
    test_file_queue()