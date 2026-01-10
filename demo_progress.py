#!/usr/bin/env python3
"""Demo script to show progress indicators in action."""

import asyncio
import time

from src.bot.file_queue import create_progress_bar, get_loading_animation


def demo_progress_bar():
    """Demonstrate the progress bar functionality."""
    print("\n🎯 Demo: Progress Bar Visualization")
    print("=" * 50)
    
    total_steps = 5
    for step in range(total_steps + 1):
        bar = create_progress_bar(step, total_steps, length=25)
        print(f"Step {step}/{total_steps}: {bar}")
        time.sleep(0.5)
    
    print("\n✅ Progress bar demo complete!")


def demo_loading_animation():
    """Demonstrate the loading animation."""
    print("\n⏳ Demo: Loading Animation")
    print("=" * 50)
    
    duration = 5  # seconds
    frames_shown = 0
    
    start_time = time.time()
    while time.time() - start_time < duration:
        animation_char = get_loading_animation(frames_shown)
        
        # Simulate different status messages
        if frames_shown < 10:
            status = "Подготовка к конвертации..."
        elif frames_shown < 20:
            status = "Анализ документа..."
        elif frames_shown < 30:
            status = "Конвертирую: document.pdf..."
        else:
            status = "Завершение обработки..."
        
        print(f"\r{animation_char} {status}", end="", flush=True)
        time.sleep(0.2)
        frames_shown += 1
    
    print("\n✅ Loading animation demo complete!")


async def demo_combined_progress():
    """Demonstrate combined progress visualization."""
    print("\n🚀 Demo: Combined Progress with Animation")
    print("=" * 50)
    
    files = ["document1.pdf", "report.doc", "presentation.pdf", "data.doc"]
    total_files = len(files)
    
    for i, filename in enumerate(files, 1):
        # Show file progress
        progress_bar = create_progress_bar(i - 1, total_files, length=20)
        print(f"\n📊 Обрабатываю файлы ({i}/{total_files})")
        print(f"{progress_bar}")
        
        # Simulate processing with animation
        short_name = filename[:30] + "..." if len(filename) > 30 else filename
        
        # Animation during processing
        processing_time = 2  # seconds per file
        frames = int(processing_time / 0.3)
        
        for frame in range(frames):
            animation_char = get_loading_animation(frame)
            print(f"\r⚙️ {animation_char} Конвертирую: {short_name}", end="", flush=True)
            await asyncio.sleep(0.3)
        
        print(f"\r✅ Готово: {filename}")
    
    # Final state
    final_bar = create_progress_bar(total_files, total_files, length=20)
    print(f"\n🎉 **Обработка завершена!**")
    print(f"{final_bar}")
    print(f"📁 Готово: {total_files}/{total_files} файл(ов)")


async def main():
    """Run all demos."""
    print("🎭 DEMO: Dynamic Progress Indicators")
    print("=" * 70)
    print("Демонстрация новой системы индикации прогресса для Telegram бота")
    print("=" * 70)
    
    # Demo 1: Basic progress bar
    demo_progress_bar()
    
    await asyncio.sleep(1)
    
    # Demo 2: Loading animation
    demo_loading_animation()
    
    await asyncio.sleep(1)
    
    # Demo 3: Combined progress
    await demo_combined_progress()
    
    print("\n" + "=" * 70)
    print("✨ Все демо завершены! Теперь пользователи будут видеть:")
    print("  • Прогресс-бары с процентами")
    print("  • Анимированные индикаторы загрузки")
    print("  • Динамически обновляемые сообщения")
    print("  • Подробную информацию о каждом этапе")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())