"""
Модуль для работы с распознаванием речи через Whisper
"""
import os
import logging
import torch
from faster_whisper import WhisperModel

def initialize_whisper_model(model_size, compute_type='float16', cpu_threads=6, model_path=None):
    """Инициализирует модель Whisper и обрабатывает ошибки CUDA"""
    try:
        # Определяем устройство
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        if device == "cpu" and compute_type == "float16":
            # Для CPU используем float32 или int8
            compute_type = "int8"
        
        # Путь для моделей
        if model_path is None:
            model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
        
        logging.info(f"Инициализация модели: {model_size}, устройство: {device}, тип вычислений: {compute_type}")
        
        # Создаем модель
        model = WhisperModel(
            model_size,
            device=device,
            compute_type=compute_type,
            download_root=model_path,
            cpu_threads=cpu_threads
        )
        
        logging.info(f"Модель Whisper инициализирована: {model_size} на {device}")
        return model
        
    except Exception as e:
        logging.error(f"Ошибка инициализации модели Whisper: {str(e)}")
        # Запасной вариант - использовать CPU
        logging.info("Переключение на резервную CPU модель")
        return WhisperModel(
            "medium",  # Используем medium как запасной вариант
            device="cpu",
            compute_type="int8",
            cpu_threads=cpu_threads
        )

def transcribe_audio(model, audio_np, language='ru', beam_size=3, vad_filter=False):
    """Транскрибирует аудио с использованием модели Whisper"""
    try:
        segments, info = model.transcribe(
            audio_np,
            language=language,
            beam_size=beam_size,
            vad_filter=vad_filter,
            initial_prompt="Это текст на русском языке."
        )
        
        logging.info(f"Определен язык: {info.language} с вероятностью {info.language_probability}")
        
        # Собираем текст из всех сегментов
        text = " ".join(segment.text for segment in segments if segment.text)
        
        return text
    except Exception as e:
        logging.error(f"Ошибка при транскрибации: {e}")
        return "" 