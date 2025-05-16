"""
Модуль для обработки аудио, включая Voice Activity Detection (VAD)
"""
import numpy as np
import logging

class SimpleVAD:
    """Энергетический Voice Activity Detector"""
    def __init__(self, energy_threshold=0.005, min_silence_duration=1.0, fs=16000):
        self.energy_threshold = energy_threshold
        self.min_silence_samples = int(min_silence_duration * fs)
        self.silence_counter = 0
        
    def is_speech(self, audio_chunk):
        """Определяет, содержит ли аудио фрагмент речь на основе энергии сигнала"""
        # Рассчитываем энергию сигнала
        energy = np.mean(np.abs(audio_chunk))
        
        if energy < self.energy_threshold:
            self.silence_counter += len(audio_chunk)
            # Возвращает True, если молчание не превысило порог
            return self.silence_counter < self.min_silence_samples
        else:
            # Сбрасываем счетчик тишины
            self.silence_counter = 0
            return True
            
    def reset(self):
        """Сбрасывает счетчик тишины"""
        self.silence_counter = 0

def normalize_audio(audio_np):
    """Нормализует аудио данные до диапазона [-1, 1]"""
    # Приводим к одномерному массиву
    audio_np = audio_np.flatten()
    
    # Убедимся, что значения в правильном диапазоне [-1, 1]
    if np.max(np.abs(audio_np)) > 1.0:
        audio_np = audio_np / 32767.0
        
    return audio_np.astype(np.float32)

def filter_unwanted_phrases(text, unwanted_phrases):
    """Удаляет нежелательные фразы из распознанного текста"""
    if not text or not unwanted_phrases:
        return text
        
    original_text = text
    
    # Удаляем каждую нежелательную фразу из текста
    for phrase in unwanted_phrases:
        if phrase:  # Проверяем, что фраза не пустая
            text = text.replace(phrase, "")
    
    # Удаляем лишние пробелы
    text = ' '.join(text.split())
    
    # Логируем, если были произведены изменения
    if original_text != text:
        logging.info(f"Удалены нежелательные фразы. Было: '{original_text}' Стало: '{text}'")
    
    return text 