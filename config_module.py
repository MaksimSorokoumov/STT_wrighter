"""
Модуль с константами и настройками для приложения голосового транскрайбера
"""
import os
import logging
import configparser

# Определяем режимы производительности
PERFORMANCE_MODES = {
    'fast': {
        'model_size': 'small',
        'compute_type': 'float16',
        'beam_size': 2,
        'vad_filter': True,
        'char_delay': 0.02
    },
    'balanced': {
        'model_size': 'large-v3-turbo',  # Оптимальный вариант для CUDA
        'compute_type': 'float16',  # Используем float16 так как есть CUDA
        'beam_size': 3,
        'vad_filter': True,
        'char_delay': 0.025
    },
    'accurate': {
        'model_size': 'large-v3',
        'compute_type': 'float16',  # Используем float16 так как есть CUDA
        'beam_size': 4,
        'vad_filter': False,
        'char_delay': 0.03
    }
}

# Список нежелательных фраз по умолчанию
DEFAULT_UNWANTED_PHRASES = [
    "Субтитры создавал DimaTorzok",
    "Продолжение следует...",
    "Редактор субтитров А.Семкин Корректор А.Егорова",
    "Текст на русском языке",
    "Субтитры сделал DimaTorzok",
    "Редактор субтитров М.Лосева Корректор А.Егорова",
    "Текст на английском.",
    "Редактор субтитров А.Синецкая Корректор А.Егорова",
    "Редактор субтитров Т.Горелова Корректор А.Егорова",
    "Редактор субтитров Е.Жукова Корректор А.Егорова",
    "Смотрите продолжение во второй части видео",
    "Смотрите продолжение в следующей части",
    "Смотрите продолжение в следующей части видео",
    "Смотрите продолжение в 4 части видео",
    "Смотрите продолжение в следующей серии...",
    "Смотрите продолжение во второй части",
    "Спасибо за субтитры!",
    "Субтитры добавил DimaTorzok",
    "Редактор субтитров А.Семкин",
    "Спасибо Спасибо",
    "Это текст на русском языке",
    "Субтитры подогнал «Симон»",
    "Всем пока!",
    "Всем пока! Субтитры подогнал «Симон»!" 
]

# Настройки по умолчанию
DEFAULT_SETTINGS = {
    'input_method': 'direct_input',
    'char_delay': '0.02',
    'remove_duplicates': 'True',
    'model_size': 'large-v3',
    'language': 'ru',
    'max_recording_seconds': '60',
    'hotkey': 'f8',
    'performance_mode': 'balanced',
    'vad_threshold': '0.005',
    'batch_size': '5',
    'max_workers': '2',
    'cpu_threads': '8'
}

def load_or_create_config():
    """Загружает или создает файл конфигурации с обработкой ошибок"""
    config = configparser.ConfigParser()
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.ini")
    
    try:
        if os.path.exists(config_path):
            config.read(config_path, encoding='utf-8')
            
            # Создаем секцию настроек, если её нет
            if 'settings' not in config:
                config['settings'] = {}
                
            # Устанавливаем значения по умолчанию для отсутствующих настроек
            for key, value in DEFAULT_SETTINGS.items():
                if key not in config['settings']:
                    config['settings'][key] = value
                    
            # Создаем секцию для нежелательных фраз, если её нет
            if 'unwanted_phrases' not in config:
                config['unwanted_phrases'] = {}
                for i, phrase in enumerate(DEFAULT_UNWANTED_PHRASES, 1):
                    config['unwanted_phrases'][f'phrase{i}'] = phrase
                    
            logging.info("Конфигурация успешно загружена")
        else:
            # Создаем конфигурацию по умолчанию
            config['settings'] = DEFAULT_SETTINGS.copy()
            
            # Добавляем нежелательные фразы
            config['unwanted_phrases'] = {}
            for i, phrase in enumerate(DEFAULT_UNWANTED_PHRASES, 1):
                config['unwanted_phrases'][f'phrase{i}'] = phrase
                
            # Сохраняем конфигурацию
            with open(config_path, 'w', encoding='utf-8') as f:
                config.write(f)
                
            logging.info("Создана конфигурация по умолчанию")
    except Exception as e:
        logging.error(f"Ошибка загрузки конфигурации: {e}")
        # Создаем конфигурацию в памяти
        config['settings'] = DEFAULT_SETTINGS.copy()
        config['unwanted_phrases'] = {}
        for i, phrase in enumerate(DEFAULT_UNWANTED_PHRASES, 1):
            config['unwanted_phrases'][f'phrase{i}'] = phrase
            
    return config

def save_config(config):
    """Сохраняет конфигурацию в файл"""
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.ini")
    try:
        with open(config_path, 'w', encoding='utf-8') as f:
            config.write(f)
        logging.info("Конфигурация успешно сохранена")
        return True
    except Exception as e:
        logging.error(f"Ошибка сохранения конфигурации: {e}")
        return False 