import sys
import keyboard
import sounddevice as sd
import numpy as np
from tkinter import *
from tkinter import ttk
from PIL import Image, ImageTk
import torch
from faster_whisper import WhisperModel
import pyautogui
from pystray import MenuItem as item, Icon
import threading
import os
from pynput.keyboard import Controller
import time
import queue
from concurrent.futures import ThreadPoolExecutor
import logging
import win32con
import win32api
import win32gui
import configparser
import ctypes
import re
import traceback

# Проверка доступности Windows API
try:
    import win32con
    import win32api
    import win32gui
    WIN32_AVAILABLE = True
    
    # Скрываем консоль при запуске
    if hasattr(sys, 'frozen'):  # Проверяем, запущено ли приложение как exe
        hwnd = ctypes.windll.kernel32.GetConsoleWindow()
        if hwnd != 0:
            ctypes.windll.user32.ShowWindow(hwnd, 0)  # SW_HIDE = 0
except ImportError:
    WIN32_AVAILABLE = False
    logging.warning("pywin32 не установлен. Прямой ввод через WinAPI будет недоступен")

# Настраиваем логирование также в консоль для отладки
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler('app.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)

# Логируем запуск приложения
logging.info("=== Запуск приложения ===")
logging.info(f"Рабочая директория: {os.getcwd()}")
logging.info(f"Доступность CUDA: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    logging.info(f"Версия CUDA: {torch.version.cuda}")
    logging.info(f"Устройство CUDA: {torch.cuda.get_device_name(0)}")

# Определяем режимы производительности
PERFORMANCE_MODES = {
    'fast': {
        'model_size': 'medium',
        'compute_type': 'int8',
        'beam_size': 2,
        'vad_filter': True,
        'char_delay': 0.02
    },
    'balanced': {
        'model_size': 'large-v3',  # Оптимальный вариант для CUDA
        'compute_type': 'float16',  # Используем float16 так как есть CUDA
        'beam_size': 3,
        'vad_filter': True,
        'char_delay': 0.025
    },
    'accurate': {
        'model_size': 'large-v3-turbo',
        'compute_type': 'float16',  # Используем float16 так как есть CUDA
        'beam_size': 5,
        'vad_filter': False,
        'char_delay': 0.03
    }
}

# Энергетический Voice Activity Detector
class SimpleVAD:
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

class TranscriberApp:
    def __init__(self):
        # Загружаем настройки перед инициализацией модели
        self.config = configparser.ConfigParser()
        self.load_or_create_config()
        
        # Выбираем текущий режим производительности
        self.current_performance_mode = self.config['settings'].get('performance_mode', 'balanced')
        
        # Инициализация VAD с настраиваемым порогом энергии
        vad_threshold = float(self.config['settings'].get('vad_threshold', '0.005'))
        self.vad = SimpleVAD(energy_threshold=vad_threshold)
        
        # Кеш для хранения загруженных моделей 
        self.model_cache = {}
        
        # Инициализация модели Whisper с обработкой ошибок CUDA
        self.initialize_whisper_model()
        
        # Настройки аудио
        self.fs = 16000
        self.is_recording = False
        self.audio_data = []
        
        # Очередь для обработки аудио в реальном времени
        self.audio_queue = queue.Queue()
        # Executor для асинхронной транскрипции
        max_workers = int(self.config['settings'].get('max_workers', '2'))
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        # Запускаем отдельный поток для фоновой обработки аудио
        self.start_audio_processing_thread()
        
        # Создание GUI
        self.root = Tk()
        self.root.withdraw()  # Скрываем основное окно
        self.setup_recording_indicator()  # Новое: настройка индикатора записи
        
        # Регистрация настраиваемой горячей клавиши для переключения записи
        self.register_hotkey()
        
        # Добавляем обработку ошибок для иконки
        try:
            self.setup_systray()
        except Exception as icon_error:
            logging.error(f"Ошибка при настройке системного трея: {str(icon_error)}")
            # Создаем запасной вариант без иконки
            self.root.protocol("WM_DELETE_WINDOW", self.quit_app)
            self.root.title("Голосовой транскрайбер")
            self.root.deiconify()  # Показываем основное окно как запасной вариант

    def initialize_whisper_model(self):
        """Инициализирует модель Whisper для CUDA"""
        try:
            # Получаем параметры из текущего режима
            mode_settings = PERFORMANCE_MODES[self.current_performance_mode]
            model_size = mode_settings['model_size']
            compute_type = mode_settings['compute_type']
            
            logging.info(f"Инициализация модели: {model_size}, устройство: cuda, тип вычислений: {compute_type}")
            
            # Создаем ключ для кеша
            cache_key = f"{model_size}_{compute_type}"
            
            # Проверяем, есть ли модель в кеше
            if cache_key in self.model_cache:
                logging.info(f"Используется кешированная модель: {cache_key}")
                self.model = self.model_cache[cache_key]
            else:
                # Создаем новую модель
                model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
                self.model = WhisperModel(
                    model_size,
                    device="cuda",
                    compute_type=compute_type,
                    download_root=model_path,
                    cpu_threads=int(self.config['settings'].get('cpu_threads', '6'))
                )
                # Кешируем модель для возможного повторного использования
                self.model_cache[cache_key] = self.model
                
            logging.info(f"Модель Whisper инициализирована: {model_size} на CUDA")
            
        except Exception as e:
            logging.error(f"Ошибка инициализации модели Whisper: {str(e)}")
            # Запасной вариант - использовать CPU
            logging.info("Переключение на резервную CPU модель")
            self.model = WhisperModel(
                "medium",  # Используем medium как запасной вариант
                device="cpu",
                compute_type="int8",
                cpu_threads=12  # Увеличиваем количество потоков CPU
            )

    def setup_systray(self):
        """Настраивает иконку в системном трее с дополнительными опциями"""
        try:
            image = Image.open("icon.png")
            
            # Создаем меню с выбором режима производительности
            menu = (
                item('Режим: Быстрый', self.switch_to_fast_mode, checked=lambda _: self.current_performance_mode == 'fast'),
                item('Режим: Сбалансированный', self.switch_to_balanced_mode, checked=lambda _: self.current_performance_mode == 'balanced'),
                item('Режим: Точный', self.switch_to_accurate_mode, checked=lambda _: self.current_performance_mode == 'accurate'),
                item('Выход', self.quit_app),
            )
            
            self.icon = Icon("STT", image, "Голосовой транскрайбер", menu)
            threading.Thread(target=self.icon.run, daemon=True).start()
        except Exception as e:
            logging.error(f"Ошибка настройки системного трея: {e}")
            raise  # Пробрасываем ошибку для обработки в конструкторе
    
    def switch_to_fast_mode(self):
        """Переключает на быстрый режим работы"""
        self.change_performance_mode('fast')
    
    def switch_to_balanced_mode(self):
        """Переключает на сбалансированный режим работы"""
        self.change_performance_mode('balanced')
    
    def switch_to_accurate_mode(self):
        """Переключает на точный режим работы"""
        self.change_performance_mode('accurate')
    
    def change_performance_mode(self, mode):
        """Изменяет режим производительности и переинициализирует модель"""
        if mode not in PERFORMANCE_MODES:
            logging.error(f"Неизвестный режим производительности: {mode}")
            return
            
        logging.info(f"Переключение на режим: {mode}")
        self.current_performance_mode = mode
        
        # Обновляем конфигурацию
        self.config['settings']['performance_mode'] = mode
        self._save_config()
        
        # Обновляем задержку между символами из настроек режима
        self.char_delay = PERFORMANCE_MODES[mode]['char_delay']
        
        # Переинициализируем модель с новыми параметрами
        self.initialize_whisper_model()

    def quit_app(self):
        """Корректно закрывает приложение"""
        if hasattr(self, 'icon'):
            self.icon.stop()
        
        # Корректно завершаем executor
        logging.info("Завершение работы ThreadPoolExecutor...")
        self.executor.shutdown(wait=False)  # wait=False, чтобы не блокировать выход
        logging.info("ThreadPoolExecutor завершен.")
        
        if hasattr(self, 'root'):
            self.root.destroy()
        
        os._exit(0)

    def start_recording(self, _):
        """Начинает запись аудио"""
        logging.info(f"Вызван метод start_recording, текущий статус is_recording: {self.is_recording}")
        
        # Если уже идет запись, останавливаем ее сначала
        if self.is_recording:
            logging.info("Уже идет запись. Останавливаем текущую запись перед началом новой")
            self.stop_recording(None)
        
        # Устанавливаем флаг записи
        self.is_recording = True
        
        # Сбрасываем предыдущие аудио данные
        self.audio_data = []
        
        # Сбрасываем счетчик тишины VAD
        self.vad.reset()
        
        logging.info("Начало записи...")
        try:
            # Закрываем предыдущий поток, если он существует
            if hasattr(self, 'stream') and self.stream is not None:
                try:
                    self.stream.stop()
                    self.stream.close()
                except Exception as e:
                    logging.error(f"Ошибка при закрытии предыдущего аудио потока: {e}")
            
            # Инициализируем новый поток
            self.stream = sd.InputStream(
                samplerate=self.fs,
                channels=1,
                callback=self.audio_callback
            )
            self.stream.start()
            
            # Добавляем автоматическую остановку записи
            max_seconds = int(self.config['settings'].get('max_recording_seconds', '120'))
            if hasattr(self, 'auto_stop_timer') and self.auto_stop_timer:
                self.auto_stop_timer.cancel()
            self.auto_stop_timer = threading.Timer(max_seconds, self.auto_stop_recording)
            self.auto_stop_timer.start()
            
            # Настраиваем индикатор записи
            try:
                x, y = pyautogui.position()
                indicator_width = 20
                indicator_height = 5
                self.recording_indicator.geometry(f"{indicator_width}x{indicator_height}+{x}+{y}")
                # Показываем индикатор записи и запускаем мигание
                self.recording_indicator.deiconify()
                self.blink_indicator()
            except Exception as e:
                logging.error(f"Ошибка позиционирования индикатора записи: {e}")
        except Exception as e:
            logging.error(f"Ошибка при запуске записи: {e}")
            logging.error(traceback.format_exc())
            self.is_recording = False

    def audio_callback(self, indata, frames, time, status):
        """Обработчик аудио данных от микрофона с использованием VAD"""
        # Проверка на ошибки и статус
        if status:
            logging.warning(f"Ошибка в аудио коллбэке: {status}")
            
        # Проверка, что флаг записи все еще активен
        if not self.is_recording:
            # Если флаг сброшен, но коллбэк все еще вызывается, это может быть ошибкой
            logging.warning("audio_callback вызван, но флаг is_recording=False")
            return
            
        try:
            audio_chunk = indata.copy()
            
            # Используем VAD только если он включен в текущем режиме
            use_vad = PERFORMANCE_MODES[self.current_performance_mode]['vad_filter']
            
            # Добавляем в аудио данные
            if not use_vad or self.vad.is_speech(audio_chunk):
                self.audio_data.append(audio_chunk)
                self.audio_queue.put(audio_chunk)
            elif len(self.audio_data) > 0 and use_vad:
                # Если был хотя бы один чанк с речью и сейчас тишина дольше порога
                if self.vad.silence_counter >= self.vad.min_silence_samples:
                    # Автоматически останавливаем запись после длительной тишины
                    logging.info("Обнаружена длительная тишина, автоматическая остановка записи")
                    threading.Thread(target=self.auto_stop_after_silence).start()
        except Exception as e:
            logging.error(f"Ошибка в audio_callback: {e}")
            logging.error(traceback.format_exc())

    def auto_stop_after_silence(self):
        """Автоматически останавливает запись после обнаружения тишины"""
        # Проверяем, что запись все еще идет
        if self.is_recording:
            logging.info("Автоматическая остановка записи после обнаружения тишины")
            self.stop_recording(None)

    def stop_recording(self, _):
        """Останавливает запись и обрабатывает аудио"""
        logging.info(f"Вызван метод stop_recording, текущий статус is_recording: {self.is_recording}")
        
        # Проверяем, инициализирован ли стрим и существует ли он
        stream_exists = hasattr(self, 'stream') and self.stream is not None
        
        # Сбрасываем флаг записи в любом случае
        self.is_recording = False
        
        # Отменяем таймер, если он существует
        if hasattr(self, 'auto_stop_timer') and self.auto_stop_timer:
            self.auto_stop_timer.cancel()
            
        # Если стрим существует, останавливаем его
        if stream_exists:
            try:
                self.stream.stop()
                self.stream.close()
                logging.info("Аудио стрим успешно остановлен")
            except Exception as e:
                logging.error(f"Ошибка при остановке аудио потока: {e}")
                
        # Скрываем индикатор записи в любом случае
        if hasattr(self, 'recording_indicator'):
            self.recording_indicator.withdraw()
        
        # Очищаем очередь аудио для предотвращения фоновой обработки
        try:
            while not self.audio_queue.empty():
                self.audio_queue.get_nowait()
            logging.info("Очередь аудио очищена")
        except Exception as e:
            logging.error(f"Ошибка при очистке очереди аудио: {e}")
        
        # Если нет аудио данных, прекращаем обработку
        if not hasattr(self, 'audio_data') or not self.audio_data:
            logging.info("Нет аудио данных, запись пропущена")
            return
        
        logging.info("Обработка аудио...")
        try:
            audio_array = np.concatenate(self.audio_data)
            
            # Добавляем отладочную информацию
            logging.info(f"Длина аудио: {len(audio_array)} сэмплов")
            logging.info(f"Максимальная амплитуда: {np.max(np.abs(audio_array))}")
            
            if len(audio_array) < self.fs * 0.5:  # меньше 0.5 секунды
                logging.info("Слишком короткая запись")
                return
            
            # Отправляем аудио на транскрипцию
            self.transcribe_and_insert(audio_array)
        except Exception as e:
            logging.error(f"Ошибка при обработке аудио: {e}")
            logging.error(traceback.format_exc())

    def transcribe_and_insert(self, audio_np):
        """Распознает речь в аудио и вставляет текст"""
        # Проверяем формат входных данных
        logging.info(f"Форма аудио массива: {audio_np.shape}")
        logging.info(f"Тип данных: {audio_np.dtype}")
        
        # Нормализация аудио данных
        audio_np = audio_np.flatten()
        
        # Убедимся, что значения в правильном диапазоне [-1, 1]
        if np.max(np.abs(audio_np)) > 1.0:
            audio_np = audio_np / 32767.0
        
        try:
            # Получаем параметры из текущего режима производительности
            mode_settings = PERFORMANCE_MODES[self.current_performance_mode]
            beam_size = mode_settings['beam_size']
            
            segments, info = self.model.transcribe(
                audio_np,
                language='ru',
                beam_size=beam_size,
                vad_filter=False,  # Мы используем свой VAD, поэтому отключаем встроенный
            )
            
            logging.info(f"Detected language: {info.language} with probability {info.language_probability}")
            
            text = " ".join(segment.text for segment in segments if segment.text)
            logging.info(f"Распознанный текст: {text}")
            
            if text:
                # Фильтруем нежелательные фразы перед вставкой
                text = self.filter_unwanted_phrases(text)
                self.insert_text(text)
            else:
                logging.info("Не удалось распознать речь")
            
        except Exception as e:
            logging.error(f"Ошибка транскрипции: {str(e)}")
            logging.error(traceback.format_exc())

    def start_audio_processing_thread(self):
        """Запускает поток для обработки аудио"""
        self.audio_processing_thread = threading.Thread(target=self.process_audio_queue, daemon=True)
        self.audio_processing_thread.start()

    def process_audio_queue(self):
        """
        Фоновый поток: обрабатывает накопленные аудиофрагменты
        и отправляет их на транскрипцию при достижении достаточного объема.
        """
        # Флаг для отслеживания состояния обработки
        is_processing = False
        last_processed_text = ""
        batch_size = int(self.config['settings'].get('batch_size', '5'))  # Размер пакета аудио фрагментов
        
        while True:
            # Обрабатываем данные только если идет запись и есть что обрабатывать
            if self.is_recording and not self.audio_queue.empty() and not is_processing:
                chunks = []
                # Извлекаем все данные из очереди с ограничением на размер пакета
                try:
                    chunk_count = 0
                    while not self.audio_queue.empty() and chunk_count < batch_size:
                        chunks.append(self.audio_queue.get())
                        chunk_count += 1
                except Exception as e:
                    logging.error(f"Ошибка при извлечении данных из очереди: {e}")
                    time.sleep(0.2)
                    continue
                
                # Если нет данных, пропускаем итерацию
                if not chunks:
                    time.sleep(0.2)
                    continue
                    
                # Объединяем сегменты
                audio_array = np.concatenate(chunks, axis=0)
                
                # Проверяем минимальную длительность
                if len(audio_array) < self.fs:
                    time.sleep(0.5)
                    continue
                
                # Устанавливаем флаг обработки
                is_processing = True
                
                def on_transcription_done(future):
                    nonlocal is_processing, last_processed_text
                    is_processing = False
                    try:
                        result = future.result()
                        if isinstance(result, str):
                            last_processed_text = result
                    except Exception as e:
                        logging.error(f"Ошибка в асинхронной транскрипции: {e}")
                
                # Отправляем задачу транскрипции асинхронно
                future = self.executor.submit(self.transcribe_and_get_text, audio_array)
                future.add_done_callback(on_transcription_done)
            else:
                # Если запись не идет, очищаем очередь
                if not self.is_recording and not self.audio_queue.empty():
                    try:
                        while not self.audio_queue.empty():
                            self.audio_queue.get_nowait()
                    except:
                        pass
                time.sleep(0.2)

    def transcribe_and_get_text(self, audio_np):
        """
        Выполняет транскрипцию аудио и возвращает текст.
        Этот метод используется для асинхронной обработки в process_audio_queue.
        """
        try:
            # Нормализация аудио данных
            audio_np = audio_np.flatten().astype(np.float32)
            
            # Нормализуем до диапазона [-1, 1]
            if np.max(np.abs(audio_np)) > 1.0:
                audio_np = audio_np / 32767.0
                
            # Получаем параметры из текущего режима производительности
            mode_settings = PERFORMANCE_MODES[self.current_performance_mode]
            beam_size = mode_settings['beam_size']
            
            # Настраиваем параметры транскрипции
            params = {
                'language': 'ru',
                'beam_size': beam_size,
                'vad_filter': False,  # Мы используем собственный VAD
                'initial_prompt': "Это текст на русском языке."
            }
            
            segments, info = self.model.transcribe(audio_np, **params)
            
            # Сбор результатов
            text = " ".join(segment.text for segment in segments if segment.text)
            
            logging.info(f"Распознано (до фильтрации): {text}")
            
            # Фильтруем нежелательные фразы
            text = self.filter_unwanted_phrases(text)
            
            logging.info(f"Распознано (после фильтрации): {text}")
            
            if text:
                self.insert_text(text)
                return text
            else:
                logging.info("Не удалось распознать речь")
                return ""
                
        except Exception as e:
            logging.error(f"Ошибка транскрипции: {str(e)}")
            logging.error(traceback.format_exc())
            return ""

    def toggle_recording(self, event):
        """
        Переключает состояние записи:
        если запись не запущена — начинает запись,
        если запись запущена — останавливает запись и запускает транскрипцию.
        """
        logging.info(f"Вызвана функция toggle_recording, текущий статус записи: {self.is_recording}")
        
        # Защита от повторного вызова во время обработки предыдущего
        if hasattr(self, '_toggle_lock') and self._toggle_lock:
            logging.info("Игнорирование повторного вызова toggle_recording (блокировка активна)")
            return
            
        try:
            # Устанавливаем блокировку
            self._toggle_lock = True
            
            if not self.is_recording:
                self.start_recording(event)
            else:
                logging.info("Останавливаем запись по горячей клавише")
                self.is_recording = False  # Явно устанавливаем флаг перед вызовом stop_recording
                self.stop_recording(event)
        finally:
            # Снимаем блокировку после выполнения
            self._toggle_lock = False

    def auto_stop_recording(self):
        """Автоматически останавливает запись по истечении максимального времени"""
        if self.is_recording:
            max_seconds = int(self.config['settings'].get('max_recording_seconds', '120'))
            logging.info(f"Автоматическая остановка записи по истечении {max_seconds} секунд")
            self.stop_recording(None)

    def run(self):
        """Запускает основной цикл приложения"""
        self.root.mainloop()

    def ensure_russian_layout(self):
        """Проверяет раскладку клавиатуры активного окна и логирует предупреждение, если она не русская."""
        if not WIN32_AVAILABLE:
            logging.warning("Проверка раскладки недоступна: pywin32 не установлен.")
            return

        try:
            hwnd = ctypes.windll.user32.GetForegroundWindow()
            if hwnd == 0:
                logging.warning("Не удалось получить активное окно для проверки раскладки.")
                return
                
            thread_id = ctypes.windll.user32.GetWindowThreadProcessId(hwnd, 0)
            layout_id = ctypes.windll.user32.GetKeyboardLayout(thread_id)
            # HKL (Keyboard Layout Handle) младшее слово содержит Language ID.
            # 0x0419 - Русский
            lang_id = layout_id & 0xFFFF
            
            if lang_id != 0x0419:  # Не русский язык
                logging.warning(f"Обнаружена не русская раскладка клавиатуры (ID: {hex(lang_id)}). Ввод текста может быть некорректным.")
            else:
                logging.info("Обнаружена русская раскладка клавиатуры.")
                
        except Exception as e:
            logging.error(f"Ошибка при проверке раскладки клавиатуры: {e}")

    def setup_recording_indicator(self):
        """
        Создаёт и настраивает индикатор записи - мигающий красный прямоугольник,
        показывающий, что идёт запись голоса.
        """
        # Создаём всплывающее окно
        self.recording_indicator = Toplevel(self.root)
        self.recording_indicator.overrideredirect(True)  # Убираем рамки окна
        self.recording_indicator.attributes('-topmost', True)  # Всегда поверх других окон
        
        # Задаем размер и начальный цвет фона
        self.indicator_width = 20
        self.indicator_height = 5
        self.indicator_on_color = "red"
        # Используем системный цвет фона окна как цвет "выключения" для мигания
        # Чтобы сделать его полупрозрачным или использовать прозрачность, 
        # можно установить цвет 'black' и использовать '-transparentcolor'
        self.indicator_off_color = self.root.cget('bg') # Или 'black' для прозрачности
        # self.recording_indicator.attributes('-transparentcolor', 'black')
        
        self.recording_indicator.configure(bg=self.indicator_on_color)
        # Начальная геометрия (позиция будет обновляться при старте записи)
        self.recording_indicator.geometry(f"{self.indicator_width}x{self.indicator_height}+0+0") 
        
        self.recording_indicator.withdraw()  # Изначально скрываем индикатор

    def blink_indicator(self):
        # Функция мигания красного прямоугольника
        if not self.is_recording:
            # Убедимся, что индикатор скрыт и имеет цвет "включено" для следующего показа
            self.recording_indicator.withdraw()
            self.recording_indicator.configure(bg=self.indicator_on_color) 
            return
            
        # Проверяем, видимо ли окно (на всякий случай)
        if not self.recording_indicator.winfo_viewable():
             self.recording_indicator.configure(bg=self.indicator_on_color)
             return # Не мигаем, если скрыто

        try:
            current_color = self.recording_indicator.cget("bg")
            new_color = self.indicator_off_color if current_color == self.indicator_on_color else self.indicator_on_color
            self.recording_indicator.configure(bg=new_color)
            # Планируем следующее мигание
            self.recording_indicator.after(500, self.blink_indicator)
        except TclError as e:
            # Окно может быть уже уничтожено
            logging.warning(f"Ошибка при мигании индикатора (возможно, окно закрыто): {e}")

    # Методы вставки текста
    def insert_text(self, text):
        """Вставка текста без использования буфера обмена"""
        try:
            # Убираем лишние пробелы и добавляем один пробел в конце
            text = text.rstrip() + ' '
            logging.info(f"Текст для вставки: '{text}'")
            # Активируем окно, симулируя клик для восстановления фокуса ввода
            pyautogui.click()
            
            # Используем метод ввода из конфигурации
            if self.config['settings'].get('input_method', 'direct_input') == 'direct_input' and WIN32_AVAILABLE:
                self.insert_via_direct_input(text)
            else:
                self.insert_via_keyboard(text)
        except Exception as e:
            logging.error(f"Ошибка вставки текста: {e}")

    def insert_via_keyboard(self, text):
        """Вставка текста через эмуляцию клавиатуры с батчингом"""
        try:
            logging.info("Вставка текста через эмуляцию клавиатуры")
            kb = Controller()
            
            # Разбиваем на блоки для более эффективного ввода
            batch_size = 5  # Вводим по 5 символов за раз
            for i in range(0, len(text), batch_size):
                batch = text[i:i+batch_size]
                kb.type(batch)
                # Небольшая задержка между батчами
                time.sleep(self.char_delay)
                
            logging.info(f"Текст '{text}' вставлен через эмуляцию клавиатуры")
        except Exception as e:
            logging.error(f"Ошибка вставки текста через клавиатуру: {e}")

    def insert_via_direct_input(self, text):
        """Вставка текста напрямую через WinAPI с оптимизацией батчинга"""
        try:
            if not WIN32_AVAILABLE:
                raise Exception("WinAPI недоступен")
            
            hwnd = win32gui.GetForegroundWindow()
            if hwnd == 0:
                raise Exception("Не удалось получить активное окно")
            
            # Используем батчинг для более эффективной вставки
            batch_size = 5  # Размер пакета символов
            for i in range(0, len(text), batch_size):
                batch = text[i:i+batch_size]
                # Отправляем пакет символов
                for char in batch:
                    win32api.SendMessage(hwnd, win32con.WM_CHAR, ord(char), 0)
                # Задержка между батчами
                time.sleep(self.char_delay)
            
            logging.info(f"Текст '{text}' вставлен через WinAPI")
        except Exception as e:
            logging.error(f"Ошибка прямого ввода текста: {e}")
            # В случае ошибки используем метод с клавиатурой
            self.insert_via_keyboard(text)

    def get_model_size_from_config(self):
        """Получает размер модели из конфигурации или переменной окружения"""
        # Сначала проверяем переменную окружения
        model_size = os.environ.get("WHISPER_MODEL_SIZE")
        if model_size:
            logging.info(f"Используется размер модели из переменной окружения: {model_size}")
            return model_size
        
        # Затем проверяем конфигурацию
        try:
            if hasattr(self, 'config') and 'settings' in self.config and 'model_size' in self.config['settings']:
                model_size = self.config['settings']['model_size']
                return model_size
        except Exception as e:
            logging.error(f"Ошибка чтения размера модели из конфигурации: {e}")
        
        # По умолчанию используем large-v3-turbo
        return "large-v3-turbo"

    def load_or_create_config(self):
        """Загружает или создает файл конфигурации с обработкой ошибок"""
        config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.ini")
        
        try:
            if os.path.exists(config_path):
                self.config.read(config_path, encoding='utf-8')
                
                # Загружаем настройки, если секция существует
                if 'settings' in self.config:
                    config_settings = self.config['settings']
                    
                    # Используем get с значениями по умолчанию для безопасного извлечения
                    self.input_method = config_settings.get('input_method', 'keyboard')
                    self.char_delay = float(config_settings.get('char_delay', '0.03'))
                    self.remove_duplicates = config_settings.getboolean('remove_duplicates', True)
                    self.max_recording_seconds = int(config_settings.get('max_recording_seconds', '120'))
                    self.hotkey = config_settings.get('hotkey', 'f8')
                    self.current_performance_mode = config_settings.get('performance_mode', 'balanced')
                    
                    logging.info(f"Загружены настройки: метод ввода={self.input_method}, "
                                 f"задержка={self.char_delay}, "
                                 f"удаление дубликатов={self.remove_duplicates}, "
                                 f"горячая клавиша={self.hotkey}, "
                                 f"режим производительности={self.current_performance_mode}")
                else:
                    self._set_default_settings()
            else:
                # Создаем конфигурацию по умолчанию
                self._set_default_settings()
                self._save_config()
        except Exception as e:
            logging.error(f"Ошибка загрузки конфигурации: {e}")
            self._set_default_settings()

    def _set_default_settings(self):
        """Устанавливает настройки по умолчанию"""
        if 'settings' not in self.config:
            self.config['settings'] = {}
        
        # Устанавливаем прямой ввод как метод по умолчанию
        self.input_method = 'direct_input' 
        self.char_delay = 0.02  # Уменьшаем задержку для более быстрого ввода
        self.remove_duplicates = True
        self.max_recording_seconds = 60  # Уменьшаем до 1 минуты для более быстрой обработки
        self.hotkey = 'f8'  # Горячая клавиша по умолчанию
        self.current_performance_mode = 'balanced'  # Режим по умолчанию
        
        self.config['settings']['input_method'] = self.input_method
        self.config['settings']['char_delay'] = str(self.char_delay)
        self.config['settings']['remove_duplicates'] = str(self.remove_duplicates)
        self.config['settings']['model_size'] = 'large-v3'  # Используем large-v3 для лучшего баланса
        self.config['settings']['language'] = 'ru'
        self.config['settings']['max_recording_seconds'] = str(self.max_recording_seconds)
        self.config['settings']['hotkey'] = self.hotkey
        self.config['settings']['performance_mode'] = self.current_performance_mode
        self.config['settings']['vad_threshold'] = '0.005'
        self.config['settings']['batch_size'] = '5'
        self.config['settings']['max_workers'] = '2'
        self.config['settings']['cpu_threads'] = '8'  # Увеличиваем для лучшей производительности
        
        # Добавляем секцию для нежелательных фраз, если её нет
        if 'unwanted_phrases' not in self.config:
            self.config['unwanted_phrases'] = {}
            self.config['unwanted_phrases']['phrase1'] = "Субтитры создавал DimaTorzok"
            self.config['unwanted_phrases']['phrase2'] = "Продолжение следует..."
            self.config['unwanted_phrases']['phrase3'] = "Редактор субтитров А.Семкин Корректор А.Егорова"
            self.config['unwanted_phrases']['phrase4'] = "Текст на русском языке"
            self.config['unwanted_phrases']['phrase5'] = "Субтитры сделал DimaTorzok"
            self.config['unwanted_phrases']['phrase6'] = "Редактор субтитров М.Лосева Корректор А.Егорова"
            self.config['unwanted_phrases']['phrase7'] = "Текст на английском."
            self.config['unwanted_phrases']['phrase8'] = "Редактор субтитров А.Синецкая Корректор А.Егорова"
            self.config['unwanted_phrases']['phrase9'] = "Редактор субтитров Т.Горелова Корректор А.Егорова"
            self.config['unwanted_phrases']['phrase10'] = "Редактор субтитров Е.Жукова Корректор А.Егорова"
            self.config['unwanted_phrases']['phrase11'] = "Смотрите продолжение во второй части видео"
            self.config['unwanted_phrases']['phrase12'] = "Смотрите продолжение в следующей части" 
            self.config['unwanted_phrases']['phrase13'] = "Смотрите продолжение в следующей части видео"
            self.config['unwanted_phrases']['phrase14'] = "Смотрите продолжение в 4 части видео"
            self.config['unwanted_phrases']['phrase15'] = "Смотрите продолжение в следующей серии..."
            self.config['unwanted_phrases']['phrase16'] = "Смотрите продолжение во второй части"
            self.config['unwanted_phrases']['phrase17'] = "Спасибо за субтитры!"
            self.config['unwanted_phrases']['phrase18'] = "Субтитры добавил DimaTorzok"
            self.config['unwanted_phrases']['phrase19'] = "Редактор субтитров А.Семкин"
            self.config['unwanted_phrases']['phrase20'] = "Спасибо Спасибо"
            self.config['unwanted_phrases']['phrase21'] = "Это текст на русском языке"         
            self.config['unwanted_phrases']['phrase22'] = "Субтитры подогнал «Симон»"

    def _save_config(self):
        """Сохраняет конфигурацию в файл"""
        config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.ini")
        try:
            with open(config_path, 'w', encoding='utf-8') as f:
                self.config.write(f)
        except Exception as e:
            logging.error(f"Ошибка сохранения конфигурации: {e}")

    def filter_unwanted_phrases(self, text):
        """Удаляет нежелательные фразы из распознанного текста"""
        try:
            unwanted_phrases = []
            # Загружаем фразы из конфигурации, если секция существует
            if 'unwanted_phrases' in self.config:
                unwanted_phrases = [phrase for key, phrase in self.config['unwanted_phrases'].items() if phrase]
            
            # Если список пуст, ничего не делаем
            if not unwanted_phrases:
                return text

            # Удаляем каждую нежелательную фразу из текста
            original_text = text
            for phrase in unwanted_phrases:
                # Используем re.escape для корректной обработки спецсимволов в фразах
                text = text.replace(phrase, "")
            
            # Удаляем лишние пробелы, которые могли образоваться после удаления фраз
            text = re.sub(r'\s+', ' ', text).strip()
            
            # Логируем, если были произведены изменения
            if original_text != text:
                logging.info(f"Удалены нежелательные фразы. Было: '{original_text}' Стало: '{text}'")
            
            return text
        except Exception as e:
            logging.error(f"Ошибка при фильтрации нежелательных фраз: {e}")
            return text  # Возвращаем оригинальный текст в случае ошибки

    def register_hotkey(self):
        """Регистрирует горячую клавишу из конфигурации"""
        try:
            # Отменяем предыдущую регистрацию горячей клавиши, если она была
            if hasattr(self, 'hotkey') and self.hotkey:
                keyboard.unhook_key(self.hotkey)
            
            # Регистрируем новую горячую клавишу
            self.hotkey = self.config['settings'].get('hotkey', 'f8').lower()
            logging.info(f"Регистрация горячей клавиши: {self.hotkey}")
            keyboard.on_press_key(self.hotkey, self.toggle_recording)
        except Exception as e:
            logging.error(f"Ошибка при регистрации горячей клавиши: {e}")
            # В случае ошибки используем F8 как резервную клавишу
            logging.info("Используем F8 как резервную горячую клавишу")
            self.hotkey = 'f8'
            keyboard.on_press_key('f8', self.toggle_recording)

    def update_settings(self):
        """Обновляет настройки приложения и применяет их"""
        try:
            # Сохраняем конфигурацию в файл
            self._save_config()
            
            # Перерегистрируем горячую клавишу, если она изменилась
            self.register_hotkey()
            
            logging.info("Настройки успешно обновлены")
        except Exception as e:
            logging.error(f"Ошибка при обновлении настроек: {e}")

if __name__ == "__main__":
    # Скрываем консоль при запуске через python.exe
    if WIN32_AVAILABLE and not hasattr(sys, 'frozen'):
        try:
            hwnd = ctypes.windll.kernel32.GetConsoleWindow()
            if hwnd != 0:
                ctypes.windll.user32.ShowWindow(hwnd, 0)  # SW_HIDE = 0
        except Exception as e:
            logging.error(f"Ошибка при скрытии консоли: {e}")
    
    app = TranscriberApp()
    app.run() 