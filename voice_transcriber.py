import sys
import keyboard
import sounddevice as sd
import numpy as np
from tkinter import Tk
import pyautogui
import threading
import os
import time
import queue
from concurrent.futures import ThreadPoolExecutor
import logging
import ctypes
import traceback

# Настраиваем логирование
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler('app.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)

# Проверяем импорт torch для логирования CUDA
try:
    import torch
    HAS_TORCH = True
    logging.info(f"PyTorch версия: {torch.__version__}")
except ImportError:
    HAS_TORCH = False
    logging.error("PyTorch не установлен!")

# Импортируем созданные модули
from config_module import PERFORMANCE_MODES, load_or_create_config, save_config
from audio_processing import SimpleVAD, normalize_audio, filter_unwanted_phrases
from text_input import insert_text, ensure_russian_layout
from gui_utils import setup_recording_indicator, blink_indicator, setup_systray
from speech_recognition import initialize_whisper_model, transcribe_audio

# Проверка доступности Windows API
try:
    import win32con
    WIN32_AVAILABLE = True
    
    # Скрываем консоль при запуске
    if hasattr(sys, 'frozen'):  # Проверяем, запущено ли приложение как exe
        hwnd = ctypes.windll.kernel32.GetConsoleWindow()
        if hwnd != 0:
            ctypes.windll.user32.ShowWindow(hwnd, 0)  # SW_HIDE = 0
except ImportError:
    WIN32_AVAILABLE = False
    logging.warning("pywin32 не установлен. Некоторые функции будут недоступны")

# Логируем запуск приложения
logging.info("=== Запуск приложения ===")
logging.info(f"Рабочая директория: {os.getcwd()}")

# Логируем информацию о CUDA, если torch доступен
if HAS_TORCH:
    logging.info(f"Доступность CUDA: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logging.info(f"Версия CUDA: {torch.version.cuda}")
        logging.info(f"Устройство CUDA: {torch.cuda.get_device_name(0)}")
else:
    logging.warning("PyTorch не установлен, CUDA недоступен")

class TranscriberApp:
    def __init__(self):
        # Загружаем настройки перед инициализацией модели
        self.config = load_or_create_config()
        
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
        
        # Настройка индикатора записи
        self.recording_indicator, self.indicator_on_color, self.indicator_off_color = setup_recording_indicator(self.root)
        
        # Регистрация настраиваемой горячей клавиши для переключения записи
        self.register_hotkey()
        
        # Добавляем обработку ошибок для иконки
        try:
            self.icon = setup_systray(self)
        except Exception as icon_error:
            logging.error(f"Ошибка при настройке системного трея: {str(icon_error)}")
            # Создаем запасной вариант без иконки
            self.root.protocol("WM_DELETE_WINDOW", self.quit_app)
            self.root.title("Голосовой транскрайбер")
            self.root.deiconify()  # Показываем основное окно как запасной вариант

    def initialize_whisper_model(self):
        """Инициализирует модель Whisper для CUDA используя кеширование"""
        try:
            # Получаем параметры из текущего режима
            mode_settings = PERFORMANCE_MODES[self.current_performance_mode]
            model_size = mode_settings['model_size']
            compute_type = mode_settings['compute_type']
            
            # Создаем ключ для кеша
            cache_key = f"{model_size}_{compute_type}"
            
            # Проверяем, есть ли модель в кеше
            if cache_key in self.model_cache:
                logging.info(f"Используется кешированная модель: {cache_key}")
                self.model = self.model_cache[cache_key]
            else:
                # Создаем новую модель
                cpu_threads = int(self.config['settings'].get('cpu_threads', '6'))
                self.model = initialize_whisper_model(
                    model_size=model_size,
                    compute_type=compute_type,
                    cpu_threads=cpu_threads
                )
                # Кешируем модель для возможного повторного использования
                self.model_cache[cache_key] = self.model
                
        except Exception as e:
            logging.error(f"Ошибка инициализации модели Whisper: {str(e)}")
            logging.error(traceback.format_exc())
            # Запасной вариант
            cpu_threads = int(self.config['settings'].get('cpu_threads', '6'))
            self.model = initialize_whisper_model("medium", "int8", cpu_threads)

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
        save_config(self.config)
        
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
                blink_indicator(self.recording_indicator, self.indicator_on_color, 
                               self.indicator_off_color, self.is_recording)
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
            # Сохраняем cursor_position перед вызовом stop_recording, так как
            # self.cursor_position должен сохраниться с момента начала записи
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
        audio_np = normalize_audio(audio_np)
        
        try:
            # Получаем параметры из текущего режима производительности
            mode_settings = PERFORMANCE_MODES[self.current_performance_mode]
            beam_size = mode_settings['beam_size']
            vad_filter = mode_settings['vad_filter']
            
            # Транскрибируем аудио
            text = transcribe_audio(
                model=self.model,
                audio_np=audio_np,
                language='ru',
                beam_size=beam_size,
                vad_filter=False  # Используем свой VAD
            )
            
            logging.info(f"Распознанный текст: {text}")
            
            if text:
                # Получаем список нежелательных фраз
                unwanted_phrases = [phrase for key, phrase in self.config['unwanted_phrases'].items() if phrase]
                
                # Фильтруем нежелательные фразы перед вставкой
                text = filter_unwanted_phrases(text, unwanted_phrases)
                
                # Выполняем вставку текста
                input_method = self.config['settings'].get('input_method', 'direct_input')
                char_delay = float(self.config['settings'].get('char_delay', PERFORMANCE_MODES[self.current_performance_mode]['char_delay']))
                
                # Проверяем раскладку клавиатуры перед вставкой
                ensure_russian_layout()
                
                # Вставляем текст (не передаем позицию курсора)
                insert_text(
                    text=text,
                    input_method=input_method,
                    char_delay=char_delay
                )
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
            audio_np = normalize_audio(audio_np)
                
            # Получаем параметры из текущего режима производительности
            mode_settings = PERFORMANCE_MODES[self.current_performance_mode]
            beam_size = mode_settings['beam_size']
            
            # Транскрибируем аудио
            text = transcribe_audio(
                model=self.model,
                audio_np=audio_np,
                language='ru',
                beam_size=beam_size,
                vad_filter=False
            )
            
            logging.info(f"Распознано (до фильтрации): {text}")
            
            if text:
                # Получаем список нежелательных фраз
                unwanted_phrases = [phrase for key, phrase in self.config['unwanted_phrases'].items() if phrase]
                
                # Фильтруем нежелательные фразы
                text = filter_unwanted_phrases(text, unwanted_phrases)
                
                logging.info(f"Распознано (после фильтрации): {text}")
                
                # Выполняем вставку текста
                input_method = self.config['settings'].get('input_method', 'direct_input')
                char_delay = float(self.config['settings'].get('char_delay', PERFORMANCE_MODES[self.current_performance_mode]['char_delay']))
                
                # Проверяем раскладку клавиатуры перед вставкой
                ensure_russian_layout()
                
                # Вставляем текст (не передаем позицию курсора)
                insert_text(
                    text=text,
                    input_method=input_method,
                    char_delay=char_delay
                )
                
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
                # Начинаем запись и сохраняем текущую позицию курсора
                self.start_recording(event)
            else:
                # Останавливаем запись, позиция курсора уже сохранена
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
            # cursor_position уже сохранен с момента начала записи
            self.stop_recording(None)

    def run(self):
        """Запускает основной цикл приложения"""
        self.root.mainloop()

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

if __name__ == "__main__":
    # Скрываем консоль при запуске через python.exe
    if WIN32_AVAILABLE and not hasattr(sys, 'frozen'):
        try:
            hwnd = ctypes.windll.kernel32.GetConsoleWindow()
            if hwnd != 0:
                ctypes.windll.user32.ShowWindow(hwnd, 0)  # SW_HIDE = 0
        except Exception as e:
            logging.error(f"Ошибка при скрытии консоли: {e}")
    
    # Запускаем приложение
    app = TranscriberApp()
    app.run() 