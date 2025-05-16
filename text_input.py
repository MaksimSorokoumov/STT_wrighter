"""
Модуль для ввода текста различными методами: прямой ввод или эмуляция клавиатуры
"""
import time
import logging
import pyautogui
from pynput.keyboard import Controller

# Проверка доступности Windows API
try:
    import win32con
    import win32api
    import win32gui
    WIN32_AVAILABLE = True
except ImportError:
    WIN32_AVAILABLE = False
    logging.warning("pywin32 не установлен. Прямой ввод через WinAPI будет недоступен")

def insert_text(text, cursor_position=None, input_method='direct_input', char_delay=0.02):
    """Вставка текста без использования буфера обмена"""
    try:
        # Убираем лишние пробелы и добавляем один пробел в конце
        text = text.rstrip() + ' '
        logging.info(f"Текст для вставки: '{text}'")
        
        # Мы НЕ используем клик мыши, чтобы не сбрасывать текстовый курсор в документе
        # Текстовый курсор остается там, где был до нажатия F8
        
        # Используем метод ввода из конфигурации
        if input_method == 'direct_input' and WIN32_AVAILABLE:
            insert_via_direct_input(text, char_delay)
        else:
            insert_via_keyboard(text, char_delay)
    except Exception as e:
        logging.error(f"Ошибка вставки текста: {e}")

def insert_via_keyboard(text, char_delay=0.02):
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
            time.sleep(char_delay)
            
        logging.info(f"Текст '{text}' вставлен через эмуляцию клавиатуры")
    except Exception as e:
        logging.error(f"Ошибка вставки текста через клавиатуру: {e}")

def insert_via_direct_input(text, char_delay=0.02):
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
            time.sleep(char_delay)
        
        logging.info(f"Текст '{text}' вставлен через WinAPI")
    except Exception as e:
        logging.error(f"Ошибка прямого ввода текста: {e}")
        # В случае ошибки используем метод с клавиатурой
        insert_via_keyboard(text, char_delay)

def ensure_russian_layout():
    """Проверяет раскладку клавиатуры активного окна и логирует предупреждение, если она не русская."""
    if not WIN32_AVAILABLE:
        logging.warning("Проверка раскладки недоступна: pywin32 не установлен.")
        return

    try:
        import ctypes
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