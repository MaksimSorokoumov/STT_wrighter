"""
Модуль с функциями для управления графическим интерфейсом
"""
import logging
import threading
from tkinter import Toplevel, TclError
from PIL import Image
from pystray import MenuItem as item, Icon

def setup_recording_indicator(root):
    """
    Создаёт и настраивает индикатор записи - мигающий красный прямоугольник,
    показывающий, что идёт запись голоса.
    """
    # Создаём всплывающее окно
    recording_indicator = Toplevel(root)
    recording_indicator.overrideredirect(True)  # Убираем рамки окна
    recording_indicator.attributes('-topmost', True)  # Всегда поверх других окон
    
    # Задаем размер и начальный цвет фона
    indicator_width = 20
    indicator_height = 5
    indicator_on_color = "red"
    # Используем системный цвет фона окна как цвет "выключения" для мигания
    indicator_off_color = root.cget('bg')
    
    recording_indicator.configure(bg=indicator_on_color)
    # Начальная геометрия (позиция будет обновляться при старте записи)
    recording_indicator.geometry(f"{indicator_width}x{indicator_height}+0+0") 
    
    recording_indicator.withdraw()  # Изначально скрываем индикатор
    
    return recording_indicator, indicator_on_color, indicator_off_color

def blink_indicator(recording_indicator, indicator_on_color, indicator_off_color, is_recording):
    """Функция мигания красного прямоугольника"""
    if not is_recording:
        # Убедимся, что индикатор скрыт и имеет цвет "включено" для следующего показа
        recording_indicator.withdraw()
        recording_indicator.configure(bg=indicator_on_color) 
        return
        
    # Проверяем, видимо ли окно (на всякий случай)
    if not recording_indicator.winfo_viewable():
         recording_indicator.configure(bg=indicator_on_color)
         return # Не мигаем, если скрыто

    try:
        current_color = recording_indicator.cget("bg")
        new_color = indicator_off_color if current_color == indicator_on_color else indicator_on_color
        recording_indicator.configure(bg=new_color)
        # Планируем следующее мигание
        recording_indicator.after(500, lambda: blink_indicator(
            recording_indicator, indicator_on_color, indicator_off_color, is_recording))
    except TclError as e:
        # Окно может быть уже уничтожено
        logging.warning(f"Ошибка при мигании индикатора (возможно, окно закрыто): {e}")

def setup_systray(app, icon_path="icon.png"):
    """Настраивает иконку в системном трее с дополнительными опциями"""
    try:
        image = Image.open(icon_path)
        
        # Создаем меню с выбором режима производительности
        menu = (
            item('Режим: Быстрый', app.switch_to_fast_mode, 
                 checked=lambda _: app.current_performance_mode == 'fast'),
            item('Режим: Сбалансированный', app.switch_to_balanced_mode, 
                 checked=lambda _: app.current_performance_mode == 'balanced'),
            item('Режим: Точный', app.switch_to_accurate_mode, 
                 checked=lambda _: app.current_performance_mode == 'accurate'),
            item('Выход', app.quit_app),
        )
        
        icon = Icon("STT", image, "Голосовой транскрайбер", menu)
        threading.Thread(target=icon.run, daemon=True).start()
        return icon
    except Exception as e:
        logging.error(f"Ошибка настройки системного трея: {e}")
        raise 