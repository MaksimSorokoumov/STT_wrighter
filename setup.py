import os
import sys
import subprocess
import platform
import venv
import logging
import shutil
import time
import winshell
from pathlib import Path
import ctypes
from win32com.client import Dispatch
from PIL import Image

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler('setup.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)

def is_admin():
    """Проверяет, запущен ли скрипт с правами администратора"""
    try:
        return ctypes.windll.shell32.IsUserAnAdmin()
    except:
        return False

def create_virtual_env():
    """Создает виртуальное окружение Python"""
    logging.info("Проверка виртуального окружения...")
    
    venv_dir = Path("venv")
    
    if venv_dir.exists():
        logging.info("Обнаружено существующее виртуальное окружение")
        print("\n=== ВНИМАНИЕ ===")
        print("Обнаружено существующее виртуальное окружение в папке 'venv'.")
        print("Для корректной установки рекомендуется удалить его и создать заново.")
        
        choice = input("\nУдалить существующее окружение и продолжить установку? (д/н): ").lower()
        
        if choice in ['д', 'да', 'y', 'yes']:
            logging.info("Удаление существующего виртуального окружения...")
            try:
                # Используем shutil.rmtree для удаления директории
                shutil.rmtree(venv_dir, ignore_errors=True)
                time.sleep(1)  # Даем время на завершение процесса удаления
                
                if venv_dir.exists():
                    logging.error("Не удалось полностью удалить виртуальное окружение")
                    print("\nОшибка: Не удалось удалить виртуальное окружение.")
                    print("Пожалуйста, закройте все программы, которые могут использовать файлы в папке venv,")
                    print("и попробуйте запустить установку снова.")
                    sys.exit(1)
                
                logging.info("Существующее виртуальное окружение успешно удалено")
            except Exception as e:
                logging.error(f"Ошибка при удалении виртуального окружения: {str(e)}")
                print(f"\nОшибка при удалении виртуального окружения: {str(e)}")
                print("Пожалуйста, удалите папку 'venv' вручную и запустите установку снова.")
                sys.exit(1)
        else:
            logging.info("Пользователь отказался от удаления существующего окружения")
            print("\nУстановка будет продолжена с использованием существующего окружения.")
            print("Это может привести к ошибкам, если окружение повреждено или несовместимо.")
            time.sleep(2)  # Даем пользователю время прочитать сообщение
            return str(venv_dir)
    
    # Создаем новое виртуальное окружение
    try:
        logging.info("Создание нового виртуального окружения...")
        venv.create(venv_dir, with_pip=True)
        logging.info(f"Виртуальное окружение успешно создано в {venv_dir}")
        return str(venv_dir)
    except Exception as e:
        logging.error(f"Ошибка создания виртуального окружения: {str(e)}")
        print(f"\nОшибка создания виртуального окружения: {str(e)}")
        sys.exit(1)

def get_python_executable(venv_path):
    """Возвращает путь к исполняемому файлу Python в виртуальном окружении"""
    if platform.system() == "Windows":
        # Сначала проверяем стандартный путь к Python в venv
        python_path = os.path.join(venv_path, "Scripts", "python.exe")
        if os.path.exists(python_path):
            logging.info(f"Найден Python в виртуальном окружении: {python_path}")
            return python_path
            
        # Проверяем py launcher
        try:
            result = subprocess.run(["py", "-3.11", "--version"], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                logging.info("Используем Python через py launcher")
                return "py -3.11"
        except FileNotFoundError:
            logging.warning("py launcher не найден")
        
        # Ищем Python в PATH
        python_exe = shutil.which("python")
        if python_exe:
            # Проверяем версию
            try:
                version_result = subprocess.run([python_exe, "--version"], 
                                              capture_output=True, text=True)
                if "Python 3.11" in version_result.stdout or "Python 3.11" in version_result.stderr:
                    logging.info(f"Используем системный Python 3.11: {python_exe}")
                    return python_exe
            except Exception as e:
                logging.warning(f"Ошибка проверки версии Python: {e}")
        
        # Поиск в стандартных местах установки
        logging.info("Поиск Python 3.11 в стандартных местах установки")
        standard_paths = [
            os.path.join(os.environ.get("ProgramFiles", "C:\\Program Files"), "Python311", "python.exe"),
            os.path.join(os.environ.get("ProgramFiles(x86)", "C:\\Program Files (x86)"), "Python311", "python.exe"),
            "C:\\Python311\\python.exe",
            os.path.join(os.environ.get("LOCALAPPDATA", ""), "Programs", "Python", "Python311", "python.exe"),
            os.path.join(os.path.expanduser("~"), "AppData", "Local", "Programs", "Python", "Python311", "python.exe")
        ]
        
        for path in standard_paths:
            if os.path.exists(path):
                logging.info(f"Найден Python 3.11 по пути: {path}")
                return path
        
        logging.error("Не удалось найти Python 3.11 в системе")
        sys.exit(1)
    else:
        return os.path.join(venv_path, "bin", "python")

def install_dependencies(python_exe):
    """Устанавливает зависимости из requirements.txt"""
    logging.info("Установка зависимостей...")
    
    try:
        # Проверяем работоспособность Python
        subprocess.run([python_exe, "--version"], 
                      capture_output=True, text=True, check=True)
                      
        # Обновляем pip
        logging.info("Обновление pip...")
        subprocess.run([python_exe, "-m", "pip", "install", "--upgrade", "pip"],
                      capture_output=True, text=True, check=True)
        
        # Удаление текущего PyTorch
        subprocess.run([python_exe, "-m", "pip", "uninstall", "-y", "torch"], 
                       capture_output=True, text=True, check=False)
        
        # Установка PyTorch с CUDA
        logging.info("Установка PyTorch с поддержкой CUDA...")
        subprocess.run([python_exe, "-m", "pip", "install", "torch", "--index-url", 
                       "https://download.pytorch.org/whl/cu121"], 
                       capture_output=True, text=True, check=True)
        
        # Установка pywin32 отдельно и запуск скрипта постустановки
        logging.info("Установка pywin32...")
        subprocess.run([python_exe, "-m", "pip", "install", "--upgrade", "pywin32"],
                      capture_output=True, text=True, check=True)
        
        # Создаем скрипт для регистрации pywin32
        with open("register_pywin32.py", "w", encoding="utf-8") as f:
            f.write("""
import sys
import os
import site

# Добавляем пути к pywin32
site_packages = site.getsitepackages()[0]
pywin32_path = os.path.join(site_packages, "pywin32_system32")
if os.path.exists(pywin32_path):
    sys.path.append(pywin32_path)

# Проверяем импорт
try:
    import win32api
    print("win32api успешно импортирован")
except ImportError as e:
    print(f"Ошибка импорта win32api: {e}")
    
print("Пути Python:", sys.path)
            """)
        
        # Запускаем скрипт регистрации
        logging.info("Регистрация pywin32...")
        subprocess.run([python_exe, "register_pywin32.py"],
                      capture_output=False, check=True)
        
        # Установка остальных зависимостей
        logging.info("Установка остальных зависимостей...")
        subprocess.run([python_exe, "-m", "pip", "install", "-r", "requirements.txt"], 
                       capture_output=True, text=True, check=True)
                       
        logging.info("Зависимости успешно установлены")
    except subprocess.CalledProcessError as e:
        logging.error(f"Ошибка установки зависимостей: {e.stderr}")
        sys.exit(1)
    finally:
        if os.path.exists("register_pywin32.py"):
            os.remove("register_pywin32.py")

def download_models(python_exe):
    """Скачивает модели Whisper в папку models"""
    logging.info("Скачивание моделей Whisper...")
    
    models_dir = Path("models")
    if not models_dir.exists():
        models_dir.mkdir(parents=True)
    
    # Создаем временный скрипт для загрузки модели
    download_script = """
import os
from faster_whisper import WhisperModel

# Установка переменной окружения для указания пути к моделям
os.environ["WHISPER_MODELS_DIR"] = os.path.abspath("models")

# Создаем экземпляр модели, что автоматически загрузит её в указанную директорию
model_size = "large-v3-turbo"  # Изменено на turbo-версию
print(f"Загрузка модели {model_size}...")
model = WhisperModel(model_size, device="cpu", download_root="models")
print(f"Модель {model_size} успешно загружена в папку models")
"""
    
    with open("download_model.py", "w", encoding="utf-8") as f:
        f.write(download_script)
    
    try:
        logging.info("Запуск скрипта загрузки моделей...")
        subprocess.run([python_exe, "download_model.py"], 
                      capture_output=True, text=True, check=True)
        logging.info("Модели успешно загружены")
    except subprocess.CalledProcessError as e:
        logging.error(f"Ошибка загрузки моделей: {e.stderr}")
        sys.exit(1)
    finally:
        if os.path.exists("download_model.py"):
            os.remove("download_model.py")

def create_shortcut(python_exe):
    """Создает ярлык на рабочем столе для запуска без консольного окна"""
    logging.info("Создание ярлыка на рабочем столе...")
    
    try:
        # Используем pythonw.exe вместо python.exe для запуска без консоли
        pythonw_exe = python_exe.replace("python.exe", "pythonw.exe")
        
        # Путь к исполняемому файлу Python и скрипту
        target_script = os.path.abspath("voice_transcriber.py")
        
        # Создаем VBS-скрипт для запуска без консоли
        vbs_path = os.path.abspath("run_voice_transcriber.vbs")
        with open(vbs_path, "w", encoding="utf-8") as vbs_file:
            vbs_file.write('Set WshShell = CreateObject("WScript.Shell")\n')
            vbs_file.write(f'WshShell.Environment("Process")("WHISPER_MODELS_DIR") = "models"\n')
            vbs_file.write(f'WshShell.Run """venv\\Scripts\\pythonw.exe"" ""voice_transcriber.py""", 0, False')
        
        # Создаем также BAT-файл как запасной вариант
        bat_path = os.path.abspath("run_app.bat")
        with open(bat_path, "w", encoding="utf-8") as bat_file:
            bat_file.write("@echo off\n")
            bat_file.write("set WHISPER_MODELS_DIR=models\n")
            bat_file.write("venv\\Scripts\\pythonw.exe voice_transcriber.py\n")
        
        # Создаем ICO файл из PNG (если PNG существует)
        icon_path = os.path.abspath("icon.png")
        ico_path = os.path.abspath("app_icon.ico")
        
        if os.path.exists(icon_path):
            try:
                img = Image.open(icon_path)
                img.save(ico_path, format='ICO', sizes=[(32, 32)])
                logging.info(f"Создан ICO файл: {ico_path}")
                icon_for_shortcut = ico_path
            except Exception as icon_error:
                logging.error(f"Ошибка создания ICO: {str(icon_error)}")
                icon_for_shortcut = icon_path
        else:
            logging.warning(f"Файл иконки не найден: {icon_path}")
            icon_for_shortcut = ""
        
        # Путь к рабочему столу
        desktop = Path(winshell.desktop())
        shortcut_path = desktop / "Голосовой транскрайбер.lnk"
        
        # Создаем ярлык
        shell = Dispatch('WScript.Shell')
        shortcut = shell.CreateShortCut(str(shortcut_path))
        shortcut.Targetpath = vbs_path  # Можно использовать bat_path вместо vbs_path
        shortcut.WorkingDirectory = os.path.dirname(os.path.abspath(vbs_path))
        if icon_for_shortcut:
            shortcut.IconLocation = icon_for_shortcut
        shortcut.Description = "Голосовой транскрайбер на основе Whisper"
        shortcut.save()
        
        logging.info(f"Ярлык создан на рабочем столе: {shortcut_path}")
    except Exception as e:
        logging.error(f"Ошибка создания ярлыка: {str(e)}")
        logging.warning("Ярлык не был создан, но установка продолжится")

def verify_installation(python_exe):
    """Проверяет корректность установки"""
    logging.info("Проверка установки...")
    
    try:
        # Запускаем временный скрипт для проверки импорта основных зависимостей
        verify_script = """
import torch
import numpy as np
import sounddevice
from faster_whisper import WhisperModel
import os

print(f"PyTorch версия: {torch.__version__}")
print(f"CUDA доступность: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA версия: {torch.version.cuda}")
    print(f"Устройство: {torch.cuda.get_device_name(0)}")

# Проверяем доступность папки с моделями
models_dir = os.path.abspath("models")
print(f"Папка моделей: {models_dir}")
print(f"Папка моделей существует: {os.path.exists(models_dir)}")
if os.path.exists(models_dir):
    print(f"Содержимое папки моделей: {os.listdir(models_dir)}")

print("Проверка установки успешно завершена!")
"""
        
        with open("verify_installation.py", "w", encoding="utf-8") as f:
            f.write(verify_script)
        
        result = subprocess.run([python_exe, "verify_installation.py"], 
                              capture_output=True, text=True, check=True)
        
        logging.info("Результаты проверки установки:")
        for line in result.stdout.splitlines():
            logging.info(f"  {line}")
        
        logging.info("Установка успешно завершена!")
    except subprocess.CalledProcessError as e:
        logging.error(f"Ошибка проверки установки: {e.stderr}")
    finally:
        if os.path.exists("verify_installation.py"):
            os.remove("verify_installation.py")

def main():
    logging.info("=== Начало установки голосового транскрайбера ===")
    
    # Проверка наличия прав администратора
    if not is_admin():
        logging.warning("Скрипт запущен без прав администратора. Некоторые функции могут не работать.")
    
    # Проверка наличия необходимых файлов
    required_files = ["voice_transcriber.py", "requirements.txt", "icon.png"]
    for file in required_files:
        if not os.path.exists(file):
            logging.error(f"Не найден необходимый файл: {file}")
            sys.exit(1)
    
    # Создание виртуального окружения
    venv_path = create_virtual_env()
    python_exe = get_python_executable(venv_path)
    
    # Установка зависимостей
    install_dependencies(python_exe)
    
    # Скачивание моделей
    download_models(python_exe)
    
    # Создание ярлыка
    create_shortcut(python_exe)
    
    # Проверка установки
    verify_installation(python_exe)
    
    logging.info("=== Установка успешно завершена! ===")
    logging.info("Вы можете запустить приложение через ярлык на рабочем столе")
    
    # Пауза перед закрытием окна
    print("\nУстановка завершена! Нажмите Enter для выхода...")
    input()

if __name__ == "__main__":
    main() 