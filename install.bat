@echo off
chcp 65001 > nul
:: Переходим в директорию скрипта
cd /d "%~dp0"
echo === Установка Голосового транскрайбера ===
echo.

:: Проверка прав администратора
net session >nul 2>&1
if %errorlevel% neq 0 (
    echo Требуются права администратора для установки!
    echo Пожалуйста, запустите этот файл от имени администратора.
    echo.
    echo Щелкните правой кнопкой мыши на файле install.bat
    echo и выберите "Запустить от имени администратора".
    echo.
    pause
    exit /b 1
)

:: Улучшенная проверка наличия Python
echo Проверка установки Python...
set PYTHON_FOUND=0
set PYTHON_EXE=

:: Проверка через py launcher
py --version >nul 2>&1
if %errorlevel% == 0 (
    set PYTHON_FOUND=1
    set PYTHON_EXE=py
)

:: Проверка через python.exe
where python >nul 2>&1
if %errorlevel% == 0 (
    set PYTHON_FOUND=1
    if not defined PYTHON_EXE (
        set PYTHON_EXE=python
    )
)

if %PYTHON_FOUND% == 0 (
    echo Python не найден!
    echo Убедитесь, что:
    echo 1. Python 3.11 установлен
    echo 2. При установке отмечена галочка "Add Python to PATH"
    echo 3. Вы перезапустили командную строку после установки Python
    echo.
    echo Скачать Python можно с официального сайта: https://www.python.org/downloads/
    echo.
    pause
    exit /b 1
)

:: Выбор версии Python
echo Выбор версии Python:
echo 1. Python 3.11
echo 2. Python 3.12
echo 3. Использовать версию Python по умолчанию
choice /c 123 /n /m "Выберите версию Python (1-3): "

if %errorlevel% == 1 (
    :: Улучшенная проверка Python 3.11
    echo Поиск Python 3.11...
    
    :: Проверяем через py launcher (самый надежный метод)
    py -3.11 --version >nul 2>&1
    if %errorlevel% == 0 (
        echo Найден Python 3.11 (py launcher)
        set PYTHON_CMD=py -3.11
        goto python_found
    )
    
    :: Проверяем через стандартные пути установки
    set FOUND_PATH=
    echo Поиск Python 3.11 в стандартных местах установки...
    
    :: Используем %USERPROFILE% вместо жесткого имени пользователя
    set PYTHON_PATHS=^
    "C:\Program Files\Python311\python.exe" ^
    "C:\Program Files (x86)\Python311\python.exe" ^
    "C:\Python311\python.exe" ^
    "%LOCALAPPDATA%\Programs\Python\Python311\python.exe" ^
    "%USERPROFILE%\AppData\Local\Programs\Python\Python311\python.exe"
    
    for %%p in (%PYTHON_PATHS%) do (
        if exist %%p (
            echo Найден Python 3.11 в: %%p
            set FOUND_PATH=%%p
            goto found_python_path
        )
    )
    
    :: Если не нашли в стандартных местах, проверяем системный Python
    for /f "tokens=* usebackq" %%x in (`where python 2^>nul`) do (
        set PYTHON_PATH=%%x
        :: Проверяем версию
        "%%x" --version 2>&1 | find "Python 3.11" >nul
        if !errorlevel! == 0 (
            echo Найден Python 3.11: %%x
            set FOUND_PATH=%%x
            goto found_python_path
        )
    )
    
    echo Python 3.11 не найден! Убедитесь, что он установлен и добавлен в PATH.
    echo.
    echo Возможные решения проблемы:
    echo 1. Полностью перезапустите компьютер после установки Python
    echo 2. Проверьте, отмечен ли флажок "Add Python to PATH" при установке
    echo 3. Укажите полный путь к Python в командной строке
    echo.
    echo Текущие пути в PATH:
    echo %PATH%
    echo.
    pause
    exit /b 1
    
    :found_python_path
    set PYTHON_CMD=%FOUND_PATH%
    goto python_found
    
) else if %errorlevel% == 2 (
    :: Проверка Python 3.12
    py -3.12 --version >nul 2>&1
    if %errorlevel% == 0 (
        echo Найден Python 3.12 (py launcher)
        set PYTHON_CMD=py -3.12
        goto python_found
    )
    
    :: То же самое для Python 3.12
    set FOUND_PATH=
    echo Поиск Python 3.12 в стандартных местах установки...
    
    set PYTHON_PATHS=^
    "C:\Program Files\Python312\python.exe" ^
    "C:\Program Files (x86)\Python312\python.exe" ^
    "C:\Python312\python.exe" ^
    "%LOCALAPPDATA%\Programs\Python\Python312\python.exe" ^
    "%USERPROFILE%\AppData\Local\Programs\Python\Python312\python.exe"
    
    for %%p in (%PYTHON_PATHS%) do (
        if exist %%p (
            echo Найден Python 3.12 в: %%p
            set FOUND_PATH=%%p
            goto found_python312_path
        )
    )
    
    echo Python 3.12 не найден! Убедитесь, что он установлен и добавлен в PATH.
    pause
    exit /b 1
    
    :found_python312_path
    set PYTHON_CMD=%FOUND_PATH%
    goto python_found
) else (
    :: Использовать Python по умолчанию
    if defined PYTHON_EXE (
        set PYTHON_CMD=%PYTHON_EXE%
    ) else (
        set PYTHON_CMD=python
    )
)

:python_found
echo Используем: %PYTHON_CMD%

:: Проверка версии Python
%PYTHON_CMD% --version
if %errorlevel% neq 0 (
    echo Ошибка: Не удалось запустить Python. Проверьте установку.
    pause
    exit /b 1
)

:: Проверка существования виртуального окружения
if exist venv (
    echo Удаляем существующее виртуальное окружение...
    rd /s /q venv
)
echo Создание виртуального окружения...
%PYTHON_CMD% -m venv venv
if %errorlevel% neq 0 (
    echo Ошибка создания виртуального окружения!
    pause
    exit /b 1
)
:: Активация виртуального окружения
call venv\Scripts\activate.bat

:: Установка необходимых для setup.py зависимостей
echo.
echo Установка необходимых компонентов...
python -m pip install --upgrade pip
python -m pip install winshell pywin32 pillow
if %errorlevel% neq 0 (
    echo.
    echo Ошибка установки зависимостей! Проверьте подключение к интернету.
    echo.
    pause
    exit /b 1
)

:: Запуск скрипта установки
python setup.py
if %errorlevel% neq 0 (
    echo.
    echo Ошибка установки! Проверьте журнал ошибок в файле setup.log.
    echo.
    pause
    exit /b 1
)

echo.
echo Установка успешно завершена!
echo Ярлык приложения создан на рабочем столе.
echo.

:: Динамически определяем путь Python для добавления в PATH
set PYTHON_INSTALL_PATH=
if "%PYTHON_CMD:~0,2%"=="py" (
    :: Если используется py launcher, находим фактический путь Python
    for /f "tokens=* usebackq" %%x in (`%PYTHON_CMD% -c "import sys,os; print(os.path.dirname(sys.executable))"`) do (
        set PYTHON_INSTALL_PATH=%%x
    )
) else (
    :: Иначе берем директорию из пути к python.exe
    for /f "tokens=* usebackq" %%x in (`%PYTHON_CMD% -c "import sys,os; print(os.path.dirname(sys.executable))"`) do (
        set PYTHON_INSTALL_PATH=%%x
    )
)

:: Добавление в PATH если нужно
if defined PYTHON_INSTALL_PATH (
    echo Найден путь установки Python: %PYTHON_INSTALL_PATH%
    set PYTHON_SCRIPTS_PATH=%PYTHON_INSTALL_PATH%\Scripts
    
    echo Проверка наличия Python в PATH...
    echo %PATH% | find /i "%PYTHON_INSTALL_PATH%" >nul
    if %errorlevel% neq 0 (
        echo Python не найден в PATH, добавляем...
        setx PATH "%PYTHON_INSTALL_PATH%;%PYTHON_SCRIPTS_PATH%;%PATH%"
        echo PATH обновлен. Перезапустите терминал для применения изменений.
    ) else (
        echo Python уже в PATH
    )
) else (
    echo Не удалось определить путь установки Python для добавления в PATH
)

pause
exit /b 0 