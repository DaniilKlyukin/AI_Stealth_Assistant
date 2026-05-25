@echo off
:: Установка кодировки UTF-8 для корректного отображения кириллицы
chcp 65001 > nul
setlocal

set MODEL_PATH=C:\AI_Models\Qwen3.5-2B-Aggr-Q4_K_M.gguf
set PORT=8080
set CONTEXT_SIZE=4096

echo Запуск сервера llama.cpp в фоновом режиме...
start "Llama.cpp Server" /min llama-server.exe --model "%MODEL_PATH%" --ctx-size %CONTEXT_SIZE% --port %PORT%

echo Сервер запущен на порту %PORT%.
echo Ожидание инициализации...
timeout /t 5 /nobreak > nul

echo.
echo ====================================================
echo Сервер готов к работе. Окно сервера свернуто.
echo Для завершения работы сервера нажмите ЛЮБУЮ клавишу.
echo ====================================================
echo.

pause

echo Завершение работы сервера...
taskkill /FI "WINDOWTITLE eq Llama.cpp Server" /F > nul