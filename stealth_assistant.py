import os
import threading
import tkinter as tk
import mss
import keyboard
import queue
import cv2
import numpy as np
import easyocr
import logging
import warnings
import requests
from dotenv import load_dotenv
import subprocess
import sys
import atexit

# --- НАСТРОЙКИ И ФИЛЬТРЫ ---
warnings.filterwarnings("ignore", category=UserWarning, module="torch")
logging.getLogger('easyocr').setLevel(logging.ERROR)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


class AreaSelector:
    """Окно для визуального выбора области экрана."""

    def __init__(self, master, on_select_callback):
        self.master = master
        self.on_select_callback = on_select_callback
        self.window = tk.Toplevel(master)
        self.window.attributes('-alpha', 0.3, '-fullscreen', True, "-topmost", True)
        self.window.config(cursor="cross")
        self.canvas = tk.Canvas(self.window, cursor="cross", bg="grey")
        self.canvas.pack(fill="both", expand=True)
        self.start_x = self.start_y = self.rect = None
        self.canvas.bind("<ButtonPress-1>", self.on_button_press)
        self.canvas.bind("<B1-Motion>", self.on_move_press)
        self.canvas.bind("<ButtonRelease-1>", self.on_button_release)

    def on_button_press(self, event):
        self.start_x, self.start_y = event.x, event.y
        self.rect = self.canvas.create_rectangle(self.start_x, self.start_y, 1, 1, outline='red', width=3)

    def on_move_press(self, event):
        self.canvas.coords(self.rect, self.start_x, self.start_y, event.x, event.y)

    def on_button_release(self, event):
        left, top = min(self.start_x, event.x), min(self.start_y, event.y)
        w, h = abs(self.start_x - event.x), abs(self.start_y - event.y)
        if w > 10 and h > 10:
            region = {"left": left, "top": top, "width": w, "height": h}
            self.on_select_callback(region)
        self.window.destroy()


class AnswerOverlay:
    """Управляющее окно для вывода ответов с возможностью перемещения мышью."""

    def __init__(self, parent, overlay_x, overlay_y):
        self.window = tk.Toplevel(parent)
        self.window.overrideredirect(True)
        self.window.attributes("-topmost", True, "-alpha", 0.95)

        self.label = tk.Label(
            self.window, text="Ожидание запроса...", font=("Consolas", 11, "bold"),
            bg="#121212", fg="#00FF7F", wraplength=450,
            padx=12, pady=12, justify="left", relief="solid", borderwidth=1
        )
        self.label.pack()

        # Позиционирование окна
        if overlay_y is not None:
            self.window.geometry(f"+{overlay_x}+{overlay_y}")
        else:
            sh = self.window.winfo_screenheight()
            self.window.geometry(f"+{overlay_x}+{int(sh * 0.65)}")

        # Возможность перемещения окна мышью (зажав ЛКМ на тексте)
        self.label.bind("<Button-1>", self.start_drag)
        self.label.bind("<B1-Motion>", self.on_drag)

        self.is_visible = False
        self.window.withdraw()

    def start_drag(self, event):
        self.drag_x = event.x
        self.drag_y = event.y

    def on_drag(self, event):
        dx = event.x - self.drag_x
        dy = event.y - self.drag_y
        x = self.window.winfo_x() + dx
        y = self.window.winfo_y() + dy
        self.window.geometry(f"+{x}+{y}")

    def display(self, text):
        self.label.config(text=text)
        self.window.deiconify()
        self.is_visible = True

    def toggle(self):
        if self.is_visible:
            self.window.withdraw()
            self.is_visible = False
        else:
            self.window.deiconify()
            self.is_visible = True


class StealthAssistant:
    """Основной класс приложения, инкапсулирующий всю логику."""

    def __init__(self):
        load_dotenv()
        self.load_config()

        self.server_process = None
        self.selected_region = None
        self.gui_queue = queue.Queue()

        # Инициализация OCR в фоновом режиме
        self.reader = easyocr.Reader(['ru', 'en'], gpu=False)

        # Автозапуск сервера, если включено в .env
        self.start_server_if_needed()

        # Инициализация графического интерфейса
        self.root = tk.Tk()
        self.root.withdraw()

        self.overlay = AnswerOverlay(self.root, self.overlay_x, self.overlay_y)

        # Показываем статус запуска при старте программы
        self.overlay.display("Программа запущена!")

        # Регистрация функций завершения и привязка клавиш
        atexit.register(self.cleanup_server)
        self.setup_hotkeys()

        self.root.after(100, self.check_queue)

    def load_config(self):
        """Загрузка конфигурационных параметров из окружения."""
        self.api_base_url = os.getenv("API_BASE_URL", "http://127.0.0.1:8080")
        self.model_id = os.getenv("LLM_MODEL", "Qwen3.5-9B.Q4_K_M")
        self.prompt = os.getenv("PROMPT", "Дай краткий ответ.")
        self.api_timeout = int(os.getenv("API_TIMEOUT", "300"))

        self.overlay_x = int(os.getenv("OVERLAY_X", "50"))
        overlay_y_raw = os.getenv("OVERLAY_Y")
        self.overlay_y = int(overlay_y_raw) if overlay_y_raw and overlay_y_raw.strip() else None

        self.hotkey_select = os.getenv("HOTKEY_SELECT", "ctrl+f8")
        self.hotkey_manual = os.getenv("HOTKEY_MANUAL", "ctrl+f9")
        self.hotkey_toggle = os.getenv("HOTKEY_TOGGLE_OVERLAY", "ctrl+f10")
        self.hotkey_clipboard = os.getenv("HOTKEY_CLIPBOARD", "ctrl+f11")

        self.auto_start_server = os.getenv("AUTO_START_SERVER", "False").lower() == "true"
        self.server_path = os.getenv("SERVER_EXE_PATH")
        self.model_path = os.getenv("MODEL_PATH")
        self.context_size = os.getenv("CONTEXT_SIZE", "8192")
        self.gpu_layers = os.getenv("GPU_LAYERS", "99")

    def start_server_if_needed(self):
        """Запуск локального сервера llama.cpp при соответствующей настройке."""
        if self.auto_start_server:
            port = self.api_base_url.split(":")[-1].replace("/", "")
            cmd = [
                self.server_path,
                "--model", self.model_path,
                "--ctx-size", self.context_size,
                "--port", port,
                "--n-gpu-layers", self.gpu_layers
            ]

            creationflags = 0x08000000 if sys.platform == "win32" else 0

            server_dir = os.path.dirname(self.server_path) if self.server_path else None

            try:
                self.server_process = subprocess.Popen(
                    cmd,
                    creationflags=creationflags,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.PIPE,  # Получаем ошибки, если они возникнут
                    cwd=server_dir  # Указываем рабочую директорию сервера
                )

                try:
                    outs, errs = self.server_process.communicate(timeout=1.0)
                    if errs:
                        print(f"[Ошибка сервера]: {errs.decode('utf-8', errors='ignore')}")
                except subprocess.TimeoutExpired:
                    # Если процесс не завершился за секунду, значит он работает штатно
                    print("Сервер запущен!")

            except Exception as e:
                print(f"[Ошибка автозапуска сервера]: {e}")

    def setup_hotkeys(self):
        """Привязка обработчиков горячих клавиш."""
        keyboard.add_hotkey(self.hotkey_select, self.on_select_pressed)
        keyboard.add_hotkey(self.hotkey_manual, self.on_manual_pressed)
        keyboard.add_hotkey(self.hotkey_toggle, self.on_toggle_pressed)
        keyboard.add_hotkey(self.hotkey_clipboard, self.on_clipboard_pressed)

    def set_selected_region(self, region):
        """Установка выбранной области экрана."""
        self.selected_region = region

    def on_select_pressed(self):
        self.root.after(0, lambda: AreaSelector(self.root, self.set_selected_region))

    def on_manual_pressed(self):
        threading.Thread(target=self.process_screenshot, daemon=True).start()

    def on_toggle_pressed(self):
        self.gui_queue.put(("TOGGLE", None))

    def on_clipboard_pressed(self):
        self.root.after(0, self.process_clipboard)

    def get_text_from_image(self, img_np):
        """Извлечение текста через EasyOCR."""
        try:
            gray = cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY)
            h, w = gray.shape
            if w > 1200:
                scale = 1200 / w
                gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

            results = self.reader.readtext(gray, detail=0)
            return " ".join(results).strip().lower()
        except Exception:
            return ""

    def query_local_llm(self, text):
        """Отправка текстового запроса на локальный сервер llama.cpp."""
        try:
            url = f"{self.api_base_url.rstrip('/')}/chat/completions"
            headers = {"Content-Type": "application/json"}
            user_content = f"{self.prompt}\n\n[Текст]:\n{text}"

            data = {
                "model": self.model_id,
                "messages": [
                    {"role": "user", "content": user_content}
                ],
                "temperature": 0.2
            }

            response = requests.post(url, json=data, headers=headers, timeout=self.api_timeout)
            response.raise_for_status()

            result_json = response.json()
            answer = result_json['choices'][0]['message']['content'].strip()
            return answer
        except requests.exceptions.Timeout:
            return "Превышено время ожидания ответа от сервера (таймаут)."
        except Exception as e:
            return f"Ошибка при подключении к LLM-серверу: {e}"

    def process_screenshot(self):
        """Снятие скриншота, OCR-обработка и отправка запроса в LLM."""
        self.gui_queue.put(("SHOW", "[Думаю...]"))
        try:
            with mss.mss() as sct:
                monitor = self.selected_region if self.selected_region else sct.monitors[1]
                screenshot = sct.grab(monitor)

                img_np = np.array(screenshot)
                img_np = cv2.cvtColor(img_np, cv2.COLOR_BGRA2BGR)

                extracted_text = self.get_text_from_image(img_np)

                if not extracted_text.strip():
                    self.gui_queue.put(("SHOW", "Текст не обнаружен в выделенной области."))
                    return

                answer = self.query_local_llm(extracted_text)
                self.gui_queue.put(("SHOW", answer))

        except Exception as e:
            self.gui_queue.put(("SHOW", f"Ошибка обработки: {e}"))

    def process_clipboard(self):
        """Чтение текста из буфера обмена и отправка в LLM (без OCR)."""
        self.gui_queue.put(("SHOW", "[Думаю...]"))
        try:
            text = self.root.clipboard_get()
            if not text or not text.strip():
                self.gui_queue.put(("SHOW", "Буфер обмена пуст."))
                return

            def run_query():
                answer = self.query_local_llm(text)
                self.gui_queue.put(("SHOW", answer))

            threading.Thread(target=run_query, daemon=True).start()
        except Exception:
            self.gui_queue.put(("SHOW", "Не удалось прочитать буфер обмена."))

    def check_queue(self):
        """Проверка очереди сообщений в главном потоке GUI."""
        try:
            while True:
                action, data = self.gui_queue.get_nowait()
                if action == "SHOW":
                    self.overlay.display(data)
                elif action == "TOGGLE":
                    self.overlay.toggle()
        except queue.Empty:
            pass
        self.root.after(100, self.check_queue)

    def cleanup_server(self):
        """Завершение процесса сервера при выходе из программы."""
        if self.server_process:
            self.server_process.terminate()
            self.server_process.wait()

    def run(self):
        """Запуск главного цикла приложения."""
        self.root.mainloop()


if __name__ == "__main__":
    app = StealthAssistant()
    app.run()