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

# --- НАСТРОЙКИ И ФИЛЬТРЫ ---
warnings.filterwarnings("ignore", category=UserWarning, module="torch")
logging.getLogger('easyocr').setLevel(logging.ERROR)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

load_dotenv()

# Загрузка параметров сервера LLM
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8080/v1")
MODEL_ID = os.getenv("LLM_MODEL", "gemma-3-27b-it")
PROMPT = os.getenv("PROMPT", "Дай краткий ответ.")

# Загрузка конфигурации горячих клавиш
HOTKEY_SELECT = os.getenv("HOTKEY_SELECT", "F8")
HOTKEY_MANUAL = os.getenv("HOTKEY_MANUAL", "F9")
HOTKEY_TOGGLE = os.getenv("HOTKEY_TOGGLE_OVERLAY", "F10")

selected_region = None
gui_queue = queue.Queue()

print("[Система] Запуск OCR (CPU Optimized)...")
reader = easyocr.Reader(['ru', 'en'], gpu=False)


class AreaSelector:
    """Окно для визуального выбора области экрана."""

    def __init__(self, master):
        self.master = master
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
        global selected_region
        left, top = min(self.start_x, event.x), min(self.start_y, event.y)
        w, h = abs(self.start_x - event.x), abs(self.start_y - event.y)
        if w > 10 and h > 10:
            selected_region = {"left": left, "top": top, "width": w, "height": h}
            print(f"[Система] Область установлена: {selected_region}")
        self.window.destroy()


class AnswerOverlay:
    """Управляющее окно для вывода ответов и ручного переключения видимости."""

    def __init__(self, parent):
        self.window = tk.Toplevel(parent)
        self.window.overrideredirect(True)
        self.window.attributes("-topmost", True, "-alpha", 0.95)

        self.label = tk.Label(
            self.window, text="Ожидание запроса...", font=("Consolas", 12, "bold"),
            bg="#121212", fg="#00FF7F", wraplength=450,
            padx=15, pady=15, justify="left", relief="solid", borderwidth=1
        )
        self.label.pack()

        # Размещение окна в нижней части экрана
        sh = self.window.winfo_screenheight()
        self.window.geometry(f"+50+{int(sh * 0.65)}")

        self.is_visible = False
        self.window.withdraw()  # Окно скрыто при запуске

    def display(self, text):
        self.label.config(text=text)
        self.window.deiconify()
        self.is_visible = True

    def toggle(self):
        if self.is_visible:
            self.window.withdraw()
            self.is_visible = False
            print("[Интерфейс] Окно скрыто")
        else:
            self.window.deiconify()
            self.is_visible = True
            print("[Интерфейс] Окно показано")


def get_text_from_image(img_np):
    """Извлечение текста через EasyOCR."""
    try:
        gray = cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        if w > 1200:
            scale = 1200 / w
            gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

        results = reader.readtext(gray, detail=0)
        return " ".join(results).strip().lower()
    except Exception as e:
        print(f"[Ошибка OCR]: {e}")
        return ""


def query_local_llm(text):
    """Отправка текстового запроса на локальный сервер llama.cpp."""
    try:
        url = f"{API_BASE_URL.rstrip('/')}/chat/completions"
        headers = {"Content-Type": "application/json"}

        user_content = f"{PROMPT}\n\n[Текст с экрана]:\n{text}"

        data = {
            "model": MODEL_ID,
            "messages": [
                {"role": "user", "content": user_content}
            ],
            "temperature": 0.2
        }

        response = requests.post(url, json=data, headers=headers, timeout=30)
        response.raise_for_status()

        result_json = response.json()
        answer = result_json['choices'][0]['message']['content'].strip()
        return answer
    except Exception as e:
        return f"Ошибка при подключении к LLM-серверу: {e}"


def process_screenshot():
    """Снятие скриншота выделенной области, OCR-обработка и отправка запроса в LLM."""
    global selected_region
    try:
        with mss.mss() as sct:
            # Если область не выбрана, скриншотится весь первый монитор
            monitor = selected_region if selected_region else sct.monitors[1]
            screenshot = sct.grab(monitor)

            img_np = np.array(screenshot)
            img_np = cv2.cvtColor(img_np, cv2.COLOR_BGRA2BGR)

            extracted_text = get_text_from_image(img_np)

            if not extracted_text.strip():
                print("[Система] Текст на выбранном участке экрана не обнаружен.")
                return

            print(f"[Локальная LLM] Отправка запроса...")
            answer = query_local_llm(extracted_text)

            print(f"\n[ОТВЕТ]: {answer}\n" + "-" * 30)
            gui_queue.put(("SHOW", answer))

    except Exception as e:
        print(f"[Ошибка обработки]: {e}")


def check_queue():
    """Проверка очереди сообщений в главном потоке GUI."""
    try:
        while True:
            action, data = gui_queue.get_nowait()
            if action == "SHOW":
                overlay.display(data)
            elif action == "TOGGLE":
                overlay.toggle()
    except queue.Empty:
        pass
    root.after(200, check_queue)


def on_manual_pressed():
    threading.Thread(target=process_screenshot, daemon=True).start()


def on_select_pressed():
    root.after(0, lambda: AreaSelector(root))


def on_toggle_pressed():
    gui_queue.put(("TOGGLE", None))


if __name__ == "__main__":
    root = tk.Tk()
    root.withdraw()

    # Инициализация окна отображения ответов
    overlay = AnswerOverlay(root)

    print(f"Приложение запущено и готово к работе.")
    print(f"Горячие клавиши:")
    print(f"  {HOTKEY_SELECT}: Выделить область экрана для анализа")
    print(f"  {HOTKEY_MANUAL}: Сделать снимок выделенной области и отправить в LLM")
    print(f"  {HOTKEY_TOGGLE}: Скрыть / Показать окно ответа на экране")

    keyboard.add_hotkey(HOTKEY_SELECT, on_select_pressed)
    keyboard.add_hotkey(HOTKEY_MANUAL, on_manual_pressed)
    keyboard.add_hotkey(HOTKEY_TOGGLE, on_toggle_pressed)

    root.after(200, check_queue)
    root.mainloop()