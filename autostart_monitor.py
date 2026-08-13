import tkinter as tk
from tkinter import scrolledtext
import threading
import time
import datetime
import subprocess
import os
import sys

# Installiere psutil automatisch, falls nicht vorhanden, da es für die Prozesserkennung am besten funktioniert
try:
    import psutil
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "psutil"])
    import psutil

TARGET_APP = "birds_ai_sound_classifier.py"
APP_DIR = os.path.dirname(os.path.abspath(__file__))

class MonitorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Bird Classifier Monitor")
        self.root.geometry("600x400")
        
        # Titel
        title_label = tk.Label(self.root, text="Birds AI Sound Classifier - Überwachung", font=("Arial", 14, "bold"))
        title_label.pack(pady=10)
        
        # Log-Fenster
        self.log_area = scrolledtext.ScrolledText(self.root, wrap=tk.WORD, state='disabled', font=("Consolas", 10), bg="#f4f4f4")
        self.log_area.pack(expand=True, fill='both', padx=15, pady=10)
        
        self.running = True
        self.log_message("Monitoring gestartet. Überprüfe alle 60 Sekunden...")
        
        # Starte den Überwachungs-Thread
        self.monitor_thread = threading.Thread(target=self.monitor_loop, daemon=True)
        self.monitor_thread.start()
        
    def log_message(self, message):
        """Fügt eine Nachricht mit Zeitstempel in das Log-Fenster ein."""
        timestamp = datetime.datetime.now().strftime("%d.%m.%Y %H:%M:%S")
        log_entry = f"[{timestamp}] {message}\n"
        
        self.log_area.config(state='normal')
        self.log_area.insert(tk.END, log_entry)
        self.log_area.see(tk.END)
        self.log_area.config(state='disabled')
        
    def is_app_running(self):
        """Prüft, ob die Ziel-Anwendung derzeit ausgeführt wird."""
        for p in psutil.process_iter(['name', 'cmdline']):
            try:
                cmdline = p.info.get('cmdline')
                if cmdline:
                    # Wandle Befehlszeile in String um für eine einfache Suche
                    cmd_str = ' '.join(cmdline).lower()
                    if TARGET_APP.lower() in cmd_str and "python" in p.info.get('name', '').lower():
                        return True
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                pass
        return False
        
    def monitor_loop(self):
        """Die Hauptschleife, die periodisch den Prozessstatus überprüft."""
        while self.running:
            if not self.is_app_running():
                self.log_message(f"'{TARGET_APP}' läuft nicht. Starte neu...")
                self.start_app()
            
            # Überprüfe jede Minute (60 Sekunden).
            # Wir verwenden eine Schleife mit kurzem Sleep, damit die App 
            # schnell geschlossen werden kann, wenn man das Fenster schließt.
            for _ in range(60):
                if not self.running:
                    break
                time.sleep(1)
            
    def start_app(self):
        """Startet die Ziel-Anwendung über das start.bat Skript, falls vorhanden, oder direkt."""
        try:
            bat_path = os.path.join(APP_DIR, "start.bat")
            if os.path.exists(bat_path):
                # Starte per start.bat in einem neuen Konsolenfenster
                subprocess.Popen(bat_path, cwd=APP_DIR, creationflags=subprocess.CREATE_NEW_CONSOLE)
            else:
                target_path = os.path.join(APP_DIR, TARGET_APP)
                if not os.path.exists(target_path):
                    self.log_message(f"Fehler: '{TARGET_APP}' nicht gefunden.")
                    return
                # Starte das Python-Skript in einem neuen Konsolenfenster
                subprocess.Popen(["python", TARGET_APP], cwd=APP_DIR, creationflags=subprocess.CREATE_NEW_CONSOLE)
            
            self.log_message("Anwendung wurde erfolgreich gestartet.")
        except Exception as e:
            self.log_message(f"Fehler beim Starten: {e}")

    def on_closing(self):
        """Wird aufgerufen, wenn das Fenster geschlossen wird."""
        self.running = False
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = MonitorApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()
