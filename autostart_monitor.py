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
MONITOR_VERSION = "V1.0"

# Pfadweiche: Unterscheide zwischen kompilierter .exe (z.B. PyInstaller) und normalem Python-Skript
if getattr(sys, 'frozen', False):
    APP_DIR = os.path.dirname(os.path.abspath(sys.executable))
else:
    APP_DIR = os.path.dirname(os.path.abspath(__file__))

# Falls die .exe direkt aus einem Unterordner (wie "dist") gestartet wird, eine Ordnerebene höher prüfen
if not os.path.exists(os.path.join(APP_DIR, TARGET_APP)) and os.path.exists(os.path.join(os.path.dirname(APP_DIR), TARGET_APP)):
    APP_DIR = os.path.dirname(APP_DIR)

class MonitorApp:
    def __init__(self, root):
        self.root = root
        self.root.title(f"Bird Classifier Monitor ({MONITOR_VERSION})")
        self.root.geometry("600x400")
        
        # Icon für Fenster setzen (falls vorhanden)
        icon_path = os.path.join(APP_DIR, "static", "feather.ico")
        if os.path.exists(icon_path):
            try:
                self.root.iconbitmap(icon_path)
            except Exception:
                pass

        # Kopfbereich mit Versionsangabe oben links
        header_frame = tk.Frame(self.root)
        header_frame.pack(fill='x', padx=10, pady=(6, 0))
        version_label = tk.Label(header_frame, text=MONITOR_VERSION, font=("Arial", 9, "bold"), fg="#666666")
        version_label.pack(side='left')

        # Titel
        title_label = tk.Label(self.root, text="Birds AI Sound Classifier - Überwachung", font=("Arial", 14, "bold"))
        title_label.pack(pady=(0, 10))
        
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
        
    def get_running_instances(self):
        """Findet alle laufenden Instanzen der Ziel-Anwendung, sortiert nach Alter (ältester zuerst)."""
        instances = []
        for p in psutil.process_iter(['name', 'cmdline', 'create_time']):
            try:
                cmdline = p.info.get('cmdline')
                if cmdline:
                    # Wandle Befehlszeile in String um für eine einfache Suche
                    cmd_str = ' '.join(cmdline).lower()
                    if TARGET_APP.lower() in cmd_str and "python" in p.info.get('name', '').lower():
                        instances.append(p)
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                pass
        
        # Sortiere aufsteigend nach Startzeit, damit wir die älteste Instanz behalten können
        instances.sort(key=lambda x: x.info.get('create_time', 0))
        return instances
        
    def kill_process_tree(self, proc):
        """Beendet einen Prozess inklusive aller Kindprozesse (z.B. Audio-Kindprozess) und Konsolenfenster."""
        try:
            pid = proc.pid
            # 1. Zugehöriges cmd.exe-Elternfenster merken (von start.bat)
            parent_cmd = None
            try:
                par = proc.parent()
                if par and par.name().lower() == 'cmd.exe':
                    parent_cmd = par
            except Exception:
                pass

            # 2. Kindprozesse ermitteln (z.B. python.exe für Audioanalyse)
            children = []
            try:
                children = proc.children(recursive=True)
            except Exception:
                pass

            for child in children:
                try:
                    self.log_message(f"Beende Kindprozess PID {child.pid}...")
                    child.terminate()
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass

            # 3. Hauptprozess beenden
            try:
                self.log_message(f"Beende Prozess PID {pid}...")
                proc.terminate()
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass

            # 4. Kurz warten, ob die Prozesse reagieren; falls nicht, hart beenden (kill)
            try:
                _, alive = psutil.wait_procs(children + [proc], timeout=3)
                for p in alive:
                    try:
                        p.kill()
                    except Exception:
                        pass
            except Exception:
                pass

            # 5. Falls vorhanden, das cmd.exe-Fenster schließen
            if parent_cmd:
                try:
                    parent_cmd.terminate()
                except Exception:
                    pass

        except Exception as e:
            self.log_message(f"Fehler beim Beenden des Prozessbaums PID {proc.pid}: {e}")

    def cleanup_orphaned_processes(self):
        """Beendet verwaiste Audio-/Subprozesse im Anwendungsverzeichnis, bevor neu gestartet wird."""
        my_pid = os.getpid()
        cleaned_count = 0
        norm_app_dir = os.path.normpath(APP_DIR).lower()

        for p in psutil.process_iter(['name', 'cmdline', 'pid']):
            try:
                if p.pid == my_pid:
                    continue
                name = p.info.get('name', '').lower()
                cmdline = p.info.get('cmdline') or []
                cmd_str = ' '.join(cmdline).lower()

                # Den Monitor selbst niemals anfassen (egal ob .py oder compilierte .exe)
                if "autostart_monitor" in cmd_str:
                    continue

                if "python" in name:
                    is_in_app_dir = False
                    try:
                        p_cwd = p.cwd()
                        if p_cwd and os.path.normpath(p_cwd).lower() == norm_app_dir:
                            is_in_app_dir = True
                    except Exception:
                        if norm_app_dir in cmd_str:
                            is_in_app_dir = True

                    # Multiprocessing-Spawn-Worker oder verwaiste Reste im App-Verzeichnis beenden
                    if is_in_app_dir and ("multiprocessing.spawn" in cmd_str or TARGET_APP.lower() in cmd_str):
                        self.log_message(f"Verwaisten Hintergrundprozess PID {p.pid} gefunden. Beende...")
                        try:
                            p.terminate()
                            p.wait(timeout=2)
                        except (psutil.TimeoutExpired, Exception):
                            try:
                                p.kill()
                            except Exception:
                                pass
                        cleaned_count += 1
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                pass

        if cleaned_count > 0:
            self.log_message(f"{cleaned_count} verwaiste(n) Prozess(e) bereinigt.")
            time.sleep(1)

    def monitor_loop(self):
        """Die Hauptschleife, die periodisch den Prozessstatus überprüft."""
        while self.running:
            instances = self.get_running_instances()
            
            if len(instances) == 0:
                self.log_message(f"'{TARGET_APP}' läuft nicht.")
                self.cleanup_orphaned_processes()
                self.log_message(f"Starte '{TARGET_APP}' neu...")
                self.start_app()
            elif len(instances) > 1:
                self.log_message(f"Warnung: {len(instances)} Instanzen von '{TARGET_APP}' gefunden! Beende überzählige...")
                # Die erste Instanz (älteste) bleibt, die anderen werden samt Kindprozessen beendet
                for p in instances[1:]:
                    self.kill_process_tree(p)
            
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
