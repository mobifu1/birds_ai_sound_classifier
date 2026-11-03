import tkinter as tk
from tkinter import scrolledtext
import threading
import time
import datetime
import subprocess
import os
import sys
import shutil

def get_python_exe():
    """Ermittelt zuverlässig den Pfad zur 'python.exe' (Konsolen-Interpreter),
    auch wenn dieses Skript als .pyw über 'pythonw.exe' oder 'pyw.exe' gestartet wurde."""
    candidates = []
    if sys.executable:
        exe_dir = os.path.dirname(os.path.abspath(sys.executable))
        candidates.append(os.path.join(exe_dir, "python.exe"))
    for prefix in (getattr(sys, "prefix", None), getattr(sys, "base_prefix", None)):
        if prefix:
            candidates.append(os.path.join(os.path.abspath(prefix), "python.exe"))
            candidates.append(os.path.join(os.path.abspath(prefix), "Scripts", "python.exe"))
    for candidate in candidates:
        if os.path.isfile(candidate):
            return candidate
    found = shutil.which("python.exe") or shutil.which("python")
    return found if found else "python.exe"

PYTHON_EXE = get_python_exe()

# Installiere psutil automatisch, falls nicht vorhanden, da es für die Prozesserkennung am besten funktioniert
try:
    import psutil
except ImportError:
    subprocess.check_call([PYTHON_EXE, "-m", "pip", "install", "psutil"])
    import psutil

TARGET_APP = "birds_ai_sound_classifier.py"
MONITOR_VERSION = "V1.2"
APP_DIR = os.path.dirname(os.path.abspath(sys.executable)) if getattr(sys, "frozen", False) else os.path.dirname(os.path.abspath(__file__))

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
        self.log_message(f"Monitoring ({MONITOR_VERSION}) gestartet. Überprüfe alle 60 Sekunden...")
        
        # Starte den Überwachungs-Thread
        self.monitor_thread = threading.Thread(target=self.monitor_loop, daemon=True)
        self.monitor_thread.start()
        
    def log_message(self, message):
        """Fügt eine Nachricht mit Zeitstempel in das Log-Fenster ein."""
        timestamp = datetime.datetime.now().strftime("%d.%m.%Y %H:%M:%S")
        log_entry = f"[{timestamp}] {message}\n"
        
        try:
            self.log_area.config(state='normal')
            self.log_area.insert(tk.END, log_entry)
            self.log_area.see(tk.END)
            self.log_area.config(state='disabled')
        except tk.TclError:
            pass
        
    def get_running_instances(self):
        """Findet alle laufenden Instanzen der Ziel-Anwendung, sortiert nach Alter (ältester zuerst)."""
        instances = []
        for p in psutil.process_iter(['name', 'cmdline', 'create_time']):
            try:
                cmdline = p.info.get('cmdline')
                if cmdline:
                    # Wandle Befehlszeile in String um für eine einfache Suche
                    cmd_str = ' '.join(cmdline).lower()
                    name_str = (p.info.get('name') or '').lower()
                    if TARGET_APP.lower() in cmd_str and "python" in name_str and "multiprocessing.spawn" not in cmd_str:
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
            # 1. Zugehörige cmd.exe-Elternfenster über die gesamte Elternkette merken (von start.bat)
            parent_cmds = []
            seen_cmd_pids = set()
            try:
                for anc in proc.parents():
                    if anc and anc.name().lower() == 'cmd.exe' and anc.pid not in seen_cmd_pids:
                        parent_cmds.append(anc)
                        seen_cmd_pids.add(anc.pid)
            except Exception:
                pass
            try:
                curr = proc
                for _ in range(6):
                    ppid = curr.ppid()
                    if not ppid or ppid <= 0:
                        break
                    par = psutil.Process(ppid)
                    if par.name().lower() == 'cmd.exe' and par.pid not in seen_cmd_pids:
                        parent_cmds.append(par)
                        seen_cmd_pids.add(par.pid)
                    curr = par
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

            # 5. Falls vorhanden, die zugehörigen cmd.exe-Fenster hart schließen
            for parent_cmd in parent_cmds:
                try:
                    parent_cmd.kill()
                except Exception:
                    pass

        except Exception as e:
            self.log_message(f"Fehler beim Beenden des Prozessbaums PID {proc.pid}: {e}")

    def cleanup_orphaned_processes(self):
        """Beendet verwaiste Audio-/Subprozesse sowie verwaiste CMD-Fenster im Anwendungsverzeichnis, bevor neu gestartet wird."""
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

                # Den Monitor selbst niemals anfassen (egal ob .py oder .pyw)
                if "autostart_monitor" in cmd_str:
                    continue

                is_in_app_dir = False
                try:
                    p_cwd = p.cwd()
                    if p_cwd and os.path.normpath(p_cwd).lower() == norm_app_dir:
                        is_in_app_dir = True
                except Exception:
                    if norm_app_dir in cmd_str:
                        is_in_app_dir = True

                if "python" in name:
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
                elif name == "cmd.exe":
                    # Verwaistes cmd.exe-Fenster von start.bat (z.B. hängend bei "Batchvorgang abbrechen (J/N)?") schließen
                    if "start.bat" in cmd_str and (is_in_app_dir or norm_app_dir in cmd_str):
                        self.log_message(f"Verwaistes CMD-Fenster PID {p.pid} gefunden. Schließe...")
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

    def cleanup_legacy_exe_monitor(self):
        """Prüft, ob noch ältere Monitor-Prozesse oder die alte 'autostart_monitor.exe' (App1)
        laufen bzw. im Installationsordner existieren, beendet diese und löscht 'autostart_monitor.exe'."""
        if getattr(sys, "frozen", False):
            return

        my_pid = os.getpid()
        my_launcher_pids = set()
        try:
            parent_proc = psutil.Process(my_pid).parent()
            if parent_proc and parent_proc.name().lower() in ('py.exe', 'pyw.exe'):
                my_launcher_pids.add(parent_proc.pid)
        except Exception:
            pass

        # 1. Eventuell noch laufende ältere Instanzen von autostart_monitor.pyw / .py sowie
        # 2. laufende Prozesse von autostart_monitor.exe (App1) erfassen
        old_py_monitors = []
        legacy_exe_name = "autostart_monitor.exe"
        legacy_exe_procs = []

        for p in psutil.process_iter(['pid', 'name', 'cmdline', 'exe']):
            try:
                pid = p.info['pid']
                if pid == my_pid:
                    continue
                p_name = (p.info.get('name') or '').lower()
                p_exe = os.path.basename(p.info.get('exe') or '').lower()
                cmdline = p.info.get('cmdline') or []
                cmd_str = ' '.join(cmdline).lower()

                # App1 (autostart_monitor.exe) IMMER erfassen, auch wenn sie Großelternprozess ist
                if p_name == legacy_exe_name or p_exe == legacy_exe_name or (legacy_exe_name in cmd_str and "python" not in p_name):
                    legacy_exe_procs.append(p)
                elif pid not in my_launcher_pids and ('python' in p_name or p_name in ('py.exe', 'pyw.exe')) and (
                    'autostart_monitor.pyw' in cmd_str or 'autostart_monitor.py' in cmd_str
                ):
                    old_py_monitors.append(p)
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                pass

        if old_py_monitors:
            for p in old_py_monitors:
                try:
                    p.terminate()
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
            try:
                _, alive_py = psutil.wait_procs(old_py_monitors, timeout=2)
                for p in alive_py:
                    try:
                        p.kill()
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        pass
            except Exception:
                pass

        if legacy_exe_procs:
            self.log_message(f"Alte '{legacy_exe_name}' erkannt ({len(legacy_exe_procs)} Prozess(e)). Beende...")
            for p in legacy_exe_procs:
                try:
                    p.terminate()
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
            try:
                _, alive_exe = psutil.wait_procs(legacy_exe_procs, timeout=3)
                for p in alive_exe:
                    try:
                        p.kill()
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        pass
                if alive_exe:
                    psutil.wait_procs(alive_exe, timeout=2)
            except Exception:
                pass
            self.log_message(f"Prozess '{legacy_exe_name}' wurde beendet.")

        # 3. Datei 'autostart_monitor.exe' aus dem Installationsordner löschen
        legacy_exe_path = os.path.join(APP_DIR, legacy_exe_name)
        if os.path.isfile(legacy_exe_path):
            removed = False
            last_err = None
            for _ in range(10):
                try:
                    if os.path.isfile(legacy_exe_path):
                        try:
                            os.chmod(legacy_exe_path, 0o777)
                        except Exception:
                            pass
                        os.remove(legacy_exe_path)
                    if not os.path.exists(legacy_exe_path):
                        removed = True
                        break
                except Exception as e:
                    last_err = e
                    time.sleep(0.3)
            if removed:
                self.log_message(f"Alte Datei '{legacy_exe_name}' wurde aus dem Installationsordner gelöscht.")
            else:
                self.log_message(f"Hinweis: '{legacy_exe_name}' konnte noch nicht gelöscht werden ({last_err}).")

    def monitor_loop(self):
        """Die Hauptschleife, die periodisch den Prozessstatus überprüft."""
        # Beim Start prüfen, ob noch eine alte autostart_monitor.exe (App1) läuft oder existiert,
        # diese beenden und aus dem Installationsordner löschen
        self.cleanup_legacy_exe_monitor()

        while self.running:
            # Falls autostart_monitor.exe beim ersten Versuch noch kurz durch Windows gesperrt war, erneut bereinigen
            if os.path.exists(os.path.join(APP_DIR, "autostart_monitor.exe")):
                self.cleanup_legacy_exe_monitor()

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
        """Startet die Ziel-Anwendung direkt über python.exe in einem neuen, sichtbaren Konsolenfenster."""
        try:
            target_path = os.path.join(APP_DIR, TARGET_APP)
            if not os.path.exists(target_path):
                self.log_message(f"Fehler: '{TARGET_APP}' nicht gefunden in {APP_DIR}.")
                return

            # Wichtig bei Start aus .pyw (pythonw.exe):
            # - Es MUSS python.exe (Konsolen-Subsystem) statt pythonw.exe verwendet werden.
            # - stdin/stdout/stderr dürfen NICHT umgeleitet werden (kein stdin=DEVNULL),
            #   da Python sonst STARTF_USESTDHANDLES setzt und unter pythonw.exe ungültige
            #   Standard-Handles an das neue Konsolenfenster übergibt.
            startupinfo = subprocess.STARTUPINFO()
            startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            startupinfo.wShowWindow = 1  # SW_SHOWNORMAL: Konsolenfenster garantiert sichtbar öffnen

            proc = subprocess.Popen(
                [PYTHON_EXE, target_path],
                cwd=APP_DIR,
                creationflags=subprocess.CREATE_NEW_CONSOLE,
                startupinfo=startupinfo,
                close_fds=True
            )
            
            self.log_message(f"Anwendung wurde erfolgreich gestartet (PID {proc.pid}).")
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
