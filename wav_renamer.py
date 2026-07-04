import os
import re
import datetime
import tkinter as tk
from tkinter import filedialog, messagebox

def process_files():
    file_paths = filedialog.askopenfilenames(
        title="Wähle .wav Dateien aus",
        filetypes=[("WAV Dateien", "*.wav *.WAV"), ("Alle Dateien", "*.*")]
    )
    if not file_paths:
        return
    rename_files(file_paths)

def process_folder():
    folder_path = filedialog.askdirectory(title="Wähle einen Ordner mit .wav Dateien")
    if not folder_path:
        return
    
    file_paths = []
    for filename in os.listdir(folder_path):
        if filename.lower().endswith('.wav'):
            file_paths.append(os.path.join(folder_path, filename))
            
    if not file_paths:
        messagebox.showinfo("Info", "Keine .wav Dateien in diesem Ordner gefunden.")
        return
        
    rename_files(file_paths)

def rename_files(file_paths):
    try:
        count = 0
        for filepath in file_paths:
            filename = os.path.basename(filepath)
            folder_path = os.path.dirname(filepath)
            
            # Art extrahieren: Entferne angehängte Zeitstempel oder Zahlen
            name = filename[:-4] # .wav entfernen
            while True:
                # Entfernt Muster wie _26-07-04-10-08-08 oder _1 vom Ende
                new_name = re.sub(r'(_\d{2}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2}|_\d+)$', '', name)
                if new_name == name:
                    break
                name = new_name
            
            species = name
            
            # Umbenennen nur fortsetzen, wenn noch eine Art übrig ist
            if species:
                
                # Änderungsdatum (Modification Time) der Datei abrufen
                # Das ist zuverlässiger, da das Erstellungsdatum sich beim Kopieren ändert
                mod_time = os.path.getmtime(filepath)
                dt = datetime.datetime.fromtimestamp(mod_time)
                
                # Formatieren in YY-MM-DD-HH-MM-SS (z.B. 26-07-04-08-34-12)
                time_str = dt.strftime("%y-%m-%d-%H-%M-%S")
                
                new_filename = f"{species}_{time_str}.wav"
                new_filepath = os.path.join(folder_path, new_filename)
                
                # Namenskollisionen bei gleicher Erstellungssekunde vermeiden
                counter = 1
                while os.path.exists(new_filepath):
                    new_filename = f"{species}_{time_str}_{counter}.wav"
                    new_filepath = os.path.join(folder_path, new_filename)
                    counter += 1
                    
                os.rename(filepath, new_filepath)
                count += 1
                    
        messagebox.showinfo("Erfolg", f"{count} .wav Dateien wurden erfolgreich umbenannt.")
    except Exception as e:
        messagebox.showerror("Fehler", f"Ein Fehler ist aufgetreten:\n{str(e)}")

def main():
    # Hauptfenster erstellen
    root = tk.Tk()
    root.title("WAV Datei Umbenenner")
    root.geometry("400x250")
    root.resizable(False, False)

    # UI Elemente
    frame = tk.Frame(root, padx=20, pady=20)
    frame.pack(expand=True, fill=tk.BOTH)

    label = tk.Label(
        frame, 
        text="Wie möchtest du die Dateien auswählen?", 
        font=("Arial", 12), 
        justify=tk.CENTER
    )
    label.pack(pady=10)

    btn_files = tk.Button(
        frame, 
        text="Einzelne Dateien auswählen", 
        command=process_files, 
        font=("Arial", 12), 
        bg="#4CAF50", 
        fg="white", 
        activebackground="#45a049"
    )
    btn_files.pack(pady=10, fill=tk.X)

    btn_folder = tk.Button(
        frame, 
        text="Ganzen Ordner auswählen\n(Dateien werden im Dialog nicht angezeigt)", 
        command=process_folder, 
        font=("Arial", 10), 
        bg="#2196F3", 
        fg="white", 
        activebackground="#0b7dda"
    )
    btn_folder.pack(pady=10, fill=tk.X)

    # Anwendung starten
    root.mainloop()

if __name__ == "__main__":
    main()
