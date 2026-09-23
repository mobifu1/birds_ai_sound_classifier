import sqlite3
import json
import tkinter as tk
from tkinter import ttk
import os

def check_mismatches():
    db_path = 'birds_audio_stats.db'
    dict_path = 'dictionary.json'
    
    if not os.path.exists(db_path) or not os.path.exists(dict_path):
        return [("Error: DB or dictionary.json not found.", "")]

    # Load dictionary
    with open(dict_path, 'r', encoding='utf-8') as f:
        bird_dict = json.load(f)
        
    valid_species = set()
    for eng_name, data in bird_dict.items():
        valid_species.add(eng_name)
        if 'translation' in data:
            valid_species.add(data['translation'])
            
    # Load from DB
    mismatches = []
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT species, COUNT(*) FROM detections GROUP BY species")
        
        for row in cursor.fetchall():
            species = row[0]
            count = row[1]
            if species not in valid_species:
                mismatches.append((species, count))
                
        conn.close()
    except Exception as e:
        return [(f"Database error: {e}", "")]
        
    return mismatches

def start_gui():
    root = tk.Tk()
    root.title("Bird Species Mismatch Checker")
    root.geometry("600x400")
    
    label = tk.Label(root, text="Mismatched Bird Species in Database", font=("Helvetica", 14, "bold"))
    label.pack(pady=10)
    
    frame = tk.Frame(root)
    frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
    
    # Treeview
    columns = ("Species", "Occurrences")
    tree = ttk.Treeview(frame, columns=columns, show="headings")
    tree.heading("Species", text="Species")
    tree.heading("Occurrences", text="Occurrences")
    
    tree.column("Species", width=300)
    tree.column("Occurrences", width=100, anchor='center')
    
    # Scrollbar
    scrollbar = ttk.Scrollbar(frame, orient=tk.VERTICAL, command=tree.yview)
    tree.configure(yscroll=scrollbar.set)
    
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    
    mismatches = check_mismatches()
    
    if not mismatches:
        tree.insert("", tk.END, values=("No mismatches found!", ""))
    else:
        # sort mismatches by count descending
        mismatches.sort(key=lambda x: x[1] if isinstance(x[1], int) else 0, reverse=True)
        for mismatch in mismatches:
            tree.insert("", tk.END, values=(mismatch[0], mismatch[1]))
            
    # Button to close
    btn_close = tk.Button(root, text="Close", command=root.destroy, width=15)
    btn_close.pack(pady=10)

    root.mainloop()

if __name__ == "__main__":
    start_gui()
