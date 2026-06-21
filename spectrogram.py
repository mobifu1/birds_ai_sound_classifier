import tkinter as tk
from tkinter import filedialog, messagebox
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import librosa
import librosa.display
import os

class SpectrogramApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Bird Sound Spectrogram Viewer")
        self.root.geometry("800x600")

        self.btn_frame = tk.Frame(self.root)
        self.btn_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=10)

        self.open_btn = tk.Button(self.btn_frame, text="Open WAV File", command=self.open_file, font=("Arial", 12))
        self.open_btn.pack(side=tk.LEFT)

        self.fig, self.ax = plt.subplots(figsize=(8, 5))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    def open_file(self):
        filepath = filedialog.askopenfilename(
            title="Select a WAV file",
            filetypes=[("WAV files", "*.wav"), ("All files", "*.*")]
        )
        if not filepath:
            return
        
        self.plot_spectrogram(filepath)

    def plot_spectrogram(self, filepath):
        try:
            # Load audio using librosa
            # sr=None preserves original sampling rate
            y, sr = librosa.load(filepath, sr=None)
            
            # The prompt mentions 3-second sound files. We'll display the whole file.
            
            # Clear previous figure
            self.fig.clf()
            self.ax = self.fig.add_subplot(111)
            
            # Compute mel spectrogram
            S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
            S_dB = librosa.power_to_db(S, ref=np.max)
            
            # Plot spectrogram
            img = librosa.display.specshow(S_dB, x_axis='time', y_axis='mel', sr=sr, fmax=8000, ax=self.ax)
            
            # Set title to the filename
            filename = os.path.basename(filepath)
            self.ax.set_title(f"Spectrogram: {filename}")
            
            # Add colorbar
            self.fig.colorbar(img, ax=self.ax, format='%+2.0f dB')
            
            self.canvas.draw()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load or process audio file:\n{e}")

if __name__ == "__main__":
    root = tk.Tk()
    app = SpectrogramApp(root)
    root.mainloop()
