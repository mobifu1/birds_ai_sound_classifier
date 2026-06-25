import tkinter as tk
from tkinter import ttk
import pyaudio
import numpy as np
from scipy import signal
import threading
from PIL import Image, ImageTk
import matplotlib.cm as cm

# Constants
CHUNK = 2048
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 48000
WATERFALL_HEIGHT = 300 # Number of time slices to show
MAX_FREQ = 20000 # Max frequency to display in Hz

class SpectrogramApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Live Spectrogram")
        
        self.p = pyaudio.PyAudio()
        
        # Audio state
        self.is_running = False
        self.stream = None
        self.audio_thread = None
        
        # Waterfall data
        # Calculate how many bins correspond to MAX_FREQ
        self.freq_resolution = RATE / CHUNK
        self.max_bin = int(MAX_FREQ / self.freq_resolution)
        self.waterfall_data = np.zeros((WATERFALL_HEIGHT, self.max_bin), dtype=np.float32)
        
        # Colormap
        self.cmap = cm.get_cmap('viridis')
        
        self._setup_ui()
        self._update_waterfall_loop()

    def _get_input_devices(self):
        devices = []
        for i in range(self.p.get_device_count()):
            dev_info = self.p.get_device_info_by_index(i)
            if dev_info.get('maxInputChannels') > 0:
                devices.append(f"{i}: {dev_info.get('name')}")
        return devices

    def _setup_ui(self):
        # Control Frame
        control_frame = ttk.Frame(self.root, padding=10)
        control_frame.pack(fill=tk.X)

        # Microphone Selection
        ttk.Label(control_frame, text="Microphone:").grid(row=0, column=0, padx=5, pady=5)
        self.mic_var = tk.StringVar()
        self.mic_combo = ttk.Combobox(control_frame, textvariable=self.mic_var, state="readonly", width=40)
        devices = self._get_input_devices()
        self.mic_combo['values'] = devices
        if devices:
            self.mic_combo.current(0)
        self.mic_combo.grid(row=0, column=1, padx=5, pady=5)

        # Start/Stop Button
        self.start_btn = ttk.Button(control_frame, text="Start", command=self.toggle_stream)
        self.start_btn.grid(row=0, column=2, padx=10, pady=5)

        # High-Pass Filter Frame
        hpf_frame = ttk.LabelFrame(control_frame, text="High-Pass Filter", padding=5)
        hpf_frame.grid(row=0, column=3, padx=20, pady=5)

        self.hpf_var = tk.BooleanVar(value=False)
        self.hpf_check = ttk.Checkbutton(hpf_frame, text="Active", variable=self.hpf_var)
        self.hpf_check.grid(row=0, column=0, padx=5)

        ttk.Label(hpf_frame, text="Freq (Hz):").grid(row=0, column=1, padx=5)
        self.hpf_freq_var = tk.StringVar(value="1000")
        self.hpf_entry = ttk.Entry(hpf_frame, textvariable=self.hpf_freq_var, width=8)
        self.hpf_entry.grid(row=0, column=2, padx=5)

        # Waterfall Display
        # Create a canvas that is properly sized
        self.canvas = tk.Canvas(self.root, width=self.max_bin, height=WATERFALL_HEIGHT, bg='black')
        self.canvas.pack(padx=10, pady=10)
        
        # Labels for frequency axis
        axis_frame = ttk.Frame(self.root)
        axis_frame.pack(fill=tk.X, padx=10, pady=(0, 10))
        # Place a few labels
        ttk.Label(axis_frame, text="0 Hz").pack(side=tk.LEFT)
        ttk.Label(axis_frame, text=f"{MAX_FREQ/1000:.0f} kHz").pack(side=tk.RIGHT)
        ttk.Label(axis_frame, text=f"{MAX_FREQ/2000:.0f} kHz").pack(expand=True)
        
        self.photo_image = None
        self.canvas_image = self.canvas.create_image(0, 0, anchor=tk.NW)

    def toggle_stream(self):
        if self.is_running:
            self.stop_stream()
        else:
            self.start_stream()

    def start_stream(self):
        if not self.mic_var.get():
            return
            
        device_id = int(self.mic_var.get().split(':')[0])
        
        try:
            self.stream = self.p.open(
                format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                input_device_index=device_id,
                frames_per_buffer=CHUNK
            )
            self.is_running = True
            self.start_btn.config(text="Stop")
            
            # Start background thread for audio processing
            self.audio_thread = threading.Thread(target=self.audio_loop, daemon=True)
            self.audio_thread.start()
        except Exception as e:
            print(f"Error opening audio stream: {e}")

    def stop_stream(self):
        self.is_running = False
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
            self.stream = None
        self.start_btn.config(text="Start")

    def audio_loop(self):
        while self.is_running:
            try:
                data = self.stream.read(CHUNK, exception_on_overflow=False)
                # Convert to numpy array
                audio_data = np.frombuffer(data, dtype=np.int16).astype(np.float32)
                
                # Apply High-Pass Filter if active
                if self.hpf_var.get():
                    try:
                        cutoff = float(self.hpf_freq_var.get())
                        if 0 < cutoff < RATE / 2:
                            nyq = 0.5 * RATE
                            normal_cutoff = cutoff / nyq
                            b, a = signal.butter(4, normal_cutoff, btype='high', analog=False)
                            audio_data = signal.filtfilt(b, a, audio_data)
                    except ValueError:
                        pass # Ignore invalid frequency input
                
                # Windowing to reduce spectral leakage
                window = np.hanning(len(audio_data))
                audio_data = audio_data * window
                
                # FFT
                fft_data = np.fft.rfft(audio_data)
                fft_mag = np.abs(fft_data)
                
                # Convert to dB
                fft_mag = 20 * np.log10(fft_mag + 1e-6)
                
                # Crop to max frequency
                fft_mag_cropped = fft_mag[:self.max_bin]
                
                # Normalize for visualization (e.g., -60dB to 60dB map to 0.0 to 1.0)
                min_db = -20
                max_db = 100
                fft_norm = np.clip((fft_mag_cropped - min_db) / (max_db - min_db), 0, 1)
                
                # Update rolling waterfall array
                # Roll down by 1 row
                self.waterfall_data = np.roll(self.waterfall_data, 1, axis=0)
                # Insert new data at the top
                self.waterfall_data[0, :] = fft_norm

            except Exception as e:
                print(f"Error in audio loop: {e}")
                self.stop_stream()
                break

    def _update_waterfall_loop(self):
        if self.is_running:
            # Apply colormap
            # mapped is an array of shape (HEIGHT, WIDTH, 4) in floats [0, 1]
            mapped = self.cmap(self.waterfall_data)
            
            # Convert to uint8 (0-255)
            img_data = (mapped[:, :, :3] * 255).astype(np.uint8)
            
            # Create PIL image
            img = Image.fromarray(img_data, mode='RGB')
            
            # Convert to PhotoImage
            self.photo_image = ImageTk.PhotoImage(image=img)
            
            # Update canvas
            self.canvas.itemconfig(self.canvas_image, image=self.photo_image)
            
        # Schedule next update (approx 30 FPS = 33ms)
        self.root.after(33, self._update_waterfall_loop)

    def on_closing(self):
        self.is_running = False
        if self.stream:
            self.stream.close()
        self.p.terminate()
        self.root.destroy()

if __name__ == "__main__":
    # Fix for matplotlib warnings
    import matplotlib
    matplotlib.use('Agg')
    
    root = tk.Tk()
    app = SpectrogramApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()
