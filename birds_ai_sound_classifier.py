import os
import time
import contextlib
import threading
import multiprocessing as mp
import sqlite3
import datetime
from astral import LocationInfo
from astral.sun import sun
import pyaudio
import wave
import json
import logging
import traceback
from collections import deque
import numpy as np
import pandas as pd
import queue
from scipy import signal
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import requests
import soundfile as sf
import base64

from flask import Flask, render_template, jsonify, request, send_file, abort, send_from_directory
from waitress import serve
import librosa
import librosa.display

# BirdNET Imports
import birdnetlib.analyzer
# Lokale Modell- und Label-Pfade konfigurieren
local_model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model")
birdnetlib.analyzer.MODEL_PATH = os.path.join(local_model_dir, "BirdNET_GLOBAL_6K_V2.4_Model_FP32.tflite")
birdnetlib.analyzer.LABEL_PATH = os.path.join(local_model_dir, "BirdNET_GLOBAL_6K_V2.4_Labels.txt")

from birdnetlib.analyzer import Analyzer
from birdnetlib import Recording

# --- KONFIGURATION ---
DB_FILE = "birds_audio_stats.db"
SETTINGS_FILE = "settings.json"
DICTIONARY_FILE = "dictionary.json"
BIRDWEATHER_FILE = "birdweather.json"
BIRDWEATHER_QUEUE_DIR = "birdweather_queue"
MAX_BIRDWEATHER_QUEUE = 500
FLASK_PORT = 5001
RECORD_SECONDS = 3
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 48000 # BirdNET standard is 48kHz
MIN_CONFIDENCE = 0.3 # Konfidenz-Schwellenwert
AUDIO_DIR = "audio_temp"
os.makedirs(AUDIO_DIR, exist_ok=True)
os.makedirs(BIRDWEATHER_QUEUE_DIR, exist_ok=True)
TEMP_WAV = os.path.join(AUDIO_DIR, "temp.wav")


app = Flask(__name__)
log_messages = deque(maxlen=100)
latest_audio_level = 0
latest_queue_length = 0

# --- MULTIPROCESSING GLOBALS ---
log_queue_global = None
shared_audio_level_global = None
shared_queue_length_global = None
shared_waterfall_global = None
monitor_running_event = None
monitor_process = None
# -------------------------------
WATERFALL_HEIGHT = 150
MAX_FREQ = 20000
freq_resolution = RATE / CHUNK
max_bin = int(MAX_FREQ / freq_resolution)
latest_waterfall_data = np.zeros((WATERFALL_HEIGHT, max_bin), dtype=np.float32)


DEFAULT_BIRD_TRANSLATIONS = {
    # Meisen & Baumläufer
    "Great Tit": "Kohlmeise", "Eurasian Blue Tit": "Blaumeise", "Coal Tit": "Tannenmeise", 
    "Crested Tit": "Haubenmeise", "Marsh Tit": "Sumpfmeise", "Willow Tit": "Weidenmeise", 
    "Long-tailed Tit": "Schwanzmeise", "Eurasian Nuthatch": "Kleiber", 
    "Eurasian Treecreeper": "Waldbaumläufer", "Short-toed Treecreeper": "Gartenbaumläufer",
    
    # Finken & Sperlinge
    "House Sparrow": "Haussperling", "Eurasian Tree Sparrow": "Feldsperling",
    "Common Chaffinch": "Buchfink", "European Greenfinch": "Grünfink", "European Goldfinch": "Stieglitz",
    "Eurasian Siskin": "Erlenzeisig", "Common Linnet": "Bluthänfling", "Eurasian Linnet": "Bluthänfling", "Eurasian Bullfinch": "Gimpel (Dompfaff)",
    "Hawfinch": "Kernbeißer", "Yellowhammer": "Goldammer", "Common Reed Bunting": "Rohrammer", "Reed Bunting": "Rohrammer", "Common Reed-Bunting": "Rohrammer",
    
    # Drosseln, Grasmücken & Fliegenschnäpper
    "Eurasian Blackbird": "Amsel", "Song Thrush": "Singdrossel", "Mistle Thrush": "Misteldrossel",
    "Fieldfare": "Wacholderdrossel", "Redwing": "Rotdrossel", "Ring Ouzel": "Ringdrossel",
    "European Robin": "Rotkehlchen", "Common Nightingale": "Nachtigall", 
    "Black Redstart": "Hausrotschwanz", "Common Redstart": "Gartenrotschwanz",
    "Eurasian Blackcap": "Mönchsgrasmücke", "Garden Warbler": "Gartengrasmücke", 
    "Common Whitethroat": "Dorngrasmücke", "Lesser Whitethroat": "Klappergrasmücke",
    "Common Chiffchaff": "Zilpzalp", "Willow Warbler": "Fitis", "Wood Warbler": "Waldlaubsänger",
    "Icterine Warbler": "Gelbspötter",
    "Goldcrest": "Wintergoldhähnchen", "Firecrest": "Sommergoldhähnchen",
    "Spotted Flycatcher": "Grauschnäpper", "European Pied Flycatcher": "Trauerschnäpper",
    "European Stonechat": "Schwarzkehlchen", "Whinchat": "Braunkehlchen", "Northern Wheatear": "Steinschmätzer",
    "Eurasian Wren": "Zaunkönig", "Dunnock": "Heckenbraunelle",
    
    # Schwalben & Segler
    "Common Swift": "Mauersegler", "Barn Swallow": "Rauchschwalbe", "Common House-Martin": "Mehlschwalbe",
    
    # Spechte
    "Great Spotted Woodpecker": "Buntspecht", "Middle Spotted Woodpecker": "Mittelspecht", 
    "Lesser Spotted Woodpecker": "Kleinspecht", "Black Woodpecker": "Schwarzspecht", 
    "European Green Woodpecker": "Grünspecht", "Eurasian Green Woodpecker": "Grünspecht", "Grey-headed Woodpecker": "Grauspecht", "Eurasian Wryneck": "Wendehals",
    
    # Rabenvögel & Stare
    "Common Starling": "Star", "Eurasian Magpie": "Elster", "Eurasian Jay": "Eichelhäher",
    "Eurasian Jackdaw": "Dohle", "Rook": "Saatkrähe", "Carrion Crow": "Rabenkrähe", 
    "Hooded Crow": "Nebelkrähe", "Northern Raven": "Kolkrabe",
    
    # Tauben, Kuckuck & Fasan
    "Common Wood-Pigeon": "Ringeltaube", "Eurasian Collared-Dove": "Türkentaube", 
    "Feral Pigeon": "Straßentaube", "Rock Pigeon": "Felsentaube", "Stock Dove": "Hohltaube", "European Turtle-Dove": "Turteltaube",
    "Common Cuckoo": "Kuckuck", "Ring-necked Pheasant": "Jagdfasan",
    
    # Greifvögel & Eulen
    "Eurasian Kestrel": "Turmfalke", "Common Buzzard": "Mäusebussard", "Red Kite": "Rotmilan",
    "Northern Goshawk": "Habicht", "Eurasian Sparrowhawk": "Sperber", 
    "Tawny Owl": "Waldkauz", "Barn Owl": "Schleiereule", "Little Owl": "Steinkauz", "Long-eared Owl": "Waldohreule",
    
    # Wasservögel & Reiher
    "Gray Heron": "Graureiher", "Grey Heron": "Graureiher", "Great Egret": "Silberreiher",
    "White Stork": "Weißstorch", "Black Stork": "Schwarzstorch",
    "Mute Swan": "Höckerschwan", "Greylag Goose": "Graugans", "Graylag Goose": "Graugans", "Canada Goose": "Kanadagans",
    "Mallard": "Stockente", "Eurasian Teal": "Krickente", "Tufted Duck": "Reiherente", "Gadwall": "Schnatterente",
    "Great Crested Grebe": "Haubentaucher", "Little Grebe": "Zwergtaucher", "Great Cormorant": "Kormoran",
    "Water Rail": "Wasserralle", "Common Moorhen": "Teichhuhn", "Eurasian Coot": "Blässhuhn",
    "Black-headed Gull": "Lachmöwe", "Common Kingfisher": "Eisvogel",
    
    # Stelzen & Pieper
    "White Wagtail": "Bachstelze", "Grey Wagtail": "Gebirgsstelze", "Gray Wagtail": "Gebirgsstelze", "Western Yellow Wagtail": "Schafstelze",
    "Tree Pipit": "Baumpieper", "Meadow Pipit": "Wiesenpieper", "Water Pipit": "Bergpieper",
    "Eurasian Hoopoe": "Wiedehopf", "Common Crane": "Kranich", "Eurasian Golden Oriole": "Pirol", "Common Raven": "Kolkrabe",
    
    # Lerchen
    "Eurasian Skylark": "Feldlerche", "Wood Lark": "Heckenlerche",
    
    # Würger
    "Red-backed Shrike": "Neuntöter"
}

def load_dictionary():
    if os.path.exists(DICTIONARY_FILE):
        try:
            with open(DICTIONARY_FILE, 'r') as f:
                return json.load(f)
        except:
            pass
    return DEFAULT_BIRD_TRANSLATIONS

def save_dictionary(dict_data):
    try:
        with open(DICTIONARY_FILE, 'w') as f:
            json.dump(dict_data, f)
    except:
        pass

def get_bird_dictionary():
    raw_dict = load_dictionary()
    flat_dict = {}
    for k, v in raw_dict.items():
        if isinstance(v, dict):
            trans = v.get("translation", "")
            flat_dict[k] = trans if str(trans).strip() != "" else k
        else:
            flat_dict[k] = v if str(v).strip() != "" else k
    return flat_dict

# Analyzer laden (wird im Hintergrundprozess initialisiert)
analyzer = None

def update_log(msg):
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    formatted = f"[{ts}] {msg}"
    # print(formatted) # Terminal-Ausgabe deaktiviert für Dauerbetrieb
    if log_queue_global is not None:
        try:
            log_queue_global.put(formatted, block=False)
        except:
            pass
    else:
        log_messages.appendleft(formatted)

def load_settings():
    if os.path.exists(SETTINGS_FILE):
        try:
            with open(SETTINGS_FILE, 'r') as f:
                return json.load(f)
        except:
            pass
    return {}

def get_effective_occurrence_threshold():
    settings = load_settings()
    if not settings.get("auto_season_lowering", False):
        return float(settings.get("occurrence_threshold", 0.03))
    
    lowering_file = "auto_season_lowering.json"
    default_val = 0.03
    if os.path.exists(lowering_file):
        try:
            import datetime, json
            with open(lowering_file, "r") as f:
                data = json.load(f)
            default_val = float(data.get("default", 0.03))
            current_cw = datetime.datetime.now().isocalendar()[1]
            weeks = data.get("weeks", {})
            cw_str = str(current_cw)
            if cw_str in weeks:
                return float(weeks[cw_str])
        except Exception:
            pass
    return default_val

def load_birdweather_settings():
    if os.path.exists(BIRDWEATHER_FILE):
        try:
            with open(BIRDWEATHER_FILE, 'r') as f:
                return json.load(f)
        except:
            pass
    return {}

def save_birdweather_setting(key, value):
    data = load_birdweather_settings()
    data[key] = value
    try:
        with open(BIRDWEATHER_FILE, 'w') as f:
            json.dump(data, f)
    except:
        pass

def save_setting(key, value):
    data = load_settings()
    data[key] = value
    try:
        with open(SETTINGS_FILE, 'w') as f:
            json.dump(data, f)
    except:
        pass

def init_db():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute('PRAGMA journal_mode=WAL;')
    c.execute('''CREATE TABLE IF NOT EXISTS detections 
                 (id INTEGER PRIMARY KEY, species TEXT, timestamp TEXT, confidence REAL)''')
    try:
        c.execute('ALTER TABLE detections ADD COLUMN snr REAL DEFAULT 0.0')
    except sqlite3.OperationalError:
        pass # Spalte existiert bereits
    try:
        c.execute('ALTER TABLE detections ADD COLUMN geo_prob REAL DEFAULT 0.0')
    except sqlite3.OperationalError:
        pass # Spalte existiert bereits
    c.execute('CREATE INDEX IF NOT EXISTS idx_timestamp ON detections(timestamp)')
    conn.commit()
    conn.close()

def save_detection(species, confidence, snr=0.0, ts=None, geo_prob=0.0):
    try:
        conn = sqlite3.connect(DB_FILE)
        c = conn.cursor()
        if ts is None:
            ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        c.execute("INSERT INTO detections (species, timestamp, confidence, snr, geo_prob) VALUES (?, ?, ?, ?, ?)",
                  (species, ts, confidence, snr, geo_prob))
        conn.commit()
        conn.close()
    except Exception as e:
        update_log(f"DB Fehler: {e}")

class AudioMonitor:
    def __init__(self):
        self.running = False
        self.record_thread = None
        self.analyze_thread = None
        self.pa = pyaudio.PyAudio()
        self.audio_queue = queue.Queue(maxsize=10) # Maximal 10 Pakete Rückstand

    def start(self):
        self.running = True
        self.record_thread = threading.Thread(target=self.loop_record, daemon=True)
        self.analyze_thread = threading.Thread(target=self.loop_analyze, daemon=True)
        self.bw_queue_thread = threading.Thread(target=self.loop_birdweather_queue, daemon=True)
        self.record_thread.start()
        self.analyze_thread.start()
        self.bw_queue_thread.start()
        update_log("Audio-Ueberwachung (Multi-Thread) gestartet.")

    def stop(self):
        self.running = False
        update_log("Audio-Ueberwachung gestoppt.")

    def loop_birdweather_queue(self):
        while self.running:
            try:
                time.sleep(60)
                bw_settings = load_birdweather_settings()
                birdweather_id = bw_settings.get("birdweather_id", "").strip()
                if not birdweather_id or not bw_settings.get("birdweather_active", False):
                    continue
                
                queue_files = sorted([f for f in os.listdir(BIRDWEATHER_QUEUE_DIR) if f.endswith('.json')])
                for qf in queue_files:
                    if not self.running: break
                    base = qf[:-5]
                    json_path = os.path.join(BIRDWEATHER_QUEUE_DIR, qf)
                    flac_path = os.path.join(BIRDWEATHER_QUEUE_DIR, base + ".flac")
                    if not os.path.exists(flac_path):
                        os.remove(json_path)
                        continue
                    
                    with open(json_path, 'r') as f:
                        qdata = json.load(f)
                    
                    with open(flac_path, 'rb') as f:
                        flac_data = f.read()
                    
                    iso_time = qdata['iso_time']
                    soundscape_url = f'https://app.birdweather.com/api/v1/stations/{birdweather_id}/soundscapes?timestamp={iso_time}'
                    resp = requests.post(url=soundscape_url, data=flac_data, timeout=10, headers={'Content-Type': 'audio/flac'})
                    sdata = resp.json()
                    
                    if sdata.get('success'):
                        soundscape_id = sdata['soundscape']['id']
                        det_url = f'https://app.birdweather.com/api/v1/stations/{birdweather_id}/detections'
                        det_data = {
                            'timestamp': iso_time,
                            'lat': qdata['lat'],
                            'lon': qdata['lon'],
                            'soundscapeId': soundscape_id,
                            'soundscapeStartTime': 0.0,
                            'soundscapeEndTime': float(RECORD_SECONDS),
                            'commonName': qdata['eng_species'],
                            'scientificName': qdata.get('scientific_name', ''),
                            'algorithm': '2p4',
                            'confidence': qdata['confidence']
                        }
                        requests.post(det_url, json=det_data, timeout=10)
                        update_log(f"BirdWeather Queue gesendet: {qdata['eng_species']}")
                        os.remove(json_path)
                        os.remove(flac_path)
                    else:
                        raise Exception(sdata.get('message', 'Unbekannt'))
            except Exception as e:
                pass # Fehler (z.B. Offline) -> nächste Runde abwarten

    def loop_record(self):
        try:
            settings = load_settings()
            mic_index = int(settings.get("mic_index", -1))
            
            stream_kwargs = {
                'format': FORMAT,
                'channels': CHANNELS,
                'rate': RATE,
                'input': True,
                'frames_per_buffer': CHUNK
            }
            if mic_index >= 0:
                stream_kwargs['input_device_index'] = mic_index

            stream = self.pa.open(**stream_kwargs)
            
            while self.running:
                buffer_frames = getattr(self, '_buffer_frames', [])
                chunk_step_needed = int(RATE / CHUNK * 1.5)  # 1.5 Sekunden Schritte
                chunks_total_needed = int(RATE / CHUNK * RECORD_SECONDS) # 3 Sekunden Puffer
                frames_step = []
                
                # Filter vorbereiten, falls aktiv (für Live-Pegel Anzeige)
                current_settings = load_settings()
                hpf_active = current_settings.get("highpass_active", False)
                if hpf_active:
                    try:
                        cutoff = float(current_settings.get("highpass_freq", 150))
                        b, a = signal.butter(4, cutoff / (0.5 * RATE), btype='high', analog=False)
                        zi = signal.lfilter_zi(b, a) * 0.0
                    except:
                        hpf_active = False

                lpf_active = current_settings.get("lowpass_active", False)
                if lpf_active:
                    try:
                        lpf_cutoff = float(current_settings.get("lowpass_freq", 12000))
                        b_lpf, a_lpf = signal.butter(4, lpf_cutoff / (0.5 * RATE), btype='low', analog=False)
                        zi_lpf = signal.lfilter_zi(b_lpf, a_lpf) * 0.0
                    except:
                        lpf_active = False

                for i in range(chunk_step_needed):
                    if not self.running:
                        break
                    data = stream.read(CHUNK, exception_on_overflow=False)
                    frames_step.append(data)
                    
                    try:
                        audio_chunk = np.frombuffer(data, dtype=np.int16)
                        audio_chunk_proc = audio_chunk
                        if hpf_active:
                            audio_chunk_proc, zi = signal.lfilter(b, a, audio_chunk_proc, zi=zi)
                        if lpf_active:
                            audio_chunk_proc, zi_lpf = signal.lfilter(b_lpf, a_lpf, audio_chunk_proc, zi=zi_lpf)
                        
                        chunk_amp = int(np.max(np.abs(audio_chunk_proc)))
                        vis_audio = audio_chunk_proc.astype(np.float32)
                        
                        if shared_audio_level_global is not None:
                            shared_audio_level_global.value = chunk_amp
                        else:
                            global latest_audio_level
                            latest_audio_level = chunk_amp

                        # Waterfall processing
                        global latest_waterfall_data, max_bin
                        window = np.hanning(len(vis_audio))
                        vis_audio = vis_audio * window
                        fft_data = np.fft.rfft(vis_audio)
                        fft_mag = np.abs(fft_data)
                        fft_mag = 20 * np.log10(fft_mag + 1e-6)
                        fft_mag_cropped = fft_mag[:max_bin]
                        
                        nr_active = current_settings.get("nr_active", False)
                        if nr_active:
                            quality = current_settings.get("nr_quality", "Medium")
                            perc = 50 if quality == "Low" else (90 if quality == "High" else 75)
                            reduction = 10 if quality == "Low" else (30 if quality == "High" else 20)
                            noise_floor = np.percentile(fft_mag_cropped, perc)
                            fft_mag_cropped = np.where(fft_mag_cropped < noise_floor, fft_mag_cropped - reduction, fft_mag_cropped)
                            
                        min_db = -20
                        max_db = 100
                        fft_norm = np.clip((fft_mag_cropped - min_db) / (max_db - min_db), 0, 1)
                        latest_waterfall_data = np.roll(latest_waterfall_data, 1, axis=0)
                        latest_waterfall_data[0, :] = fft_norm
                        if shared_waterfall_global is not None:
                            with shared_waterfall_global.get_lock():
                                np_shared = np.frombuffer(shared_waterfall_global.get_obj(), dtype=np.float32)
                                np_shared[:] = latest_waterfall_data.flatten()
                    except:
                        pass

                self._buffer_frames = buffer_frames + frames_step
                if len(self._buffer_frames) > chunks_total_needed:
                    self._buffer_frames = self._buffer_frames[-chunks_total_needed:]

                if self.running and len(self._buffer_frames) == chunks_total_needed:
                    raw_data = b''.join(self._buffer_frames)
                    # Blockiert nicht ewig, wenn Queue voll ist (verwirft im Zweifel alte Daten)
                    if self.audio_queue.full():
                        try:
                            self.audio_queue.get_nowait()
                        except:
                            pass
                    self.audio_queue.put(raw_data)
                    
                    if shared_queue_length_global is not None:
                        shared_queue_length_global.value = self.audio_queue.qsize()
                    else:
                        global latest_queue_length
                        latest_queue_length = self.audio_queue.qsize()

            stream.stop_stream()
            stream.close()
        except Exception as e:
            update_log(f"Fehler im Aufnahme-Thread: {e}")

    def _commit_detection(self, det):
        species = det['species']
        confidence = det['confidence']
        calculated_snr = det['snr']
        is_new_species = det['is_new_species']
        raw_data = det['raw_data']
        eng_species = det['eng_species']
        best = det['best']
        lat = det['lat']
        lon = det['lon']
        geo_prob = det.get('geo_prob', 0.0)
        settings = det['settings']
        
        update_log(f"Erkannt: {species} ({confidence:.0%}) | SNR: {calculated_snr:.1f}dB | Geo-Prob: {geo_prob*100:.2f}%")
        
        now_dt = datetime.datetime.now()
        ts_db = now_dt.strftime("%Y-%m-%d %H:%M:%S")
        ts_file = now_dt.strftime("%y-%m-%d-%H-%M-%S")
        save_detection(species, confidence, calculated_snr, ts_db, geo_prob)
        
        if is_new_species:
            full_bird_dict = load_dictionary()
            if eng_species not in full_bird_dict:
                full_bird_dict[eng_species] = {
                    "translation": "",
                    "blocklist": False,
                    "ind_conf": "",
                    "force_active": False,
                    "link": "",
                    "priority": False
                }
                save_dictionary(full_bird_dict)
                update_log(f"INFO: Neue Art ins Dictionary aufgenommen: {eng_species}")
        
        temp_commit_wav = "temp_commit.wav"
        try:
            wf = wave.open(temp_commit_wav, 'wb')
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(self.pa.get_sample_size(FORMAT))
            wf.setframerate(RATE)
            wf.writeframes(raw_data)
            wf.close()
        except Exception as e:
            update_log(f"Fehler beim Speichern der temporären Audio-Datei: {e}")
            return
            
        bw_settings = load_birdweather_settings()
        birdweather_id = bw_settings.get("birdweather_id", "").strip()
        birdweather_active = bw_settings.get("birdweather_active", False)
        if birdweather_active and birdweather_id:
            try:
                data_sf, samplerate_sf = sf.read(temp_commit_wav)
                buf_sf = io.BytesIO()
                sf.write(buf_sf, data_sf, samplerate_sf, format='FLAC')
                flac_data = buf_sf.getvalue()
                
                iso_time = datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S.000Z")
                soundscape_url = f'https://app.birdweather.com/api/v1/stations/{birdweather_id}/soundscapes?timestamp={iso_time}'
                
                resp = requests.post(url=soundscape_url, data=flac_data, timeout=10, headers={'Content-Type': 'audio/flac'})
                sdata = resp.json()
                if sdata.get('success'):
                    soundscape_id = sdata['soundscape']['id']
                    det_url = f'https://app.birdweather.com/api/v1/stations/{birdweather_id}/detections'
                    det_data = {
                        'timestamp': iso_time,
                        'lat': lat,
                        'lon': lon,
                        'soundscapeId': soundscape_id,
                        'soundscapeStartTime': 0.0,
                        'soundscapeEndTime': float(RECORD_SECONDS),
                        'commonName': eng_species,
                        'scientificName': best.get('scientific_name', ''),
                        'algorithm': '2p4',
                        'confidence': confidence
                    }
                    requests.post(det_url, json=det_data, timeout=10)
                    update_log(f"An BirdWeather gesendet: {eng_species}")
                else:
                    raise Exception(f"API Fehler: {sdata.get('message')}")
            except Exception as e:
                update_log(f"BirdWeather Upload verzögert ({e}), ab in Warteschlange.")
                try:
                    ts_filename = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                    flac_path = os.path.join(BIRDWEATHER_QUEUE_DIR, f"{ts_filename}.flac")
                    json_path = os.path.join(BIRDWEATHER_QUEUE_DIR, f"{ts_filename}.json")
                    with open(flac_path, 'wb') as f:
                        f.write(flac_data)
                    queue_data = {
                        'iso_time': iso_time,
                        'lat': lat,
                        'lon': lon,
                        'eng_species': eng_species,
                        'scientific_name': best.get('scientific_name', ''),
                        'confidence': confidence
                    }
                    with open(json_path, 'w') as f:
                        json.dump(queue_data, f)
                    
                    all_json = sorted([f for f in os.listdir(BIRDWEATHER_QUEUE_DIR) if f.endswith('.json')])
                    if len(all_json) > MAX_BIRDWEATHER_QUEUE:
                        for f_to_del in all_json[:-MAX_BIRDWEATHER_QUEUE]:
                            base = f_to_del[:-5]
                            try:
                                os.remove(os.path.join(BIRDWEATHER_QUEUE_DIR, base + ".json"))
                                os.remove(os.path.join(BIRDWEATHER_QUEUE_DIR, base + ".flac"))
                            except:
                                pass
                except Exception as qe:
                    update_log(f"Fehler Warteschlange: {qe}")
        
        archive_species_str = settings.get("archive_species", "")
        if archive_species_str:
            archive_list = [s.strip().lower() for s in archive_species_str.split(',') if s.strip()]
            should_archive = species.lower() in archive_list or "*" in archive_list or "alle" in archive_list
            if not should_archive and "neu" in archive_list and is_new_species:
                should_archive = True

            if should_archive:
                import shutil
                archive_dir = os.path.join(AUDIO_DIR, "archive")
                if not os.path.exists(archive_dir):
                    os.makedirs(archive_dir)
                
                safe_species = species.replace(" ", "_").replace("/", "_")
                max_archive_files = int(settings.get("max_archive_files", 0))
                can_save = True
                
                if max_archive_files > 0:
                    existing_files = [f for f in os.listdir(archive_dir) if f.startswith(safe_species + "_") and f.endswith(".wav")]
                    if len(existing_files) >= max_archive_files:
                        can_save = False
                
                if can_save:
                    new_filename = f"{safe_species}_{ts_file}.wav"
                    new_filepath = os.path.join(archive_dir, new_filename)
                    
                    try:
                        shutil.copy(temp_commit_wav, new_filepath)
                        update_log(f"Audio archiviert: {new_filename}")
                    except Exception as e:
                        update_log(f"Fehler beim Archivieren: {e}")

    def loop_analyze(self):
        previous_detected_species = set()
        pending_detections = {}
        while self.running:
            try:
                # Wartet auf neues Audio-Paket (max 1 Sekunde, um while-Bedingung regelmäßig zu prüfen)
                try:
                    raw_data = self.audio_queue.get(timeout=1.0)
                    if shared_queue_length_global is not None:
                        shared_queue_length_global.value = self.audio_queue.qsize()
                    else:
                        global latest_queue_length
                        latest_queue_length = self.audio_queue.qsize()
                except queue.Empty:
                    continue
                
                # Filters
                settings = load_settings()
                
                audio_data_proc = np.frombuffer(raw_data, dtype=np.int16).astype(np.float32)
                processed = False
                
                if settings.get("highpass_active", False):
                    try:
                        cutoff = float(settings.get("highpass_freq", 150))
                        if 0 < cutoff < RATE / 2:
                            b_hp, a_hp = signal.butter(4, cutoff / (0.5 * RATE), btype='high', analog=False)
                            audio_data_proc = signal.filtfilt(b_hp, a_hp, audio_data_proc)
                            processed = True
                    except Exception as e:
                        print(f"Fehler bei Highpass Filter: {e}")

                if settings.get("lowpass_active", False):
                    try:
                        lpf_cutoff = float(settings.get("lowpass_freq", 12000))
                        if 0 < lpf_cutoff < RATE / 2:
                            b_lpf_2, a_lpf_2 = signal.butter(4, lpf_cutoff / (0.5 * RATE), btype='low', analog=False)
                            audio_data_proc = signal.filtfilt(b_lpf_2, a_lpf_2, audio_data_proc)
                            processed = True
                    except Exception as e:
                        print(f"Fehler bei Lowpass Filter: {e}")
                        
                if processed:
                    raw_data = np.clip(audio_data_proc, -32768, 32767).astype(np.int16).tobytes()

                # Noise Reduction (using noisereduce)
                if settings.get("nr_active", False):
                    try:
                        import noisereduce as nr
                        quality = settings.get("nr_quality", "Medium")
                        if quality == "Low":
                            prop_decrease = 0.5
                        elif quality == "Medium":
                            prop_decrease = 0.8
                        elif quality == "High":
                            prop_decrease = 1.0
                        else:
                            prop_decrease = 0.8
                            
                        audio_data_nr = np.frombuffer(raw_data, dtype=np.int16).astype(np.float32)
                        
                        # Apply noisereduce
                        audio_data_nr = nr.reduce_noise(y=audio_data_nr, sr=RATE, prop_decrease=prop_decrease)
                        
                        raw_data = np.clip(audio_data_nr, -32768, 32767).astype(np.int16).tobytes()
                    except Exception as e:
                        print(f"Fehler bei Noise Reduction: {e}")

                # Speichern in Temp-Datei
                wf = wave.open(TEMP_WAV, 'wb')
                wf.setnchannels(CHANNELS)
                wf.setsampwidth(self.pa.get_sample_size(FORMAT))
                wf.setframerate(RATE)
                wf.writeframes(raw_data)
                wf.close()

                # SNR berechnen
                try:
                    audio_data = np.frombuffer(raw_data, dtype=np.int16)
                    if np.max(np.abs(audio_data)) > 0:
                        rms_signal = np.sqrt(np.mean(np.square(audio_data.astype(np.float32))))
                        window_size = RATE // 10
                        windows = [audio_data[i:i+window_size] for i in range(0, len(audio_data), window_size)]
                        rms_windows = [np.sqrt(np.mean(np.square(w.astype(np.float32)))) for w in windows if len(w) > 0]
                        noise_floor = np.percentile(rms_windows, 10) if rms_windows else 1.0
                        noise_floor = max(noise_floor, 1.0)
                        calculated_snr = float(20 * np.log10(rms_signal / noise_floor))
                    else:
                        calculated_snr = 0.0
                except Exception as e:
                    calculated_snr = 0.0
                    print(f"Fehler bei SNR Berechnung: {e}")

                # BirdNET Klassifizierung
                settings = load_settings()
                bird_dict = get_bird_dictionary()
                lat_val = settings.get("gps_lat")
                try:
                    lat = float(lat_val) if lat_val else -1.0
                except ValueError:
                    lat = -1.0
                    
                lon_val = settings.get("gps_lon")
                try:
                    lon = float(lon_val) if lon_val else -1.0
                except ValueError:
                    lon = -1.0
                
                # Update Occurrence Threshold dynamically
                occ_thresh = get_effective_occurrence_threshold()

                recording = Recording(
                    analyzer,
                    TEMP_WAV,
                    lat=lat,
                    lon=lon,
                    date=datetime.datetime.now(),
                    min_conf=MIN_CONFIDENCE,
                    filter_threshold=occ_thresh
                )

                # Check for forced species in bird_dictionary
                full_bird_dict = load_dictionary()
                forced_species = []
                for eng_name, props in full_bird_dict.items():
                    if isinstance(props, dict) and props.get("force_active", False):
                        forced_species.append(eng_name)

                with open(os.devnull, 'w') as f, contextlib.redirect_stdout(f), contextlib.redirect_stderr(f):
                    recording.analyze()
                
                # Ergebnisse verarbeiten
                valid_detections = recording.detections.copy() if recording.detections else []
                
                geo_prob_dict = {}
                try:
                    if hasattr(analyzer, 'species_class'):
                        with open(os.devnull, 'w') as f_out, contextlib.redirect_stdout(f_out), contextlib.redirect_stderr(f_out):
                            geo_probs = analyzer.species_class.return_list(lon=lon, lat=lat, date=datetime.datetime.now(), threshold=0.0)
                        geo_prob_dict = {item['common_name']: item['threshold'] for item in geo_probs}
                except Exception:
                    pass
                
                # Manuell die Forced Species hinzufügen und lokationsgefilterte Arten loggen
                if hasattr(recording, 'detection_list'):
                    allowed_labels = {d['label'] for d in valid_detections}
                    app_min_conf = float(settings.get("threshold", MIN_CONFIDENCE * 100)) / 100.0
                    
                    for raw_d in recording.detection_list:
                        eng_species = raw_d.common_name
                        species_min_conf = app_min_conf
                        
                        if eng_species in full_bird_dict and isinstance(full_bird_dict[eng_species], dict):
                            ind_conf_val = full_bird_dict[eng_species].get("ind_conf")
                            if ind_conf_val is not None and str(ind_conf_val).strip() != "":
                                try:
                                    species_min_conf = float(ind_conf_val) / 100.0
                                except (ValueError, TypeError):
                                    pass

                        if raw_d.label not in allowed_labels:
                            if forced_species and eng_species in forced_species:
                                if raw_d.confidence >= recording.minimum_confidence:
                                    try:
                                        forced_dict = recording.return_detection_dict(raw_d)
                                        valid_detections.append(forced_dict)
                                        allowed_labels.add(raw_d.label)
                                    except Exception as e:
                                        update_log(f"Fehler beim manuellen Hinzufügen von Force-Species: {e}")
                            else:
                                if raw_d.confidence >= species_min_conf:
                                    try:
                                        species = bird_dict.get(eng_species, eng_species)
                                        if settings.get("log_blocklist", True):
                                            geo_prob_val = geo_prob_dict.get(eng_species, 0.0)
                                            prob_switch_active = settings.get("log_blocklist_prob_switch", False)
                                            
                                            should_log = True
                                            if prob_switch_active and round(geo_prob_val * 100, 2) <= 0.0:
                                                should_log = False
                                                
                                            if should_log:
                                                with open("blocklist-log.txt", "a", encoding="utf-8") as f:
                                                    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                                                    f.write(f"{ts} > Low probability for your area,LAT:{lat},LON:{lon},K:{raw_d.confidence*100:.0f}%,P:{geo_prob_val*100:.2f}%,SNR:{calculated_snr:.1f}dB,Species:{species} > erkannt\n")
                                    except Exception as e:
                                        pass

                if valid_detections:
                    current_detected_species = set()
                    
                    for best in valid_detections:
                        eng_species = best['common_name']
                        
                        # Zeitbasierte Plausibilitätskorrektur: Sperber (tagaktiv) -> Waldohreule (nachtaktiv)
                        # Bettelrufe von jungen Waldohreulen werden nachts oft fälschlich als Sperber erkannt.
                        current_hour = datetime.datetime.now().hour
                        if eng_species == "Eurasian Sparrowhawk" and (current_hour >= 21 or current_hour <= 4):
                            eng_species = "Long-eared Owl"
                            update_log("INFO: Sperber in der Nacht erkannt -> Automatisch korrigiert zu Waldohreule")

                        species = bird_dict.get(eng_species, eng_species)
                        confidence = float(best['confidence'])
                        
                        min_conf = float(settings.get("threshold", MIN_CONFIDENCE * 100)) / 100.0
                        min_snr_val = float(settings.get("min_snr", 0.0))
                        full_bird_dict = load_dictionary()

                        is_blocklisted = False
                        if eng_species in full_bird_dict and isinstance(full_bird_dict[eng_species], dict):
                            is_blocklisted = full_bird_dict[eng_species].get("blocklist", False)
                            
                            ind_conf_val = full_bird_dict[eng_species].get("ind_conf")
                            if ind_conf_val is not None and str(ind_conf_val).strip() != "":
                                try:
                                    min_conf = float(ind_conf_val) / 100.0
                                except (ValueError, TypeError):
                                    pass

                        if is_blocklisted:
                            if confidence >= min_conf and calculated_snr > min_snr_val:
                                try:
                                    if settings.get("log_blocklist", True):
                                        geo_prob_val = geo_prob_dict.get(eng_species, 0.0)
                                        prob_switch_active = settings.get("log_blocklist_prob_switch", False)
                                        
                                        should_log = True
                                        if prob_switch_active and round(geo_prob_val * 100, 2) <= 0.0:
                                            should_log = False
                                            
                                        if should_log:
                                            with open("blocklist-log.txt", "a", encoding="utf-8") as f:
                                                ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                                                f.write(f"{ts}, {species}, {confidence*100:.0f}%, {calculated_snr:.1f} dB, Geo-Prob: {geo_prob_val*100:.2f}%, {lat}, {lon}\n")
                                except Exception as e:
                                    pass
                                update_log(f"Ignoriert (Blocklist): {species}")
                                
                            # If a bird is blocklisted, remove pending
                            if species in pending_detections:
                                del pending_detections[species]
                                
                        elif confidence >= min_conf and calculated_snr > min_snr_val:
                            current_detected_species.add(species)
                            
                            is_new_species = False
                            try:
                                conn_check = sqlite3.connect(DB_FILE)
                                c_check = conn_check.cursor()
                                c_check.execute("SELECT COUNT(*) FROM detections WHERE species = ?", (species,))
                                if c_check.fetchone()[0] == 0:
                                    is_new_species = True
                                conn_check.close()
                            except Exception as e:
                                print(f"Fehler bei DB-Check für neue Art: {e}")

                            geo_prob_val = geo_prob_dict.get(eng_species, 0.0)
                            current_det_data = {
                                'species': species,
                                'confidence': confidence,
                                'snr': calculated_snr,
                                'is_new_species': is_new_species,
                                'raw_data': raw_data,
                                'eng_species': eng_species,
                                'best': best,
                                'lat': lat,
                                'lon': lon,
                                'geo_prob': geo_prob_val,
                                'settings': settings
                            }

                            if species in pending_detections:
                                if confidence > pending_detections[species]['confidence']:
                                    self._commit_detection(current_det_data)
                                else:
                                    self._commit_detection(pending_detections[species])
                                previous_detected_species.add(species)
                                del pending_detections[species]
                            elif species in previous_detected_species:
                                pass
                            else:
                                pending_detections[species] = current_det_data

                    # Commit pending detections that are not in current chunk (they stopped singing or dropped below threshold)
                    for sp in list(pending_detections.keys()):
                        if sp not in current_detected_species:
                            self._commit_detection(pending_detections[sp])
                            previous_detected_species.add(sp)
                            del pending_detections[sp]
                        
                    # Maintain streak only for species in current chunk
                    new_prev = set()
                    for sp in current_detected_species:
                        if sp in previous_detected_species:
                            new_prev.add(sp)
                    previous_detected_species = new_prev
                    
                else:
                    # Keine Erkennung in diesem Chunk, alle verbleibenden pendings speichern
                    for sp in list(pending_detections.keys()):
                        self._commit_detection(pending_detections[sp])
                        previous_detected_species.add(sp)
                    pending_detections.clear()
                    previous_detected_species.clear()
                
                self.audio_queue.task_done()

            except Exception as e:
                update_log(f"Fehler in Analyse-Schleife: {e}")
                time.sleep(1)


# --- FLASK ROUTEN ---
@app.context_processor
def inject_version():
    return dict(version="V1.3.5-RC4", year="2026")

@app.route('/favicon.ico')
def favicon():
    return send_from_directory(os.path.join(app.root_path, 'static'),
                               'feather.ico', mimetype='image/vnd.microsoft.icon')

@app.route('/')
def index():
    s = load_settings()
    icon_dir = os.path.join(app.root_path, 'static', 'bird_icons')
    available_icons = [f for f in os.listdir(icon_dir) if os.path.isfile(os.path.join(icon_dir, f))] if os.path.exists(icon_dir) else []
    return render_template('index.html', s=s, available_icons=available_icons)

@app.route('/manual_pdf')
def manual_pdf():
    return send_from_directory(os.getcwd(), 'Einstellungen_Beschreibung.pdf')

@app.route('/settings')
def settings_page():
    s = load_settings()
    bw = load_birdweather_settings()
    s['birdweather_id'] = bw.get('birdweather_id', '')
    s['birdweather_active'] = bw.get('birdweather_active', False)
    s["bird_dictionary"] = load_dictionary()
    
    queue_size = 0
    if os.path.exists(BIRDWEATHER_QUEUE_DIR):
        queue_size = len([f for f in os.listdir(BIRDWEATHER_QUEUE_DIR) if f.endswith('.json')])
    
    pa = pyaudio.PyAudio()
    mics = []
    for i in range(pa.get_device_count()):
        try:
            info = pa.get_device_info_by_index(i)
            if info.get("maxInputChannels", 0) > 0:
                try:
                    name = info.get("name", f"Device {i}")
                    if isinstance(name, bytes):
                        name = name.decode('utf-8', errors='ignore')
                except Exception:
                    name = f"Unknown Device {i}"
                mics.append({"index": i, "name": name})
        except Exception:
            pass
    pa.terminate()
    return render_template('settings.html', s=s, mics=mics, queue_size=queue_size)

def create_chart(title, labels, values):
    height = max(6, len(labels) * 0.4)
    plt.figure(figsize=(10, height), facecolor='#1e1e1e')
    ax = plt.axes()
    ax.set_facecolor('#1e1e1e')
    ax.tick_params(colors='white')
    for spine in ax.spines.values():
        spine.set_color('#444')
    
    color_palette = ['#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231', '#911eb4', '#46f0f0', '#f032e6', '#bcf60c', '#fabebe', '#008080', '#e6beff', '#9a6324', '#fffac8', '#800000', '#aaffc3', '#808000', '#ffd8b1', '#000075', '#808080', '#ffffff', '#000000']
    bar_colors = [color_palette[i % len(color_palette)] for i in range(len(labels))]
    
    labels_rev = list(reversed(labels))
    values_rev = list(reversed(values))
    bar_colors_rev = list(reversed(bar_colors))
    
    bars = plt.barh(labels_rev, values_rev, color=bar_colors_rev)
    if title:
        plt.title(title, color='white')
    else:
        plt.title('Anzahl', color='white', pad=20)
        
    plt.xscale('symlog', subs=list(range(1, 10)))
    
    max_val = max(values) if values else 10
    plt.xlim(left=0, right=max(10, max_val * 1.1))
    
    # Tick parameters for both top and bottom
    ax.tick_params(axis='x', which='both', bottom=True, top=True, labelbottom=True, labeltop=True, colors='white')
    ax.tick_params(axis='y', colors='white')
    ax.set_xlabel('Anzahl', color='white')
    plt.grid(True, which='major', axis='x', color='#666', linestyle='-', alpha=0.5)
    plt.grid(True, which='minor', axis='x', color='#444', linestyle='--', alpha=0.5)
    plt.tight_layout()
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', facecolor='#1e1e1e')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()
    return f"data:image/png;base64,{img_base64}"

def create_daily_line_chart(title, all_detections):
    plt.figure(figsize=(10, 6), facecolor='#1e1e1e')
    ax = plt.axes()
    ax.set_facecolor('#1e1e1e')
    ax.tick_params(colors='white')
    for spine in ax.spines.values():
        spine.set_color('#444')
    
    hours = list(range(24))
    hour_labels = [f"{h:02d}:00" for h in hours]
    
    if not all_detections:
        plt.plot([], [])
    else:
        from collections import defaultdict
        species_hourly = defaultdict(lambda: [0]*24)
        for species, timestamp in all_detections:
            try:
                hour = int(timestamp.split(' ')[1].split(':')[0])
                species_hourly[species][hour] += 1
            except:
                pass
                
        for species, counts in species_hourly.items():
            if sum(counts) > 0:
                plt.plot(hours, counts, label=species, marker='o', markersize=4, linewidth=2)
                
        plt.legend(facecolor='#1e1e1e', labelcolor='white', edgecolor='#444', bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.title(title, color='white')
    plt.xticks(hours, hour_labels, rotation=45, ha='right', color='white')
    plt.yscale('symlog')
    plt.ylim(bottom=0)
    plt.yticks(color='white')
    plt.grid(color='#444', linestyle='--', linewidth=0.5, alpha=0.5)
    plt.tight_layout()
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()
    return f"data:image/png;base64,{img_base64}"

@app.route('/daily')
def daily_page():
    today = datetime.date.today()
    date_str = request.args.get('date', today.strftime('%Y-%m-%d'))
    try:
        dt = datetime.datetime.strptime(date_str, '%Y-%m-%d').date()
    except:
        dt = today
        date_str = dt.strftime('%Y-%m-%d')
        
    prev_date = (dt - datetime.timedelta(days=1)).strftime('%Y-%m-%d')
    next_date = (dt + datetime.timedelta(days=1)).strftime('%Y-%m-%d')
    
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT COUNT(*) as c FROM detections WHERE timestamp LIKE ?", (f"{date_str}%",))
    total_row = c.fetchone()
    total = total_row[0] if total_row else 0
    
    c.execute("SELECT species, timestamp FROM detections WHERE timestamp LIKE ? ORDER BY timestamp ASC LIMIT 1", (f"{date_str}%",))
    first = c.fetchone()
    conn.close()
    
    first_bird = first[0] if first else None
    first_bird_time = first[1].split(' ')[1][:5] if first else None

    s = load_settings()
    daily_chart = create_daily_total_chart(date_str)
    daily_pie = get_daily_pie_data(date_str)
    return render_template('daily.html', 
        s=s,
        selected_date_str=date_str, total_birds_day=total, 
        first_bird=first_bird, first_bird_time=first_bird_time,
        prev_date=prev_date, next_date=next_date,
        is_today=(dt == today), today_str=today.strftime('%Y-%m-%d'),
        table_content=generate_daily_heatmap_html(date_str),
        daily_chart=daily_chart,
        daily_pie=daily_pie
    )

def create_daily_total_chart(date_str):
    query = f"""
    SELECT 
        strftime('%H', timestamp) as hour_sort,
        COUNT(*) as counts,
        COUNT(DISTINCT species) as unique_species
    FROM detections
    WHERE timestamp LIKE '{date_str}%'
    GROUP BY hour_sort
    ORDER BY hour_sort
    """
    
    conn = sqlite3.connect(DB_FILE, timeout=10)
    try:
        grouped = pd.read_sql_query(query, conn)
    except:
        grouped = pd.DataFrame()
    finally:
        conn.close()

    fig, ax1 = plt.subplots(figsize=(10, 3), facecolor='#1e1e1e')
    ax1.set_facecolor('#1e1e1e')
    ax1.tick_params(colors='white')
    for spine in ax1.spines.values():
        spine.set_color('#444')
    
    hours = list(range(24))
    hour_labels = [f"{h:02d}:00" for h in hours]
    
    counts = [0] * 24
    species_counts = [0] * 24
    if not grouped.empty:
        for _, row in grouped.iterrows():
            if pd.notna(row['hour_sort']):
                try:
                    h = int(row['hour_sort'])
                    counts[h] = int(row['counts'])
                    if 'unique_species' in row:
                        species_counts[h] = int(row['unique_species'])
                except:
                    pass

    line1 = ax1.plot(hours, counts, color='#e5c07b', linewidth=2, marker='o', markersize=4, label='Rufe (Gesamt)')
    ax1.fill_between(hours, counts, color='#e5c07b', alpha=0.1)
    
    ax1.set_xticks(hours)
    ax1.set_xticklabels(hour_labels, rotation=45, ha='right', color='white')
    
    from matplotlib.ticker import MaxNLocator
    ax1.yaxis.set_major_locator(MaxNLocator(integer=True))
    
    ax1.set_ylim(bottom=0)
    if max(counts) == 0:
        ax1.set_ylim(top=10)
    
    ax2 = ax1.twinx()
    line2 = ax2.plot(hours, species_counts, color='#56b6c2', linewidth=2, marker='s', markersize=4, label='Arten (Diversifikation)')
    ax2.fill_between(hours, species_counts, color='#56b6c2', alpha=0.1)
    ax2.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax2.set_ylim(bottom=0)
    if max(species_counts) == 0:
        ax2.set_ylim(top=10)
    ax2.tick_params(colors='white')
    for spine in ax2.spines.values():
        spine.set_color('#444')

    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper left', facecolor='#263238', edgecolor='#444', labelcolor='white')

    ax1.grid(color='#444', linestyle='--', linewidth=0.5, alpha=0.5)
    plt.tight_layout()
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', transparent=True)
    plt.close()
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')

def get_daily_pie_data(date_str):
    query = f"SELECT species, COUNT(*) as count FROM detections WHERE timestamp LIKE '{date_str}%' GROUP BY species"
    
    conn = sqlite3.connect(DB_FILE, timeout=10)
    try:
        df = pd.read_sql_query(query, conn)
    except:
        df = pd.DataFrame()
    finally:
        conn.close()

    if df.empty:
        return None

    d = load_dictionary()
    status_counts = {}
    status_species = {}
    
    for _, row in df.iterrows():
        sp = row['species']
        # count = row['count']  # We no longer need the call count
        status = 'Unbekannt'
        if sp in d:
            status = d[sp].get('status', 'Unbekannt')
        else:
            for k, v in d.items():
                if v.get('translation') == sp:
                    status = v.get('status', 'Unbekannt')
                    break
                    
        if not status:
            status = 'Unbekannt'
            
        status_counts[status] = status_counts.get(status, 0) + 1
        if status not in status_species:
            status_species[status] = []
        if sp not in status_species[status]:
            status_species[status].append(sp)

    if not status_counts:
        return None
        
    labels = list(status_counts.keys())
    sizes = list(status_counts.values())
    
    import matplotlib.colors as mcolors
    colors_rgba = plt.cm.Set3(np.linspace(0, 1, len(labels)))
    colors_hex = [mcolors.to_hex(c) for c in colors_rgba]
    
    species_lists = [", ".join(status_species[status]) for status in labels]
    
    return {
        "labels": labels,
        "sizes": sizes,
        "colors": colors_hex,
        "species": species_lists
    }

def generate_daily_heatmap_html(date_str):
    query = f"""
    SELECT 
        CASE WHEN species = 'IGNORED_LOW_CONFIDENCE' THEN 'Unbekannt' ELSE species END as species,
        strftime('%H', timestamp) as hour_sort,
        strftime('%H', timestamp) || ':00' as hour_display,
        COUNT(*) as counts
    FROM detections
    WHERE timestamp LIKE '{date_str}%'
    GROUP BY species, hour_sort, hour_display
    ORDER BY hour_sort
    """
    
    conn = sqlite3.connect(DB_FILE, timeout=10)
    try:
        grouped = pd.read_sql_query(query, conn)
    except:
        grouped = pd.DataFrame()
    finally:
        conn.close()

    html_table = "<p>Keine Daten für diesen Tag vorhanden.</p>"

    icon_map = {}
    static_folder = os.path.join(app.root_path, 'static', 'bird_icons')
    if os.path.exists(static_folder):
        for f in os.listdir(static_folder):
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif')):
                icon_map[os.path.splitext(f)[0].lower()] = f"bird_icons/{f}"
                icon_map[f.lower()] = f"bird_icons/{f}"

    def get_bird_icon(sp):
        if not sp: return 'bird_icons/Unbekannt.png'
        clean = sp.strip().lower()
        if clean in icon_map:
            return icon_map[clean]
        if clean + '.png' in icon_map:
            return icon_map[clean + '.png']
        return 'bird_icons/Unbekannt.png'

    if not grouped.empty:
        pivot_counts = grouped.pivot(index='species', columns='hour_display', values='counts').fillna(0)
        
        # Sicherstellen dass alle 24 Stunden da sind
        all_hours = [f"{h:02d}:00" for h in range(24)]
        pivot_counts = pivot_counts.reindex(columns=all_hours, fill_value=0)
        
        hour_totals = pivot_counts.sum(axis=0)
        # Avoid division by zero
        pivot_pct = pivot_counts.div(hour_totals.replace(0, 1), axis=1).mul(100).fillna(0)
        
        total_counts = pivot_counts.sum(axis=1)
        pivot_pct['total_sort_idx'] = total_counts
        pivot_pct = pivot_pct.sort_values('total_sort_idx', ascending=False)
        pivot_pct = pivot_pct.drop('total_sort_idx', axis=1)
        
        num_species = len(pivot_counts.index)
        html_table = '<div class="table-responsive" style="margin-top:20px;"><table class="weekly-table">'
        html_table += f'<thead><tr><th style="text-align:left;">Vogelarten ({num_species})</th>'
        for col in pivot_pct.columns:
            total_in_hour = int(hour_totals[col])
            # Kürzeres Format für die Uhrzeit z.B. nur '14' statt '14:00' um Platz zu sparen, aber '14:00' ist auch ok
            html_table += f'<th title="Gesamtsumme: {total_in_hour}">{col[:2]}h</th>'
        html_table += '</tr></thead><tbody>'
        
        from flask import url_for
        for species, row in pivot_pct.iterrows():
            img_src = url_for('static', filename=get_bird_icon(species))
            if 'Unbekannt.png' in img_src:
                img_tag = '<div class="bird-icon-placeholder">?</div>'
            else:
                img_tag = f'<img src="{img_src}" class="bird-icon-small">'

            total_species_count = int(pivot_counts.loc[species].sum())
            html_table += f'<tr><td style="text-align:left; font-weight:bold;"><div class="species-wrapper">{img_tag}<span>{species} ({total_species_count})</span></div></td>'
            
            for col_name, val in row.items():
                absolute_count = int(pivot_counts.at[species, col_name])
                total_in_hour = int(hour_totals[col_name])
                
                style = 'background-color: transparent;'
                if absolute_count > 0:
                    alpha = 0.15 + (val / 50.0) * 0.85 
                    alpha = min(alpha, 1.0) 
                    style = f'background-color: rgba(76, 175, 80, {alpha});'
                
                if total_in_hour > 0:
                    tooltip = f"{val:.1f}% ({absolute_count} von {total_in_hour} Vögeln)"
                else:
                    tooltip = "0%"
                    
                html_table += f'<td title="{tooltip}" style="{style}"></td>'
            html_table += '</tr>'
        html_table += '</tbody></table></div>'
        
        html_table += """
        <div class="legend-container">
            <div class="legend-item"><div class="legend-box" style="background-color: transparent;"></div><span>0 Sichtungen</span></div>
            <div class="legend-item"><div class="legend-box" style="background-color: rgba(76, 175, 80, 0.2);"></div><span>Wenige</span></div>
            <div class="legend-item"><div class="legend-box" style="background-color: rgba(76, 175, 80, 0.6);"></div><span>Mittel</span></div>
            <div class="legend-item"><div class="legend-box" style="background-color: rgba(76, 175, 80, 1.0);"></div><span>Viele</span></div>
        </div>
        """

    return html_table

def generate_weekly_heatmap_html(year_str=None):
    if year_str:
        where_clause = "WHERE timestamp IS NOT NULL AND timestamp != '' AND timestamp LIKE ?"
        params = (f"{year_str}%",)
    else:
        where_clause = "WHERE timestamp IS NOT NULL AND timestamp != ''"
        params = ()

    query = f"""
    SELECT 
        CASE WHEN species = 'IGNORED_LOW_CONFIDENCE' THEN 'Unbekannt' ELSE species END as species,
        strftime('%Y-', timestamp) || printf('%02d', CAST(strftime('%W', timestamp) AS INTEGER) + 1) as week_sort,
        printf('%02d', CAST(strftime('%W', timestamp) AS INTEGER) + 1) || '<br><small style=''color:#aaa''>''' || substr(strftime('%Y', timestamp), 3, 2) || '</small>' as week_display,
        COUNT(*) as counts
    FROM detections
    {where_clause}
    GROUP BY species, week_sort, week_display
    ORDER BY week_sort
    """
    
    conn = sqlite3.connect(DB_FILE, timeout=10)
    try:
        grouped = pd.read_sql_query(query, conn, params=params)
    except:
        grouped = pd.DataFrame()
    finally:
        conn.close()

    import datetime
    now = datetime.datetime.now()
    current_year = int(year_str) if year_str else now.year
    short_y = str(current_year)[-2:]
    all_weeks_display_empty = [f"{w:02d}<br><small style='color:#aaa'>'{short_y}</small>" for w in range(1, 53)]
    
    current_week_str = f"{int(now.strftime('%W')) + 1:02d}<br><small style='color:#aaa'>'{str(now.year)[-2:]}</small>"

    html_table = '<div class="table-responsive" style="margin-top:30px;"><table class="weekly-table">'
    html_table += '<thead><tr><th style="text-align:left;">Vogelarten (0)</th>'
    for col in all_weeks_display_empty:
        if col == current_week_str:
            html_table += f'<th title="Gesamtsumme: 0" style="background-color: #81d4fa; color: black;">{col}</th>'
        else:
            html_table += f'<th title="Gesamtsumme: 0">{col}</th>'
    html_table += '</tr></thead><tbody>'
    html_table += f'<tr><td colspan="{len(all_weeks_display_empty)+1}" style="text-align:center;">Keine Daten vorhanden.</td></tr>'
    html_table += '</tbody></table></div>'

    icon_map = {}
    static_folder = os.path.join(app.root_path, 'static', 'bird_icons')
    if os.path.exists(static_folder):
        for f in os.listdir(static_folder):
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif')):
                icon_map[os.path.splitext(f)[0].lower()] = f"bird_icons/{f}"
                icon_map[f.lower()] = f"bird_icons/{f}"

    def get_bird_icon(sp):
        if not sp: return 'bird_icons/Unbekannt.png'
        clean = sp.strip().lower()
        if clean in icon_map:
            return icon_map[clean]
        if clean + '.png' in icon_map:
            return icon_map[clean + '.png']
        return 'bird_icons/Unbekannt.png'

    if not grouped.empty:
        pivot_counts = grouped.pivot(index='species', columns='week_display', values='counts').fillna(0)
        
        import datetime
        current_year = int(year_str) if year_str else datetime.datetime.now().year
        years_to_show = {current_year}
        for ws in grouped['week_sort']:
            try:
                years_to_show.add(int(ws.split('-')[0]))
            except:
                pass
        
        years_to_show = sorted(list(years_to_show))
        all_weeks_display = []
        for y in years_to_show:
            short_y = str(y)[-2:]
            for w in range(1, 54):
                disp = f"{w:02d}<br><small style='color:#aaa'>'{short_y}</small>"
                if w < 53 or disp in pivot_counts.columns:
                    all_weeks_display.append(disp)
                    
        pivot_counts = pivot_counts.reindex(columns=all_weeks_display, fill_value=0)
        
        week_totals = pivot_counts.sum(axis=0)
        pivot_pct = pivot_counts.div(week_totals.replace(0, 1), axis=1).mul(100).fillna(0)
        
        total_counts = pivot_counts.sum(axis=1)
        pivot_pct['total_sort_idx'] = total_counts
        pivot_pct = pivot_pct.sort_values('total_sort_idx', ascending=False)
        pivot_pct = pivot_pct.drop('total_sort_idx', axis=1)

        import datetime
        now = datetime.datetime.now()
        current_week_str = f"{int(now.strftime('%W')) + 1:02d}<br><small style='color:#aaa'>'{str(now.year)[-2:]}</small>"

        num_species = len(pivot_pct)
        html_table = '<div class="table-responsive" style="margin-top:30px;"><table class="weekly-table">'
        html_table += f'<thead><tr><th style="text-align:left;">Vogelarten ({num_species})</th>'
        for col in pivot_pct.columns:
            total_in_week = int(week_totals[col])
            if col == current_week_str:
                html_table += f'<th title="Gesamtsumme: {total_in_week}" style="background-color: #81d4fa; color: black;">{col}</th>'
            else:
                html_table += f'<th title="Gesamtsumme: {total_in_week}">{col}</th>'
        html_table += '</tr></thead><tbody>'
        
        import re
        from flask import url_for
        
        status_map = {
            "Standvogel": "SV",
            "Teilzieher": "TZ",
            "Zugvogel": "ZV",
            "Wintergast": "WG"
        }

        settings = load_settings()
        raw_bird_dict = load_dictionary()
        species_meta = {}
        for eng, data in raw_bird_dict.items():
            if isinstance(data, dict):
                trans = data.get("translation", "")
                trans = trans if str(trans).strip() != "" else eng
                species_meta[trans] = {
                    "aufenthalt": data.get("aufenthalt", ""),
                    "status": data.get("status", "")
                }

        for species, row in pivot_pct.iterrows():
            img_src = url_for('static', filename=get_bird_icon(species))
            if 'Unbekannt.png' in img_src:
                img_tag = '<div class="bird-icon-placeholder">?</div>'
            else:
                img_tag = f'<img src="{img_src}" class="bird-icon-small">'

            meta = species_meta.get(species, {})
            status_full = meta.get("status", "")
            aufenthalt = meta.get("aufenthalt", "")

            status_short = status_map.get(status_full, "")
            display_species = f"{species} ({status_short})" if status_short else species

            html_table += f'<tr><td style="text-align:left; font-weight:bold;"><div class="species-wrapper">{img_tag}<span>{display_species}</span></div></td>'
            
            start_m, end_m = 0, 0
            if aufenthalt and '-' in aufenthalt:
                try:
                    parts = aufenthalt.split('-')
                    start_m = int(parts[0])
                    end_m = int(parts[1])
                except:
                    pass
            
            in_range_list = []
            for col_name in row.keys():
                w_match = re.search(r"^(\d{2})<br>.*?\'(\d{2})</small>$", col_name)
                is_in_range = False
                if w_match and start_m and end_m:
                    w = int(w_match.group(1))
                    y = 2000 + int(w_match.group(2))
                    try:
                        month = datetime.date.fromisocalendar(y, w, 1).month
                    except (ValueError, AttributeError):
                        try:
                            month = datetime.datetime.strptime(f'{y}-W{w-1}-1', "%Y-W%W-%w").month
                        except:
                            month = ((w - 1) * 12) // 52 + 1
                    
                    if start_m <= end_m:
                        if start_m <= month <= end_m:
                            is_in_range = True
                    else:
                        if month >= start_m or month <= end_m:
                            is_in_range = True
                in_range_list.append(is_in_range)
                
            for i, (col_name, val) in enumerate(row.items()):
                absolute_count = int(pivot_counts.at[species, col_name])
                total_in_week = int(week_totals[col_name])
                
                is_in_range = in_range_list[i]
                
                style = ''
                
                if is_in_range:
                    if start_m and end_m and start_m > end_m:
                        is_start = (i > 0) and not in_range_list[i-1]
                        is_end = (i < len(in_range_list) - 1) and not in_range_list[i+1]
                    else:
                        is_start = (i == 0) or not in_range_list[i-1]
                        is_end = (i == len(in_range_list) - 1) or not in_range_list[i+1]
                    
                    shadows = []
                    
                    if is_start:
                        shadows.append('inset 2px 0 0 0 rgba(255, 235, 59, 0.8)')
                        shadows.append('inset 0 1px 0 0 rgba(255, 235, 59, 0.8)')
                        shadows.append('inset 0 -1px 0 0 rgba(255, 235, 59, 0.8)')
                        
                    if is_end:
                        shadows.append('inset -2px 0 0 0 rgba(255, 235, 59, 0.8)')
                        if not is_start:
                            shadows.append('inset 0 1px 0 0 rgba(255, 235, 59, 0.8)')
                            shadows.append('inset 0 -1px 0 0 rgba(255, 235, 59, 0.8)')
                    
                    if shadows:
                        style += ' box-shadow: ' + ', '.join(shadows) + ';'
                
                if total_in_week > 0:
                    tooltip = f"{val:.1f}% ({absolute_count} von {total_in_week} Vögeln)"
                else:
                    tooltip = "0%"
                    
                import math
                barchart_max = settings.get("barchart_max_calls_weekly", 3000)
                inner_html = ""
                if absolute_count > 0:
                    step = math.ceil((absolute_count / barchart_max) * 10)
                    step = min(step, 10)
                    step = max(step, 1)
                    height_pct = step * 10
                    inner_html = f'<div class="data-cell-inner"><div class="barchart-bar" style="height: {height_pct}%;"></div></div>'
                    
                html_table += f'<td title="{tooltip}" style="{style}">{inner_html}</td>'
            html_table += '</tr>'
        html_table += '</tbody></table></div>'
        
        html_table += """
        <div class="legend-container" id="barchartLegend" style="display: flex; align-items: center;">
            <div class="legend-item" style="display: flex; align-items: center;">
                <div style="border-left: 1px solid #fff; height: 16px; margin-right: 4px;"></div>
                <div style="width: 40px; height: 16px; background-color: #90ee90; clip-path: polygon(0 50%, 100% 0, 100% 100%); margin-right: 8px;"></div>
                <span>= selten bis oft</span>
            </div>
        </div>
        """

    return html_table

def create_weekly_total_chart(year_str):
    if year_str:
        where_clause = "WHERE timestamp IS NOT NULL AND timestamp != '' AND timestamp LIKE ?"
        params = (f"{year_str}%",)
    else:
        where_clause = "WHERE timestamp IS NOT NULL AND timestamp != ''"
        params = ()

    query = f"""
    SELECT 
        CAST(strftime('%W', timestamp) AS INTEGER) + 1 as week_num,
        COUNT(*) as counts,
        COUNT(DISTINCT species) as unique_species,
        GROUP_CONCAT(DISTINCT species) as species_list
    FROM detections
    {where_clause}
    GROUP BY week_num
    ORDER BY week_num
    """
    
    conn = sqlite3.connect(DB_FILE, timeout=10)
    try:
        grouped = pd.read_sql_query(query, conn, params=params)
    except:
        grouped = pd.DataFrame()
    finally:
        conn.close()

    fig, ax1 = plt.subplots(figsize=(10, 3), facecolor='#1e1e1e')
    ax1.set_facecolor('#1e1e1e')
    ax1.tick_params(colors='white')
    for spine in ax1.spines.values():
        spine.set_color('#444')
    
    weeks = list(range(1, 54))
    week_labels = [f"KW{w:02d}" for w in weeks]
    
    counts = [0] * 53
    species_counts = [0] * 53
    species_sets = [set() for _ in range(53)]
    
    if not grouped.empty:
        for _, row in grouped.iterrows():
            if pd.notna(row['week_num']):
                try:
                    w = int(row['week_num'])
                    if 1 <= w <= 53:
                        counts[w-1] = int(row['counts'])
                        if 'unique_species' in row:
                            species_counts[w-1] = int(row['unique_species'])
                        if 'species_list' in row and row['species_list']:
                            species_sets[w-1] = set(row['species_list'].split(','))
                except:
                    pass

    jaccard_indices = [np.nan] * 53
    for i in range(1, 53):
        set_prev = species_sets[i-1]
        set_curr = species_sets[i]
        if set_prev or set_curr:
            intersection = len(set_prev.intersection(set_curr))
            union = len(set_prev.union(set_curr))
            jaccard_indices[i] = intersection / union if union > 0 else np.nan

    line1 = ax1.plot(weeks, counts, color='#e5c07b', linewidth=2, marker='o', markersize=4, label='Rufe (Gesamt)')
    ax1.fill_between(weeks, counts, color='#e5c07b', alpha=0.1)
    
    ax1.set_xticks(weeks)
    ax1.set_xticklabels(week_labels, rotation=90, ha='center', color='white', fontsize=8)
    
    from matplotlib.ticker import MaxNLocator
    ax1.yaxis.set_major_locator(MaxNLocator(integer=True))
    
    ax1.set_ylim(bottom=0)
    if max(counts) == 0:
        ax1.set_ylim(top=10)
    
    ax2 = ax1.twinx()
    line2 = ax2.plot(weeks, species_counts, color='#56b6c2', linewidth=2, marker='s', markersize=4, label='Arten (Diversifikation)')
    ax2.fill_between(weeks, species_counts, color='#56b6c2', alpha=0.1)
    ax2.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax2.set_ylim(bottom=0)
    if max(species_counts) == 0:
        ax2.set_ylim(top=10)
    ax2.tick_params(colors='white')
    for spine in ax2.spines.values():
        spine.set_color('#444')

    ax3 = ax1.twinx()
    line3 = ax3.plot(weeks, jaccard_indices, color='#98c379', linewidth=2, linestyle='--', marker='^', markersize=4, label='Konstanz (Jaccard-Index)')
    ax3.set_ylim(0, 1)
    ax3.yaxis.set_visible(False)
    for spine in ax3.spines.values():
        spine.set_visible(False)

    lines = line1 + line2 + line3
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper left', facecolor='#263238', edgecolor='#444', labelcolor='white')

    ax1.grid(color='#444', linestyle='--', linewidth=0.5, alpha=0.5)
    plt.tight_layout()
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', transparent=True)
    plt.close()
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')

@app.route('/weekly')
def weekly_page():
    import datetime
    today = datetime.date.today()
    year_str = request.args.get('year', str(today.year))
    try:
        year = int(year_str)
    except:
        year = today.year
        year_str = str(year)

    weekly_chart = create_weekly_total_chart(year_str)

    return render_template('weekly.html', 
        table_content=generate_weekly_heatmap_html(year_str),
        weekly_chart=weekly_chart,
        selected_year=year,
        prev_year=year-1,
        next_year=year+1,
        is_current_year=(year == today.year),
        current_year=today.year
    )

def create_species_polar_chart(species, hourly_counts, time_mode='relative_sunrise'):
    plt.figure(figsize=(8, 8), facecolor='#1e1e1e')
    ax = plt.subplot(111, polar=True)
    ax.set_facecolor('#1e1e1e')
    
    # 24 hours
    theta = np.linspace(0.0, 2 * np.pi, 24, endpoint=False)
    
    # 0 degrees location depending on mode
    if time_mode == 'relative_sunset':
        ax.set_theta_zero_location("S")
    else:
        ax.set_theta_zero_location("N")
        
    # going clockwise
    ax.set_theta_direction(-1)
    
    ax.set_xticks(theta)
    if time_mode == 'relative_sunset':
        labels = ["Sonnen-\nuntergang" if i == 0 else f"+{i}h" if i <= 12 else f"-{24-i}h" for i in range(24)]
    elif time_mode == 'relative_sunrise':
        labels = ["Sonnen-\naufgang" if i == 0 else f"+{i}h" if i <= 12 else f"-{24-i}h" for i in range(24)]
    else:
        labels = [f"{i:02d}:00" for i in range(24)]
        
    ax.set_xticklabels(labels, color='white', fontsize=9)
    ax.tick_params(colors='white')
    for spine in ax.spines.values():
        spine.set_color('#444')
    
    bars = ax.bar(theta, hourly_counts, width=2*np.pi/24, bottom=0.0, color='#4CAF50', alpha=0.7, edgecolor='white')
    
    if time_mode == 'relative_sunset':
        title_text = f"Aktivität (Relativ zum Sonnenuntergang): {species}"
    elif time_mode == 'relative_sunrise':
        title_text = f"Aktivität (Relativ zum Sonnenaufgang): {species}"
    else:
        title_text = f"Aktivität über 24h: {species}"
    plt.title(title_text, color='white', y=1.08)
    plt.grid(color='#444', linestyle='--', linewidth=0.5, alpha=0.5)
    plt.tight_layout()
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()
    return f"data:image/png;base64,{img_base64}"

@app.route('/territory')
def territory_page():
    import struct
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    
    # Einstellungen laden (für Persistenz)
    current_settings = load_settings()
    
    # 1. Parameter aus dem Request holen (falls der Nutzer auf "Anwenden" geklickt hat)
    custom_threshold_str = request.args.get('threshold')
    custom_min_calls_str = request.args.get('min_calls')
    
    if custom_threshold_str is not None:
        threshold = float(custom_threshold_str)
        save_setting("territory_threshold", threshold)
    else:
        threshold = float(current_settings.get("territory_threshold", 25.0))
        
    if custom_min_calls_str is not None:
        min_calls = int(custom_min_calls_str)
        save_setting("territory_min_calls", min_calls)
    else:
        min_calls = int(current_settings.get("territory_min_calls", 20))
    
    # 2. Alle Erkennungen abrufen und im Python-Code aggregieren
    c.execute("SELECT species, snr FROM detections WHERE species != 'IGNORED_LOW_CONFIDENCE'")
    rows = c.fetchall()
    conn.close()
    
    species_calls = {}
    
    for row in rows:
        sp = row[0]
        snr_val = row[1]
        
        # SNR parsen (manche Werte wurden evtl. binär von numpy/sqlite gespeichert)
        if isinstance(snr_val, bytes):
            if len(snr_val) == 8:
                snr_val = struct.unpack('d', snr_val)[0]
            elif len(snr_val) == 4:
                snr_val = struct.unpack('f', snr_val)[0]
            else:
                snr_val = 0.0
        elif snr_val is None:
            snr_val = 0.0
        else:
            snr_val = float(snr_val)
            
        if sp not in species_calls:
            species_calls[sp] = []
        species_calls[sp].append(snr_val)
        
        if snr_val > 0:
            pass # Wir sammeln all_snrs nicht mehr für min/max Berechnung
            
    residents = []     # > 66%
    commuters = []     # 33% - 66%
    transients = []    # < 33%
    
    for sp, snrs in species_calls.items():
        total = len(snrs)
        if total >= min_calls:
            high = sum(1 for s in snrs if s >= threshold)
            pct = (high / total) * 100.0 if total > 0 else 0
            
            entry = {'species': sp, 'total': total, 'high': high, 'percent': round(pct, 1)}
            
            if pct > 66:
                residents.append(entry)
            elif pct >= 33:
                commuters.append(entry)
            else:
                transients.append(entry)
                
    # Sortieren nach Prozent absteigend
    residents.sort(key=lambda x: x['percent'], reverse=True)
    commuters.sort(key=lambda x: x['percent'], reverse=True)
    transients.sort(key=lambda x: x['percent'], reverse=True)
    
    categories = {
        'residents': residents,
        'commuters': commuters,
        'transients': transients
    }
    
    return render_template('territory.html', threshold=round(threshold, 1), min_calls=min_calls, categories=categories)


@app.route('/intersection')
def intersection_page():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT DISTINCT strftime('%Y', timestamp) FROM detections WHERE timestamp IS NOT NULL")
    years = [row[0] for row in c.fetchall() if row[0]]
    conn.close()
    
    current_year = str(datetime.datetime.now().year)
    if not years:
        years = [current_year]
    elif current_year not in years:
        years.append(current_year)
    
    years.sort(reverse=True)
        
    return render_template('intersection.html', available_years=years, current_year=current_year)

@app.route('/api/intersection_data')
def api_intersection_data():
    year = request.args.get('year', str(datetime.datetime.now().year))
    division = request.args.get('division', 'quarter')
    
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT species, timestamp FROM detections WHERE strftime('%Y', timestamp) = ?", (year,))
    rows = c.fetchall()
    conn.close()
    
    from collections import defaultdict
    import datetime as dt
    species_in_bucket = defaultdict(set)
    
    for species, ts_str in rows:
        if not ts_str: continue
        try:
            ts = dt.datetime.strptime(ts_str, '%Y-%m-%d %H:%M:%S')
        except ValueError:
            continue
            
        bucket = None
        if division == 'half':
            bucket = "H1" if ts.month <= 6 else "H2"
        elif division == 'tertial':
            if ts.month <= 4: bucket = "T1"
            elif ts.month <= 8: bucket = "T2"
            else: bucket = "T3"
        elif division == 'quarter':
            if ts.month <= 3: bucket = "Q1"
            elif ts.month <= 6: bucket = "Q2"
            elif ts.month <= 9: bucket = "Q3"
            else: bucket = "Q4"
        elif division == 'month':
            months = ["Jan", "Feb", "Mär", "Apr", "Mai", "Jun", "Jul", "Aug", "Sep", "Okt", "Nov", "Dez"]
            bucket = months[ts.month - 1]
        elif division == 'week':
            bucket = f"KW {ts.isocalendar()[1]}"
            
        if bucket:
            species_in_bucket[bucket].add(species)
            
    # Removed Top-5 limit to show all available weeks
    if len(species_in_bucket) < 2:
        return jsonify({'error': 'Zu wenig Daten für die gewählten Zeiträume vorhanden.'})
        
    # Sort buckets chronologically
    expected_order = []
    if division == 'half': expected_order = ["H1", "H2"]
    elif division == 'tertial': expected_order = ["T1", "T2", "T3"]
    elif division == 'quarter': expected_order = ["Q1", "Q2", "Q3", "Q4"]
    elif division == 'month': expected_order = ["Jan", "Feb", "Mär", "Apr", "Mai", "Jun", "Jul", "Aug", "Sep", "Okt", "Nov", "Dez"]
    elif division == 'week': expected_order = [f"KW {w}" for w in range(1, 54)]
    
    # Only keep buckets that have data, keeping their chronological order
    chronological_buckets = [b for b in expected_order if b in species_in_bucket]
    # Handle any edge case (like custom buckets in the future)
    for b in species_in_bucket.keys():
        if b not in chronological_buckets:
            chronological_buckets.append(b)

    venn_data = []
    
    # 1-way sets
    for b in chronological_buckets:
        venn_data.append({
            'sets': [b], 
            'size': len(species_in_bucket[b]), 
            'species': list(species_in_bucket[b])
        })
        
    # 2-way sets (Only chronologically adjacent intersections)
    for i in range(len(chronological_buckets) - 1):
        b1 = chronological_buckets[i]
        b2 = chronological_buckets[i+1]
        
        intersect = species_in_bucket[b1].intersection(species_in_bucket[b2])
        union_len = len(species_in_bucket[b1].union(species_in_bucket[b2]))
        
        if len(intersect) > 0:
            jaccard = len(intersect) / union_len if union_len > 0 else 0
            venn_data.append({
                'sets': [b1, b2],
                'size': len(intersect),
                'species': list(intersect),
                'jaccard': jaccard
            })
                
    return jsonify({'data': venn_data})

@app.route('/species')

def species_page():
    species_set = set(get_bird_dictionary().values())
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    try:
        c.execute("SELECT DISTINCT species FROM detections")
        for row in c.fetchall():
            if row[0] != 'IGNORED_LOW_CONFIDENCE':
                species_set.add(row[0])
    except:
        pass
        
    all_species = sorted(list(species_set))
    selected_species = request.args.get('species', '')
    time_mode = request.args.get('time_mode', 'relative_sunrise')
    
    chart_url = None
    total_count = 0
    wav_files = []
    confusion_report = []
    if selected_species:
        c.execute("SELECT COUNT(*) FROM detections WHERE species = ?", (selected_species,))
        total_row = c.fetchone()
        total_count = total_row[0] if total_row else 0
        
        hourly_counts = [0] * 24
        
        if total_count > 0:
            if time_mode in ('relative_sunrise', 'relative_sunset'):
                settings = load_settings()
                try:
                    lat = float(settings.get("gps_lat", -1.0))
                    lon = float(settings.get("gps_lon", -1.0))
                except:
                    lat, lon = -1.0, -1.0
                
                if lat != -1.0 and lon != -1.0:
                    loc = LocationInfo(latitude=lat, longitude=lon)
                    c.execute("SELECT timestamp FROM detections WHERE species = ?", (selected_species,))
                    rows = c.fetchall()
                    local_tz = datetime.datetime.now().astimezone().tzinfo
                    for r in rows:
                        if r[0] is not None:
                            try:
                                dt = datetime.datetime.strptime(r[0], "%Y-%m-%d %H:%M:%S")
                                s = sun(loc.observer, date=dt.date())
                                ref_dt = s['sunrise'] if time_mode == 'relative_sunrise' else s['sunset']
                                dt_aware = dt.replace(tzinfo=local_tz)
                                diff = dt_aware - ref_dt
                                diff_hours = diff.total_seconds() / 3600.0
                                rel_hour = int(diff_hours // 1) % 24
                                hourly_counts[rel_hour] += 1
                            except:
                                pass
                    chart_url = create_species_polar_chart(selected_species, hourly_counts, time_mode=time_mode)
                else:
                    time_mode = 'absolute'
                    
            if time_mode == 'absolute':
                c.execute("SELECT strftime('%H', timestamp) as hour, COUNT(*) FROM detections WHERE species = ? GROUP BY hour", (selected_species,))
                rows = c.fetchall()
                for r in rows:
                    if r[0] is not None:
                        try:
                            h = int(r[0])
                            if 0 <= h < 24:
                                hourly_counts[h] = r[1]
                        except:
                            pass
                chart_url = create_species_polar_chart(selected_species, hourly_counts, time_mode='absolute')
            
            try:
                c.execute('CREATE INDEX IF NOT EXISTS idx_timestamp ON detections(timestamp)')
                query = """
                    SELECT d2.species, COUNT(DISTINCT d1.rowid) as co_occurrences
                    FROM detections d1
                    CROSS JOIN detections d2 
                      ON d2.timestamp BETWEEN datetime(d1.timestamp, '-20 seconds') AND datetime(d1.timestamp, '+20 seconds')
                    WHERE d1.species = ? 
                      AND d2.species != ? 
                      AND d2.species != 'IGNORED_LOW_CONFIDENCE'
                    GROUP BY d2.species
                    ORDER BY co_occurrences DESC
                    LIMIT 10
                """
                c.execute(query, (selected_species, selected_species))
                for row in c.fetchall():
                    conf_percent = round((row[1] / total_count) * 100, 1)
                    confusion_report.append({
                        'species': row[0],
                        'count': row[1],
                        'percent': conf_percent
                    })
            except Exception as e:
                print(f"Error confusion report: {e}")
            
        archive_path = os.path.join(AUDIO_DIR, "archive")
        if os.path.exists(archive_path):
            prefix = selected_species + "_"
            wav_files = [f for f in os.listdir(archive_path) if f.startswith(prefix) and f.endswith('.wav')]
            wav_files.sort(reverse=True) # newest first, assuming IDs might correlate with time or just standard sort
            
    conn.close()
    
    return render_template('species.html', 
        all_species=all_species, 
        selected_species=selected_species,
        chart_url=chart_url,
        total_count=total_count,
        wav_files=wav_files,
        confusion_report=confusion_report,
        time_mode=time_mode
    )

@app.route('/api/archive/audio/<filename>')
def serve_archive_audio(filename):
    archive_path = os.path.join(AUDIO_DIR, "archive")
    file_path = os.path.join(archive_path, filename)
    if os.path.exists(file_path):
        return send_file(file_path)
    abort(404)

@app.route('/api/archive/spectrogram/<filename>')
def serve_archive_spectrogram(filename):
    archive_path = os.path.join(AUDIO_DIR, "archive")
    file_path = os.path.join(archive_path, filename)
    if not os.path.exists(file_path):
        abort(404)
    
    try:
        y, sr = librosa.load(file_path, sr=None)
        fig = plt.figure(figsize=(8, 4))
        ax = fig.add_axes([0.12, 0.15, 0.75, 0.75])
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=20000)
        S_dB = librosa.power_to_db(S, ref=np.max)
        img = librosa.display.specshow(S_dB, x_axis='time', y_axis='mel', sr=sr, fmax=20000, ax=ax)
        ax.set_title(f"Spectrogram: {filename}")
        
        cax = fig.add_axes([0.88, 0.15, 0.02, 0.75])
        fig.colorbar(img, cax=cax, format='%+2.0f dB')
        
        fig.canvas.draw()
        pos = ax.get_position()
        duration = librosa.get_duration(y=y, sr=sr)
        
        buf = io.BytesIO()
        fig.savefig(buf, format='png')
        buf.seek(0)
        plt.close(fig)
        
        from flask import make_response
        response = make_response(send_file(buf, mimetype='image/png'))
        trans = (ax.transScale + ax.transLimits).inverted()
        import json
        ymap = [round(float(trans.transform((0.5, yr))[1]), 1) for yr in np.linspace(0, 1, 101)]
        response.headers['X-Plot-Left'] = str(pos.x0)
        response.headers['X-Plot-Bottom'] = str(pos.y0)
        response.headers['X-Plot-Width'] = str(pos.width)
        response.headers['X-Plot-Height'] = str(pos.height)
        response.headers['X-Audio-Duration'] = str(duration)
        response.headers['X-Plot-YMap'] = json.dumps(ymap)
        response.headers['Access-Control-Expose-Headers'] = 'X-Plot-Left, X-Plot-Bottom, X-Plot-Width, X-Plot-Height, X-Audio-Duration, X-Plot-YMap'
        return response
    except Exception as e:
        print(f"Error generating spectrogram: {e}")
        abort(500)

@app.route('/api/archive/phase_spectrogram/<filename>')
def serve_archive_phase_spectrogram(filename):
    archive_path = os.path.join(AUDIO_DIR, "archive")
    file_path = os.path.join(archive_path, filename)
    if not os.path.exists(file_path):
        abort(404)
    
    try:
        y, sr = librosa.load(file_path, sr=None)
        D = librosa.stft(y)
        phase = np.angle(D)
        
        # Berechne die zeitliche Ableitung der Phase (Momentanfrequenz / Phase Stability)
        unwrapped_phase = np.unwrap(phase, axis=1)
        inst_freq = np.gradient(unwrapped_phase, axis=1)
        
        from scipy.interpolate import interp1d
        
        # Interpoliere auf die Mel-Skala, um exakt die gleiche y-Achse wie beim Frequenzspektrum zu erhalten
        mel_frequencies = librosa.mel_frequencies(n_mels=128, fmin=0.0, fmax=20000)
        linear_frequencies = librosa.fft_frequencies(sr=sr, n_fft=2048)
        
        f = interp1d(linear_frequencies, inst_freq, axis=0, bounds_error=False, fill_value='extrapolate')
        inst_freq_mel = f(mel_frequencies)
        
        # Auf -180° bis +180° wrappen und in Grad umwandeln
        inst_freq_deg = np.angle(np.exp(1j * inst_freq_mel), deg=True)
        
        fig = plt.figure(figsize=(8, 4))
        ax = fig.add_axes([0.12, 0.15, 0.75, 0.75])
        cmap = plt.get_cmap('twilight').copy()
        
        # Zeige die Momentanfrequenz fuer die gesamte Flaeche an (ohne Maskierung) auf der Mel-Skala
        img = librosa.display.specshow(inst_freq_deg, x_axis='time', y_axis='mel', sr=sr, fmax=20000, ax=ax, cmap=cmap, vmin=-180, vmax=180)
        
        # Exakte X-Achsen-Limits wie beim Spektrogramm erzwingen
        duration = len(y) / sr
        ax.set_xlim(0, duration)
        
        ax.set_title(f"Phasenstabilität: {filename}")
        
        cax = fig.add_axes([0.88, 0.15, 0.02, 0.75])
        cbar = fig.colorbar(img, cax=cax, format='%+2.0f\u00b0')
        
        fig.canvas.draw()
        pos = ax.get_position()
        
        buf = io.BytesIO()
        fig.savefig(buf, format='png')
        buf.seek(0)
        plt.close(fig)
        
        from flask import make_response
        response = make_response(send_file(buf, mimetype='image/png'))
        trans = (ax.transScale + ax.transLimits).inverted()
        import json
        ymap = [round(float(trans.transform((0.5, yr))[1]), 1) for yr in np.linspace(0, 1, 101)]
        response.headers['X-Plot-Left'] = str(pos.x0)
        response.headers['X-Plot-Bottom'] = str(pos.y0)
        response.headers['X-Plot-Width'] = str(pos.width)
        response.headers['X-Plot-Height'] = str(pos.height)
        response.headers['X-Audio-Duration'] = str(duration)
        response.headers['X-Plot-YMap'] = json.dumps(ymap)
        response.headers['Access-Control-Expose-Headers'] = 'X-Plot-Left, X-Plot-Bottom, X-Plot-Width, X-Plot-Height, X-Audio-Duration, X-Plot-YMap'
        return response
    except Exception as e:
        print(f"Error generating phase spectrogram: {e}")
        abort(500)

@app.route('/api/archive/waveform/<filename>')
def serve_archive_waveform(filename):
    archive_path = os.path.join(AUDIO_DIR, "archive")
    file_path = os.path.join(archive_path, filename)
    if not os.path.exists(file_path):
        abort(404)
    
    try:
        y, sr = librosa.load(file_path, sr=None)
        fig = plt.figure(figsize=(8, 2.5))
        ax = fig.add_axes([0.12, 0.25, 0.75, 0.65])
        librosa.display.waveshow(y, sr=sr, ax=ax, color='b')
        ax.set_title(f"Amplitudenspektrum: {filename}")
        ax.set_ylabel("Amplitude (dB)")
        ax.set_xlabel("Zeit (s)")
        
        import matplotlib.ticker as ticker
        def lin_to_db(x, pos):
            val = np.abs(x)
            if val < 1e-10:
                return '-inf'
            return f'{20 * np.log10(val):.0f}'
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(lin_to_db))
        
        # Exakte X-Achsen-Limits wie beim Spektrogramm erzwingen
        duration = len(y) / sr
        ax.set_xlim(0, duration)
        
        # Unsichtbare Platzhalter-Colorbar hinzufügen, 
        # damit die Bildbreite und Achsenausrichtung exakt mit dem Spektrogramm übereinstimmt.
        # Wir nutzen eine komplett transparente Colormap, damit kein Farbverlauf sichtbar ist.
        from matplotlib.colors import ListedColormap
        transparent_cmap = ListedColormap([(0,0,0,0)])
        sm = plt.cm.ScalarMappable(cmap=transparent_cmap)
        sm.set_array([])
        
        cax = fig.add_axes([0.88, 0.25, 0.02, 0.65])
        cbar = fig.colorbar(sm, cax=cax, format='%+2.0f dB')
        cbar.ax.tick_params(color='none', labelcolor='none')
        cbar.outline.set_visible(False)
        
        fig.canvas.draw()
        
        buf = io.BytesIO()
        fig.savefig(buf, format='png')
        buf.seek(0)
        plt.close(fig)
        return send_file(buf, mimetype='image/png')
    except Exception as e:
        print(f"Error generating waveform: {e}")
        abort(500)

@app.route('/api/archive/delete/<filename>', methods=['POST', 'DELETE'])
def delete_archive_audio(filename):
    archive_path = os.path.join(AUDIO_DIR, "archive")
    file_path = os.path.join(archive_path, filename)
    if os.path.exists(file_path):
        try:
            os.remove(file_path)
            return jsonify({"success": True, "msg": "Datei erfolgreich gelöscht."})
        except Exception as e:
            return jsonify({"success": False, "msg": str(e)}), 500
    return jsonify({"success": False, "msg": "Datei nicht gefunden."}), 404

@app.route('/yearly')
def yearly_page():
    today = datetime.date.today()
    year_str = request.args.get('year', str(today.year))
    try:
        year = int(year_str)
    except:
        year = today.year
        year_str = str(year)

    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT species, COUNT(*) as c FROM detections WHERE timestamp LIKE ? GROUP BY species ORDER BY c DESC", (f"{year_str}%",))
    rows = c.fetchall()

    # Query first & last detection per species in selected year (window functions)
    c.execute("""
        SELECT
            species,
            MIN(timestamp) AS first_time,
            MAX(timestamp) AS last_time
        FROM detections
        WHERE timestamp LIKE ?
        GROUP BY species
        ORDER BY first_time DESC
    """, (f"{year_str}%",))
    first_last_rows = c.fetchall()
    conn.close()

    total = sum([r[1] for r in rows])

    icon_map = {}
    static_folder = os.path.join(app.root_path, 'static', 'bird_icons')
    if os.path.exists(static_folder):
        for f in os.listdir(static_folder):
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif')):
                icon_map[os.path.splitext(f)[0].lower()] = f"bird_icons/{f}"
                icon_map[f.lower()] = f"bird_icons/{f}"

    def get_icon(sp):
        if not sp: return 'bird_icons/Unbekannt.png'
        clean = sp.strip().lower()
        if clean in icon_map: return icon_map[clean]
        if clean + '.png' in icon_map: return icon_map[clean + '.png']
        return 'bird_icons/Unbekannt.png'

    def format_count(c):
        if c >= 1_000_000:
            return f"{c/1_000_000:.1f}M".replace('.', ',')
        elif c >= 1_000:
            return f"{c/1_000:.1f}K".replace('.', ',')
        else:
            return str(c)

    import math
    max_val = max([r[1] for r in rows]) if rows else 0
    max_log = math.log10(max_val + 1) if max_val > 0 else 1

    color_palette = ['#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231', '#911eb4', '#46f0f0', '#f032e6', '#bcf60c', '#fabebe', '#008080', '#e6beff', '#9a6324', '#fffac8', '#800000', '#aaffc3', '#808000', '#ffd8b1', '#000075', '#808080', '#ffffff', '#000000']

    bird_data = []
    for i, r in enumerate(rows):
        species = r[0]
        count = r[1]
        width_pct = (math.log10(count + 1) / max_log) * 100
        bird_data.append({
            'species': species,
            'count': count,
            'formatted_count': format_count(count),
            'icon': get_icon(species),
            'width_pct': width_pct,
            'color': color_palette[i % len(color_palette)]
        })

    # Build first_seen_data: fetch exact row details for first and last timestamps
    # geo_prob may be stored as BLOB (bytes) – safe_float handles all cases
    def safe_float(v):
        if v is None:
            return None
        if isinstance(v, (bytes, bytearray)):
            try:
                import struct
                if len(v) == 8:
                    return struct.unpack('d', v)[0]
                elif len(v) == 4:
                    return struct.unpack('f', v)[0]
                return float(v.decode('utf-8', errors='replace'))
            except Exception:
                return None
        return float(v)

    first_seen_data = []
    conn2 = sqlite3.connect(DB_FILE)
    c2 = conn2.cursor()
    for row in first_last_rows:
        species, first_time, last_time = row[0], row[1], row[2]
        c2.execute("SELECT confidence, geo_prob, snr FROM detections WHERE species=? AND timestamp=? LIMIT 1", (species, first_time))
        fr = c2.fetchone()
        c2.execute("SELECT confidence, geo_prob, snr FROM detections WHERE species=? AND timestamp=? LIMIT 1", (species, last_time))
        lr = c2.fetchone()

        def _pct(v):
            f = safe_float(v)
            return round(f * 100, 1) if f is not None else None

        def _val(v):
            f = safe_float(v)
            return round(f, 1) if f is not None else None

        first_seen_data.append({
            'species': species,
            'icon': get_icon(species),
            'first_time': first_time,
            'first_conf': _pct(fr[0]) if fr else None,
            'first_prob': _pct(fr[1]) if fr else None,
            'first_snr':  _val(fr[2]) if fr else None,
            'last_time': last_time,
            'last_conf': _pct(lr[0]) if lr else None,
            'last_prob': _pct(lr[1]) if lr else None,
            'last_snr':  _val(lr[2]) if lr else None,
        })
    conn2.close()

    return render_template('yearly.html',
        bird_data=bird_data, selected_year=year, total_birds_year=total,
        prev_year=year-1, next_year=year+1,
        is_current_year=(year == today.year), current_year=today.year,
        unique_species_year=len(rows),
        first_seen_data=first_seen_data
    )

@app.route('/manual_entry')
def manual_entry_page():
    species_set = set(get_bird_dictionary().values())
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    try:
        c.execute("SELECT DISTINCT species FROM detections")
        for row in c.fetchall():
            species_set.add(row[0])
    except:
        pass
    conn.close()
    return render_template('manual_entry.html', species_list=sorted(list(species_set)))

@app.route('/delete_entry')
def delete_entry_page():
    species_set = set(get_bird_dictionary().values())
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    try:
        c.execute("SELECT DISTINCT species FROM detections")
        for row in c.fetchall():
            species_set.add(row[0])
    except:
        pass
    conn.close()
    return render_template('delete_entry.html', species_list=sorted(list(species_set)))

@app.route('/db_edit')
def db_edit_page():
    species_set = set(get_bird_dictionary().values())
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    try:
        c.execute("SELECT DISTINCT species FROM detections")
        for row in c.fetchall():
            species_set.add(row[0])
    except:
        pass
    conn.close()
    return render_template('db_edit.html', species_list=sorted(list(species_set)))

@app.route('/wiki')
def wiki_page():
    wiki_dir = os.path.join('static', 'wiki')
    wiki_images = []
    if os.path.exists(wiki_dir):
        wiki_images = [f for f in os.listdir(wiki_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.webp'))]
    return render_template('wiki.html', wiki_images=wiki_images)

@app.route('/prediction')
def prediction_page():
    import random
    from collections import Counter
    import datetime

    try:
        days = int(request.args.get('days', 7))
    except ValueError:
        days = 7

    end_date = datetime.datetime.now()
    start_date = end_date - datetime.timedelta(days=days)
    
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    
    # All detections in timeframe
    c.execute("SELECT species, timestamp FROM detections WHERE timestamp >= ?", (start_date.strftime('%Y-%m-%d %H:%M:%S'),))
    rows = c.fetchall()
    
    # 1. Rush-Hour
    hours = [int(row[1][11:13]) for row in rows if len(row[1]) >= 13]
    hour_counts = Counter(hours)
    rush_hour_labels = [f"{h:02d}:00" for h in range(24)]
    rush_hour_data = [hour_counts.get(h, 0) for h in range(24)]
    busiest_hour = max(hour_counts, key=hour_counts.get) if hour_counts else 12
    
    # 2. Arten-Wahrscheinlichkeit for next hour
    next_h = (end_date.hour + 1) % 24
    next_h_species = [row[0] for row in rows if len(row[1]) >= 13 and int(row[1][11:13]) == next_h]
    next_h_counts = Counter(next_h_species)
    total_next_h = sum(next_h_counts.values())
    probs = {}
    if total_next_h > 0:
        probs = {s: round((c / total_next_h) * 100, 1) for s, c in next_h_counts.most_common(10)}
    prob_labels = list(probs.keys())
    prob_data = list(probs.values())
    
    # 3. Seltenheits-Radar
    c.execute("SELECT species, COUNT(*) as cnt FROM detections GROUP BY species ORDER BY cnt ASC")
    all_counts = c.fetchall()
    total_det = sum([x[1] for x in all_counts])
    rare_species = [x[0] for x in all_counts if x[1] < max(5, total_det * 0.01)]
    recent_rare = [s for s in rare_species if s in [r[0] for r in rows]]
    if recent_rare:
        selten_text = f"Die Chance auf seltene Besucher ist gut! In letzter Zeit wurden gesichtet: {', '.join(set(recent_rare[:3]))}."
    else:
        if rare_species:
            selten_text = f"Heute ist es ruhig bei den seltenen Arten. Vielleicht lässt sich ja trotzdem bald wieder ein {random.choice(rare_species)} blicken."
        else:
            selten_text = "Bisher noch nicht genug Daten für seltene Arten."
            
    # 4. Zugvogel-Tracker
    migratory_birds = ["Mauersegler", "Nachtigall", "Kuckuck", "Rauchschwalbe", "Mehlschwalbe", "Fitis", "Zilpzalp", "Kranich", "Weißstorch"]
    detected_migratory = set([r[0] for r in rows if r[0] in migratory_birds])
    if detected_migratory:
        zugvogel_text = f"Zugvögel sind aktiv! Gesichtet wurden: {', '.join(detected_migratory)}."
    else:
        zugvogel_text = "Momentan gibt es wenig Aktivität von typischen Zugvögeln am Mikrofon."
        
    # 5. Tagesbericht (Regelbasiert)
    if total_next_h >= 750:
        berichte_high = [
            f"Vogelkonzert vom Feinsten! Für die nächste Stunde (um {next_h:02d}:00 Uhr) erwarten wir über {total_next_h} erfasste Rufe.",
            f"Es ist richtig was los! Wir rechnen um {next_h:02d}:00 Uhr mit extrem hoher akustischer Aktivität.",
            f"Ein wahres Stimmenmeer! Die nächste Stunde verspricht sehr viele Aufzeichnungen."
        ]
        bericht = random.choice(berichte_high)
    elif total_next_h >= 350:
        berichte_med_high = [
            f"Gute akustische Aktivität! Wir erwarten um {next_h:02d}:00 Uhr rege Rufaktivität.",
            f"Es wird musikalisch: Für die nächste Stunde deuten die Daten auf viele Vogelstimmen hin.",
            f"Reger Flugverkehr am Mikrofon: Um die {total_next_h} Rufe werden in der nächsten Stunde erwartet."
        ]
        bericht = random.choice(berichte_med_high)
    elif total_next_h >= 100:
        berichte_med = [
            f"Normale Rufaktivität erwartet. Schauen Sie mal um {busiest_hour:02d}:00 Uhr rein, da ist historisch gesehen am meisten zu hören.",
            f"Gelegentliche Rufe sind in der nächsten Stunde zu erwarten. Die höchste Aktivität gibt es meist gegen {busiest_hour:02d}:00 Uhr.",
            f"Gemütliche Stimmung. Wir erwarten eine durchschnittliche Anzahl an Aufzeichnungen."
        ]
        bericht = random.choice(berichte_med)
    else:
        berichte_low = [
            "Aktuell ist es eher ruhig. Wenig bis keine Rufe in der nächsten Stunde erwartet.",
            "Stille am Mikrofon. Das ist für diese Tageszeit aber nicht ungewöhnlich.",
            "Kaum akustische Aktivität prognostiziert. Eine gute Zeit, um die Aufnahmen der letzten Rush-Hour durchzuhören."
        ]
        bericht = random.choice(berichte_low)
        
    # 6. Anomaliekarte
    baseline_days = max(14, days)
    baseline_start = end_date - datetime.timedelta(days=baseline_days)
    recent_start = end_date - datetime.timedelta(days=1)
    
    c.execute("SELECT species, COUNT(*) FROM detections WHERE timestamp >= ? AND timestamp < ? GROUP BY species", 
              (baseline_start.strftime('%Y-%m-%d %H:%M:%S'), recent_start.strftime('%Y-%m-%d %H:%M:%S')))
    hist_counts = {row[0]: row[1] for row in c.fetchall()}
    
    c.execute("SELECT species, COUNT(*) FROM detections WHERE timestamp >= ? GROUP BY species", 
              (recent_start.strftime('%Y-%m-%d %H:%M:%S'),))
    recent_counts = {row[0]: row[1] for row in c.fetchall()}
    
    anomalies = []
    hist_days = baseline_days - 1
    if hist_days <= 0: hist_days = 1
    
    all_species_anomaly = set(list(hist_counts.keys()) + list(recent_counts.keys()))
    for s in all_species_anomaly:
        avg_daily = hist_counts.get(s, 0) / hist_days
        current = recent_counts.get(s, 0)
        
        if current >= 5 and current > avg_daily * 3:
            anomalies.append({
                'species': s,
                'status': 'influx',
                'title': 'Massenansturm',
                'desc': f'Ungewöhnlich viele Sichtungen ({current} in 24h, normal sind {avg_daily:.1f}/Tag).'
            })
        elif avg_daily >= 3 and current == 0:
            anomalies.append({
                'species': s,
                'status': 'missing',
                'title': 'Verschwunden',
                'desc': f'Heute noch nicht gesichtet (normal sind {avg_daily:.1f}/Tag).'
            })
            
    anomalies = sorted(anomalies, key=lambda x: x['status'])[:9] # max 9 anzeigen
        
    conn.close()
    
    return render_template('prediction.html',
                           days=days,
                           rush_hour_labels=rush_hour_labels,
                           rush_hour_data=rush_hour_data,
                           busiest_hour=busiest_hour,
                           prob_labels=prob_labels,
                           prob_data=prob_data,
                           next_hour=next_h,
                           selten_text=selten_text,
                           zugvogel_text=zugvogel_text,
                           tagesbericht=bericht,
                           anomalies=anomalies)

@app.route('/api/settings/save', methods=['POST'])
def api_save_settings():
    data = request.json
    if "birdweather_id" in data:
        save_birdweather_setting("birdweather_id", data.get("birdweather_id", ""))
    if "birdweather_active" in data:
        save_birdweather_setting("birdweather_active", bool(data.get("birdweather_active", False)))
    save_setting("threshold", data.get("threshold", 30))
    save_setting("occurrence_threshold", float(data.get("occurrence_threshold", 0.03)))
    if "auto_season_lowering" in data:
        save_setting("auto_season_lowering", bool(data.get("auto_season_lowering", False)))
    save_setting("min_snr", data.get("min_snr", 0.0))
    save_setting("gps_lat", data.get("gps_lat"))
    save_setting("gps_lon", data.get("gps_lon"))
    save_setting("radar_zoom", data.get("radar_zoom", 1.0))
    save_setting("radar_max_birds", data.get("radar_max_birds", 10))
    save_setting("radar_time_range", data.get("radar_time_range", 24))
    save_setting("radar_snr_max", data.get("radar_snr_max", 20.0))
    save_setting("radar_snr_min", data.get("radar_snr_min", 5.0))
    save_setting("barchart_max_calls_weekly", int(data.get("barchart_max_calls_weekly", 3000)))
    if "device_hostname" in data:
        save_setting("device_hostname", data.get("device_hostname", "bird-ai-sound-classifier"))
    if "mic_index" in data:
        save_setting("mic_index", data.get("mic_index", -1))
    if "archive_species" in data:
        save_setting("archive_species", data.get("archive_species", ""))
    if "max_archive_files" in data:
        save_setting("max_archive_files", int(data.get("max_archive_files", 0)))
    if "alarm_active" in data:
        save_setting("alarm_active", bool(data.get("alarm_active", False)))
    if "click_sound_active" in data:
        save_setting("click_sound_active", bool(data.get("click_sound_active", False)))
    if "highpass_active" in data:
        save_setting("highpass_active", bool(data.get("highpass_active", False)))
    if "highpass_freq" in data:
        save_setting("highpass_freq", int(data.get("highpass_freq", 1000)))
    if "lowpass_active" in data:
        save_setting("lowpass_active", bool(data.get("lowpass_active", False)))
    if "lowpass_freq" in data:
        save_setting("lowpass_freq", int(data.get("lowpass_freq", 12000)))
    if "nr_active" in data:
        save_setting("nr_active", bool(data.get("nr_active", False)))
    if "nr_quality" in data:
        save_setting("nr_quality", str(data.get("nr_quality", "Medium")))
    if "log_blocklist" in data:
        save_setting("log_blocklist", bool(data.get("log_blocklist", True)))
    if "log_blocklist_prob_switch" in data:
        save_setting("log_blocklist_prob_switch", bool(data.get("log_blocklist_prob_switch", False)))
    if "bird_dictionary" in data:
        save_dictionary(data.get("bird_dictionary", {}))
    return jsonify({"msg": "Einstellungen gespeichert!"})

@app.route('/api/birdweather/test', methods=['POST'])
def api_birdweather_test():
    data = request.json
    token = data.get("birdweather_id", "").strip()
    if not token:
        return jsonify({"success": False, "msg": "Kein Token angegeben."})
    try:
        url = f"https://app.birdweather.com/api/v1/stations/{token}"
        r = requests.get(url, timeout=10)
        try:
            resp_data = r.json()
        except:
            resp_data = {}
        
        if r.status_code == 200 or resp_data.get("success"):
            return jsonify({"success": True, "msg": "Verbindung erfolgreich! Token ist gültig."})
        else:
            return jsonify({"success": False, "msg": f"Zugriff verweigert oder ungültiger Token. Meldung: {resp_data.get('message', 'Unbekannt')}"})
    except Exception as e:
        return jsonify({"success": False, "msg": f"Fehler bei der Verbindung: {e}"})

@app.route('/api/control/apply_dictionary', methods=['POST'])
def api_control_apply_dictionary():
    bird_dict = get_bird_dictionary()
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    updated_count = 0
    renamed_files_count = 0
    try:
        archive_dir = os.path.join(AUDIO_DIR, "archive")
        if os.path.exists(archive_dir):
            archive_files = os.listdir(archive_dir)
        else:
            archive_files = []

        for eng, trans in bird_dict.items():
            if eng != trans and trans.strip():
                # DB Update
                c.execute("UPDATE detections SET species = ? WHERE species = ?", (trans, eng))
                updated_count += c.rowcount
                
                # Archive Renaming
                safe_eng = eng.replace(" ", "_").replace("/", "_")
                safe_trans = trans.replace(" ", "_").replace("/", "_")
                prefix = f"{safe_eng}_"
                new_prefix = f"{safe_trans}_"
                
                for f in archive_files:
                    if f.startswith(prefix) and f.endswith(".wav"):
                        old_path = os.path.join(archive_dir, f)
                        new_name = f.replace(prefix, new_prefix, 1)
                        new_path = os.path.join(archive_dir, new_name)
                        try:
                            os.rename(old_path, new_path)
                            renamed_files_count += 1
                        except Exception as e:
                            print(f"Fehler beim Umbenennen von {f}: {e}")
                            
        # Re-list files if needed for subsequent operations, but not strictly necessary here.
        conn.commit()
    except Exception as e:
        conn.rollback()
        return jsonify({"error": str(e)}), 500
    finally:
        conn.close()
    return jsonify({"msg": f"Wörterbuch angewendet. {updated_count} DB-Einträge und {renamed_files_count} Dateien aktualisiert."})

@app.route('/api/control/delete_single_occurrences', methods=['POST'])
def api_control_delete_single_occurrences():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    deleted_count = 0
    deleted_files = 0
    try:
        c.execute("""
            SELECT species, timestamp FROM detections 
            WHERE species IN (
                SELECT species FROM detections 
                GROUP BY species 
                HAVING COUNT(*) = 1
            )
        """)
        rows = c.fetchall()

        c.execute("""
            DELETE FROM detections 
            WHERE species IN (
                SELECT species FROM detections 
                GROUP BY species 
                HAVING COUNT(*) = 1
            )
        """)
        deleted_count = c.rowcount
        conn.commit()

        archive_dir = os.path.join(AUDIO_DIR, "archive")
        if os.path.exists(archive_dir):
            for species, ts in rows:
                safe_species = species.replace(" ", "_").replace("/", "_")
                prefix = f"{safe_species}_"
                for filename in os.listdir(archive_dir):
                    if filename.startswith(prefix) and filename.endswith(".wav"):
                        filepath = os.path.join(archive_dir, filename)
                        try:
                            os.remove(filepath)
                            deleted_files += 1
                        except:
                            pass

    except Exception as e:
        conn.rollback()
        return jsonify({"error": str(e)}), 500
    finally:
        conn.close()
    return jsonify({"msg": f"{deleted_count} einzelne Vogelarten und {deleted_files} Dateien wurden gelöscht."})

@app.route('/api/control/bulk_delete_species', methods=['POST'])
def api_control_bulk_delete_species():
    data = request.json
    species = data.get("species")
    if not species:
        return jsonify({"error": "Keine Vogelart angegeben."}), 400
        
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    deleted_count = 0
    deleted_files = 0
    try:
        c.execute("SELECT timestamp FROM detections WHERE species = ?", (species,))
        rows = c.fetchall()

        c.execute("DELETE FROM detections WHERE species = ?", (species,))
        deleted_count = c.rowcount
        conn.commit()

        safe_species = species.replace(" ", "_").replace("/", "_")
        archive_dir = os.path.join(AUDIO_DIR, "archive")
        if os.path.exists(archive_dir):
            prefix = f"{safe_species}_"
            for filename in os.listdir(archive_dir):
                if filename.startswith(prefix) and filename.endswith(".wav"):
                    filepath = os.path.join(archive_dir, filename)
                    try:
                        os.remove(filepath)
                        deleted_files += 1
                    except:
                        pass

    except Exception as e:
        conn.rollback()
        return jsonify({"error": str(e)}), 500
    finally:
        conn.close()
    return jsonify({"success": True, "msg": f"{deleted_count} Einträge und {deleted_files} Dateien für '{species}' wurden gelöscht."})

@app.route('/api/control/bulk_rename_species', methods=['POST'])
def api_control_bulk_rename_species():
    data = request.json
    old_species = data.get("old_species")
    new_species = data.get("new_species")
    if not old_species or not new_species:
        return jsonify({"error": "Vogelarten nicht vollständig angegeben."}), 400
        
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    updated_count = 0
    renamed_files = 0
    try:
        c.execute("SELECT timestamp FROM detections WHERE species = ?", (old_species,))
        rows = c.fetchall()

        c.execute("UPDATE detections SET species = ? WHERE species = ?", (new_species, old_species))
        updated_count = c.rowcount
        conn.commit()

        safe_old_species = old_species.replace(" ", "_").replace("/", "_")
        safe_new_species = new_species.replace(" ", "_").replace("/", "_")
        archive_dir = os.path.join(AUDIO_DIR, "archive")

        if os.path.exists(archive_dir):
            for filename in os.listdir(archive_dir):
                if filename.startswith(safe_old_species + "_") and filename.endswith(".wav"):
                    old_filepath = os.path.join(archive_dir, filename)
                    new_filename = filename.replace(safe_old_species + "_", safe_new_species + "_", 1)
                    new_filepath = os.path.join(archive_dir, new_filename)
                    try:
                        os.rename(old_filepath, new_filepath)
                        renamed_files += 1
                    except Exception as fe:
                        print(f"Failed to rename {filename}: {fe}")

    except Exception as e:
        conn.rollback()
        return jsonify({"error": str(e)}), 500
    finally:
        conn.close()
    return jsonify({"success": True, "msg": f"{updated_count} Einträge und {renamed_files} Dateien wurden von '{old_species}' in '{new_species}' umbenannt."})

@app.route('/api/model_labels')
def api_model_labels():
    try:
        model_dir = os.path.join(os.path.dirname(__file__), 'model')
        labels_path = None
        if os.path.exists(model_dir):
            for file in os.listdir(model_dir):
                if file.endswith('_Labels.txt') or file.endswith('Labels.txt'):
                    labels_path = os.path.join(model_dir, file)
                    break
        
        if not labels_path or not os.path.exists(labels_path):
            return jsonify({"error": "Label-Datei nicht gefunden."}), 404
        with open(labels_path, 'r', encoding='utf-8') as f:
            labels = [line.strip() for line in f if line.strip()]
        return jsonify({"labels": labels})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/status')
def api_status():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT COUNT(*) FROM detections")
    total_count = c.fetchone()[0]
    
    # Heute
    today = datetime.datetime.now().strftime("%Y-%m-%d")
    c.execute("SELECT COUNT(*) FROM detections WHERE timestamp LIKE ?", (f"{today}%",))
    today_count = c.fetchone()[0]
    
    conn.close()

    return jsonify({
        "status": "Online (Mikrofon aktiv)" if (monitor_running_event.is_set() if monitor_running_event else False) else "Offline (Gestoppt)",
        "total_detections": total_count,
        "today_detections": today_count,
        "effective_occurrence_threshold": get_effective_occurrence_threshold()
    })

@app.route('/api/check_model_update')
def check_model_update():
    current_version = "V2.4"
    try:
        import urllib.request
        import json
        req = urllib.request.Request(
            'https://api.github.com/repos/birdnet-team/BirdNET-Analyzer/releases/latest',
            headers={'User-Agent': 'Mozilla/5.0'}
        )
        with urllib.request.urlopen(req, timeout=5) as response:
            data = json.loads(response.read().decode())
            latest_version = data.get('tag_name', '')
            html_url = data.get('html_url', 'https://github.com/birdnet-team/BirdNET-Analyzer/releases')
            
            cur_norm = current_version.lower().replace('v', '').strip()
            lat_norm = latest_version.lower().replace('v', '').strip()
            
            if lat_norm.startswith(cur_norm) and len(lat_norm) <= len(cur_norm) + 2:
                is_newer = False
            elif lat_norm != cur_norm:
                is_newer = True
            else:
                is_newer = False
                
            return jsonify({
                "success": True,
                "current_version": current_version,
                "latest_version": latest_version,
                "is_newer": is_newer,
                "download_url": html_url
            })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

@app.route('/api/check_app_update')
def check_app_update():
    current_version = "V1.3.5-RC4"
    try:
        import urllib.request
        import json
        req = urllib.request.Request(
            'https://api.github.com/repos/mobifu1/birds_ai_sound_classifier/tags',
            headers={'User-Agent': 'Mozilla/5.0'}
        )
        with urllib.request.urlopen(req, timeout=5) as response:
            data = json.loads(response.read().decode())
            if not data:
                return jsonify({"success": False, "error": "Keine Tags/Versionen auf GitHub gefunden."})
                
            latest_version = data[0].get('name', '')
            html_url = f"https://github.com/mobifu1/birds_ai_sound_classifier/releases/tag/{latest_version}"
            
            cur_norm = current_version.lower().replace('v', '').strip()
            lat_norm = latest_version.lower().replace('v', '').strip()
            
            is_newer = False
            if lat_norm and cur_norm:
                try:
                    lat_parts = [int(x) for x in lat_norm.split('.')]
                    cur_parts = [int(x) for x in cur_norm.split('.')]
                    
                    # Pad lists if lengths differ
                    while len(lat_parts) < max(len(lat_parts), len(cur_parts)): lat_parts.append(0)
                    while len(cur_parts) < max(len(lat_parts), len(cur_parts)): cur_parts.append(0)
                        
                    for l, c in zip(lat_parts, cur_parts):
                        if l > c:
                            is_newer = True
                            break
                        elif l < c:
                            is_newer = False
                            break
                except ValueError:
                    # Fallback
                    if lat_norm != cur_norm and lat_norm > cur_norm:
                        is_newer = True
                
            return jsonify({
                "success": True,
                "current_version": current_version,
                "latest_version": latest_version,
                "is_newer": is_newer,
                "download_url": html_url
            })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

from flask import Response
import matplotlib.cm as cm
from PIL import Image

def generate_waterfall_stream():
    import matplotlib.cm as cm
    cmap = matplotlib.colormaps['viridis']
    while True:
        if shared_waterfall_global is not None:
            with shared_waterfall_global.get_lock():
                data = np.frombuffer(shared_waterfall_global.get_obj(), dtype=np.float32).reshape((WATERFALL_HEIGHT, max_bin)).copy()
        else:
            global latest_waterfall_data
            data = latest_waterfall_data.copy()
        mapped = cmap(data)
        img_data = (mapped[:, :, :3] * 255).astype(np.uint8)
        img = Image.fromarray(img_data)
        buf = io.BytesIO()
        img.save(buf, format='JPEG', quality=80)
        frame = buf.getvalue()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        time.sleep(0.05)

@app.route('/api/waterfall_feed')
def waterfall_feed():
    return Response(generate_waterfall_stream(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/audio_level')
def api_audio_level():
    lvl = shared_audio_level_global.value if shared_audio_level_global is not None else latest_audio_level
    ql = shared_queue_length_global.value if shared_queue_length_global is not None else latest_queue_length
    return jsonify({
        "level": lvl,
        "queue_length": ql
    })

@app.route('/api/latest_logs')
def api_latest_logs():
    return jsonify(list(log_messages))

@app.route('/api/live_audio')
def api_live_audio():
    if os.path.exists(TEMP_WAV):
        return send_file(TEMP_WAV, mimetype="audio/wav")
    return abort(404)


def run_audio_process(running_event, log_q, shared_al, shared_ql, shared_wf):
    global log_queue_global, shared_audio_level_global, shared_queue_length_global, shared_waterfall_global
    global analyzer
    
    print("Initialisiere BirdNET Analyzer...")
    with open(os.devnull, 'w') as f, contextlib.redirect_stdout(f), contextlib.redirect_stderr(f):
        analyzer = Analyzer()
    print("OK: BirdNET Analyzer bereit.")
    
    log_queue_global = log_q
    shared_audio_level_global = shared_al
    shared_queue_length_global = shared_ql
    shared_waterfall_global = shared_wf
    
    monitor = AudioMonitor()
    
    update_log("Audio-Prozess gestartet. Warte auf Start-Signal...")
    
    was_running = False
    
    while True:
        is_set = running_event.is_set()
        if is_set and not was_running:
            monitor.start()
            was_running = True
        elif not is_set and was_running:
            monitor.stop()
            was_running = False
            
        time.sleep(0.5)

# --- CONTROL ROUTEN ---
@app.route('/api/control/start', methods=['POST'])
def api_control_start():
    if monitor_running_event:
        monitor_running_event.set()
        return jsonify({"msg": "Gestartet"})
    return jsonify({"error": "Process event not initialized"})

@app.route('/api/control/stop', methods=['POST'])
def api_control_stop():
    if monitor_running_event:
        monitor_running_event.clear()
    return jsonify({"msg": "Gestoppt"})

@app.route('/api/control/dbsync', methods=['POST'])
def api_control_dbsync():
    # Einfache Sortierung der DB nach Timestamp
    try:
        conn = sqlite3.connect(DB_FILE)
        c = conn.cursor()
        c.execute("CREATE TABLE detections_temp AS SELECT * FROM detections ORDER BY timestamp ASC")
        c.execute("DROP TABLE detections")
        c.execute("ALTER TABLE detections_temp RENAME TO detections")
        conn.commit()
        conn.close()
        return jsonify({"msg": "Datenbank zeitlich sortiert!"})
    except Exception as e:
        return jsonify({"msg": f"Fehler: {e}"})

@app.route('/api/control/dbreset', methods=['POST'])
def api_control_dbreset():
    try:
        conn = sqlite3.connect(DB_FILE)
        c = conn.cursor()
        c.execute("DELETE FROM detections")
        conn.commit()
        conn.close()
        
        deleted_files = 0
        archive_dir = os.path.join(AUDIO_DIR, "archive")
        if os.path.exists(archive_dir):
            for f in os.listdir(archive_dir):
                if f.endswith(".wav"):
                    try:
                        os.remove(os.path.join(archive_dir, f))
                        deleted_files += 1
                    except:
                        pass

        return jsonify({"msg": f"Datenbank wurde erfolgreich geleert! {deleted_files} Dateien wurden gelöscht."})
    except Exception as e:
        return jsonify({"msg": f"Fehler beim Leeren: {e}"})

@app.route('/api/control/dbbackup', methods=['POST'])
def api_control_dbbackup():
    try:
        import shutil
        import os
        backup_dir = "backup_db"
        if not os.path.exists(backup_dir):
            os.makedirs(backup_dir)
        backup_file = os.path.join(backup_dir, f"birds_log_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.db")
        shutil.copy2(DB_FILE, backup_file)
        return jsonify({"msg": f"Backup erstellt: {backup_file}"})
    except Exception as e:
        return jsonify({"msg": f"Fehler beim Backup: {e}"})

@app.route('/api/top_species')
def api_top_species():
    s = load_settings()
    max_birds = int(s.get("radar_max_birds", 10))
    radar_time_range = int(s.get("radar_time_range", 24))
    
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    query = f"""
        SELECT d1.species, COUNT(*) as count,
               (SELECT snr FROM detections d2 WHERE d2.species = d1.species ORDER BY timestamp DESC LIMIT 1) as snr,
               (SELECT confidence FROM detections d2 WHERE d2.species = d1.species ORDER BY timestamp DESC LIMIT 1) as confidence,
               (SELECT timestamp FROM detections d2 WHERE d2.species = d1.species ORDER BY timestamp DESC LIMIT 1) as timestamp,
               (SELECT geo_prob FROM detections d2 WHERE d2.species = d1.species ORDER BY timestamp DESC LIMIT 1) as geo_prob
        FROM detections d1
        WHERE date(timestamp) = date('now', 'localtime')
          AND timestamp >= datetime('now', '-{radar_time_range} hours', 'localtime')
        GROUP BY species
        ORDER BY count DESC
        LIMIT {max_birds}
    """
    c.execute(query)
    raw_data = c.fetchall()
    top_data = []
    for r in raw_data:
        snr_val = r[2]
        if isinstance(snr_val, bytes):
            import struct
            try:
                if len(snr_val) == 8:
                    snr_val = struct.unpack('d', snr_val)[0]
                elif len(snr_val) == 4:
                    snr_val = struct.unpack('f', snr_val)[0]
                else:
                    snr_val = 0.0
            except:
                snr_val = 0.0
        elif snr_val is None:
            snr_val = 0.0
            
        conf_val = r[3]
        if isinstance(conf_val, bytes):
            import struct
            try:
                if len(conf_val) == 8:
                    conf_val = struct.unpack('d', conf_val)[0]
                elif len(conf_val) == 4:
                    conf_val = struct.unpack('f', conf_val)[0]
                else:
                    conf_val = 0.0
            except:
                conf_val = 0.0
        elif conf_val is None:
            conf_val = 0.0
            
        geo_prob_val = r[5] if len(r) > 5 and r[5] is not None else 0.0
        if isinstance(geo_prob_val, bytes):
            import struct
            try:
                if len(geo_prob_val) == 8:
                    geo_prob_val = struct.unpack('d', geo_prob_val)[0]
                elif len(geo_prob_val) == 4:
                    geo_prob_val = struct.unpack('f', geo_prob_val)[0]
                else:
                    geo_prob_val = 0.0
            except:
                geo_prob_val = 0.0

        top_data.append({
            "species": r[0], 
            "count": r[1], 
            "snr": float(snr_val),
            "confidence": float(conf_val),
            "timestamp": r[4],
            "geo_prob": float(geo_prob_val)
        })

    
    c.execute("SELECT species, rowid, timestamp FROM detections ORDER BY timestamp DESC LIMIT 1")
    last = c.fetchone()
    latest_species = last[0] if last else None
    latest_id = last[1] if last else None
    
    seconds_since_latest = None
    if last and last[2]:
        try:
            # Handle potential fractional seconds or different formats
            ts_str = last[2].split(".")[0]
            ts = datetime.datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S")
            diff = (datetime.datetime.now() - ts).total_seconds()
            seconds_since_latest = max(0, int(diff))
        except Exception as e:
            print(f"Error parsing timestamp {last[2]}: {e}")
            pass
    
    c.execute(f"SELECT COUNT(DISTINCT species) FROM detections WHERE date(timestamp) = date('now', 'localtime') AND timestamp >= datetime('now', '-{radar_time_range} hours', 'localtime')")
    unique_count = c.fetchone()[0]
    
    c.execute("SELECT COUNT(*) FROM detections WHERE timestamp >= datetime('now', '-1 hours', 'localtime')")
    last_hour_count = c.fetchone()[0]
    c.execute("SELECT COUNT(*) FROM detections WHERE timestamp >= datetime('now', '-24 hours', 'localtime')")
    last_24h_count = c.fetchone()[0]
    
    c.execute("""
        SELECT species
        FROM detections
        GROUP BY species
        HAVING date(MIN(timestamp)) = date('now', 'localtime')
        ORDER BY MIN(timestamp) DESC
        LIMIT 1
    """)
    new_record = c.fetchone()
    new_record_species = new_record[0] if new_record else None
    
    c.execute("SELECT COUNT(*) FROM detections WHERE date(timestamp) = date('now', 'localtime')")
    today_total_count = c.fetchone()[0]
    
    conn.close()
    return jsonify({
        "top": top_data, 
        "latest": latest_species, 
        "latest_id": latest_id, 
        "unique_species_count": unique_count, 
        "last_hour_count": last_hour_count,
        "last_24h_count": last_24h_count,
        "seconds_since_latest": seconds_since_latest,
        "new_record_species": new_record_species,
        "today_total_count": today_total_count,
        "radar_time_range": radar_time_range
    })

# --- DATENBANK MANAGEMENT ROUTEN ---
@app.route('/api/detections/by_date')
def api_detections_by_date():
    date_str = request.args.get('date', '')
    if not date_str: return jsonify({"success": False, "error": "No date provided"})
    try:
        conn = sqlite3.connect(DB_FILE)
        c = conn.cursor()
        c.execute("SELECT rowid, timestamp, species, confidence FROM detections WHERE timestamp LIKE ? ORDER BY timestamp DESC", (f"{date_str}%",))
        rows = c.fetchall()
        conn.close()
        entries = [{"id": r[0], "timestamp": r[1], "species": r[2], "confidence": r[3]} for r in rows]
        return jsonify({"success": True, "entries": entries})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

@app.route('/api/detections/delete', methods=['POST'])
def api_detections_delete():
    data = request.json
    try:
        conn = sqlite3.connect(DB_FILE)
        c = conn.cursor()
        
        c.execute("SELECT species, timestamp FROM detections WHERE rowid = ?", (data['id'],))
        row = c.fetchone()
        
        c.execute("DELETE FROM detections WHERE rowid = ?", (data['id'],))
        conn.commit()
        conn.close()
        
        if row:
            old_species, old_ts = row
            try:
                dt = datetime.datetime.strptime(old_ts, "%Y-%m-%d %H:%M:%S")
                fn_ts = dt.strftime("%y-%m-%d-%H-%M-%S")
                safe_species = old_species.replace(" ", "_").replace("/", "_")
                old_filename = f"{safe_species}_{fn_ts}.wav"
                old_filepath = os.path.join(AUDIO_DIR, "archive", old_filename)
                
                if os.path.exists(old_filepath):
                    os.remove(old_filepath)
                    update_log(f"Zugehörige Audiodatei gelöscht: {old_filename}")
            except Exception as fe:
                update_log(f"Fehler beim Löschen der Audiodatei: {fe}")
                
        return jsonify({"success": True, "msg": "Eintrag gelöscht."})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

@app.route('/api/detections/update', methods=['POST'])
def api_detections_update():
    data = request.json
    try:
        conn = sqlite3.connect(DB_FILE)
        c = conn.cursor()
        
        c.execute("SELECT species, timestamp FROM detections WHERE id = ?", (data['id'],))
        row = c.fetchone()
        
        c.execute("UPDATE detections SET species = ? WHERE id = ?", (data['species'], data['id']))
        conn.commit()
        conn.close()
        
        if row:
            old_species, old_ts = row
            new_species = data['species']
            if old_species != new_species:
                try:
                    dt = datetime.datetime.strptime(old_ts, "%Y-%m-%d %H:%M:%S")
                    fn_ts = dt.strftime("%y-%m-%d-%H-%M-%S")
                    
                    safe_old_species = old_species.replace(" ", "_").replace("/", "_")
                    old_filename = f"{safe_old_species}_{fn_ts}.wav"
                    old_filepath = os.path.join(AUDIO_DIR, "archive", old_filename)
                    
                    safe_new_species = new_species.replace(" ", "_").replace("/", "_")
                    new_filename = f"{safe_new_species}_{fn_ts}.wav"
                    new_filepath = os.path.join(AUDIO_DIR, "archive", new_filename)
                    
                    if os.path.exists(old_filepath):
                        os.rename(old_filepath, new_filepath)
                        update_log(f"Audiodatei umbenannt: {old_filename} -> {new_filename}")
                except Exception as fe:
                    update_log(f"Fehler beim Umbenennen der Audiodatei: {fe}")
                    
        return jsonify({"success": True, "msg": "Eintrag aktualisiert."})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

@app.route('/api/detections/add', methods=['POST'])
def api_detections_add():
    data = request.json
    try:
        conn = sqlite3.connect(DB_FILE)
        c = conn.cursor()
        c.execute("INSERT INTO detections (species, timestamp, confidence) VALUES (?, ?, ?)", 
                  (data['species'], data['timestamp'], 100.0))
        conn.commit()
        conn.close()
        return jsonify({"success": True, "msg": "Eintrag manuell hinzugefügt."})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

@app.route('/api/recent_events')
def api_recent_events():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT species, timestamp, confidence FROM detections ORDER BY timestamp DESC LIMIT 20")
    data = c.fetchall()
    conn.close()
    
    events = []
    for row in data:
        events.append({
            "species": row[0],
            "time": row[1].split()[1],
            "date": row[1].split()[0],
            "confidence": f"{row[2]:.0%}"
        })
    return jsonify(events)

@app.route('/api/export/weekly_csv')
def export_weekly_csv():
    from flask import Response
    year_str = request.args.get('year')
    if year_str:
        where_clause = "WHERE timestamp IS NOT NULL AND timestamp != '' AND timestamp LIKE ?"
        params = (f"{year_str}%",)
    else:
        where_clause = "WHERE timestamp IS NOT NULL AND timestamp != ''"
        params = ()

    query = f"""
    SELECT 
        CASE WHEN species = 'IGNORED_LOW_CONFIDENCE' THEN 'Unbekannt' ELSE species END as species,
        printf('%02d', CAST(strftime('%W', timestamp) AS INTEGER) + 1) as week,
        COUNT(*) as counts
    FROM detections
    {where_clause}
    GROUP BY species, week
    ORDER BY week
    """
    
    conn = sqlite3.connect(DB_FILE, timeout=10)
    try:
        grouped = pd.read_sql_query(query, conn, params=params)
    except:
        grouped = pd.DataFrame()
    finally:
        conn.close()

    if grouped.empty:
        return "Keine Daten vorhanden für dieses Jahr.", 404

    pivot_counts = grouped.pivot(index='species', columns='week', values='counts').fillna(0).astype(int)
    csv_data = pivot_counts.to_csv(sep=';')
    
    filename = f"weekly_export_{year_str if year_str else 'all'}.csv"
    return Response(
        csv_data,
        mimetype="text/csv",
        headers={"Content-disposition": f"attachment; filename={filename}"}
    )

@app.route('/api/export/blocklist_log')
def export_blocklist_log():
    try:
        return send_file('blocklist-log.txt', as_attachment=True)
    except Exception as e:
        return f"Fehler beim Herunterladen: {e}", 404

@app.route('/api/control/clear_blocklist_log', methods=['POST'])
def clear_blocklist_log():
    try:
        open('blocklist-log.txt', 'w').close()
        return jsonify({'success': True, 'msg': 'Blocklist-Log wurde erfolgreich gelöscht.'})
    except Exception as e:
        return jsonify({'success': False, 'msg': f'Fehler beim Löschen: {e}'})

@app.route('/api/control/update_bird_icons', methods=['POST'])
def update_bird_icons():
    try:
        response = requests.get("https://api.github.com/repos/mobifu1/birds_ai_sound_classifier/contents/static/bird_icons", timeout=10)
        if response.status_code != 200:
            return jsonify({'success': False, 'msg': f'Fehler beim Abrufen der Github-Daten. Statuscode: {response.status_code}'})
        
        files = response.json()
        downloaded = 0
        icon_dir = os.path.join("static", "bird_icons")
        os.makedirs(icon_dir, exist_ok=True)
        
        for file in files:
            if file.get("type") == "file":
                file_name = file.get("name")
                download_url = file.get("download_url")
                local_path = os.path.join(icon_dir, file_name)
                
                if not os.path.exists(local_path) and download_url:
                    img_resp = requests.get(download_url, timeout=10)
                    if img_resp.status_code == 200:
                        with open(local_path, "wb") as f:
                            f.write(img_resp.content)
                        downloaded += 1
                        
        return jsonify({'success': True, 'msg': f'Update abgeschlossen. {downloaded} neue Icons heruntergeladen.'})
    except Exception as e:
        return jsonify({'success': False, 'msg': f'Fehler: {str(e)}'})

@app.route('/api/control/update_dictionary', methods=['POST'])
def update_dictionary():
    try:
        response = requests.get("https://raw.githubusercontent.com/mobifu1/birds_ai_sound_classifier/main/dictionary.json", timeout=10)
        if response.status_code != 200:
            return jsonify({'success': False, 'msg': f'Fehler beim Abrufen der Dictionary-Daten. Statuscode: {response.status_code}'})
        
        new_dict = response.json()
        local_dict = load_dictionary()
        added = 0
        
        for key, value in new_dict.items():
            if key not in local_dict:
                local_dict[key] = value
                added += 1
                
        if added > 0:
            save_dictionary(local_dict)
            
        return jsonify({'success': True, 'msg': f'Update abgeschlossen. {added} neue Vögel dem Wörterbuch hinzugefügt.'})
    except Exception as e:
        return jsonify({'success': False, 'msg': f'Fehler: {str(e)}'})

@app.route('/api/control/check_probability', methods=['GET'])
def check_probability_route():
    try:
        from birdnetlib.species import SpeciesList
        import datetime
        
        local_settings = load_settings()
        lat = float(local_settings.get('gps_lat', 0.0))
        lon = float(local_settings.get('gps_lon', 0.0))
        
        threshold_val = get_effective_occurrence_threshold()
        
        if not os.path.exists(DICTIONARY_FILE):
            return jsonify({'success': False, 'msg': 'dictionary.json existiert nicht.'})
        with open(DICTIONARY_FILE, 'r', encoding='utf-8') as f:
            dictionary = json.load(f)
            
        with open(os.devnull, 'w') as f, contextlib.redirect_stdout(f), contextlib.redirect_stderr(f):
            sl = SpeciesList()
            predicted = sl.return_list(lat=lat, lon=lon, date=datetime.datetime.now(), threshold=0.0)
        
        results = []
        for vogel_name, props in dictionary.items():
            trans = props.get("translation", "")
            translated_name = trans if str(trans).strip() != "" else vogel_name
            parts = vogel_name.split('_')
            if len(parts) >= 2:
                sci_name = parts[0]
                prob = 0.0
                for p in predicted:
                    if p['scientific_name'] == sci_name:
                        prob = float(p['threshold'])
                        break
                results.append((vogel_name, translated_name, prob))
            else:
                # Fallback if dictionary keys are different (which they are)
                prob = 0.0
                for p in predicted:
                    if p['scientific_name'] in vogel_name or p['common_name'] == vogel_name:
                        prob = float(p['threshold'])
                        break
                results.append((vogel_name, translated_name, prob))
                
        results.sort(key=lambda x: x[2], reverse=True)
        
        msg_lines = []
        for eng_name, trans_name, prob in results:
            color = "#4CAF50" if prob >= threshold_val else "#F44336"
            msg_lines.append(f'<div style="color: {color};">{eng_name} ({trans_name}): {prob*100:.1f}%</div>')
            
        msg = "".join(msg_lines)
        return jsonify({'success': True, 'msg': msg})
        
    except Exception as e:
        return jsonify({'success': False, 'msg': f'Fehler: {str(e)}'})

@app.route('/api/control/check_dictionary', methods=['GET'])
def check_dictionary_route():
    issues = []
    if not os.path.exists(DICTIONARY_FILE):
        return jsonify({'success': False, 'msg': 'dictionary.json existiert nicht.'})
        
    with open(DICTIONARY_FILE, 'r', encoding='utf-8') as f:
        content = f.read()
        
    seen_keys = set()
    duplicates_keys = set()
    
    def object_pairs_hook(pairs):
        for key, value in pairs:
            if key in seen_keys:
                duplicates_keys.add(key)
            else:
                seen_keys.add(key)
        return dict(pairs)
        
    try:
        parsed = json.loads(content, object_pairs_hook=object_pairs_hook)
    except Exception as e:
        return jsonify({'success': False, 'msg': f'Fehler beim Parsen der JSON: {e}'})
        
    if duplicates_keys:
        issues.append(f"Doppelte englische Begriffe (Keys): {', '.join(duplicates_keys)}")
        
    translation_to_keys = {}
    for key, value in parsed.items():
        if isinstance(value, dict):
            trans = value.get("translation", "").strip()
        else:
            trans = str(value).strip()
            
        if trans:
            if trans not in translation_to_keys:
                translation_to_keys[trans] = []
            translation_to_keys[trans].append(key)
            
    dup_trans = {t: ks for t, ks in translation_to_keys.items() if len(ks) > 1}
    for t, ks in dup_trans.items():
        issues.append(f"Doppelte Übersetzung '{t}' verwendet bei: {', '.join(ks)}")
        
    total_entries = len(seen_keys)
    info_msg = f"Das Wörterbuch besitzt {total_entries} Einträge (Schlüsselwörter).\n\n"

    if not issues:
        return jsonify({'success': True, 'msg': info_msg + 'Keine doppelten Einträge gefunden. Alles in Ordnung!'})
    else:
        return jsonify({'success': True, 'msg': info_msg + 'Folgende mögliche Probleme wurden gefunden:\n\n' + '\n'.join(issues)})


def log_reader_thread():
    while True:
        if log_queue_global is not None:
            try:
                # get_nowait can throw queue.Empty from stdlib queue, but log_queue_global is multiprocessing.Queue
                # so we can use queue.Empty (since queue is imported)
                msg = log_queue_global.get(timeout=1.0)
                log_messages.appendleft(msg)
            except queue.Empty:
                pass
        else:
            time.sleep(1)

if __name__ == '__main__':
    mp.freeze_support()
    init_db()
    
    # Initialize shared multiprocessing variables
    log_queue_global = mp.Queue()
    shared_audio_level_global = mp.Value('i', 0)
    shared_queue_length_global = mp.Value('i', 0)
    shared_waterfall_global = mp.Array('f', WATERFALL_HEIGHT * max_bin)
    monitor_running_event = mp.Event()
    
    # Start log reader thread
    t = threading.Thread(target=log_reader_thread, daemon=True)
    t.start()
    
    # Start audio process
    monitor_process = mp.Process(target=run_audio_process, args=(
        monitor_running_event, 
        log_queue_global, 
        shared_audio_level_global, 
        shared_queue_length_global, 
        shared_waterfall_global
    ))
    monitor_process.daemon = True
    monitor_process.start()
    
    # Automatically start monitoring
    monitor_running_event.set()
    
    print(f"Starte Webserver auf http://127.0.0.1:{FLASK_PORT}")
    
    # --- mDNS (Zeroconf) Setup ---
    zc = None
    try:
        from zeroconf import ServiceInfo, Zeroconf
        import socket
        
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            s.connect(('10.255.255.255', 1))
            local_ip = s.getsockname()[0]
        except Exception:
            local_ip = '127.0.0.1'
        finally:
            s.close()
            
        local_settings = load_settings()
        raw_hostname = local_settings.get('device_hostname', 'bird-ai-sound-classifier')
        import re
        safe_hostname = re.sub(r'[^a-zA-Z0-9-]', '-', raw_hostname).strip('-').lower()
        if not safe_hostname:
            safe_hostname = "bird-ai-sound-classifier"

        desc = {'path': '/'}
        info = ServiceInfo(
            "_http._tcp.local.",
            f"Birds AI Sound Classifier - {safe_hostname}._http._tcp.local.",
            addresses=[socket.inet_aton(local_ip)],
            port=FLASK_PORT,
            properties=desc,
            server=f"{safe_hostname}.local."
        )
        
        zc = Zeroconf()
        zc.register_service(info)
        print(f"mDNS Broadcasting gestartet: http://{safe_hostname}.local:{FLASK_PORT}")
    except Exception as e:
        print(f"Fehler beim Starten von mDNS/Zeroconf: {e}")

    try:
        serve(app, host='0.0.0.0', port=FLASK_PORT)
    except KeyboardInterrupt:
        if zc is not None:
            zc.unregister_service(info)
            zc.close()
        monitor_running_event.clear()
        monitor_process.join(timeout=2)
        print("Server beendet.")
