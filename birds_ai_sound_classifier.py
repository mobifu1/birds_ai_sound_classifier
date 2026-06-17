import os
import time
import threading
import sqlite3
import datetime
import pyaudio
import wave
import json
import logging
import traceback
from collections import deque
import numpy as np
import pandas as pd
import queue
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import base64

from flask import Flask, render_template, jsonify, request
from waitress import serve

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
FLASK_PORT = 5001
RECORD_SECONDS = 3
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 48000 # BirdNET standard is 48kHz
MIN_CONFIDENCE = 0.3 # Konfidenz-Schwellenwert
AUDIO_DIR = "audio_records"

os.makedirs(AUDIO_DIR, exist_ok=True)
TEMP_WAV = os.path.join(AUDIO_DIR, "temp.wav")

app = Flask(__name__)
log_messages = deque(maxlen=100)
latest_audio_level = 0


BIRD_TRANSLATIONS = {
    # Meisen & Baumläufer
    "Great Tit": "Kohlmeise", "Eurasian Blue Tit": "Blaumeise", "Coal Tit": "Tannenmeise", 
    "Crested Tit": "Haubenmeise", "Marsh Tit": "Sumpfmeise", "Willow Tit": "Weidenmeise", 
    "Long-tailed Tit": "Schwanzmeise", "Eurasian Nuthatch": "Kleiber", 
    "Eurasian Treecreeper": "Waldbaumläufer", "Short-toed Treecreeper": "Gartenbaumläufer",
    
    # Finken & Sperlinge
    "House Sparrow": "Haussperling", "Eurasian Tree Sparrow": "Feldsperling",
    "Common Chaffinch": "Buchfink", "European Greenfinch": "Grünfink", "European Goldfinch": "Stieglitz",
    "Eurasian Siskin": "Erlenzeisig", "Common Linnet": "Bluthänfling", "Eurasian Linnet": "Bluthänfling", "Eurasian Bullfinch": "Gimpel (Dompfaff)",
    "Hawfinch": "Kernbeißer", "Yellowhammer": "Goldammer", "Common Reed Bunting": "Rohrammer",
    
    # Drosseln, Grasmücken & Fliegenschnäpper
    "Eurasian Blackbird": "Amsel", "Song Thrush": "Singdrossel", "Mistle Thrush": "Misteldrossel",
    "Fieldfare": "Wacholderdrossel", "Redwing": "Rotdrossel", "Ring Ouzel": "Ringdrossel",
    "European Robin": "Rotkehlchen", "Common Nightingale": "Nachtigall", 
    "Black Redstart": "Hausrotschwanz", "Common Redstart": "Gartenrotschwanz",
    "Eurasian Blackcap": "Mönchsgrasmücke", "Garden Warbler": "Gartengrasmücke", 
    "Common Whitethroat": "Dorngrasmücke", "Lesser Whitethroat": "Klappergrasmücke",
    "Common Chiffchaff": "Zilpzalp", "Willow Warbler": "Fitis", "Wood Warbler": "Waldlaubsänger",
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
    "Tawny Owl": "Waldkauz", "Barn Owl": "Schleiereule", "Little Owl": "Steinkauz",
    
    # Wasservögel & Reiher
    "Gray Heron": "Graureiher", "Grey Heron": "Graureiher", "Great Egret": "Silberreiher",
    "White Stork": "Weißstorch", "Black Stork": "Schwarzstorch",
    "Mute Swan": "Höckerschwan", "Greylag Goose": "Graugans", "Graylag Goose": "Graugans", "Canada Goose": "Kanadagans",
    "Mallard": "Stockente", "Eurasian Teal": "Krickente", "Tufted Duck": "Reiherente",
    "Great Crested Grebe": "Haubentaucher", "Little Grebe": "Zwergtaucher", "Great Cormorant": "Kormoran",
    "Water Rail": "Wasserralle", "Common Moorhen": "Teichhuhn", "Eurasian Coot": "Blässhuhn",
    "Black-headed Gull": "Lachmöwe", "Common Kingfisher": "Eisvogel",
    
    # Stelzen & Pieper
    "White Wagtail": "Bachstelze", "Grey Wagtail": "Gebirgsstelze", "Gray Wagtail": "Gebirgsstelze", "Western Yellow Wagtail": "Schafstelze",
    "Tree Pipit": "Baumpieper", "Meadow Pipit": "Wiesenpieper", "Water Pipit": "Bergpieper",
    "Eurasian Hoopoe": "Wiedehopf", "Common Crane": "Kranich", "Eurasian Golden Oriole": "Pirol", "Common Raven": "Kolkrabe",
    
    # Lerchen
    "Eurasian Skylark": "Feldlerche",
    
    # Würger
    "Red-backed Shrike": "Neuntöter"
}

# Analyzer laden (lädt Modelle beim ersten Start herunter falls nicht vorhanden)
print("Initialisiere BirdNET Analyzer...")
analyzer = Analyzer()
print("OK: BirdNET Analyzer bereit.")

def update_log(msg):
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    formatted = f"[{ts}] {msg}"
    print(formatted)
    log_messages.appendleft(formatted)

def load_settings():
    if os.path.exists(SETTINGS_FILE):
        try:
            with open(SETTINGS_FILE, 'r') as f:
                return json.load(f)
        except:
            pass
    return {}

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
    conn.commit()
    conn.close()

def save_detection(species, confidence, snr=0.0):
    try:
        conn = sqlite3.connect(DB_FILE)
        c = conn.cursor()
        ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        c.execute("INSERT INTO detections (species, timestamp, confidence, snr) VALUES (?, ?, ?, ?)",
                  (species, ts, confidence, snr))
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
        self.audio_queue = queue.Queue(maxsize=3) # Maximal 3 Pakete Rückstand

    def start(self):
        self.running = True
        self.record_thread = threading.Thread(target=self.loop_record, daemon=True)
        self.analyze_thread = threading.Thread(target=self.loop_analyze, daemon=True)
        self.record_thread.start()
        self.analyze_thread.start()
        update_log("Audio-Ueberwachung (Multi-Thread) gestartet.")

    def stop(self):
        self.running = False
        update_log("Audio-Ueberwachung gestoppt.")

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
                frames = []
                chunks_needed = int(RATE / CHUNK * RECORD_SECONDS)
                
                for i in range(chunks_needed):
                    if not self.running:
                        break
                    data = stream.read(CHUNK, exception_on_overflow=False)
                    frames.append(data)
                    
                    try:
                        chunk_amp = int(np.max(np.abs(np.frombuffer(data, dtype=np.int16))))
                        global latest_audio_level
                        latest_audio_level = chunk_amp
                    except:
                        pass

                if self.running and len(frames) == chunks_needed:
                    raw_data = b''.join(frames)
                    # Blockiert nicht ewig, wenn Queue voll ist (verwirft im Zweifel alte Daten)
                    if self.audio_queue.full():
                        try:
                            self.audio_queue.get_nowait()
                        except:
                            pass
                    self.audio_queue.put(raw_data)

            stream.stop_stream()
            stream.close()
        except Exception as e:
            update_log(f"Fehler im Aufnahme-Thread: {e}")

    def loop_analyze(self):
        while self.running:
            try:
                # Wartet auf neues Audio-Paket (max 1 Sekunde, um while-Bedingung regelmäßig zu prüfen)
                try:
                    raw_data = self.audio_queue.get(timeout=1.0)
                except queue.Empty:
                    continue
                
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
                lat = float(settings.get("gps_lat", 51.165691))
                lon = float(settings.get("gps_lon", 10.451526))
                
                recording = Recording(
                    analyzer,
                    TEMP_WAV,
                    lat=lat,
                    lon=lon,
                    date=datetime.datetime.now(),
                    min_conf=MIN_CONFIDENCE
                )
                recording.analyze()
                
                # Ergebnisse verarbeiten
                if recording.detections:
                    for det in recording.detections:
                        eng_spec = det['common_name']
                        conf = float(det['confidence'])
                        spec_de = BIRD_TRANSLATIONS.get(eng_spec, eng_spec)
                        print(f"[KI] {spec_de}: {conf:.0%}")
                    
                    best = recording.detections[0] # höchste Konfidenz
                    eng_species = best['common_name']
                    species = BIRD_TRANSLATIONS.get(eng_species, eng_species)
                    confidence = float(best['confidence'])
                    
                    min_conf = float(settings.get("threshold", MIN_CONFIDENCE * 100)) / 100.0
                    
                    if confidence >= min_conf:
                        update_log(f"Erkannt: {species} ({confidence:.0%}) | SNR: {calculated_snr:.1f}dB")
                        save_detection(species, confidence, calculated_snr)
                else:
                    print("[KI] Nichts erkannt.")
                
                self.audio_queue.task_done()

            except Exception as e:
                update_log(f"Fehler in Analyse-Schleife: {e}")
                time.sleep(1)


# --- FLASK ROUTEN ---
@app.route('/')
def index():
    s = load_settings()
    icon_dir = os.path.join(app.root_path, 'static', 'bird_icons')
    available_icons = [f for f in os.listdir(icon_dir) if os.path.isfile(os.path.join(icon_dir, f))] if os.path.exists(icon_dir) else []
    return render_template('index.html', version="1.0", s=s, available_icons=available_icons)

@app.route('/settings')
def settings_page():
    s = load_settings()
    
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

    return render_template('settings.html', s=s, mics=mics)

def create_chart(title, labels, values):
    plt.figure(figsize=(10, 6), facecolor='#1e1e1e')
    ax = plt.axes()
    ax.set_facecolor('#1e1e1e')
    ax.tick_params(colors='white')
    for spine in ax.spines.values():
        spine.set_color('#444')
    
    color_palette = ['#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231', '#911eb4', '#46f0f0', '#f032e6', '#bcf60c', '#fabebe', '#008080', '#e6beff', '#9a6324', '#fffac8', '#800000', '#aaffc3', '#808000', '#ffd8b1', '#000075', '#808080', '#ffffff', '#000000']
    bar_colors = [color_palette[i % len(color_palette)] for i in range(len(labels))]
    bars = plt.bar(labels, values, color=bar_colors)
    plt.title(title, color='white')
    plt.xticks(rotation=45, ha='right')
    plt.yscale('symlog')
    plt.ylim(bottom=0)
    plt.yticks(color='white')
    plt.tight_layout()
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
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

    return render_template('daily.html', 
        selected_date_str=date_str, total_birds_day=total, 
        first_bird=first_bird, first_bird_time=first_bird_time,
        prev_date=prev_date, next_date=next_date,
        is_today=(dt == today), today_str=today.strftime('%Y-%m-%d'),
        table_content=generate_daily_heatmap_html(date_str)
    )

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
        
        special_cases = {
            "Rabenkrähe": "Rabe.png",
            "Nebelkrähe": "Rabe.png",
            "Aaskrähe": "Rabe.png",
            "Dohle": "Rabe.png",
            "Turmfalke": "Falke.png",
            "Mäusebussard": "Falke.png",
            "Rotmilan": "Milan.png",
            "Graugans": "Gans.png",
            "Kanadagans": "Gans.png",
            "Stockente": "Ente.png",
            "Waldkauz": "Uhu.png",
            "Schleiereule": "Uhu.png",
            "Rauchschwalbe": "Schwalbe.png",
            "Mehlschwalbe": "Schwalbe.png",
            "Mauersegler": "Mauersegler.png"
        }
        if sp in special_cases and special_cases[sp].lower() in icon_map:
            return icon_map[special_cases[sp].lower()]
            
        clean = sp.replace('ä', 'ae').replace('ö', 'oe').replace('ü', 'ue')\
                  .replace('ß', 'ss').replace('Ä', 'Ae').replace('Ö', 'Oe').replace('Ü', 'Ue')
        clean = clean.split(' (')[0].lower()
        
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
        
        html_table = '<div class="table-responsive" style="margin-top:20px;"><table class="weekly-table">'
        html_table += '<thead><tr><th style="text-align:left;">Vogelart</th>'
        for col in pivot_pct.columns:
            total_in_hour = int(hour_totals[col])
            # Kürzeres Format für die Uhrzeit z.B. nur '14' statt '14:00' um Platz zu sparen, aber '14:00' ist auch ok
            html_table += f'<th>{col[:2]}h<br><small style="color:#81d4fa;">(∑ {total_in_hour})</small></th>'
        html_table += '</tr></thead><tbody>'
        
        from flask import url_for
        for species, row in pivot_pct.iterrows():
            img_src = url_for('static', filename=get_bird_icon(species))
            if 'Unbekannt.png' in img_src:
                img_tag = '<div class="bird-icon-placeholder">?</div>'
            else:
                img_tag = f'<img src="{img_src}" class="bird-icon-small">'

            html_table += f'<tr><td style="text-align:left; font-weight:bold;"><div class="species-wrapper">{img_tag}<span>{species}</span></div></td>'
            
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

def generate_weekly_heatmap_html():
    query = """
    SELECT 
        CASE WHEN species = 'IGNORED_LOW_CONFIDENCE' THEN 'Unbekannt' ELSE species END as species,
        strftime('%Y-', timestamp) || printf('%02d', CAST(strftime('%W', timestamp) AS INTEGER) + 1) as week_sort,
        printf('%02d', CAST(strftime('%W', timestamp) AS INTEGER) + 1) || '<br><small style=''color:#aaa''>''' || substr(strftime('%Y', timestamp), 3, 2) || '</small>' as week_display,
        COUNT(*) as counts
    FROM detections
    WHERE timestamp IS NOT NULL AND timestamp != ''
    GROUP BY species, week_sort, week_display
    ORDER BY week_sort
    """
    
    conn = sqlite3.connect(DB_FILE, timeout=10)
    try:
        grouped = pd.read_sql_query(query, conn)
    except:
        grouped = pd.DataFrame()
    finally:
        conn.close()

    html_table = "<p>Keine Daten für die Wochenansicht.</p>"

    icon_map = {}
    static_folder = os.path.join(app.root_path, 'static', 'bird_icons')
    if os.path.exists(static_folder):
        for f in os.listdir(static_folder):
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif')):
                icon_map[os.path.splitext(f)[0].lower()] = f"bird_icons/{f}"
                icon_map[f.lower()] = f"bird_icons/{f}"

    def get_bird_icon(sp):
        if not sp: return 'bird_icons/Unbekannt.png'
        
        special_cases = {
            "Rabenkrähe": "Rabe.png",
            "Nebelkrähe": "Rabe.png",
            "Aaskrähe": "Rabe.png",
            "Dohle": "Rabe.png",
            "Turmfalke": "Falke.png",
            "Mäusebussard": "Falke.png",
            "Rotmilan": "Milan.png",
            "Graugans": "Gans.png",
            "Kanadagans": "Gans.png",
            "Stockente": "Ente.png",
            "Waldkauz": "Uhu.png",
            "Schleiereule": "Uhu.png",
            "Rauchschwalbe": "Schwalbe.png",
            "Mehlschwalbe": "Schwalbe.png",
            "Mauersegler": "Mauersegler.png"
        }
        if sp in special_cases and special_cases[sp].lower() in icon_map:
            return icon_map[special_cases[sp].lower()]
            
        clean = sp.replace('ä', 'ae').replace('ö', 'oe').replace('ü', 'ue')\
                  .replace('ß', 'ss').replace('Ä', 'Ae').replace('Ö', 'Oe').replace('Ü', 'Ue')
        clean = clean.split(' (')[0].lower()
        
        if clean in icon_map:
            return icon_map[clean]
        if clean + '.png' in icon_map:
            return icon_map[clean + '.png']
            
        return 'bird_icons/Unbekannt.png'

    if not grouped.empty:
        pivot_counts = grouped.pivot(index='species', columns='week_display', values='counts').fillna(0)
        week_totals = pivot_counts.sum(axis=0)
        pivot_pct = pivot_counts.div(week_totals, axis=1).mul(100).fillna(0)
        
        week_mapping = grouped[['week_sort', 'week_display']].drop_duplicates().sort_values('week_sort')
        sorted_columns = week_mapping['week_display'].tolist()
        pivot_pct = pivot_pct.reindex(columns=sorted_columns)
        
        total_counts = pivot_counts.sum(axis=1)
        pivot_pct['total_sort_idx'] = total_counts
        pivot_pct = pivot_pct.sort_values('total_sort_idx', ascending=False)
        pivot_pct = pivot_pct.drop('total_sort_idx', axis=1)

        html_table = '<div class="table-responsive" style="margin-top:30px;"><h2 style="color:#4CAF50;">Ganzjahres-Heatmap</h2><table class="weekly-table">'
        html_table += '<thead><tr><th style="text-align:left;">Vogelart</th>'
        for col in pivot_pct.columns:
            total_in_week = int(week_totals[col])
            html_table += f'<th>{col}<br><small style="color:#81d4fa;">(∑ {total_in_week})</small></th>'
        html_table += '</tr></thead><tbody>'
        
        from flask import url_for
        for species, row in pivot_pct.iterrows():
            img_src = url_for('static', filename=get_bird_icon(species))
            if 'Unbekannt.png' in img_src:
                img_tag = '<div class="bird-icon-placeholder">?</div>'
            else:
                img_tag = f'<img src="{img_src}" class="bird-icon-small">'

            html_table += f'<tr><td style="text-align:left; font-weight:bold;"><div class="species-wrapper">{img_tag}<span>{species}</span></div></td>'
            
            for col_name, val in row.items():
                absolute_count = int(pivot_counts.at[species, col_name])
                total_in_week = int(week_totals[col_name])
                
                style = 'background-color: transparent;'
                if val > 0:
                    alpha = 0.15 + (val / 50.0) * 0.85 
                    alpha = min(alpha, 1.0) 
                    style = f'background-color: rgba(76, 175, 80, {alpha});'
                
                if total_in_week > 0:
                    tooltip = f"{val:.1f}% ({absolute_count} von {total_in_week} Vögeln)"
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

@app.route('/weekly')
def weekly_page():
    return render_template('weekly.html', 
        table_content=generate_weekly_heatmap_html()
    )

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
    conn.close()
    
    total = sum([r[1] for r in rows])
    chart_url = None
    if rows:
        labels = [r[0] for r in rows]
        values = [r[1] for r in rows]
        chart_url = create_chart(f"Vögel Jahr {year_str}", labels, values)
        
    return render_template('yearly.html', 
        chart_url=chart_url, selected_year=year, total_birds_year=total,
        prev_year=year-1, next_year=year+1,
        is_current_year=(year == today.year), current_year=today.year
    )

@app.route('/manual_entry')
def manual_entry_page():
    return render_template('manual_entry.html', species_list=list(BIRD_TRANSLATIONS.values()))

@app.route('/delete_entry')
def delete_entry_page():
    return render_template('delete_entry.html', species_list=list(BIRD_TRANSLATIONS.values()))

@app.route('/api/settings/save', methods=['POST'])
def api_save_settings():
    data = request.json
    save_setting("threshold", data.get("threshold", 30))
    save_setting("gps_lat", data.get("gps_lat", 51.165691))
    save_setting("gps_lon", data.get("gps_lon", 10.451526))
    save_setting("radar_zoom", data.get("radar_zoom", 1.0))
    save_setting("radar_max_birds", data.get("radar_max_birds", 10))
    save_setting("radar_snr_max", data.get("radar_snr_max", 20.0))
    save_setting("radar_snr_min", data.get("radar_snr_min", 5.0))
    if "mic_index" in data:
        save_setting("mic_index", data.get("mic_index", -1))
    return jsonify({"msg": "Einstellungen gespeichert!"})

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
        "status": "Online (Mikrofon aktiv)" if monitor.running else "Offline (Gestoppt)",
        "total_detections": total_count,
        "today_detections": today_count
    })

@app.route('/api/audio_level')
def api_audio_level():
    return jsonify({"level": latest_audio_level})

@app.route('/api/latest_logs')
def api_latest_logs():
    return jsonify(list(log_messages))

# --- CONTROL ROUTEN ---
@app.route('/api/control/start', methods=['POST'])
def api_control_start():
    if not monitor.running:
        monitor.start()
        return jsonify({"msg": "Gestartet"})
    return jsonify({"error": "Läuft bereits"})

@app.route('/api/control/stop', methods=['POST'])
def api_control_stop():
    if monitor.running:
        monitor.stop()
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
        return jsonify({"msg": "Datenbank wurde erfolgreich geleert!"})
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
    
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    query = f"""
        SELECT d1.species, COUNT(*) as count,
               (SELECT snr FROM detections d2 WHERE d2.species = d1.species ORDER BY timestamp DESC LIMIT 1) as snr
        FROM detections d1
        WHERE date(timestamp) = date('now', 'localtime')
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
            
        top_data.append({"species": r[0], "count": r[1], "snr": float(snr_val)})

    
    c.execute("SELECT species FROM detections ORDER BY timestamp DESC LIMIT 1")
    last = c.fetchone()
    latest_species = last[0] if last else None
    
    c.execute("SELECT COUNT(DISTINCT species) FROM detections WHERE date(timestamp) = date('now', 'localtime')")
    unique_count = c.fetchone()[0]
    
    conn.close()
    return jsonify({"top": top_data, "latest": latest_species, "unique_species_count": unique_count})

# --- DATENBANK MANAGEMENT ROUTEN ---
@app.route('/api/detections/by_date')
def api_detections_by_date():
    date_str = request.args.get('date', '')
    if not date_str: return jsonify({"success": False, "error": "No date provided"})
    try:
        conn = sqlite3.connect(DB_FILE)
        c = conn.cursor()
        c.execute("SELECT id, timestamp, species, confidence FROM detections WHERE timestamp LIKE ? ORDER BY timestamp DESC", (f"{date_str}%",))
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
        c.execute("DELETE FROM detections WHERE id = ?", (data['id'],))
        conn.commit()
        conn.close()
        return jsonify({"success": True, "msg": "Eintrag gelöscht."})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

@app.route('/api/detections/update', methods=['POST'])
def api_detections_update():
    data = request.json
    try:
        conn = sqlite3.connect(DB_FILE)
        c = conn.cursor()
        c.execute("UPDATE detections SET species = ? WHERE id = ?", (data['species'], data['id']))
        conn.commit()
        conn.close()
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

if __name__ == '__main__':
    init_db()
    monitor = AudioMonitor()
    monitor.start()
    
    print(f"Starte Webserver auf http://127.0.0.1:{FLASK_PORT}")
    try:
        serve(app, host='0.0.0.0', port=FLASK_PORT)
    except KeyboardInterrupt:
        monitor.stop()
        print("Server beendet.")
