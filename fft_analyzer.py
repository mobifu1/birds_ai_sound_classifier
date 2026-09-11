"""
=============================================================
  Bird Detection FFT Analyzer
  Standalone-Tool - liest die DB nur lesend.
=============================================================
"""
import sqlite3
import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox
from pathlib import Path
import datetime

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure

DB_PATH = Path(__file__).parent / "birds_audio_stats.db"

BG       = "#1e1e2e"
PANEL    = "#2a2a3e"
ACCENT   = "#7c9ded"
ACCENT2  = "#f38ba8"
TEXT     = "#cdd6f4"
SUBTEXT  = "#a6adc8"
YELLOW   = "#f9e2af"
FONT     = ("Segoe UI", 10)

plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": PANEL,
    "axes.edgecolor": SUBTEXT, "axes.labelcolor": TEXT,
    "xtick.color": SUBTEXT, "ytick.color": SUBTEXT,
    "text.color": TEXT, "grid.color": "#3a3a5e",
    "grid.linestyle": "--", "grid.alpha": 0.5,
    "legend.facecolor": PANEL, "legend.edgecolor": SUBTEXT,
})

BIN_OPTIONS = {
    "1 Stunde": 60, "2 Stunden": 120, "6 Stunden": 360,
    "12 Stunden": 720, "1 Tag": 1440,
}

PERIOD_MARKERS = {
    "12h": 1/12, "24h": 1/24, "48h": 1/48,
    "7d": 1/(7*24), "14d": 1/(14*24), "30d": 1/(30*24),
}


def get_species_list():
    with sqlite3.connect(DB_PATH) as conn:
        rows = conn.execute(
            "SELECT species, COUNT(*) as n FROM detections GROUP BY species ORDER BY n DESC"
        ).fetchall()
    return [f"{r[0]}  ({r[1]:,} Eintr.)" for r in rows], [r[0] for r in rows]


def load_detections(species):
    with sqlite3.connect(DB_PATH) as conn:
        rows = conn.execute(
            "SELECT timestamp, confidence, snr FROM detections WHERE species=? ORDER BY timestamp",
            (species,)
        ).fetchall()
    return rows


def build_timeseries(rows, bin_min, signal):
    if not rows:
        return [], np.array([]), bin_min
    fmt = "%Y-%m-%d %H:%M:%S"
    parsed = []
    for ts, conf, snr in rows:
        try:
            dt = datetime.datetime.strptime(ts, fmt)
            try:
                c_val = float(conf) if conf is not None else 0.0
            except (ValueError, TypeError):
                c_val = 0.0
            try:
                s_val = float(snr) if snr is not None else 0.0
            except (ValueError, TypeError):
                s_val = 0.0
            parsed.append((dt, c_val, s_val))
        except ValueError:
            pass
    if not parsed:
        return [], np.array([]), bin_min
    t0, t1 = parsed[0][0], parsed[-1][0]
    n_bins = int((t1 - t0).total_seconds() / 60) // bin_min + 2
    cnt = np.zeros(n_bins); cs = np.zeros(n_bins); ss = np.zeros(n_bins)
    for dt, c, s in parsed:
        i = min(int((dt - t0).total_seconds() / 60) // bin_min, n_bins - 1)
        cnt[i] += 1; cs[i] += c; ss[i] += s
    with np.errstate(invalid="ignore", divide="ignore"):
        mc = np.where(cnt > 0, cs / cnt, 0.0)
        ms = np.where(cnt > 0, ss / cnt, 0.0)
    sig = {"count": cnt, "confidence": mc, "snr": ms, "weighted": cnt * mc}.get(signal, cnt)
    times = [t0 + datetime.timedelta(minutes=i * bin_min) for i in range(n_bins)]
    return times, sig, bin_min


def compute_fft(sig, bin_min, window_key):
    N = len(sig)
    if N < 4:
        return np.array([]), np.array([])
    sc = sig - np.mean(sig)
    wmap = {"hann": np.hanning, "hamm": np.hamming, "black": np.blackman}
    w = wmap[window_key](N) if window_key in wmap else np.ones(N)
    fft_v = np.fft.rfft(sc * w)
    freqs = np.fft.rfftfreq(N, d=bin_min / 60.0)
    power = (np.abs(fft_v) ** 2) / N
    return freqs[1:], power[1:]


def period_fmt(x, _):
    if x <= 0: return ""
    h = 1.0 / x
    if h >= 720: return f"{h/24:.0f}d"
    if h >= 24:  return f"{h/24:.1f}d"
    if h >= 1:   return f"{h:.0f}h"
    return f"{h*60:.0f}m"


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Bird FFT Spectrum Analyzer")
        self.configure(bg=BG)
        self.geometry("1220x800")
        self.minsize(900, 600)
        self.display, self.species = get_species_list()
        self._build()

    def _lbl(self, p, t, fg=TEXT, font=FONT, **kw):
        return tk.Label(p, text=t, bg=p.cget("bg"), fg=fg, font=font, **kw)

    def _build(self):
        sb = tk.Frame(self, bg=PANEL, padx=14, pady=14)
        sb.pack(side=tk.LEFT, fill=tk.Y, padx=(10,0), pady=10)

        self._lbl(sb, "Bird FFT Analyzer", fg=ACCENT,
                  font=("Segoe UI Semibold", 13)).pack(anchor="w", pady=(0,2))
        self._lbl(sb, "Frequenzspektrum aus Detektionen",
                  fg=SUBTEXT, font=("Segoe UI", 9)).pack(anchor="w", pady=(0,10))
        ttk.Separator(sb).pack(fill="x", pady=5)

        def row(parent, label, var, values, default=None):
            self._lbl(parent, label, fg=SUBTEXT).pack(anchor="w", pady=(6,1))
            cb = ttk.Combobox(parent, textvariable=var, values=values, state="readonly", width=29)
            cb.pack(anchor="w", pady=(0,8))
            if default is not None:
                cb.current(default)
            return cb

        self.sp_var  = tk.StringVar()
        self.bin_var = tk.StringVar(value="1 Stunde")

        self.sp_cb = row(sb, "Vogelart:", self.sp_var, self.display, 0)
        row(sb, "Zeitauflösung:", self.bin_var, list(BIN_OPTIONS.keys()))

        ttk.Separator(sb).pack(fill="x", pady=5)
        self._lbl(sb, "Optionen:", fg=SUBTEXT).pack(anchor="w", pady=(4,2))

        def chk(parent, text, var):
            tk.Checkbutton(parent, text=text, variable=var, bg=PANEL, fg=TEXT,
                           selectcolor=BG, activebackground=PANEL,
                           activeforeground=TEXT, font=FONT).pack(anchor="w")

        self.raw_v  = tk.BooleanVar(value=True)
        chk(sb, "Zeitreihe anzeigen", self.raw_v)


        # Schwellwert-Regler
        ttk.Separator(sb).pack(fill="x", pady=(10, 5))
        thresh_frame = tk.Frame(sb, bg=PANEL)
        thresh_frame.pack(fill="x", pady=(4, 2))
        self._lbl(thresh_frame, "Labelschwellwert:", fg=SUBTEXT).pack(anchor="w")
        ctrl_frame = tk.Frame(thresh_frame, bg=PANEL)
        ctrl_frame.pack(fill="x", pady=(2, 0))
        self.thresh_v = tk.DoubleVar(value=15.0)
        thresh_spin = tk.Spinbox(
            ctrl_frame, from_=0.5, to=50.0, increment=0.5,
            textvariable=self.thresh_v, width=6, format="%.1f",
            bg=BG, fg=TEXT, buttonbackground=PANEL,
            insertbackground=TEXT, relief="flat", font=FONT,
        )
        thresh_spin.pack(side=tk.LEFT)
        self._lbl(ctrl_frame, "% des stärksten Peaks", fg=SUBTEXT,
                  font=("Segoe UI", 9)).pack(side=tk.LEFT, padx=(5, 0))
        ttk.Separator(sb).pack(fill="x", pady=(10, 10))

        tk.Button(sb, text="  Analyse starten", bg=ACCENT, fg=BG,
                  font=("Segoe UI Semibold", 11), relief="flat",
                  cursor="hand2", padx=10, pady=7,
                  activebackground="#5a7dcc", activeforeground=BG,
                  command=self._run).pack(fill="x", pady=(0,4))

        self.info = tk.Label(sb, text="", bg=PANEL, fg=SUBTEXT,
                             font=("Segoe UI", 9), justify="left", wraplength=215)
        self.info.pack(anchor="w", pady=(10,0))

        pf = tk.Frame(self, bg=BG)
        pf.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)
        self.fig = Figure(figsize=(9,6), dpi=100, facecolor=BG)
        self.canvas = FigureCanvasTkAgg(self.fig, master=pf)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        tf = tk.Frame(pf, bg=BG)
        tf.pack(fill=tk.X)
        NavigationToolbar2Tk(self.canvas, tf).update()
        self._welcome()

    def _welcome(self):
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        ax.text(0.5, 0.55, "FFT", fontsize=80, ha="center", va="center",
                transform=ax.transAxes, color=ACCENT, alpha=0.25, fontweight="bold")
        ax.text(0.5, 0.38, "Vogelart waehlen und  Analyse starten",
                fontsize=13, ha="center", va="center",
                color=SUBTEXT, transform=ax.transAxes)
        ax.set_axis_off()
        self.canvas.draw()

    def _run(self):
        idx = self.sp_cb.current()
        if idx < 0:
            messagebox.showwarning("", "Bitte eine Vogelart auswaehlen.")
            return
        sp  = self.species[idx]
        bm  = BIN_OPTIONS[self.bin_var.get()]
        sk  = "count"
        wk  = "hann"
        rows = load_detections(sp)
        if not rows:
            messagebox.showinfo("", f"Keine Eintraege fuer '{sp}'."); return
        times, sig, _ = build_timeseries(rows, bm, sk)
        if len(sig) < 8:
            messagebox.showwarning("", "Zu wenige Datenpunkte."); return
        freqs, power = compute_fft(sig, bm, wk)
        t0s = times[0].strftime("%d.%m.%Y")
        t1s = times[-1].strftime("%d.%m.%Y")
        nd  = (times[-1] - times[0]).days
        self.info.config(text=(
            f"Art: {sp}\nDetektionen: {len(rows):,}\n"
            f"Zeitraum: {t0s} - {t1s}\nDauer: {nd} Tage\n"
            f"Bins: {len(sig):,}  |  FFT-Punkte: {len(freqs):,}"
        ))
        self._draw(sp, times, sig, freqs, power, bm)

    def _draw(self, sp, times, sig, freqs, power, bm):
        self.fig.clear()
        show_raw = self.raw_v.get()
        if show_raw:
            gs  = self.fig.add_gridspec(2,1,hspace=0.42,top=0.91,bottom=0.08,
                                         left=0.12,right=0.97,height_ratios=[1,2])
            a0  = self.fig.add_subplot(gs[0])
            a1  = self.fig.add_subplot(gs[1])
            a0.plot(times, sig, color=ACCENT, lw=0.8, alpha=0.9)
            a0.fill_between(times, sig, alpha=0.18, color=ACCENT)
            a0.set_ylabel("Detektionen / Bin", fontsize=9, color=SUBTEXT)
            a0.set_title("Zeitreihe (Eingangssignal)", fontsize=10, color=TEXT, pad=4)
            a0.grid(True)
            a0.tick_params(axis="x", labelrotation=20, labelsize=8)
        else:
            gs = self.fig.add_gridspec(1,1,top=0.91,bottom=0.10,left=0.12,right=0.97)
            a1 = self.fig.add_subplot(gs[0])


        # Spektrum
        a1.plot(freqs, power, color=ACCENT, lw=1.0, zorder=3)
        a1.fill_between(freqs, power, alpha=0.22, color=ACCENT, zorder=2)


        # Alle signifikanten Peaks automatisch erkennen und labeln
        if len(power) > 0:
            # 1) Lokale Maxima finden (jeder Bin der groesser als beide Nachbarn ist)
            local_max = np.where(
                (power[1:-1] > power[:-2]) & (power[1:-1] > power[2:])
            )[0] + 1  # Index-Korrektur wegen Slicing

            # 2) Mindestabstand: benachbarte Bins desselben physischen Peaks zusammenfassen
            min_dist = max(3, len(freqs) // 80)
            candidates = sorted(local_max, key=lambda i: power[i], reverse=True)
            selected = []
            for pi in candidates:
                if all(abs(pi - s) >= min_dist for s in selected):
                    selected.append(pi)

            # 3) Nur Peaks ab dem eingestellten Schwellwert behalten
            if selected:
                pct = max(0.1, self.thresh_v.get()) / 100.0
                threshold = power[selected[0]] * pct
                selected = [pi for pi in selected if power[pi] >= threshold]
                selected = selected[:6]  # Nur die 6 stärksten behalten


            ylims = a1.get_ylim()
            y_range = ylims[1] - ylims[0]
            offs = y_range * 0.10

            # Gesamtpower aller sichtbaren Peaks → Basis für Prozentangabe
            total_peak_power = sum(power[pi] for pi in selected) if selected else 1.0

            for pi in selected:
                fp, pp = freqs[pi], power[pi]
                if fp <= 0:
                    continue
                ph = 1.0 / fp
                period_str = (f"{ph/24:.1f}d" if ph >= 24 else
                              f"{ph:.1f}h"    if ph >= 1  else
                              f"{ph*60:.0f}m")
                share = pp / total_peak_power * 100.0
                pl = f"{period_str}\n{share:.1f}%"

                # Label nach oben, nach unten spiegeln wenn nahe am oberen Rand
                if (pp + offs) > ylims[1] - y_range * 0.05:
                    y_text  = pp - offs
                    va_text = "top"
                else:
                    y_text  = pp + offs
                    va_text = "bottom"

                a1.plot(fp, pp, "o", color=YELLOW, ms=5, zorder=6)
                a1.annotate(
                    pl,
                    xy=(fp, pp),
                    xytext=(fp, y_text),
                    color=YELLOW, fontsize=8, ha="center", va=va_text,
                    fontweight="bold",
                    arrowprops=dict(arrowstyle="->", color=YELLOW, lw=0.8),
                    annotation_clip=False,
                    zorder=7,
                )

        a1.xaxis.set_major_formatter(ticker.FuncFormatter(period_fmt))
        a1.set_xlabel("Periode  (d=Tage, h=Stunden, m=Minuten)", fontsize=10, color=TEXT)
        a1.set_ylabel("Power  |X(f)|^2", fontsize=10, color=TEXT)
        a1.set_title(
            f"Frequenzspektrum  --  {sp}  |  Zeitauflösung: {self.bin_var.get()}  |  Fenster: Hanning",
            fontsize=11, color=TEXT, pad=10, fontweight="bold"
        )
        x_pad = (freqs.max() - freqs.min()) * 0.05
        a1.set_xlim(freqs.min() - x_pad, freqs.max() + x_pad)
        a1.grid(True, which="major")
        a1.text(0.99, 0.97, f"Signal: Detektions-Anzahl\nZeitauflösung: {bm} min",
                transform=a1.transAxes, fontsize=8, color=SUBTEXT, ha="right", va="top")
        self.canvas.draw()


def _style():
    s = ttk.Style()
    s.theme_use("clam")
    s.configure("TCombobox", fieldbackground=BG, background=PANEL, foreground=TEXT,
                 selectbackground=ACCENT, selectforeground=BG, bordercolor=SUBTEXT)
    s.configure("TSeparator", background=SUBTEXT)
    s.map("TCombobox", fieldbackground=[("readonly", BG)])


if __name__ == "__main__":
    if not DB_PATH.exists():
        import sys; print(f"FEHLER: DB nicht gefunden: {DB_PATH}"); sys.exit(1)
    app = App()
    _style()
    app.mainloop()