import sys
from fpdf import FPDF

pdf = FPDF()
pdf.add_page()
pdf.set_font("Arial", size=12)

# title
pdf.set_font("Arial", 'B', 16)
pdf.cell(200, 10, txt="Einstellungen - Beschreibung", ln=1, align='C')
pdf.ln(10)

pdf.set_font("Arial", size=12)

lines = [
    "Erkennungs-Einstellungen:",
    "- Mindest-Konfidenz (Threshold %): Prozentsatz der Sicherheit, ab der ein Vogel gespeichert wird",
    "- Mindest-SNR (dB): Signal-Rausch-Verhaeltnis, ab dem ein Vogel gespeichert wird",
    "- GPS Breitengrad / Laengengrad: Koordinaten fuer lokale Vogelarten-Filterung",
    "",
    "Radar-Einstellungen:",
    "- Radar Zoom-Faktor: Groesse und Sichtbarkeit der Icons im Radar",
    "- Max. Voegel im Radar: Maximale Anzahl gleichzeitig angezeigter Voegel",
    "- Radar Historie (Stunden): Zeitraum in die Vergangenheit fuer das Radar",
    "- Radar Max SNR / Min SNR: Steuert die Positionierung im Radar (Mitte vs. Rand)",
    "",
    "Archivierung & Woerterbuch:",
    "- Audio Archivierung (Vogelarten): Liste zu archivierender Arten ('alle', 'neu', Kombinationen)",
    "- Max. Archiv-Dateien pro Art: Begrenzung der gespeicherten Aufnahmen (0 = unbegrenzt)",
    "- Vogelarten Woerterbuch: Uebersetzung von englischen zu deutschen Bezeichnungen",
    "- Aufenthalt (Woerterbuch): Aufenthaltszeitraum in Monaten (z.B. 4-10) fuer die Wochen-Statistik",
    "- Status (Woerterbuch): Vogel-Klassifizierung (Standvogel, Zugvogel, etc.) fuer die Wochen-Statistik",
    "- Konf. % (Woerterbuch): Individuelle Erkennungssicherheit, ueberschreibt globales Limit",
    "- Blocked (Woerterbuch): Markierte Voegel ignorieren und nicht speichern/anzeigen",
    "",
    "Audio Filter & System:",
    "- High-Pass Filter: Herausfiltern von tiefen Stoergeraeuschen (inkl. Grenzfrequenz in Hz)",
    "- Noise Reduction: Rauschunterdrueckung aktivieren und Qualitaet einstellen",
    "- Mikrofon auswaehlen: Auswahl des Audio-Eingabegeraets",
    "- Alarm-Ton aktivieren: Akustisches Signal bei jeder neuen Erkennung",
    "",
    "Audio-Erfassung & Auswertung (System-Intern):",
    "- Aufnahmetakt: Alle 1,5 Sek. wird ein 3-Sekunden-Audiofenster analysiert.",
    "- Deduplizierung: Durchgehender Gesang in ueberlappenden Fenstern zaehlt 1x."
]

for line in lines:
    if not line.startswith("-") and line.strip() != "":
        pdf.set_font("Arial", 'B', 12)
        pdf.cell(0, 8, txt=line, ln=1)
        pdf.set_font("Arial", size=12)
    else:
        pdf.cell(0, 8, txt=line, ln=1)

pdf.output("Einstellungen_Beschreibung.pdf")
print("PDF erfolgreich erstellt!")
