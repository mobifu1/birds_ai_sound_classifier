import sys
from fpdf import FPDF

pdf = FPDF()
pdf.set_auto_page_break(auto=True, margin=15)
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
    "- Occurrence Threshold: Historische eBird-Wahrscheinlichkeit fuer deinen Standort",
    "- Auto Season Lowering: Ignoriert manuellen Threshold. Nutzt stattdessen kalenderwochen-",
    "  spezifische Werte (aus 'auto_season_lowering.json') oder einen Standardwert.",
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
    "- blocklist-log.txt: Protokolliert Vogelarten, die aufgrund lokaler GPS-Filter oder Blockierung (Woerterbuch) aussortiert wurden.",
    "- Blocklist-Log-Probability Umschalter: Wenn aktiviert, werden nur Vogelarten mit Geo-Probability > 0% im blocklist-log protokolliert.",
    "",
    "Audio Filter & System:",
    "- High-Pass Filter: Herausfiltern von tiefen Stoergeraeuschen (inkl. Grenzfrequenz in Hz)",
    "- Low-Pass Filter: Herausfiltern von hohen Stoergeraeuschen (inkl. Grenzfrequenz in Hz)",
    "- Noise Reduction: Rauschunterdrueckung aktivieren und Qualitaet einstellen",
    "- Mikrofon auswaehlen: Auswahl des Audio-Eingabegeraets",
    "- Alarm-Ton aktivieren: Akustisches Signal bei jeder neuen Erkennung",
    "- Klick-Ton aktivieren: Dezentes akustisches Feedback im Browser bei Erkennungen",
    "",
    "Steuerung & Datenbank:",
    "- Woerterbuch anwenden: Wendet das Woerterbuch rueckwirkend auf alle Eintraege an (inkl. Dateiumbenennung)",
    "- Datenbank synchronisieren: Sortiert die komplette Datenbankliste chronologisch nach dem Zeitstempel",
    "- Einzelvorkommen loeschen: Loescht alle Vogelarten (inkl. Dateien), die bisher nur ein einziges Mal erkannt wurden",
    "- Arten zusammenfuehren: Benennt alle Eintraege und Audiodateien einer bestimmten Art in eine andere Art um",
    "- Art-Eintraege loeschen: Loescht saemtliche Eintraege und Audiodateien einer bestimmten Vogelart",
    "- Datenbank Backup: Erstellt sofort eine Sicherungskopie der aktuellen Datenbank-Datei",
    "- Datenbank zuruecksetzen: Loescht alle Erkennungen aus der Datenbank und leert das komplette Audio-Archiv",
    "",
    "Audio-Erfassung & Auswertung (System-Intern):"
]

for line in lines:
    if not line.startswith("-") and line.strip() != "":
        pdf.set_font("Arial", 'B', 12)
        pdf.cell(0, 8, txt=line, ln=1)
        pdf.set_font("Arial", size=12)
    else:
        pdf.cell(0, 8, txt=line, ln=1)

description_text = "Die Soundanalyse wertet die Umgebungsgeraeusche in 3-Sekunden-Fenstern aus, die sich um jeweils 1,5 Sekunden ueberschneiden (Overlapping). Wird ein Vogel positiv erkannt, erfolgt die Speicherung nicht mehr sofort. Stattdessen wird die Erkennung zunaechst fuer genau ein Fenster (1,5 Sekunden) zurueckgehalten. In dieser Zeit wird das naechste, ueberlappende Audio-Fenster analysiert. Da sich der Vogelruf oft erst im zweiten Fenster in der zeitlichen Mitte befindet, liefert dieses haeufig einen besseren Konfidenzwert. Die KI vergleicht dann beide Erkennungen und speichert nur das Fenster mit der hoeheren Konfidenz in der Datenbank und im Archiv ab. Anschliessend greift ein automatischer Spam-Schutz fuer diese Vogelart. Fortlaufende, direkte Folge-Erkennungen desselben Vogels werden ignoriert, um die Datenbank nicht mit unzaehligen Eintraegen desselben Rufs zu ueberfluten.\n\nWie funktioniert der Spam-Schutz im Detail? Sobald eine Vogelart erfolgreich mit der hoechsten Konfidenz aus zwei ueberlappenden Fenstern gespeichert wurde, merkt sich das System diesen Vogel als 'aktiv rufend'. Solange dieser Vogel in den unmittelbar folgenden 3-Sekunden-Fenstern weiterhin ununterbrochen erkannt wird (auch wenn die Konfidenz schwankt), wird er ignoriert und nicht erneut in die Datenbank geschrieben. Erst wenn in einem Fenster der Vogel nicht mehr erkannt wird oder die Konfidenz unter den eingestellten Schwellenwert (Threshold) faellt, gilt die durchgehende Erkennungs-Serie als beendet. Der Spam-Schutz fuer diese Vogelart wird dann zurueckgesetzt. Taucht der Vogel in einem spaeteren Fenster wieder auf, wird dies als komplett neuer, eigenstaendiger Ruf gewertet und der gesamte Prozess (inklusive 1-Fenster-Verzoegerung) beginnt von vorn.\n\nMultitasking bei der Artenerkennung: Die Erkennung verfuegt nun ueber eine Multitasking-Faehigkeit. Das System wertet alle erkannten Vogelstimmen innerhalb eines 3-Sekunden-Fensters parallel aus. Rufen beispielsweise eine Amsel und ein Wiedehopf gleichzeitig, werden beide Arten individuell geprueft, aufgezeichnet und auf dem Radar angezeigt, sofern sie ihre jeweiligen Schwellenwerte uebertreffen. Der Spam-Schutz (Streak-Logik) laeuft fuer jeden erkannten Vogel voellig unabhaengig im Hintergrund ab."

pdf.multi_cell(0, 6, txt=description_text)

pdf.output("Einstellungen_Beschreibung.pdf")
print("PDF erfolgreich erstellt!")
