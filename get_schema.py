import sqlite3

conn = sqlite3.connect('birds_audio_stats.db')
cursor = conn.cursor()
cursor.execute("SELECT name, sql FROM sqlite_master WHERE type='table'")
for row in cursor.fetchall():
    print(f"Table: {row[0]}")
    print(f"Schema: {row[1]}\n")
conn.close()
