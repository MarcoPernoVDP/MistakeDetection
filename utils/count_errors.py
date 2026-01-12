import json

# Leggi il file JSON
with open('data/annotation_json/video_level_annotations.json', 'r') as f:
    data = json.load(f)

# Conta i recording con has_errors = true
count_with_errors = 0
total_recordings = len(data)

for recording_id, recording_data in data.items():
    if recording_data.get('has_errors', False):
        count_with_errors += 1

# Stampa i risultati
print(f"Totale recording: {total_recordings}")
print(f"Recording con has_errors = true: {count_with_errors}")
print(f"Recording con has_errors = false: {total_recordings - count_with_errors}")
print(f"Percentuale con errori: {(count_with_errors / total_recordings) * 100:.2f}%")
