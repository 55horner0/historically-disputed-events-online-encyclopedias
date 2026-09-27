import json
import re

articles = [
    'falkland_islands',
    'status_of_gibraltar',
    'guantanamo_bay',
    'belize_guatemala',
    'bajo_nuevo',
    'san_andres',
    'serranilla_bank',
    'geography_puerto_rico',
    'south_georgia'
]

def get_article_sections(art_id):
    with open(f"data/articles_parsed/{art_id}_en.json", encoding='utf-8') as f:
        en_data = json.load(f)
    with open(f"data/articles_parsed/{art_id}_es.json", encoding='utf-8') as f:
        es_data = json.load(f)
    return en_data, es_data

for art in articles:
    en, es = get_article_sections(art)
    print(f"=== {art} ===")
    print("EN:", [s['title'] for s in en['sections']])
    print("ES:", [s['title'] for s in es['sections']])
    print()
