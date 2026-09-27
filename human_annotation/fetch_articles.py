import os
import re
import json
import urllib.parse
import requests
from bs4 import BeautifulSoup

HEADERS = {
    "User-Agent": "WikipediaControversyResearch/1.0 (academic_research@example.com)"
}

ARTICLES = [
    {
        "id": "falkland_islands",
        "name": "Falkland Islands / Islas Malvinas",
        "en_url": "https://en.wikipedia.org/wiki/Falkland_Islands",
        "es_url": "https://es.wikipedia.org/wiki/Islas_Malvinas_(territorio_brit%C3%A1nico_de_ultramar)"
    },
    {
        "id": "status_of_gibraltar",
        "name": "Status of Gibraltar / Disputa territorial de Gibraltar",
        "en_url": "https://en.wikipedia.org/wiki/Status_of_Gibraltar",
        "es_url": "https://es.wikipedia.org/wiki/Disputa_territorial_de_Gibraltar"
    },
    {
        "id": "guantanamo_bay",
        "name": "Guantanamo Bay Naval Base / Base Naval de la Bahía de Guantánamo",
        "en_url": "https://en.wikipedia.org/wiki/Guantanamo_Bay_Naval_Base",
        "es_url": "https://es.wikipedia.org/wiki/Base_Naval_de_la_Bah%C3%ADa_de_Guant%C3%A1namo"
    },
    {
        "id": "belize_guatemala",
        "name": "Belizean–Guatemalan territorial dispute / Diferendo territorial entre Belice y Guatemala",
        "en_url": "https://en.wikipedia.org/wiki/Belizean%E2%80%93Guatemalan_territorial_dispute",
        "es_url": "https://es.wikipedia.org/wiki/Diferendo_territorial_entre_Belice_y_Guatemala"
    },
    {
        "id": "bajo_nuevo",
        "name": "Bajo Nuevo Bank / Bajo Nuevo",
        "en_url": "https://en.wikipedia.org/wiki/Bajo_Nuevo_Bank",
        "es_url": "https://es.wikipedia.org/wiki/Bajo_Nuevo"
    },
    {
        "id": "san_andres",
        "name": "San Andrés and Providencia / Archipiélago de San Andrés, Providencia y Santa Catalina",
        "en_url": "https://en.wikipedia.org/wiki/San_Andr%C3%A9s_and_Providencia",
        "es_url": "https://es.wikipedia.org/wiki/Archipi%C3%A9lago_de_San_Andr%C3%A9s,_Providencia_y_Santa_Catalina"
    },
    {
        "id": "serranilla_bank",
        "name": "Serranilla Bank / Isla Serranilla",
        "en_url": "https://en.wikipedia.org/wiki/Serranilla_Bank",
        "es_url": "https://es.wikipedia.org/wiki/Isla_Serranilla"
    },
    {
        "id": "geography_puerto_rico",
        "name": "Geography of Puerto Rico / Geografía de Puerto Rico",
        "en_url": "https://en.wikipedia.org/wiki/Geography_of_Puerto_Rico",
        "es_url": "https://es.wikipedia.org/wiki/Geograf%C3%ADa_de_Puerto_Rico"
    },
    {
        "id": "south_georgia",
        "name": "South Georgia and South Sandwich / Islas Georgias del Sur y Sandwich del Sur",
        "en_url": "https://en.wikipedia.org/wiki/South_Georgia_and_the_South_Sandwich_Islands",
        "es_url": "https://es.wikipedia.org/wiki/Islas_Georgias_del_Sur_y_Sandwich_del_Sur"
    }
]

def clean_text(text):
    # Remove citation brackets like [1], [note 2], etc.
    text = re.sub(r'\[\s*\d+\s*\]', '', text)
    text = re.sub(r'\[\s*note\s*\d+\s*\]', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\[\s*citation needed\s*\]', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\[\s*aclaración requerida\s*\]', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\[\s*cita requerida\s*\]', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def extract_sections_from_html(html_content):
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # Remove unwanted tags
    for tag in soup(['script', 'style', 'table', 'figure', 'nav', 'sup', 'footer']):
        tag.decompose()
        
    content_div = soup.find('div', {'id': 'mw-content-text'})
    if not content_div:
        content_div = soup
        
    parser_output = content_div.find('div', {'class': 'mw-parser-output'})
    if parser_output:
        content_div = parser_output

    sections = []
    current_section = {
        "title": "Lead",
        "level": 1,
        "paragraphs": []
    }
    
    # Find all h2, h3, and p tags in order of appearance
    for elem in content_div.find_all(['h2', 'h3', 'p']):
        if elem.name in ['h2', 'h3']:
            title = elem.get_text().strip()
            title = re.sub(r'\[editar?\]', '', title, flags=re.IGNORECASE).strip()
            
            # Skip references, see also, external links, bibliography
            skip_titles = [
                'see also', 'references', 'further reading', 'external links', 'notes',
                'véase también', 'referencias', 'bibliografía', 'enlaces externos', 'notas'
            ]
            if title.lower() in skip_titles:
                continue

            if current_section["paragraphs"]:
                sections.append(current_section)
                
            level = int(elem.name[1])
            current_section = {
                "title": title,
                "level": level,
                "paragraphs": []
            }
        elif elem.name == 'p':
            txt = clean_text(elem.get_text())
            if len(txt) > 30: # Filter trivial text
                current_section["paragraphs"].append(txt)
                
    if current_section["paragraphs"]:
        sections.append(current_section)
        
    return sections

def download_all():
    os.makedirs("data/articles_raw", exist_ok=True)
    os.makedirs("data/articles_parsed", exist_ok=True)
    
    for item in ARTICLES:
        art_id = item["id"]
        print(f"Downloading {item['name']}...")
        
        # English
        en_resp = requests.get(item["en_url"], headers=HEADERS, timeout=30)
        with open(f"data/articles_raw/{art_id}_en.html", "w", encoding="utf-8") as f:
            f.write(en_resp.text)
        en_sections = extract_sections_from_html(en_resp.text)
        with open(f"data/articles_parsed/{art_id}_en.json", "w", encoding="utf-8") as f:
            json.dump({"id": art_id, "url": item["en_url"], "sections": en_sections}, f, ensure_ascii=False, indent=2)
            
        # Spanish
        es_resp = requests.get(item["es_url"], headers=HEADERS, timeout=30)
        with open(f"data/articles_raw/{art_id}_es.html", "w", encoding="utf-8") as f:
            f.write(es_resp.text)
        es_sections = extract_sections_from_html(es_resp.text)
        with open(f"data/articles_parsed/{art_id}_es.json", "w", encoding="utf-8") as f:
            json.dump({"id": art_id, "url": item["es_url"], "sections": es_sections}, f, ensure_ascii=False, indent=2)
            
        print(f"Done {art_id}: {len(en_sections)} EN sections, {len(es_sections)} ES sections.")

if __name__ == "__main__":
    download_all()
