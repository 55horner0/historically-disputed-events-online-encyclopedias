"""
Wikipedia QID annotator — importable module.

Public API:
    from extract_qids_alias import extract_qids
    data = extract_qids("en", "Falkland_Islands")
    data = extract_qids("es", "Islas_Malvinas", output_path="out.json")
"""

import re
import json
import time
import random
import unicodedata
from collections import defaultdict, Counter
from urllib.parse import unquote

import requests
from bs4 import BeautifulSoup

# ── NLTK setup ───────────────────────────────────────────────────────────
try:
    import nltk
    from nltk import word_tokenize, pos_tag
    from nltk.corpus import stopwords as nltk_stopwords
    from nltk.corpus import brown
    for _pkg in ("punkt", "punkt_tab",
                 "averaged_perceptron_tagger",
                 "averaged_perceptron_tagger_eng",
                 "stopwords",
                 "brown"):
        try:
            nltk.download(_pkg, quiet=True)
        except Exception:
            pass
    _NLTK_OK = True
    STOPWORDS = set(nltk_stopwords.words("english"))
    _BROWN_FREQ = Counter(w.lower() for w in brown.words() if w.isalpha())
    _BROWN_TOTAL = sum(_BROWN_FREQ.values())
    _BROWN_THRESHOLD = _BROWN_TOTAL * 0.00005
except Exception as _e:
    _NLTK_OK = False
    STOPWORDS = set()
    _BROWN_FREQ = Counter()
    _BROWN_THRESHOLD = float("inf")
    print(f"WARNING: NLTK resources unavailable ({_e})")

try:
    from spacy.lang.en.stop_words import STOP_WORDS as _SPACY_STOPWORDS
    STOPWORDS |= set(_SPACY_STOPWORDS)
except Exception:
    pass


# ── HTTP headers (module-level; not article-specific) ───────────────────
HEADERS = {
    "User-Agent": (
        "QID-Annotator/1.0 "
        "(https://github.com/55horner0; "
        "marina.hornero-merino@universite-paris-saclay.fr) "
        "python-requests"
    ),
    "Accept": "application/json",
}


def wiki_api_url(lang):
    return f"https://{lang}.wikipedia.org/w/api.php"


# ── HTTP helper ──────────────────────────────────────────────────────────
def api_get(url, params, max_retries=6):
    for attempt in range(max_retries):
        try:
            r = requests.get(url, params=params, headers=HEADERS, timeout=30)

            if r.status_code == 429:
                ra = r.headers.get("Retry-After")
                if ra is not None:
                    try:
                        wait = float(ra)
                    except ValueError:
                        wait = 5.0
                else:
                    wait = min(60.0, 2 ** attempt + random.random())
                print(f"  [429] rate limited, sleeping {wait:.1f}s "
                      f"(attempt {attempt+1}/{max_retries})")
                time.sleep(wait)
                continue

            if 500 <= r.status_code < 600:
                wait = min(60.0, 2 ** attempt + random.random())
                print(f"  [{r.status_code}] server error, sleeping "
                      f"{wait:.1f}s (attempt {attempt+1}/{max_retries})")
                time.sleep(wait)
                continue

            r.raise_for_status()
            return r.json()

        except requests.exceptions.RequestException as e:
            if attempt == max_retries - 1:
                raise
            wait = min(60.0, 2 ** attempt + random.random())
            print(f"  [net] {type(e).__name__}, sleeping {wait:.1f}s "
                  f"(attempt {attempt+1}/{max_retries})")
            time.sleep(wait)

    raise RuntimeError(f"api_get failed after {max_retries} retries: {url}")


# ── Title normalization ─────────────────────────────────────────────────
def normalize_title(title):
    if title is None:
        return ""
    title = unquote(title)
    title = title.replace("_", " ").strip()
    title = unicodedata.normalize("NFC", title)
    return title


# ── Step 1: Fetch article HTML ───────────────────────────────────────────
def fetch_article_html(wiki_api, title):
    data = api_get(wiki_api, {
        "action": "parse",
        "page": title,
        "prop": "text",
        "format": "json",
        "formatversion": "2",
    })
    return data["parse"]["text"]


# ── Step 2: Sections and explicit wikilinks ─────────────────────────────
def extract_sections_and_links(html):
    """
    Treat only <h2> as a top-level section boundary. Subsection content
    (<h3>..<h6>) is absorbed into the current <h2> section. The Lead is
    the prose before the first <h2>.
    """
    soup = BeautifulSoup(html, "html.parser")

    for tag in soup.select(
        "sup.reference, .mw-editsection, .infobox, .navbox, .metadata, "
        ".hatnote, .thumb, table, style, script, "
        ".sidebar, .quotebox, .gallery, .gallerytext, .reflist, "
        ".navbox-styles, .mw-empty-elt, .shortdescription"
    ):
        tag.decompose()

    for sec in soup.find_all("section"):
        try:
            sec.unwrap()
        except Exception:
            pass

    sections = []
    links = []
    current_heading = "Lead"
    current_level = 0

    def ensure_section():
        if not sections:
            sections.append({
                "heading": "Lead", "level": 0, "text": "",
            })

    content = soup.find("div", {"class": "mw-parser-output"}) or soup

    for element in content.find_all(
        ["h2", "h3", "h4", "h5", "h6", "p"],
        recursive=True,
    ):
        if element.name == "h2":
            heading = element.get_text(" ", strip=True)
            heading = re.sub(r"\[.*?\]", "", heading).strip()
            if heading:
                current_heading = heading
                current_level = 2
                sections.append({
                    "heading": heading, "level": 2, "text": "",
                })
            continue

        if element.name in ("h3", "h4", "h5", "h6"):
            sub = element.get_text(" ", strip=True)
            sub = re.sub(r"\[.*?\]", "", sub).strip()
            if sub:
                ensure_section()
                sections[-1]["text"] += sub + ". "
            continue

        if element.name == "p":
            text = element.get_text(" ", strip=True)
            if not text:
                continue
            ensure_section()
            sections[-1]["text"] += text + " "

            for a in element.find_all("a", href=True):
                href = a["href"]
                if not href.startswith("/wiki/") or ":" in href:
                    continue
                raw_title = href.replace("/wiki/", "").split("#")[0]
                title = normalize_title(raw_title)
                surface = a.get_text(" ", strip=True)
                if not surface or not title:
                    continue
                links.append({
                    "surface": surface,
                    "title": title,
                    "section": current_heading,
                })

    return sections, links


# ── Step 3: Resolve titles to QIDs (with redirect chain) ────────────────
def titles_to_qids(wiki_api, titles, batch_size=25):
    title_to_qid = {}
    original_to_resolved = {}
    unique_titles = list(set(t for t in titles if t))
    n_batches = (len(unique_titles) + batch_size - 1) // batch_size
    for i in range(0, len(unique_titles), batch_size):
        batch = unique_titles[i:i + batch_size]
        print(f"   batch {i // batch_size + 1}/{n_batches} "
              f"({len(batch)} titles)")
        data = api_get(wiki_api, {
            "action": "query",
            "prop": "pageprops",
            "titles": "|".join(batch),
            "redirects": "1",
            "format": "json",
            "formatversion": "2",
            "maxlag": "5",
        })
        query = data.get("query", {})

        for norm in query.get("normalized", []):
            original_to_resolved[normalize_title(norm["from"])] = \
                normalize_title(norm["to"])

        for redir in query.get("redirects", []):
            original_to_resolved[normalize_title(redir["from"])] = \
                normalize_title(redir["to"])

        for page in query.get("pages", []):
            title = page.get("title")
            qid = page.get("pageprops", {}).get("wikibase_item")
            if title and qid:
                title_to_qid[normalize_title(title)] = qid

        time.sleep(0.5)
    return title_to_qid, original_to_resolved


def resolve_chain(title, original_to_resolved, max_hops=5):
    seen = set()
    current = normalize_title(title)
    for _ in range(max_hops):
        if current in seen:
            break
        seen.add(current)
        nxt = original_to_resolved.get(current)
        if not nxt or nxt == current:
            break
        current = nxt
    return current


# ── Step 4: Redirects to linked pages ───────────────────────────────────
def fetch_redirects(wiki_api, titles, batch_size=25):
    result = defaultdict(set)
    unique_titles = list(set(normalize_title(t) for t in titles if t))
    n_batches = (len(unique_titles) + batch_size - 1) // batch_size
    for i in range(0, len(unique_titles), batch_size):
        batch = unique_titles[i:i + batch_size]
        print(f"   batch {i // batch_size + 1}/{n_batches} "
              f"({len(batch)} titles)")
        data = api_get(wiki_api, {
            "action": "query",
            "prop": "redirects",
            "titles": "|".join(batch),
            "rdlimit": "max",
            "format": "json",
            "formatversion": "2",
            "maxlag": "5",
        })
        for page in data.get("query", {}).get("pages", []):
            title = page.get("title")
            if not title:
                continue
            key = normalize_title(title)
            for r in page.get("redirects", []):
                result[key].add(normalize_title(r["title"]))
        time.sleep(0.5)
    return dict(result)


# ── Filters ─────────────────────────────────────────────────────────────
_PHRASE_CACHE = {}

def is_phrase_link(surface):
    if surface in _PHRASE_CACHE:
        return _PHRASE_CACHE[surface]
    result = False
    if _NLTK_OK:
        try:
            tokens = word_tokenize(surface)
            if len(tokens) > 4:
                result = True
            elif tokens:
                tagged = pos_tag(tokens)
                result = any(tag.startswith("VB") for _, tag in tagged)
        except Exception:
            result = False
    else:
        result = len(surface.split()) > 4
    _PHRASE_CACHE[surface] = result
    return result


def is_function_word(surface):
    tokens = re.findall(r"\w+", surface.lower())
    if not tokens:
        return False
    if all(t in STOPWORDS for t in tokens):
        return True
    if len(tokens) == 1 and not surface[0].isupper():
        freq = _BROWN_FREQ.get(surface.lower(), 0)
        if freq >= _BROWN_THRESHOLD:
            return True
    return False


def is_numeric(surface):
    return bool(re.fullmatch(r"[\d\s,.\-–]+", surface.strip()))


_APOSTROPHES = "['\u2019\u02BC]"

def _has_capitalized_neighbor_before(text, start):
    window = text[max(0, start - 30):start]
    m = re.search(r"([A-Z][a-z]+)\s*$", window)
    if not m:
        return False
    return m.group(1).lower() not in STOPWORDS


def _has_capitalized_neighbor_after(text, end):
    window = text[end:end + 30]
    pattern = r"^(?:" + _APOSTROPHES + r"s\s+|\s+)([A-Z][a-z]+)"
    m = re.search(pattern, window)
    if not m:
        return False
    return m.group(1).lower() not in STOPWORDS


def should_suppress_single_token(text, start, end, alias):
    if len(alias.split()) != 1:
        return False
    if not alias[:1].isupper():
        return False
    if _has_capitalized_neighbor_before(text, start):
        return True
    if _has_capitalized_neighbor_after(text, end):
        return True
    return False


# ── Alias table ─────────────────────────────────────────────────────────
def build_alias_table(
    links,
    title_to_redirects,
    title_to_qid,
    original_to_resolved,
    article_title,
    article_qid,
):
    qid_to_surfaces = defaultdict(set)
    qid_target_titles = defaultdict(set)

    for link in links:
        qid = link.get("qid")
        if not qid:
            continue
        qid_to_surfaces[qid].add(link["surface"])
        resolved = link.get("resolved_title") or link["title"]
        qid_to_surfaces[qid].add(resolved)
        qid_target_titles[qid].add(resolved)

    for canonical_title, redirects in title_to_redirects.items():
        qid = title_to_qid.get(canonical_title)
        if not qid:
            resolved = resolve_chain(canonical_title, original_to_resolved)
            qid = title_to_qid.get(normalize_title(resolved))
        if not qid:
            continue
        for r in redirects:
            qid_to_surfaces[qid].add(r)

    if article_qid:
        article_canonical = normalize_title(article_title)
        qid_to_surfaces[article_qid].add(article_canonical)
        qid_target_titles[article_qid].add(article_canonical)
        for r in title_to_redirects.get(article_canonical, []):
            qid_to_surfaces[article_qid].add(r)

    for qid in list(qid_to_surfaces.keys()):
        cleaned = set()
        for s in qid_to_surfaces[qid]:
            s = s.strip()
            if not s:
                continue
            if len(s) <= 2:
                continue
            if is_numeric(s):
                continue
            if is_phrase_link(s):
                continue
            if is_function_word(s):
                continue
            cleaned.add(s)
        qid_to_surfaces[qid] = cleaned

    surface_to_qids = defaultdict(set)
    for qid, surfaces in qid_to_surfaces.items():
        for s in surfaces:
            surface_to_qids[s].add(qid)

    cleaned = defaultdict(set)
    for surface, qids in surface_to_qids.items():
        if len(qids) == 1:
            for q in qids:
                cleaned[q].add(surface)
            continue
        matches = [
            q for q in qids
            if surface.lower() in {t.lower() for t in qid_target_titles[q]}
        ]
        if len(matches) == 1:
            cleaned[matches[0]].add(surface)

    return {q: s for q, s in cleaned.items() if s}


# ── Matching ────────────────────────────────────────────────────────────
def make_pattern(surface):
    escaped = re.escape(surface)
    if surface and surface[0].isupper():
        return re.compile(r"\b" + escaped + r"\b")
    return re.compile(r"\b" + escaped + r"\b", re.IGNORECASE)


def overlaps(start, end, occupied):
    for s, e in occupied:
        if not (end <= s or start >= e):
            return True
    return False


def find_alias_mentions(text, qid_to_aliases, existing_spans, section):
    flat = [(a, q) for q, aliases in qid_to_aliases.items() for a in aliases]
    flat.sort(key=lambda x: len(x[0]), reverse=True)

    occupied = [(s["start"], s["end"]) for s in existing_spans]
    new_spans = []
    for alias, qid in flat:
        pattern = make_pattern(alias)
        for match in pattern.finditer(text):
            start, end = match.span()
            if overlaps(start, end, occupied):
                continue
            if should_suppress_single_token(text, start, end, alias):
                continue
            new_spans.append({
                "qid": qid,
                "surface": match.group(0),
                "start": start,
                "end": end,
                "section": section,
                "source": "alias",
            })
            occupied.append((start, end))
    return new_spans


# ── Section annotation ──────────────────────────────────────────────────
def annotate_sections(sections, links, qid_to_aliases):
    links_by_section = defaultdict(list)
    for link in links:
        links_by_section[link["section"]].append(link)

    annotated = []
    for sec in sections:
        heading = sec["heading"]
        text = sec["text"]

        explicit_spans = []
        search_from = 0
        for link in links_by_section.get(heading, []):
            surface = link["surface"]
            if is_phrase_link(surface):
                continue
            pos = text.find(surface, search_from)
            if pos == -1:
                continue
            explicit_spans.append({
                "qid": link.get("qid"),
                "surface": surface,
                "start": pos,
                "end": pos + len(surface),
                "section": heading,
                "source": "link",
            })
            search_from = pos + len(surface)

        alias_spans = find_alias_mentions(
            text, qid_to_aliases, explicit_spans, heading
        )
        all_spans = explicit_spans + alias_spans
        all_spans.sort(key=lambda s: s["start"])

        annotated.append({
            "heading": heading,
            "level": sec["level"],
            "text": text,
            "spans": all_spans,
        })
    return annotated


# ── Render annotated text ───────────────────────────────────────────────
def render_text_with_qids(text, spans):
    if not spans:
        return text
    ordered = sorted(spans, key=lambda s: s["start"])
    out = []
    last = 0
    for sp in ordered:
        s, e, qid = sp["start"], sp["end"], sp.get("qid")
        if s < last or qid is None:
            continue
        out.append(text[last:s])
        out.append(f"{sp['surface']} ({qid})")
        last = e
    out.append(text[last:])
    return "".join(out)


# ════════════════════════════════════════════════════════════════════════
# Public entry point
# ════════════════════════════════════════════════════════════════════════
def extract_qids(lang, title, output_path=None, verbose=True):
    """
    Annotate a Wikipedia article with QIDs per top-level section.

    Parameters
    ----------
    lang : str
        Wikipedia language code, e.g. "en", "es".
    title : str
        Article title with underscores, without the "/wiki/" prefix,
        e.g. "Falkland_Islands".
    output_path : str or None
        If given, the JSON result is written to this path.
        If None, only returned in memory.
    verbose : bool
        If True, prints progress to stdout.

    Returns
    -------
    dict with keys:
        title, language, article_qid, article_id, sections, qid_to_aliases
    """
    def log(msg):
        if verbose:
            print(msg)

    wiki_api = wiki_api_url(lang)

    log(f"Stopwords loaded: {len(STOPWORDS)}")
    log(f"Brown corpus: {len(_BROWN_FREQ)} unique words, "
        f"threshold={_BROWN_THRESHOLD:.0f}")

    log(f"\n[{lang}] {title} — fetching article HTML …")
    html = fetch_article_html(wiki_api, title)

    log("Extracting sections and explicit wikilinks …")
    sections, links = extract_sections_and_links(html)
    log(f"   → {len(sections)} sections, {len(links)} explicit links")
    for s in sections[:8]:
        log(f"      [{s['level']}] {s['heading']!r} "
            f"({len(s['text'])} chars)")

    log("Resolving Wikipedia titles to QIDs …")
    all_titles = [l["title"] for l in links] + [normalize_title(title)]
    title_to_qid, original_to_resolved = titles_to_qids(
        wiki_api, all_titles
    )

    for link in links:
        resolved = resolve_chain(link["title"], original_to_resolved)
        link["resolved_title"] = resolved
        link["qid"] = title_to_qid.get(normalize_title(resolved))

    article_qid = title_to_qid.get(normalize_title(title))
    resolved_links = sum(1 for l in links if l.get("qid"))
    log(f"   → resolved {len(title_to_qid)} titles to QIDs")
    log(f"   → {resolved_links}/{len(links)} links got a QID")
    log(f"   → article QID = {article_qid}")

    log("Fetching Wikipedia redirects …")
    linked_titles = [l["title"] for l in links if l.get("qid")]
    linked_titles.append(normalize_title(title))
    title_to_redirects = fetch_redirects(wiki_api, linked_titles)
    log(f"   → fetched redirects for {len(title_to_redirects)} titles")

    log("Building alias table …")
    qid_to_aliases = build_alias_table(
        links,
        title_to_redirects,
        title_to_qid,
        original_to_resolved,
        title,
        article_qid,
    )
    total = sum(len(v) for v in qid_to_aliases.values())
    log(f"   → {total} surfaces across {len(qid_to_aliases)} QIDs")
    if article_qid:
        sample = sorted(qid_to_aliases.get(article_qid, []))
        log(f"   → article aliases ({len(sample)}): {sample[:12]}")

    log("Annotating sections …")
    annotated = annotate_sections(sections, links, qid_to_aliases)
    total_spans = sum(len(s["spans"]) for s in annotated)
    link_spans = sum(1 for s in annotated for sp in s["spans"]
                     if sp["source"] == "link")
    alias_spans = total_spans - link_spans
    null_spans = sum(1 for s in annotated for sp in s["spans"]
                     if sp.get("qid") is None)
    log(f"   → {total_spans} spans "
        f"({link_spans} links, {alias_spans} aliases, "
        f"{null_spans} null QIDs)")

    article_id = f"{lang}:{article_qid}" if article_qid else f"{lang}:UNKNOWN"
    enriched = []
    for idx, sec in enumerate(annotated):
        section_id = f"{article_id}:{idx}"
        text_with_qids = render_text_with_qids(sec["text"], sec["spans"])
        enriched.append({
            "article_id": article_id,
            "section_id": section_id,
            "section_idx": idx,
            "language": lang,
            "heading": sec["heading"],
            "level": sec["level"],
            "text": sec["text"],
            "text_with_qids": text_with_qids,
            "spans": sec["spans"],
        })

    output = {
        "title": title,
        "language": lang,
        "article_qid": article_qid,
        "article_id": article_id,
        "sections": enriched,
        "qid_to_aliases": {q: sorted(a) for q, a in qid_to_aliases.items()},
    }

    if output_path:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)
        log(f"\n→ Saved to {output_path}")

    return output


# ── Standalone usage ────────────────────────────────────────────────────
if __name__ == "__main__":
    # Default test: English Falkland Islands
    extract_qids(
        lang="en",
        title="Falkland_Islands",
        output_path="falkland_annotated.json",
    )