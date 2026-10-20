#!/usr/bin/env python3
"""
Replication of the Contropedia controversy score (Borra et al., CHI 2015), per QID.

    C(q) = sum over revisions R_i, sum over sentences S in D(R_i) with q in S : 1 / w(S)

D(R_i)  = sentences of R_{i-1} that have a substantive, disagreeing edit in R_i
          (the edit contains a deletion; insert-only edits and full section
          inserts/deletes are ignored; vandalism and its reverts are ignored).
w(S)    = number of distinct linked entities (QIDs) in the edited sentence.

Design choices (A3 B1 C1 D3 E2 F2 G1):
  -  fetch once from the Wikipedia API into a local cache, then everything is local
  -  full vandalism rules of the paper (comment, anti-vandal bots, IP revert < 60 s,
      automatic blank/replace summaries); reverts are detected by SHA1 equality
  -  regex sentence splitter (links, templates, refs protected/stripped)
  -  links of a replaced sentence = union(old, new); of a deleted sentence = old only
  -  word-level check on replaced pairs: pure insertions inside a sentence are discarded
  -  w(S) counts distinct QIDs
  -  titles -> QIDs through the API (redirects resolved), red links dropped

Interpretation note: a revision flagged as vandalism or as a vandalism revert still
advances the baseline text (so that the next diff is against the true previous state)
but emits no events.

Usage:
  python scripts/controversy_qid_with_sections_simple.py fetch "en:Falkland Islands" "es:Islas Malvinas (territorio británico de ultramar)" \
         --user-agent "my-project/0.1 (me@univ.fr)"
  python scripts/controversy_qid_with_sections_simple.py score "en:Falkland Islands" --out-dir results --user-agent "my-project/0.1 (me@email.com)"

Optional last step, section-level scores for the CURRENT version (one JSON per article,
same order as the articles, as produced by extract_qids):
  python scripts/controversy_qid_with_sections_simple.py score "es:Islas Malvinas (territorio británico de ultramar)" --annotated "Islas_Malvinas_(territorio_británico_de_ultramar).json" --out-dir results --user-agent "my-project/0.1 (me@email.com)"
  Each QID's full-history score C(q) is split over the sections of the JSON in proportion to
  how many times the QID is mentioned in each section (JSON spans). A section's score is the
  sum of its shares. QIDs that are not in the current version are not distributed: they keep
  their original score in the ranking (in_current_version = False).

Outputs per article (in --out-dir):
  <name>.events.csv    rev_id, timestamp, qid, weight      (one row per edited sentence x QID)
  <name>.ranking.csv   rank, qid, title, controversy, n_edits [, in_current_version]
  <name>.sections.csv  section_id, section_idx, heading, section_score,
                       section_score_per_token (= score / words of the section text; empty if no text)
                       (only with --annotated)
  cumulative C(q) up to revision R_r  =  sum of `weight` over events with rev order <= r.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import os
import re
import sys
import time
import urllib.parse
import urllib.request
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from datetime import datetime
from difflib import SequenceMatcher

try:  # optional, ~5-10x faster JSON
    import orjson

    def jloads(b):
        return orjson.loads(b)

    def jdumps(o) -> bytes:
        return orjson.dumps(o)

except ImportError:  # pragma: no cover

    def jloads(b):
        return json.loads(b)

    def jdumps(o) -> bytes:
        return json.dumps(o, ensure_ascii=False).encode("utf-8")


# --------------------------------------------------------------------------- #
# Configuration (extend freely): here are all the vandalism bots that are normally known in Wikipedia.
# --------------------------------------------------------------------------- #
BOTS = {  # anti-vandalism bots, lower-case
    "cluebot ng", "cluebot", "xlinkbot", "antivandalbot", "voabot ii",
    "salebot", "avbot", "eskimbot", "sz-iwbot", "lupin anti-vandal tool",
}
AUTO_BLANK = re.compile(
    r"^(?:blanked|page blanked|replaced (?:the )?(?:content|page) with|content was:"
    r"|blanqueo|página en blanco|blanchi)",
    re.I,
)
NS_PREFIXES = {  # non-article namespaces (en/es/fr) -> links ignored
    "file", "image", "category", "template", "wikipedia", "wp", "help", "portal",
    "special", "talk", "user", "media", "draft", "module", "mediawiki", "book", "topic",
    "archivo", "imagen", "categoría", "categoria", "plantilla", "wikipedia", "ayuda",
    "portal", "especial", "usuario", "discusión", "medio",
    "fichier", "catégorie", "modèle", "aide", "spécial", "utilisateur", "discussion",
    "wikt", "wiktionary", "commons", "wikidata", "d", "w", "s", "b", "q", "n", "v",
}
ABBR = {
    "mr", "mrs", "ms", "dr", "prof", "sr", "sra", "srta", "dra", "jr", "st", "mt", "vs",
    "etc", "inc", "ltd", "co", "no", "gen", "col", "lt", "capt", "cmdr", "adm", "sgt",
    "rev", "fr", "hon", "gov", "pres", "approx", "ca", "c", "fl", "b", "d", "r", "vol",
    "u.s", "u.k", "e.g", "i.e", "a.m", "p.m", "av", "avda", "lic", "ing", "pág", "pp",
    "núm", "ed", "eds", "est", "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep",
    "sept", "oct", "nov", "dec",
}
API_URL = "https://{lang}.wikipedia.org/w/api.php"
DEFAULT_UA = "contropedia-replication/0.1 (set --user-agent with your contact)"


# --------------------------------------------------------------------------- #
# HTTP helpers + fetching (A3)
# --------------------------------------------------------------------------- #
def api_get(lang: str, params: dict, ua: str, retries: int = 6) -> dict:
    url = API_URL.format(lang=lang) + "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"User-Agent": ua, "Accept-Encoding": "gzip"})
    last = None
    for k in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=90) as r:
                data = r.read()
                if r.headers.get("Content-Encoding") == "gzip":
                    data = gzip.decompress(data)
            return jloads(data)
        except Exception as e:  # network / 429 / 5xx
            last = e
            time.sleep(min(2 ** k, 30))
    raise RuntimeError(f"API request failed: {last}")


def safe_name(lang: str, title: str) -> str:
    return f"{lang}_" + re.sub(r"[^\w\-]+", "_", title).strip("_")


def parse_spec(spec: str) -> tuple[str, str]:
    if ":" not in spec:
        raise SystemExit(f"Article spec must look like 'en:Title', got {spec!r}")
    lang, title = spec.split(":", 1)
    return lang.strip(), title.strip()


def _count_lines(path: str) -> int:
    n = 0
    with open(path, "rb") as f:
        for _ in f:
            n += 1
    return n


def _truncate_lines(path: str, n: int) -> None:
    with open(path, "rb") as f:
        lines = [next(f) for _ in range(n)]
    with open(path, "wb") as f:
        f.writelines(lines)


def fetch_article(spec: str, cache_dir: str, ua: str) -> str:
    """Download full revision history (wikitext) into <cache>/<name>.meta.jsonl + .text.jsonl.
    Resumable. Returns the cache name."""
    lang, title = parse_spec(spec)
    os.makedirs(cache_dir, exist_ok=True)
    name = safe_name(lang, title)
    meta_p = os.path.join(cache_dir, name + ".meta.jsonl")
    text_p = os.path.join(cache_dir, name + ".text.jsonl")
    seen: set[int] = set()
    last_ts = None
    if os.path.exists(meta_p) and os.path.exists(text_p):
        nm, nt = _count_lines(meta_p), _count_lines(text_p)
        if nt > nm:
            _truncate_lines(text_p, nm)
        elif nm > nt:
            _truncate_lines(meta_p, nt)
        with open(meta_p, "rb") as f:
            for line in f:
                m = jloads(line)
                seen.add(m["revid"])
                last_ts = m["ts"]
    params = dict(
        action="query", prop="revisions", titles=title, format="json", formatversion=2,
        rvprop="ids|timestamp|user|comment|sha1|content", rvslots="main",
        rvlimit=50, rvdir="newer",
    )
    if last_ts:
        params["rvstart"] = last_ts
    total = len(seen)
    with open(text_p, "ab") as ft, open(meta_p, "ab") as fm:
        while True:
            d = api_get(lang, params, ua)
            pages = d.get("query", {}).get("pages", [])
            if not pages or pages[0].get("missing"):
                raise SystemExit(f"Page not found: {spec}")
            tb, mb = [], []
            for r in pages[0].get("revisions", []):
                if r["revid"] in seen:
                    continue
                main = r.get("slots", {}).get("main", {})
                if main.get("texthidden") or "content" not in main:
                    continue
                seen.add(r["revid"])
                tb.append(jdumps(main["content"]))
                mb.append(jdumps({
                    "revid": r["revid"], "ts": r["timestamp"],
                    "user": "" if r.get("userhidden") else r.get("user", ""),
                    "anon": bool(r.get("anon")),
                    "comment": "" if r.get("commenthidden") else r.get("comment", ""),
                    "sha1": r.get("sha1", ""),
                }))
            if tb:  # text first, meta second -> meta never ahead of text
                ft.write(b"\n".join(tb) + b"\n")
                ft.flush()
                fm.write(b"\n".join(mb) + b"\n")
                fm.flush()
                total += len(tb)
                print(f"[{name}] {total} revisions", file=sys.stderr, end="\r")
            if "continue" in d:
                params.update(d["continue"])
            else:
                break
    print(f"[{name}] done: {total} revisions", file=sys.stderr)
    return name


# --------------------------------------------------------------------------- #
# Pass 1: revision filtering (collapse + vandalism)
# --------------------------------------------------------------------------- #
_SECTION_PREFIX = re.compile(r"^/\*.*?\*/\s*")


def _epoch(ts: str) -> float:
    return datetime.fromisoformat(ts.replace("Z", "+00:00")).timestamp()


def flag_revisions(meta: list[dict]) -> tuple[list[bool], list[bool]]:
    """Returns (kept, drop).
    kept[i]: revision survives the 'consecutive edits by same user -> keep last' rule.
    drop[i]: kept revision is vandalism or a vandalism revert -> advances baseline, no events."""
    n = len(meta)
    users = [m["user"] for m in meta]
    ts = [_epoch(m["ts"]) for m in meta]
    kept = [False] * n
    for i in range(n):
        if i == n - 1 or not users[i] or users[i + 1] != users[i]:
            kept[i] = True
    keep_idx = [i for i in range(n) if kept[i]]
    drop = [False] * n
    last_pos_by_sha: dict[str, int] = {}
    for p, i in enumerate(keep_idx):
        m = meta[i]
        if AUTO_BLANK.match(_SECTION_PREFIX.sub("", m["comment"])):
            drop[i] = True  # blanking / replaced-by-junk (automatic edit summary)
        sha = m["sha1"]
        p0 = last_pos_by_sha.get(sha)
        if sha and p0 is not None and p0 < p - 1:  # revision i restores the state of keep_idx[p0]
            rng = keep_idx[p0 + 1:p]
            if "vandal" in m["comment"].lower() or users[i].lower() in BOTS:
                for q in rng:
                    drop[q] = True
                drop[i] = True
            else:
                hit = [q for q in rng if meta[q]["anon"] and ts[i] - ts[q] <= 60]
                if hit or any(drop[q] for q in rng):
                    for q in hit:
                        drop[q] = True
                    drop[i] = True
        if sha:
            last_pos_by_sha[sha] = p
    return kept, drop


# --------------------------------------------------------------------------- #
# Wikitext processing
# --------------------------------------------------------------------------- #
_HEAD = re.compile(r"^(={2,6})[ \t]*([^=\n][^\n]*?)[ \t]*\1[ \t]*$", re.M)
_COMMENT = re.compile(r"<!--.*?-->", re.S)
_REF = re.compile(r"<ref\b[^>]*?/>|<ref\b[^>]*>.*?</ref\s*>", re.S | re.I)
_TMPL = re.compile(r"\{\{[^{}]*\}\}")
_FILE = re.compile(
    r"\[\[\s*(?:File|Image|Archivo|Imagen|Fichier|Datei)\s*:(?:[^\[\]]|\[\[[^\[\]]*\]\])*\]\]", re.I)
_TABLE = re.compile(r"^[ \t]*(?:\{\||\|\}|\|-|[|!]).*$", re.M)
_HTML = re.compile(r"</?[a-zA-Z][^>]*>")
_QUOTES = re.compile(r"'{2,5}")
_LISTMARK = re.compile(r"^[*#:;]+[ \t]*", re.M)
_LINK_ANY = re.compile(r"\[\[[^\[\]]*\]\]")
_LINK = re.compile(r"\[\[([^\[\]|]+)(?:\|[^\[\]]*)?\]\]")
_BOUND = re.compile(r"[.!?]+[\"')\]»”]*\s+")
_WORD = re.compile(r"\w+")
_OPEN = "\"'[(¿¡«“‘"
_PROT = str.maketrans({".": "\x00", "!": "\x01", "?": "\x02"})
_UNPROT = str.maketrans({"\x00": ".", "\x01": "!", "\x02": "?"})


def split_sections(text: str) -> dict[str, str]:
    parts = _HEAD.split(text)
    secs = {"(lead)": parts[0]}
    for k in range(1, len(parts) - 2, 3):
        title = parts[k + 1].strip()
        key, c = title, 1
        while key in secs:
            c += 1
            key = f"{title}#{c}"
        secs[key] = parts[k + 2]
    return secs


def clean(body: str) -> str:
    t = body
    if "<!--" in t:
        t = _COMMENT.sub("", t)
    if "<ref" in t:
        t = _REF.sub("", t)
    if "{{" in t:
        while True:
            t2 = _TMPL.sub("", t)
            if t2 == t:
                break
            t = t2
    t = _FILE.sub("", t)
    t = _TABLE.sub("", t)
    t = _HTML.sub("", t)
    t = _QUOTES.sub("", t)
    return _LISTMARK.sub("", t)


def split_sentences(text: str) -> list[str]:
    text = _LINK_ANY.sub(lambda m: m.group().translate(_PROT), text)
    out: list[str] = []
    for para in text.split("\n"):
        para = para.strip()
        if not para:
            continue
        start = 0
        for m in _BOUND.finditer(para):
            end = m.end()
            if end >= len(para):
                break
            nxt = para[end]
            if not (nxt.isupper() or nxt.isdigit() or nxt in _OPEN):
                continue
            if para[m.start()] == ".":
                toks = para[start:m.start()].split()
                word = toks[-1].lstrip("(\"'[«").lower() if toks else ""
                if word in ABBR or (len(word) == 1 and word.isalpha()):
                    continue
            s = para[start:end].strip()
            if s:
                out.append(s)
            start = end
        tail = para[start:].strip()
        if tail:
            out.append(tail)
    return [s.translate(_UNPROT) for s in out]


def norm_title(t: str) -> str:
    t = " ".join(t.replace("_", " ").split())
    return t[:1].upper() + t[1:]


def extract_links(sentence: str) -> tuple[str, ...]:
    seen: dict[str, None] = {}
    for m in _LINK.finditer(sentence):
        t = m.group(1).split("#", 1)[0].strip()
        if not t or t.startswith(":"):
            continue
        if ":" in t:
            pre = t.split(":", 1)[0].strip().lower()
            if pre in NS_PREFIXES or (pre.isalpha() and len(pre) <= 3):
                continue
        t = norm_title(t)
        if t:
            seen[t] = None
    return tuple(seen)


def _words(s: str) -> list[str]:
    return _WORD.findall(s.lower())


def _is_subseq(small: list[str], big: list[str]) -> bool:
    it = iter(big)
    return all(w in it for w in small)


# --------------------------------------------------------------------------- #
# Pass 2: walk revisions, collect events
# --------------------------------------------------------------------------- #
class _Scorer:
    def __init__(self):
        self.intern: dict[str, int] = {}
        self.sent_text: list[str] = []
        self.sent_links: list[tuple[str, ...]] = []
        self.linkset_idx: dict[frozenset, int] = {}
        self.linksets: list[tuple[str, ...]] = []
        self.events: list[tuple[int, str, int]] = []  # (revid, ts, linkset idx)

    def to_ids(self, body: str) -> list[int]:
        ids = []
        intern, st, sl = self.intern, self.sent_text, self.sent_links
        for s in split_sentences(clean(body)):
            i = intern.get(s)
            if i is None:
                i = len(st)
                intern[s] = i
                st.append(s)
                sl.append(extract_links(s))
            ids.append(i)
        return ids

    def emit(self, links: tuple[str, ...], revid: int, ts: str) -> None:
        if not links:
            return
        links = tuple(dict.fromkeys(links))
        key = frozenset(links)
        idx = self.linkset_idx.get(key)
        if idx is None:
            idx = len(self.linksets)
            self.linkset_idx[key] = idx
            self.linksets.append(links)
        self.events.append((revid, ts, idx))

    def diff(self, a: list[int], b: list[int], revid: int, ts: str) -> None:
        la, lb = len(a), len(b)
        lo = 0
        while lo < la and lo < lb and a[lo] == b[lo]:
            lo += 1
        ha, hb = la, lb
        while ha > lo and hb > lo and a[ha - 1] == b[hb - 1]:
            ha -= 1
            hb -= 1
        A, B = a[lo:ha], b[lo:hb]
        if not A:  # insert only
            return
        sl = self.sent_links
        if not B:  # pure deletion: old links only (D3)
            for o in A:
                self.emit(sl[o], revid, ts)
            return
        for tag, i1, i2, j1, j2 in SequenceMatcher(None, A, B, autojunk=False).get_opcodes():
            if tag == "delete":
                for o in A[i1:i2]:
                    self.emit(sl[o], revid, ts)
            elif tag == "replace":
                olds, news = A[i1:i2], B[j1:j2]
                same_len = len(olds) == len(news)
                block = None
                for k, o in enumerate(olds):
                    partner = news[k] if k < len(news) else None
                    if same_len:
                        ref = _words(self.sent_text[partner])
                    else:
                        if block is None:
                            block = [w for n in news for w in _words(self.sent_text[n])]
                        ref = block
                    if _is_subseq(_words(self.sent_text[o]), ref):  # E2: pure insertion
                        continue
                    links = sl[o] + (sl[partner] if partner is not None else ())  # D3
                    self.emit(links, revid, ts)


def score_article(meta_p: str, text_p: str) -> dict:
    with open(meta_p, "rb") as f:
        meta = [jloads(l) for l in f if l.strip()]
    n = len(meta)
    kept, drop = flag_revisions(meta)
    sc = _Scorer()
    prev_text = None
    prev_secs: dict[str, str] | None = None
    prev_sents: dict[str, list[int]] = {}
    with open(text_p, "rb") as f:
        for i, line in enumerate(f):
            if i >= n:
                break
            if not kept[i]:
                continue
            text = jloads(line)
            if text == prev_text:
                continue
            secs = split_sections(text)
            cur_sents: dict[str, list[int]] = {}
            if prev_secs is not None:
                emit_ok = not drop[i]
                revid, ts = meta[i]["revid"], meta[i]["ts"]
                for key, body in secs.items():
                    pb = prev_secs.get(key)
                    if pb is None:  # new section: ignored (paper: section insert)
                        continue
                    if pb == body:
                        s = prev_sents.get(key)
                        if s is not None:
                            cur_sents[key] = s
                        continue
                    if not emit_ok:
                        continue
                    a = prev_sents.get(key)
                    if a is None:
                        a = sc.to_ids(pb)
                    b = sc.to_ids(body)
                    cur_sents[key] = b
                    if a != b:
                        sc.diff(a, b, revid, ts)
            prev_text, prev_secs, prev_sents = text, secs, cur_sents
    return {
        "n_revisions": n, "n_kept": sum(kept), "n_dropped": sum(1 for i in range(n) if kept[i] and drop[i]),
        "linksets": sc.linksets, "events": sc.events,
    }


# --------------------------------------------------------------------------- #
# Title -> QID resolution
# --------------------------------------------------------------------------- #
def resolve_titles(titles: set[str], lang: str, ua: str, cache_dir: str, workers: int = 3) -> dict:
    cache_p = os.path.join(cache_dir, f"titles_{lang}.json")
    cache: dict = {}
    if os.path.exists(cache_p):
        with open(cache_p, encoding="utf-8") as f:
            cache = json.load(f)
    todo = sorted(t for t in titles if t not in cache)

    def work(batch: list[str]) -> dict:
        d = api_get(lang, dict(
            action="query", titles="|".join(batch), prop="pageprops", ppprop="wikibase_item",
            redirects=1, format="json", formatversion=2), ua)
        q = d.get("query", {})
        norm = {x["from"]: x["to"] for x in q.get("normalized", [])}
        red = {x["from"]: x["to"] for x in q.get("redirects", [])}
        pages = {p["title"]: p for p in q.get("pages", [])}
        res = {}
        for t in batch:
            t1 = norm.get(t, t)
            t2 = red.get(t1, t1)
            p = pages.get(t2)
            qid = p.get("pageprops", {}).get("wikibase_item") if p and not p.get("missing") \
                and not p.get("invalid") else None
            res[t] = [qid, t2] if qid else None  # None = red link / no item
        return res

    batches = [todo[i:i + 50] for i in range(0, len(todo), 50)]
    if batches:
        print(f"[resolve:{lang}] {len(todo)} titles in {len(batches)} calls", file=sys.stderr)
        with ThreadPoolExecutor(max_workers=workers) as ex:
            for k, res in enumerate(ex.map(work, batches), 1):
                cache.update(res)
                if k % 20 == 0 or k == len(batches):
                    with open(cache_p, "w", encoding="utf-8") as f:
                        json.dump(cache, f, ensure_ascii=False)
    return cache


# --------------------------------------------------------------------------- #
# Output
# --------------------------------------------------------------------------- #
def section_scores(annotated_path: str, score: Counter) -> tuple[list[dict], Counter]:
    """Split each QID's C(q) over the sections of the current version, in proportion to
    how many times the QID is mentioned in each section (JSON spans)."""
    with open(annotated_path, encoding="utf-8") as f:
        data = json.load(f)
    sections = data["sections"] if isinstance(data, dict) else data
    mentions = [Counter(sp["qid"] for sp in sec.get("spans", []) if sp.get("qid"))
                for sec in sections]
    totals: Counter = Counter()
    for m in mentions:
        totals.update(m)
    rows = []
    for sec, m in zip(sections, mentions):
        value = sum(score.get(q, 0.0) * n / totals[q] for q, n in m.items())
        n_tokens = len(_WORD.findall(sec.get("text") or ""))  # words of the section text
        rows.append({"section_id": sec.get("section_id"), "section_idx": sec.get("section_idx"),
                     "heading": sec.get("heading"), "section_score": round(value, 6),
                     "section_score_per_token": round(value / n_tokens, 8) if n_tokens else None})
    return rows, totals


def finalize(res: dict, tmap: dict, name: str, out_dir: str, annotated: str | None = None) -> str:
    """tmap: title -> [qid, canonical_title] or None. Writes events + ranking CSVs."""
    ls_q: list[list[tuple[str, float]]] = []
    label: dict[str, str] = {}
    for titles in res["linksets"]:
        qids: dict[str, None] = {}
        for t in titles:
            r = tmap.get(t)
            if r:
                qids[r[0]] = None
                label.setdefault(r[0], r[1])
        w = len(qids)  # F2: distinct QIDs
        ls_q.append([(q, 1.0 / w) for q in qids])
    os.makedirs(out_dir, exist_ok=True)
    score: Counter = Counter()
    n_edits: Counter = Counter()
    with open(os.path.join(out_dir, name + ".events.csv"), "w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(["rev_id", "timestamp", "qid", "weight"])
        for revid, ts, idx in res["events"]:
            for q, w in ls_q[idx]:
                wr.writerow([revid, ts, q, f"{w:.6f}"])
                score[q] += w
                n_edits[q] += 1
    rank_p = os.path.join(out_dir, name + ".ranking.csv")
    with open(rank_p, "w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        sec_rows, totals = section_scores(annotated, score) if annotated else ([], Counter())
        wr.writerow(["rank", "qid", "title", "controversy", "n_edits"]
                    + (["in_current_version"] if annotated else []))
        for r, (q, s) in enumerate(score.most_common(), 1):
            wr.writerow([r, q, label.get(q, ""), f"{s:.4f}", n_edits[q]]
                        + ([q in totals] if annotated else []))
    if annotated:
        with open(os.path.join(out_dir, name + ".sections.csv"), "w", newline="", encoding="utf-8") as f:
            wr = csv.DictWriter(f, fieldnames=["section_id", "section_idx", "heading", "section_score",
                                              "section_score_per_token"])
            wr.writeheader()
            wr.writerows(sec_rows)
        kept = sum(v for q, v in score.items() if q not in totals)
        print(f"[{name}] total C = {sum(score.values()):.2f}; "
              f"{kept:.2f} belongs to QIDs not in the current version (kept in ranking only)",
              file=sys.stderr)
    return rank_p


def _score_job(args):
    name, meta_p, text_p = args
    return name, score_article(meta_p, text_p)


def cmd_score(a) -> None:
    if a.annotated and a.no_resolve:
        raise SystemExit("--annotated needs QIDs: do not combine it with --no-resolve")
    if a.annotated and len(a.annotated) != len(a.articles):
        raise SystemExit("--annotated needs one JSON per article, in the same order")
    jobs, langs = [], {}
    for spec in a.articles:
        lang, title = parse_spec(spec)
        name = safe_name(lang, title)
        jobs.append((name, os.path.join(a.cache_dir, name + ".meta.jsonl"),
                     os.path.join(a.cache_dir, name + ".text.jsonl")))
        langs[name] = lang
    t0 = time.time()
    if len(jobs) == 1 or a.procs == 1:
        results = dict(_score_job(j) for j in jobs)
    else:
        with ProcessPoolExecutor(max_workers=a.procs) as ex:
            results = dict(ex.map(_score_job, jobs))
    print(f"scored {len(jobs)} article(s) in {time.time() - t0:.1f}s", file=sys.stderr)
    by_lang: dict[str, set[str]] = {}
    for name, res in results.items():
        by_lang.setdefault(langs[name], set()).update(t for ls in res["linksets"] for t in ls)
    tmaps = {}
    for lang, titles in by_lang.items():
        if a.no_resolve:  # title-level scores; the "qid" column then holds the title
            tmaps[lang] = {t: [t, t] for t in titles}
        else:
            tmaps[lang] = resolve_titles(titles, lang, a.user_agent, a.cache_dir, a.resolve_workers)
    ann = dict(zip(langs, a.annotated)) if a.annotated else {}
    for name, res in results.items():
        p = finalize(res, tmaps[langs[name]], name, a.out_dir, ann.get(name))
        print(f"{name}: {res['n_revisions']} revs, {res['n_kept']} kept after collapse, "
              f"{res['n_dropped']} vandalism-related, {len(res['events'])} events -> {p}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fetch", help="download revision histories into the local cache")
    f.add_argument("articles", nargs="+", help="e.g. 'en:Falkland Islands'")
    f.add_argument("--cache-dir", default="cache")
    f.add_argument("--user-agent", default=DEFAULT_UA)
    f.add_argument("--workers", type=int, default=2, help="articles fetched in parallel")
    s = sub.add_parser("score", help="compute controversy scores from the cache")
    s.add_argument("articles", nargs="+")
    s.add_argument("--cache-dir", default="cache")
    s.add_argument("--out-dir", default="results")
    s.add_argument("--user-agent", default=DEFAULT_UA)
    s.add_argument("--procs", type=int, default=os.cpu_count() or 1)
    s.add_argument("--resolve-workers", type=int, default=3)
    s.add_argument("--no-resolve", action="store_true", help="skip Wikidata; score per link title")
    s.add_argument("--annotated", nargs="+", help="one extract_qids JSON per article (same order): "
                   "adds <name>.sections.csv with section scores for the current version")
    a = ap.parse_args()
    if a.cmd == "fetch":
        with ThreadPoolExecutor(max_workers=a.workers) as ex:
            list(ex.map(lambda sp: fetch_article(sp, a.cache_dir, a.user_agent), a.articles))
    else:
        cmd_score(a)


if __name__ == "__main__":
    main()