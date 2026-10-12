#!/usr/bin/env python3
"""
This needs both extract_qids.py and controversy_qid_with_sections_simple.py
The articles which we want to extract are defined in the PAIRS variable.

Workflow for the 9 article pairs (en + es):
  1. extract  extract_qids.py            -> annotated/<name>.json   (sections, QIDs, aliases)
  2. fetch    controversy_qid_with_sections_simple.py fetch   -> cache/   (full histories)
  3. score    controversy_qid_with_sections_simple.py score   -> results/<name>.sections.csv
                                                                 results/<name>.ranking.csv
                                                                 results/<name>.events.csv
  4. combine  all section tables in one file                  -> results/all_sections.csv

Put this file in the same folder as extract_qids.py and
controversy_qid_with_sections_simple.py, then run:

    python run_workflow.py                      # everything (fetching big histories takes a while)
    python run_workflow.py --steps score combine   # re-run only some steps
    python run_workflow.py --only 1 2           # only pairs 1 and 2
    python run_workflow.py --force-extract      # rebuild the JSONs

Every step can be re-run: JSONs are skipped if they exist, history downloads resume,
and Wikidata lookups are cached.
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import unquote, urlparse

HERE = Path(__file__).resolve().parent
SCORER = "controversy_qid_with_sections_simple"      # name of your scoring file (no .py)

PAIRS = [
    ("Falkland Islands", [
        "https://en.wikipedia.org/wiki/Falkland_Islands",
        "https://es.wikipedia.org/wiki/Islas_Malvinas_(territorio_brit%C3%A1nico_de_ultramar)"]),
    ("Status of Gibraltar", [
        "https://en.wikipedia.org/wiki/Status_of_Gibraltar",
        "https://es.wikipedia.org/wiki/Disputa_territorial_de_Gibraltar"]),
    ("Guantanamo Bay Naval Base", [
        "https://en.wikipedia.org/wiki/Guantanamo_Bay_Naval_Base",
        "https://es.wikipedia.org/wiki/Base_Naval_de_la_Bah%C3%ADa_de_Guant%C3%A1namo"]),
    ("Belizean-Guatemalan territorial dispute", [
        "https://en.wikipedia.org/wiki/Belizean%E2%80%93Guatemalan_territorial_dispute",
        "https://es.wikipedia.org/wiki/Diferendo_territorial_entre_Belice_y_Guatemala"]),
    ("Bajo Nuevo Bank", [
        "https://en.wikipedia.org/wiki/Bajo_Nuevo_Bank",
        "https://es.wikipedia.org/wiki/Bajo_Nuevo"]),
    ("San Andres and Providencia", [
        "https://en.wikipedia.org/wiki/San_Andr%C3%A9s_and_Providencia",
        "https://es.wikipedia.org/wiki/Archipi%C3%A9lago_de_San_Andr%C3%A9s,_Providencia_y_Santa_Catalina"]),
    ("Serranilla Bank", [
        "https://en.wikipedia.org/wiki/Serranilla_Bank",
        "https://es.wikipedia.org/wiki/Isla_Serranilla"]),
    ("Geography of Puerto Rico", [
        "https://en.wikipedia.org/wiki/Geography_of_Puerto_Rico",
        "https://es.wikipedia.org/wiki/Geograf%C3%ADa_de_Puerto_Rico"]),
    ("South Georgia and the South Sandwich Islands", [
        "https://en.wikipedia.org/wiki/South_Georgia_and_the_South_Sandwich_Islands",
        "https://es.wikipedia.org/wiki/Islas_Georgias_del_Sur_y_Sandwich_del_Sur"]),
]

COMBINED_COLUMNS = ["pair_no", "pair", "lang", "title", "article_id", "section_id",
                    "section_idx", "heading", "section_score", "section_score_per_token"]


@dataclass
class Article:
    pair_no: int
    pair: str
    lang: str
    url: str
    title: str = ""        # canonical title (spaces), filled by resolve_titles
    name: str = ""         # file stem used by the scorer, e.g. "es_Islas_Malvinas_..."
    json_path: Path | None = None

    @property
    def spec(self) -> str:
        return f"{self.lang}:{self.title}"


def url_parts(url: str) -> tuple[str, str]:
    p = urlparse(url)
    return p.netloc.split(".")[0], unquote(p.path.split("/wiki/", 1)[1])


def build_articles(only: list[int] | None) -> list[Article]:
    arts = []
    for no, (pair, urls) in enumerate(PAIRS, 1):
        if only and no not in only:
            continue
        for url in urls:
            lang, _ = url_parts(url)
            arts.append(Article(no, pair, lang, url))
    return arts


# --------------------------------------------------------------------------- #
# Canonical titles (the URL may be a redirect; histories belong to the target page)
# --------------------------------------------------------------------------- #
def resolve_titles(arts: list[Article], cache_file: Path) -> None:
    cache = json.loads(cache_file.read_text(encoding="utf-8")) if cache_file.exists() else {}
    api = None
    for a in arts:
        if a.url not in cache:
            if api is None:
                from utilities.extract_qids import api_get, wiki_api_url  # needs network, imported lazily
                api = (api_get, wiki_api_url)
            _, raw = url_parts(a.url)
            d = api[0](api[1](a.lang), {"action": "query", "titles": raw.replace("_", " "),
                                        "redirects": "1", "format": "json", "formatversion": "2"})
            page = d["query"]["pages"][0]
            if page.get("missing") or page.get("invalid"):
                raise SystemExit(f"Page not found: {a.url}")
            cache[a.url] = page["title"]
            if page["title"].replace(" ", "_") != raw:
                print(f"  note: {a.url.split('/wiki/')[1]} redirects to '{page['title']}'")
    cache_file.write_text(json.dumps(cache, ensure_ascii=False, indent=1), encoding="utf-8")

    sys.path.insert(0, str(HERE))
    scorer = __import__(SCORER)
    for a in arts:
        a.title = cache[a.url]
        a.name = scorer.safe_name(a.lang, a.title)


# --------------------------------------------------------------------------- #
# Steps
# --------------------------------------------------------------------------- #
def step_extract(arts: list[Article], force: bool) -> None:
    from utilities.extract_qids import extract_qids  # lazy: pulls in requests / bs4 / nltk
    for a in arts:
        if a.json_path.exists() and not force:
            print(f"[extract] skip (exists): {a.json_path.name}")
            continue
        print(f"[extract] {a.spec}")
        try:
            extract_qids(a.lang, a.title.replace(" ", "_"),
                         output_path=str(a.json_path), verbose=False)
        except Exception as e:  # keep going with the other articles
            print(f"[extract] FAILED {a.spec}: {e}")


def run_scorer(command: str, args: list[str]) -> None:
    cmd = [sys.executable, str(HERE / f"{SCORER}.py"), command, *args]
    print("[run]", command, f"({sum(1 for x in args if ':' in x and x[:3] in ('en:', 'es:'))} articles)")
    subprocess.run(cmd, check=True)


def with_json(arts: list[Article]) -> list[Article]:
    ok = [a for a in arts if a.json_path.exists()]
    for a in arts:
        if a not in ok:
            print(f"[warn] no JSON for {a.spec}, skipped (run the extract step)")
    return ok


def step_fetch(arts: list[Article], a_args) -> None:
    arts = with_json(arts)
    run_scorer("fetch", [x.spec for x in arts] + [
        "--cache-dir", a_args.cache_dir, "--user-agent", a_args.user_agent,
        "--workers", str(a_args.fetch_workers)])


def step_score(arts: list[Article], a_args) -> None:
    arts = with_json(arts)
    run_scorer("score", [x.spec for x in arts] + [
        "--annotated", *[str(x.json_path) for x in arts],
        "--cache-dir", a_args.cache_dir, "--out-dir", a_args.out_dir,
        "--user-agent", a_args.user_agent, "--procs", str(a_args.procs)])


def step_combine(arts: list[Article], out_dir: Path) -> Path:
    rows, missing = [], []
    for a in arts:
        p = out_dir / f"{a.name}.sections.csv"
        if not p.exists():
            missing.append(a.spec)
            continue
        with open(p, newline="", encoding="utf-8-sig") as f:
            for r in csv.DictReader(f):
                rows.append({"pair_no": a.pair_no, "pair": a.pair, "lang": a.lang,
                             "title": a.title, "article_id": r["section_id"].rsplit(":", 1)[0],
                             **{k: r.get(k, "") for k in COMBINED_COLUMNS[5:]}})
    out = out_dir / "all_sections.csv"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=COMBINED_COLUMNS)
        w.writeheader()
        w.writerows(rows)
    print(f"[combine] {len(rows)} section rows from {len(arts) - len(missing)} articles -> {out}")
    for m in missing:
        print(f"[combine] missing sections file for {m}")
    return out


# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--steps", nargs="+", default=["extract", "fetch", "score", "combine"],
                    choices=["extract", "fetch", "score", "combine"])
    ap.add_argument("--only", nargs="+", type=int, help="pair numbers to run, e.g. --only 1 2")
    ap.add_argument("--force-extract", action="store_true", help="rebuild existing JSONs")
    ap.add_argument("--annotated-dir", default="annotated")
    ap.add_argument("--cache-dir", default="cache")
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--user-agent", default=None,
                    help="default: the User-Agent defined in extract_qids.py")
    ap.add_argument("--fetch-workers", type=int, default=2, help="histories downloaded in parallel")
    ap.add_argument("--procs", type=int, default=4, help="articles scored in parallel")
    a = ap.parse_args()

    if a.user_agent is None:
        sys.path.insert(0, str(HERE))
        from utilities.extract_qids import HEADERS
        a.user_agent = HEADERS["User-Agent"]

    arts = build_articles(a.only)
    ann_dir, out_dir = Path(a.annotated_dir), Path(a.out_dir)
    ann_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    resolve_titles(arts, ann_dir / "canonical_titles.json")
    for x in arts:
        x.json_path = ann_dir / f"{x.name}.json"

    if "extract" in a.steps:
        step_extract(arts, a.force_extract)
    if "fetch" in a.steps:
        step_fetch(arts, a)
    if "score" in a.steps:
        step_score(arts, a)
    if "combine" in a.steps:
        step_combine(arts, out_dir)


if __name__ == "__main__":
    main()