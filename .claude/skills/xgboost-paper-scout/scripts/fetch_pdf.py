#!/usr/bin/env python3
"""
fetch_pdf.py — download open-access PDFs for shortlisted papers.

Standard library only.

Policy: **open access only.** This script follows the OA link that OpenAlex or
arXiv already published, or resolves a DOI through Unpaywall. It does not, and
must not be extended to, bypass paywalls, use institutional-proxy credentials,
or scrape publisher pages that decline automated access. If a paper has no OA
copy the script records `paywalled` and moves on — read it through your
university library instead.

Usage
-----
  python3 fetch_pdf.py --from-json papers/candidates.json --top 5
  python3 fetch_pdf.py --url https://arxiv.org/pdf/2501.01234 --name my-paper
  python3 fetch_pdf.py --from-json papers/candidates.json --top 10 --out-dir papers

Outputs
-------
  <out-dir>/pdf/<slug>.pdf
  <out-dir>/MANIFEST.md     — what was fetched, from where, and what failed
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import date
from pathlib import Path

UA = "xgboost-paper-scout/1.0 (academic use; contact via --mailto)"
PDF_MAGIC = b"%PDF"


def slugify(title: str, year: str | int | None, limit: int = 70) -> str:
    s = re.sub(r"[^a-zA-Z0-9]+", "-", (title or "paper").lower()).strip("-")
    s = re.sub(r"-+", "-", s)[:limit].strip("-")
    return f"{year}-{s}" if year else s


def http(url: str, mailto: str, timeout: int = 90) -> tuple[bytes | None, str]:
    """Return (body, note). Never raises."""
    req = urllib.request.Request(url, headers={
        "User-Agent": UA.replace("--mailto", mailto),
        "Accept": "application/pdf,text/html,*/*",
    })
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.read(), f"HTTP {r.status}"
    except urllib.error.HTTPError as e:
        return None, f"HTTP {e.code}"
    except Exception as e:  # noqa: BLE001
        return None, type(e).__name__


def unpaywall_pdf(doi: str, mailto: str) -> str | None:
    """Ask Unpaywall for a legal OA copy of a DOI."""
    if not doi:
        return None
    url = (f"https://api.unpaywall.org/v2/{urllib.parse.quote(doi)}"
           f"?email={urllib.parse.quote(mailto)}")
    body, _ = http(url, mailto, timeout=30)
    if not body:
        return None
    try:
        d = json.loads(body)
    except json.JSONDecodeError:
        return None
    loc = d.get("best_oa_location") or {}
    return loc.get("url_for_pdf") or loc.get("url") or None


def arxiv_pdf_url(arxiv_id: str) -> str | None:
    return f"https://arxiv.org/pdf/{arxiv_id}" if arxiv_id else None


def candidate_urls(paper: dict, mailto: str) -> list[tuple[str, str]]:
    """Ordered (label, url) attempts, most reliable first."""
    out: list[tuple[str, str]] = []
    if paper.get("arxiv_id"):
        out.append(("arxiv", arxiv_pdf_url(paper["arxiv_id"])))
    if paper.get("pdf_url"):
        out.append(("openalex-oa", paper["pdf_url"]))
    doi = paper.get("doi") or ""
    if doi:
        up = unpaywall_pdf(doi, mailto)
        if up:
            out.append(("unpaywall", up))
    seen, uniq = set(), []
    for label, url in out:
        if url and url not in seen:
            seen.add(url)
            uniq.append((label, url))
    return uniq


def fetch_one(paper: dict, out_dir: Path, mailto: str) -> dict:
    title = paper.get("title") or "untitled"
    slug = slugify(title, (paper.get("date") or "")[:4])
    dest = out_dir / "pdf" / f"{slug}.pdf"
    record = {
        "title": title,
        "date": paper.get("date", ""),
        "venue": paper.get("venue", ""),
        "doi": paper.get("doi", ""),
        "arxiv_id": paper.get("arxiv_id", ""),
        "citations": paper.get("citations", 0),
        "file": "",
        "source": "",
        "status": "",
        "notes": [],
    }

    if dest.exists() and dest.stat().st_size > 4096:
        record.update(file=str(dest), source="cache", status="ok")
        return record

    for label, url in candidate_urls(paper, mailto):
        body, note = http(url, mailto)
        if not body:
            record["notes"].append(f"{label}: {note}")
            continue
        if not body.startswith(PDF_MAGIC):
            # Publisher served an HTML interstitial / consent wall, not a PDF.
            record["notes"].append(f"{label}: not-a-pdf ({len(body)}B)")
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(body)
        record.update(file=str(dest), source=label, status="ok")
        record["notes"].append(f"{label}: {len(body) // 1024} KB")
        return record
        # (loop continues to the next candidate only on failure)

    record["status"] = "paywalled" if not record["notes"] else "failed"
    return record


def write_manifest(records: list[dict], path: Path, out_dir: Path) -> None:
    ok = [r for r in records if r["status"] == "ok"]
    bad = [r for r in records if r["status"] != "ok"]
    lines = [
        "# Downloaded papers",
        "",
        f"Fetched {len(ok)} of {len(records)} attempted on {date.today()}. "
        f"PDFs are in `{out_dir / 'pdf'}`.",
        "",
        "Open access only — no paywall circumvention. Anything listed as "
        "`paywalled` below needs your university library.",
        "",
        "## Retrieved",
        "",
        "| File | Date | Venue | Cites | Via |",
        "|------|------|-------|-------|-----|",
    ]
    for r in ok:
        fname = Path(r["file"]).name
        lines.append(
            f"| [`{fname}`](pdf/{fname}) | {r['date']} "
            f"| {(r['venue'] or 'preprint')[:32]} | {r['citations']} | {r['source']} |")
    lines += ["", f"### Titles", ""]
    for r in ok:
        ref = r["doi"] or r["arxiv_id"] or ""
        lines.append(f"- **{r['title']}**" + (f"  \n  `{ref}`" if ref else ""))

    if bad:
        lines += ["", "## Not retrieved", "",
                  "| Title | Status | Attempts |", "|-------|--------|----------|"]
        for r in bad:
            lines.append(
                f"| {r['title'][:64].replace('|', '/')} | {r['status']} "
                f"| {'; '.join(r['notes'])[:80] or 'no OA link published'} |")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--from-json", help="candidates.json produced by find_papers.py")
    ap.add_argument("--top", type=int, default=5, help="how many to fetch (default 5)")
    ap.add_argument("--url", help="fetch a single explicit PDF URL instead")
    ap.add_argument("--name", help="filename stem when using --url")
    ap.add_argument("--out-dir", default="papers", help="output dir (default papers/)")
    ap.add_argument("--mailto", default="anonymous@example.com",
                    help="contact email — required by the Unpaywall API")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.url:
        papers = [{"title": args.name or "manual-download", "pdf_url": args.url,
                   "date": date.today().isoformat()}]
    elif args.from_json:
        src = Path(args.from_json)
        if not src.exists():
            print(f"not found: {src}. Run find_papers.py first.", file=sys.stderr)
            return 2
        papers = json.loads(src.read_text())["papers"][:args.top]
    else:
        print("pass --from-json or --url", file=sys.stderr)
        return 2

    print(f"[fetch] {len(papers)} paper(s) -> {out_dir / 'pdf'}")
    records = []
    for i, p in enumerate(papers, 1):
        print(f"  {i}/{len(papers)} {p.get('title', '?')[:64]}")
        rec = fetch_one(p, out_dir, args.mailto)
        icon = "ok " if rec["status"] == "ok" else "-- "
        print(f"      {icon}{rec['status']}"
              + (f" via {rec['source']}" if rec["source"] else "")
              + (f"  [{'; '.join(rec['notes'])}]" if rec["notes"] else ""))
        records.append(rec)
        time.sleep(1)

    write_manifest(records, out_dir / "MANIFEST.md", out_dir)
    n_ok = sum(1 for r in records if r["status"] == "ok")
    print(f"\n[done] {n_ok}/{len(records)} retrieved")
    print(f"[done] {out_dir / 'MANIFEST.md'}")
    if n_ok < len(records):
        print("       Paywalled items need your university library — this tool "
              "will not work around access controls.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
