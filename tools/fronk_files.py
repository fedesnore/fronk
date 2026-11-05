"""fronk_files.py - files and photos for the chat, prepared for a 10,240-token model.

Every attachment becomes clean text the model reads natively, once, at upload:
  PDF          PDFKit text layer; scanned pages read by Vision OCR (fronk-extract)
  photo        Vision OCR of the text in it, with its size (no image encoder here)
  Word/RTF/... textutil (macOS)          PowerPoint  slide texts      Excel  rows as "a | b | c"
  text, code, CSV, JSON, Markdown as they are
Cleaning is near lossless: repeated page headers/footers, bare page numbers,
hyphenated line breaks, wrapped lines, dot leaders and runs of blank space go;
words, numbers and code stay. The text is then cut into ~500-token passages.
Each turn gets the best version that fits its budget (context_for): the whole
file when it fits, otherwise its outline plus the passages that best match the
question (BM25, no model), in document order with their page numbers.
"""
import hashlib, html, json, math, os, re, subprocess, unicodedata, zipfile
import xml.etree.ElementTree as ET

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FILES = os.path.join(ROOT, "runs", "chats", "files")
EXTRACT = os.path.join(ROOT, "runs", "bin", "fronk-extract")
EXTRACT_SRC = os.path.join(ROOT, "tools", "native", "extract.swift")
CHARS_PER_TOKEN = 3.5     # this model on mixed Italian/English prose (ds4's counts, BENCH)
PASSAGE = 1800            # characters, ~500 tokens
IMAGES = (".png", ".jpg", ".jpeg", ".heic", ".heif", ".gif", ".tif", ".tiff", ".bmp", ".webp")
TEXTUTIL = (".doc", ".docx", ".rtf", ".rtfd", ".odt", ".html", ".htm", ".webarchive", ".wordml")
CODE = (".c", ".h", ".py", ".js", ".ts", ".swift", ".m", ".go", ".rs", ".java", ".sh", ".css", ".sql", ".yml", ".yaml", ".toml")


def tokens(text):
    return math.ceil(len(text) / CHARS_PER_TOKEN)


def extractor():
    """runs/bin/fronk-extract, built by `make fronk-extract`; rebuilt here only if
    extract.swift is newer (about 40 s)."""
    if not os.path.exists(EXTRACT) or os.path.getmtime(EXTRACT) < os.path.getmtime(EXTRACT_SRC):
        os.makedirs(os.path.dirname(EXTRACT), exist_ok=True)
        subprocess.run(["swiftc", "-O", EXTRACT_SRC, "-o", EXTRACT], check=True, capture_output=True, timeout=600)
    return EXTRACT


def native(kind, path):
    r = subprocess.run([extractor(), kind, path], capture_output=True, timeout=900)
    if r.returncode:
        raise ValueError(r.stderr.decode(errors="replace").strip() or f"cannot read this {kind}")
    return json.loads(r.stdout)


# --- cleaning ---------------------------------------------------------------

def clean_prose(text):
    text = unicodedata.normalize("NFKC", text).replace("\r\n", "\n").replace("\r", "\n").replace("­", "")
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)                   # hyphenated line breaks
    text = re.sub(r"[ \t]*\.{4,}[ \t]*(\d+)", r" \1", text)         # table-of-contents dot leaders
    text = re.sub(r"[_\-=]{6,}", "", text)
    lines = [re.sub(r"[ \t ]+", " ", l).strip() for l in text.split("\n")]
    out = []
    for l in lines:                                                  # join wrapped lines into paragraphs
        if out and l and out[-1] and not re.search(r"[.!?:;)\]»\"]$", out[-1]) and re.match(r"[a-zà-ü(]", l) \
                and not re.match(r"^([-–•●*]|\d+[.)])\s", l):
            out[-1] += " " + l
        else:
            out.append(l)
    return re.sub(r"\n{3,}", "\n\n", "\n".join(out)).strip()


def clean_code(text):
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = "\n".join(l.rstrip() for l in text.split("\n"))
    return re.sub(r"\n{4,}", "\n\n\n", text).strip("\n")


def strip_page_furniture(pages):
    """Drops lines that repeat at the top or bottom of most pages, and bare page numbers."""
    if len(pages) < 3:
        return pages
    edges = {}
    for p in pages:
        ls = [l.strip() for l in p.split("\n") if l.strip()]
        for l in set(ls[:2] + ls[-2:]):
            key = re.sub(r"\d+", "#", l)
            edges[key] = edges.get(key, 0) + 1
    common = {k for k, n in edges.items() if n >= max(3, len(pages) // 2) and len(k) < 120}
    out = []
    for p in pages:
        ls = p.split("\n")
        keep = [l for i, l in enumerate(ls) if not (re.sub(r"\d+", "#", l.strip()) in common and (i < 3 or i >= len(ls) - 3))
                and not re.fullmatch(r"\s*(-\s*)?\d{1,4}(\s*-)?\s*(/\s*\d{1,4})?\s*", l)]
        out.append("\n".join(keep))
    return out


# --- converters -------------------------------------------------------------

def layout(pieces):
    """OCR pieces back into reading rows: pieces whose vertical centers align
    form one row, left to right, with " | " where a wide gap separates columns
    (slides, tables, forms keep their grid instead of interleaving)."""
    rows = []
    for p in sorted(pieces, key=lambda p: (p["y"] + p["h"] / 2, p["x"])):
        mid = p["y"] + p["h"] / 2
        if rows and abs(rows[-1]["mid"] - mid) < max(0.6 * p["h"], 0.006):
            rows[-1]["items"].append(p)
        else:
            rows.append({"mid": mid, "items": [p]})
    out = []
    for r in rows:
        items = sorted(r["items"], key=lambda p: p["x"])
        line = items[0]["t"]
        for a, b in zip(items, items[1:]):
            gap = b["x"] - (a["x"] + a["w"])
            line += (" | " if gap > 1.0 * max(a["h"], b["h"]) else " ") + b["t"]
        out.append(line.strip())
    return [l for l in out if l]

def xml_text(data, tag):
    root = ET.fromstring(data)
    return [el.text or "" for el in root.iter() if el.tag.endswith("}" + tag) or el.tag == tag]


def xlsx(path):
    with zipfile.ZipFile(path) as z:
        shared = []
        if "xl/sharedStrings.xml" in z.namelist():
            root = ET.fromstring(z.read("xl/sharedStrings.xml"))
            for si in root:
                shared.append("".join(t.text or "" for t in si.iter() if t.tag.endswith("}t")))
        names = []
        if "xl/workbook.xml" in z.namelist():
            names = [s.get("name") for s in ET.fromstring(z.read("xl/workbook.xml")).iter() if s.tag.endswith("}sheet")]
        sheets = sorted((n for n in z.namelist() if re.match(r"xl/worksheets/sheet\d+\.xml$", n)), key=lambda n: int(re.findall(r"\d+", n)[-1]))
        parts = []
        for k, name in enumerate(sheets):
            rows = []
            for row in ET.fromstring(z.read(name)).iter():
                if not row.tag.endswith("}row"):
                    continue
                cells = []
                for c in row:
                    if not c.tag.endswith("}c"):
                        continue
                    v = next((x.text for x in c if x.tag.endswith("}v")), None)
                    if v is None:
                        v = "".join(t.text or "" for t in c.iter() if t.tag.endswith("}t"))
                    elif c.get("t") == "s":
                        v = shared[int(v)] if int(v) < len(shared) else v
                    cells.append(v.strip())
                while cells and not cells[-1]:
                    cells.pop()
                if cells:
                    rows.append(" | ".join(cells))
            title = names[k] if k < len(names) else f"Sheet {k + 1}"
            parts.append(f"## {title} ({len(rows)} rows)\n" + "\n".join(rows))
        return "\n\n".join(parts)


def pptx(path):
    with zipfile.ZipFile(path) as z:
        slides = sorted((n for n in z.namelist() if re.match(r"ppt/slides/slide\d+\.xml$", n)), key=lambda n: int(re.findall(r"\d+", n)[-1]))
        return [" ".join(t for t in xml_text(z.read(n), "t") if t.strip()) for n in slides]


def convert(name, path):
    """Returns the prepared document: kind, the cleaned text in units (pages, slides
    or one), and a one-line description of what was done."""
    ext = os.path.splitext(name)[1].lower()
    if ext == ".pdf":
        d = native("pdf", path)
        pages = strip_page_furniture([clean_prose(p) for p in d["pages"]])
        how = f"PDF, {len(pages)} pages" + (f", {len(d['ocr'])} scanned pages read by OCR" if d["ocr"] else "")
        return "pdf", [("p. %d" % (i + 1), p) for i, p in enumerate(pages)], how
    if ext in IMAGES:
        d = native("image", path)
        rows = layout(d["pieces"])
        how = f"photo, {d['width']}×{d['height']}, " + (f"text read by OCR, {len(rows)} rows (cells of one row joined by |)" if rows else "no text found in it")
        return "image", [("", "\n".join(rows))], how
    if ext in TEXTUTIL:
        r = subprocess.run(["textutil", "-convert", "txt", "-stdout", path], capture_output=True, timeout=300)
        if r.returncode:
            raise ValueError("textutil cannot read this document")
        return "document", [("", clean_prose(r.stdout.decode("utf-8", errors="replace")))], "document converted with textutil"
    if ext == ".pptx":
        slides = [clean_prose(s) for s in pptx(path)]
        return "slides", [("slide %d" % (i + 1), s) for i, s in enumerate(slides)], f"presentation, {len(slides)} slides"
    if ext in (".xlsx", ".xlsm"):
        return "sheet", [("", xlsx(path))], "spreadsheet, rows as a | b | c"
    raw = open(path, "rb").read()
    if b"\0" in raw[:4096]:
        raise ValueError("this file type cannot be read as text")
    text = raw.decode("utf-8", errors="replace")
    if ext in (".md", ".txt", ".rst", ""):
        return "text", [("", clean_prose(text))], "text"
    if ext in (".csv", ".tsv", ".json", ".xml") or ext in CODE:
        return "code", [("", clean_code(text))], "code" if ext in CODE else ext[1:].upper()
    return "text", [("", clean_prose(text))], "text"


# --- passages and retrieval -------------------------------------------------

def passages(units):
    """~PASSAGE-character passages along paragraph boundaries, each with its location."""
    out = []
    for label, text in units:
        buf = ""
        for para in re.split(r"\n\s*\n", text):
            if buf and len(buf) + len(para) > PASSAGE:
                out.append((label, buf.strip())); buf = ""
            while len(para) > PASSAGE * 1.5:                         # one huge paragraph
                cut = para.rfind(" ", 0, PASSAGE)
                cut = cut if cut > PASSAGE // 2 else PASSAGE
                out.append((label, (buf + para[:cut]).strip())); buf = ""; para = para[cut:]
            buf += para + "\n\n"
        if buf.strip():
            out.append((label, buf.strip()))
    return [{"at": l, "text": t} for l, t in out if t]


def words(text):
    t = unicodedata.normalize("NFKD", text.lower())
    t = "".join(c for c in t if not unicodedata.combining(c))
    return [w for w in re.findall(r"[a-z0-9]+", t) if len(w) > 1 and w not in STOP]


STOP = set("""the and for that with this from are was were have has had not but you your they their them what which who
when where how why can will would should could into than then there here also about over such only its it's is be
of to in on at by or an as if do does did so we our il lo la le gli un una uno di da del della dei delle che non per
con su tra fra come sono era essere ho hai ha abbiamo hanno questo questa questi quello quella nel nella nei alla
alle al ai dal dalla anche ma piu più se ci si mi ti vi ne""".split())


def bm25(query, docs, k1=1.4, b=0.75):
    q = set(words(query))
    toks = [words(d) for d in docs]
    n = len(docs) or 1
    avg = sum(map(len, toks)) / n or 1
    df = {w: sum(1 for t in toks if w in t) for w in q}
    scores = []
    for t in toks:
        tf = {}
        for w in t:
            if w in q:
                tf[w] = tf.get(w, 0) + 1
        s = sum(math.log(1 + (n - df[w] + 0.5) / (df[w] + 0.5)) * f * (k1 + 1) / (f + k1 * (1 - b + b * len(t) / avg)) for w, f in tf.items())
        scores.append(s)
    return scores


# --- store ------------------------------------------------------------------

def add(name, data):
    """Stores and prepares an upload; returns its card (id, name, kind, how, tokens)."""
    fid = hashlib.sha256(data).hexdigest()[:20]
    os.makedirs(FILES, exist_ok=True)
    meta = os.path.join(FILES, fid + ".json")
    if os.path.exists(meta):                                          # the same bytes were prepared before
        doc = json.load(open(meta))
        return card(doc)
    safe = re.sub(r"[^\w.\- ]", "_", os.path.basename(name))[:120] or "file"
    path = os.path.join(FILES, fid + os.path.splitext(safe)[1].lower())
    with open(path, "wb") as f:
        f.write(data)
    kind, units, how = convert(safe, path)
    text = "\n\n".join((f"[{l}]\n" if l else "") + t for l, t in units if t.strip())
    outline = [l for l in text.split("\n") if len(l.strip()) <= 80 and re.match(r"^(#{1,3} |\d+(\.\d+)*\.? [A-ZÀ-Ü]|[A-ZÀ-Ü][A-ZÀ-Ü0-9 ,'’\-]{6,60}$)", l.strip())][:40]
    doc = {"id": fid, "name": safe, "kind": kind, "how": how, "bytes": len(data), "text": text,
           "outline": outline, "passages": passages(units), "tokens": tokens(text)}
    tmp = meta + ".tmp"
    json.dump(doc, open(tmp, "w")); os.replace(tmp, meta)
    return card(doc)


def card(doc):
    return {k: doc[k] for k in ("id", "name", "kind", "how", "bytes", "tokens")}


def load(fid):
    if not re.fullmatch(r"[0-9a-f]{20}", fid or ""):
        raise ValueError("bad file id")
    return json.load(open(os.path.join(FILES, fid + ".json")))


def context_for(fids, query, budget, sent=None):
    """The text to put before the user's message. sent: what this chat's history
    already holds, {file id: "whole" or [passage indexes]}; nothing is sent twice.
    Every file goes whole when the new material fits the budget (tokens);
    otherwise each file gets its outline (first time only) and its best new
    passages for the question. Returns the text and, per file, what it added."""
    sent = sent or {}
    docs = [d for d in (load(f) for f in fids) if sent.get(d["id"]) != "whole"]
    blocks, report = [], []
    fresh = lambda d: [i for i in range(len(d["passages"])) if i not in set(sent.get(d["id"], []))]
    whole = sum(d["tokens"] + 40 for d in docs if not sent.get(d["id"]))
    for d in docs:
        first = not sent.get(d["id"])
        share = budget if len(docs) == 1 else budget // len(docs)
        if first and (whole <= budget or d["tokens"] + 40 <= share):
            blocks.append(f'<file name="{d["name"]}" kind="{d["kind"]}" note="{d["how"]}; complete">\n{d["text"] or "(no text)"}\n</file>')
            report.append({"id": d["id"], "name": d["name"], "mode": "whole", "tokens": d["tokens"], "sent": "whole"}); continue
        ps, todo = d["passages"], fresh(d)
        scores = bm25(query, [ps[i]["text"] for i in todo])
        head = ("Outline: " + " / ".join(d["outline"][:20]) + "\n\n") if first and d["outline"] else ""
        room = share - tokens(head) - 60
        order = sorted(range(len(todo)), key=lambda k: -scores[k]) if any(scores) else (range(len(todo)) if first else [])
        chosen = []
        for k in order:
            if any(scores) and scores[k] <= 0:
                break
            t = tokens(ps[todo[k]]["text"]) + 4
            if t <= room:
                chosen.append(todo[k]); room -= t
            if room < 80:
                break
        if not chosen and not head:
            continue
        chosen.sort()
        body = "\n\n".join((f"[{ps[i]['at']}] " if ps[i]["at"] else "") + ps[i]["text"] for i in chosen)
        what = "the passages that best match the question" if first else "more passages that match this question"
        blocks.append(f'<file name="{d["name"]}" kind="{d["kind"]}" note="{d["how"]}; about {d["tokens"]} tokens in all, '
                      f'here {len(chosen)} of {len(ps)} passages, {what}">\n{head}{body}\n</file>')
        report.append({"id": d["id"], "name": d["name"], "mode": "passages", "chosen": len(chosen), "of": len(ps),
                       "tokens": tokens(head + body), "sent": chosen})
    text = "\n\n".join(blocks)
    return {"text": text, "tokens": tokens(text), "files": report}
