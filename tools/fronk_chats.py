"""fronk_chats.py - the user's chats: a folder they own, and search across it.

~/Documents/fronk/chats (FRONK_CHATS overrides) holds one JSON per chat, which
the chat page reads and writes, and next to it the same chat as Markdown, to
open, read or share without fronk. index.sqlite is a SQLite FTS5 index of every
message (BM25 ranking, Unicode-aware, accents ignored), rebuilt from the JSON
files whenever one is newer than its entry; the files are the truth.
Topics are each chat's most distinctive words (TF-IDF against all chats).
"""
import json, math, os, re, sqlite3, threading, time, unicodedata
from collections import Counter

DIR = os.environ.get("FRONK_CHATS") or os.path.expanduser("~/Documents/fronk/chats")
INDEX = os.path.join(DIR, "index.sqlite")
LOCK = threading.Lock()
STOP = set("""the and for that with this from are was were have has had not but you your they their them what which who when
where how why can will would should could into than then there here also about over such only its is be of to in on at by or an
as if do does did so we our it il lo la le gli un una uno di da del della dei delle che non per con su tra fra come sono era
essere ho hai ha abbiamo hanno questo questa questi quello quella nel nella nei alla alle al ai dal dalla anche ma più piu se
ci si mi ti vi ne puoi può posso fammi dimmi cosa quali quale sei mio mia tuo tua empty thinking""".split())


def path(cid, ext=".json"):
    if not re.fullmatch(r"[0-9a-z]{6,40}", cid or ""):
        raise ValueError("bad chat id")
    return os.path.join(DIR, cid + ext)


def words(text):
    t = "".join(c for c in unicodedata.normalize("NFKD", text.lower()) if not unicodedata.combining(c))
    return [w for w in re.findall(r"[a-z][a-z0-9]{2,}", t) if w not in STOP]


def answer(text):
    """An assistant message without the thinking quote the page shows above it."""
    return text[text.index("\n\n") + 2:] if text.startswith("> **Thinking**") and "\n\n" in text else text


def markdown(chat):
    out = [f"# {chat.get('title') or 'Untitled'}", "",
           f"_{time.strftime('%Y-%m-%d %H:%M', time.localtime(chat.get('updated', time.time())))} · fronk_", ""]
    if chat.get("topics"):
        out += ["Topics: " + ", ".join(chat["topics"]), ""]
    for m in chat.get("messages", []):
        who = "**You**" if m.get("role") == "user" else "**fronk**"
        files = "".join(f"\n\n📎 {a.get('name')}" for a in m.get("attachments", []))
        out += [who, "", (m.get("text") if m.get("role") == "user" else answer(m.get("text", ""))) + files, ""]
    return "\n".join(out)


def db():
    os.makedirs(DIR, exist_ok=True)
    c = sqlite3.connect(INDEX, timeout=10)
    c.execute("create table if not exists chats(id text primary key, title text, updated real, topics text, mtime real)")
    c.execute("create virtual table if not exists msgs using fts5(title, topics, text, chat unindexed, idx unindexed, role unindexed,"
              " tokenize='unicode61 remove_diacritics 2')")
    return c


def topics(chat, corpus_df, n_docs, k=6):
    tf = Counter(words(chat.get("title", "") + " " + " ".join(
        m.get("text", "") if m.get("role") == "user" else answer(m.get("text", "")) for m in chat.get("messages", []))))
    score = {w: f * math.log((1 + n_docs) / (1 + corpus_df.get(w, 0))) for w, f in tf.items() if f > 1 or n_docs < 3}
    return [w for w, _ in sorted(score.items(), key=lambda x: -x[1])[:k]]


def sync():
    """Brings the index up to date with the JSON files (new, changed, deleted)."""
    with LOCK:
        c = db()
        files = {n[:-5]: os.path.getmtime(os.path.join(DIR, n)) for n in os.listdir(DIR) if n.endswith(".json")}
        known = dict(c.execute("select id, mtime from chats"))
        stale = [cid for cid, mt in files.items() if known.get(cid) != mt]
        gone = [cid for cid in known if cid not in files]
        if not stale and not gone:
            c.close(); return
        chats = {}
        for cid in files:
            try:
                chats[cid] = json.load(open(path(cid)))
            except (OSError, ValueError):
                continue
        df, n = Counter(), len(chats)
        for ch in chats.values():
            df.update(set(words(" ".join(m.get("text", "") for m in ch.get("messages", [])))))
        for cid in gone + stale:
            c.execute("delete from chats where id=?", (cid,)); c.execute("delete from msgs where chat=?", (cid,))
        for cid in stale:
            ch = chats.get(cid)
            if not ch:
                continue
            tp = topics(ch, df, n)
            ch["topics"] = tp
            c.execute("insert into chats values(?,?,?,?,?)", (cid, ch.get("title", ""), ch.get("updated", 0), " ".join(tp), files[cid]))
            for i, m in enumerate(ch.get("messages", [])):
                text = m.get("text", "") if m.get("role") == "user" else answer(m.get("text", ""))
                text += "".join(" " + a.get("name", "") for a in m.get("attachments", []))
                c.execute("insert into msgs(title, topics, text, chat, idx, role) values(?,?,?,?,?,?)",
                          (ch.get("title", ""), " ".join(tp), text, cid, i, m.get("role", "")))
            with open(path(cid, ".md"), "w") as f:
                f.write(markdown(ch))
        c.commit(); c.close()


def save(cid, chat):
    chat["id"], chat["updated"] = cid, time.time()
    os.makedirs(DIR, exist_ok=True)
    with open(path(cid) + ".tmp", "w") as f:
        json.dump(chat, f)
    os.replace(path(cid) + ".tmp", path(cid))
    sync()


def listing():
    sync()
    c = db()
    rows = c.execute("select id, title, updated, topics from chats order by updated desc").fetchall()
    c.close()
    return [{"id": r[0], "title": r[1], "updated": r[2], "topics": r[3].split()} for r in rows]


def delete(cid):
    """Moves the chat's files to DIR/deleted (nothing is erased)."""
    os.makedirs(os.path.join(DIR, "deleted"), exist_ok=True)
    for ext in (".json", ".md"):
        if os.path.exists(path(cid, ext)):
            os.replace(path(cid, ext), os.path.join(DIR, "deleted", cid + ext))
    sync()


def search(query, limit=30):
    """Best matching messages (BM25 over title x8, topics x4, text x1), one row per
    message with a snippet; words match by prefix, accents ignored."""
    terms = re.findall(r"\w+", unicodedata.normalize("NFKC", query))
    if not terms:
        return []
    sync()
    match = " ".join('"' + t.replace('"', '') + '"*' for t in terms[:12])
    c = db()
    try:
        rows = c.execute("select chat, idx, role, title, snippet(msgs, 2, '[', ']', ' … ', 14), bm25(msgs, 8.0, 4.0, 1.0) as s "
                         "from msgs where msgs match ? order by s limit ?", (match, limit)).fetchall()
        if not rows and len(terms) > 1:                                # no message has every word: any of them
            rows = c.execute("select chat, idx, role, title, snippet(msgs, 2, '[', ']', ' … ', 14), bm25(msgs, 8.0, 4.0, 1.0) as s "
                             "from msgs where msgs match ? order by s limit ?", (match.replace('" "', '" OR "'), limit)).fetchall()
    finally:
        c.close()
    return [{"chat": r[0], "idx": int(r[1]), "role": r[2], "title": r[3], "snippet": r[4]} for r in rows]


def as_document(cid):
    """A past chat as Markdown bytes, to attach to another chat like a file."""
    chat = json.load(open(path(cid)))
    return (re.sub(r"[^\w\- ]", "", chat.get("title") or "chat")[:60] or "chat") + ".md", markdown(chat).encode()


def migrate(old_dir):
    """Moves chats saved by the first version (runs/chats) into DIR, once."""
    if not os.path.isdir(old_dir):
        return
    os.makedirs(DIR, exist_ok=True)
    for n in os.listdir(old_dir):
        if n.endswith(".json") and not os.path.exists(os.path.join(DIR, n)):
            os.replace(os.path.join(old_dir, n), os.path.join(DIR, n))
