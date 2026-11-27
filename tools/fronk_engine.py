#!/usr/bin/env python3
"""fronk_engine.py -- ENGINE [arguments without --port]

The one supervisor of the head's dense engine (llama.cpp with the Fronk hook).

    chat client --HTTP--> this program --HTTP--> ONE engine --moe.sock--> fronk head --> nodes

What it does, and nothing else:
  * serves the OpenAI API on 127.0.0.1:FRONK_DS4_PORT and on the private cable
    address in state/gsq-listen; requests run one at a time, later ones queue;
  * reads the harness's system prompt and tools once at start (state/
    study-prefix-request.json, refreshed whenever a chat brings a new prefix),
    so every chat begins after them;
  * keeps chats in RAM (the engine's slot and its --cache-ram). The SSD is
    written only in an emergency: a stop while a chat is loaded, or memory
    running out. That one file is restored when the same chat returns;
  * watches memory and never lets a chat push the Mac into swap: it stops
    cleanly (chat saved) before that;
  * reports everything on GET /fronk/status: phase, speeds, memory, events.

The engine itself never dies on a cluster hiccup: its hook waits and repeats
the expert call (client/fronk_client.h), which appears here as a "paused" event.

A chat can be saved only while the engine stands between two passes. When a
piece of the cluster falls (a node, a cable, the head itself) the pass in flight
never ends, so this program asks the engine for its lifeboat: a file named like
the expert socket plus ".lifeboat" makes the engine end that pass by itself from
the model file. The request in flight is closed, the chat is saved, and the same
chat is restored without re-reading when it returns. This happens on every stop,
and by itself when the cluster has not answered for PAUSE_SAVE_S seconds.
"""
import base64
import collections
import hashlib
import http.client
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import re
import select
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time

CHAT_PATHS = ('/v1/chat/completions', '/v1/completions', '/completion')
POST_PATHS = CHAT_PATHS + ('/tokenize', '/detokenize')
WARM_USER = {'role': 'user', 'content': 'Rispondi soltanto OK.'}
WARM_KEYS = ('model', 'tools', 'tool_choice', 'parallel_tool_calls', 'chat_template_kwargs', 'reasoning_effort',
             'temperature', 'store', 'cache_prompt')
MIN_SAVE_TOKENS = 4096        # a shorter chat is re-read faster than it is worth writing
SAVED_CHATS = 2               # emergency files kept on the SSD
FREE_WARN, FREE_STOP = 9, 6   # percent of RAM reclaimable without swap (the expert cache keeps 11-14%)
SWAP_STOP_BYTES = 256 << 20   # new swap-out since start that ends the run
PAUSE_SAVE_S = 60             # a cluster silent this long: the chat is saved without it (lifeboat)
SETTLE_ASK_S = 3              # an engine blocked on the cluster this long gets the lifeboat
SETTLE_LIMIT_S = 300          # the longest wait for a pass computed by the lifeboat


class Status:
    """Everything the console shows. One lock; no prompt or reply text is kept."""

    def __init__(self):
        self.lock = threading.Lock()
        self.events = collections.deque(maxlen=200)
        self.data = dict(state='loading', started=time.time(), request=None, queued=0, last={}, profile={},
                         paused=None, memory={}, chat=dict(loaded=False, dirty=False, tokens=0), prefix={},
                         saves=dict(count=0, bytes=0), totals=dict(requests=0, read_tokens=0, written_tokens=0))

    def event(self, kind, text):
        with self.lock:
            self.events.append(dict(t=time.time(), kind=kind, text=text))
        print(f'FRONK engine [{kind}] {text}', flush=True)

    def update(self, **values):
        with self.lock:
            self.data.update(values)

    def snapshot(self):
        with self.lock:
            return json.loads(json.dumps(dict(self.data, events=list(self.events), now=time.time())))


STATUS = Status()
LINE = dict(
    read=re.compile(r'task (\d+) \| prompt processing, n_tokens = +(\d+), progress = ([0-9.]+).*?/ +([0-9.]+) tokens per second'),
    write=re.compile(r'task (\d+) \| n_gen = +(\d+), tg = +([0-9.]+) t/s'),
    read_done=re.compile(r'prompt eval time = +([0-9.]+) ms / +(\d+) tokens .*? ([0-9.]+) tokens per second'),
    write_done=re.compile(r'\|        eval time = +([0-9.]+) ms / +(\d+) tokens .*? ([0-9.]+) tokens per second'),
    draft=re.compile(r'draft acceptance = ([0-9.]+) \( *(\d+) accepted / *(\d+) generated\), mean len = +([0-9.]+)'),
    profile=re.compile(r'llama-fronk: width ([0-9.]+); per layer ([0-9.]+) ms head \+ ([0-9.]+) ms cluster; per pass ([0-9.]+) ms'),
    paused=re.compile(r'llama-fronk: cluster paused at layer (\d+)'),
    resumed=re.compile(r'llama-fronk: cluster resumed after (\d+) s'),
    boat=re.compile(r'llama-fronk: lifeboat (on at layer \d+|off after .*)'),
)


def follow_engine(stream):
    """Copy the engine's log to ours and keep the numbers the console needs."""
    for raw in iter(stream.readline, b''):
        line = raw.decode('utf-8', 'replace').rstrip('\n')
        print(line, flush=True)
        if 'llama-fronk' not in line and 'print_timing' not in line:
            continue
        with STATUS.lock:
            request, last, totals = STATUS.data['request'], STATUS.data['last'], STATUS.data['totals']
            if (m := LINE['read'].search(line)) and request is not None:
                done, fraction = int(m[2]), float(m[3])
                request.update(phase='reading', read=done, read_total=round(done/fraction) if fraction > 0 else done,
                               progress=fraction, tok_s=float(m[4]))
            elif (m := LINE['write'].search(line)) and request is not None:
                request.update(phase='writing', written=int(m[2]), tok_s=float(m[3]))
            elif m := LINE['read_done'].search(line):
                last.update(read_tokens=int(m[2]), read_tok_s=float(m[3]))
                totals['read_tokens'] += int(m[2])
            elif m := LINE['write_done'].search(line):
                last.update(written_tokens=int(m[2]), write_tok_s=float(m[3]))
                totals['written_tokens'] += int(m[2])
            elif m := LINE['draft'].search(line):
                last.update(draft_acceptance=float(m[1]), tokens_per_pass=float(m[4]))
            elif m := LINE['profile'].search(line):
                STATUS.data['profile'] = dict(width=float(m[1]), head_ms=float(m[2]), cluster_ms=float(m[3]), between_ms=float(m[4]))
        if m := LINE['paused'].search(line):
            STATUS.update(paused=dict(since=time.time(), layer=int(m[1])))
            STATUS.event('error', f'cluster in pausa al layer {m[1]}: un nodo o la head non risponde; il motore attende e il contesto resta in RAM')
        elif m := LINE['resumed'].search(line):
            STATUS.update(paused=None)
            STATUS.event('ok', f'cluster ripreso dopo {m[1]} s, nessun contesto perso')
        elif m := LINE['boat'].search(line):
            STATUS.event('warn', 'salvagente: il motore chiude il passo in corso da solo, leggendo gli esperti dal file del modello'
                         if m[1].startswith('on') else 'salvagente terminato: il cluster risponde di nuovo')


# --- images: the engine rejects lossy WebP; macOS sips rewrites it as PNG -------------------
WEBP = 'data:image/webp;base64,'
converted = collections.OrderedDict()


def has_images(body):
    for message in body.get('messages', []):
        content = message.get('content') if isinstance(message, dict) else None
        for part in content if isinstance(content, list) else []:
            if isinstance(part, dict) and part.get('type') in ('image_url', 'input_image', 'image'):
                return True
    return False


def png_images(body):
    for message in body.get('messages', []):
        content = message.get('content') if isinstance(message, dict) else None
        for part in content if isinstance(content, list) else []:
            image = part.get('image_url') if isinstance(part, dict) and part.get('type') == 'image_url' else None
            url = image.get('url') if isinstance(image, dict) else image
            if not isinstance(url, str) or not url.startswith(WEBP):
                continue
            digest = hashlib.sha256(url.encode()).hexdigest()
            try:
                if digest not in converted:
                    with tempfile.TemporaryDirectory() as work:
                        source, target = Path(work, 'in.webp'), Path(work, 'out.png')
                        source.write_bytes(base64.b64decode(url[len(WEBP):], validate=True))
                        subprocess.run(['/usr/bin/sips', '-s', 'format', 'png', str(source), '--out', str(target)],
                                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=60, check=True)
                        converted[digest] = 'data:image/png;base64,' + base64.b64encode(target.read_bytes()).decode()
                    while len(converted) > 16:
                        converted.popitem(last=False)
            except (OSError, ValueError, subprocess.SubprocessError) as error:
                STATUS.event('warn', f'immagine WebP non convertita ({type(error).__name__})')
                continue
            if isinstance(image, dict):
                image['url'] = converted[digest]
            else:
                part['image_url'] = converted[digest]


# --- the engine: one process, one request at a time ----------------------------------------
class Engine:
    def __init__(self, command, port, state):
        self.port, self.state = port, Path(state)
        self.turn = threading.Condition()
        self.waiting, self.busy, self.closing = collections.deque(), False, False
        self.upstream = set()            # sockets of the requests the engine is answering
        self.lifeboat = Path(os.environ['FRONK_MOE_SOCKET']+'.lifeboat')
        self.lifeboat.unlink(missing_ok=True)
        self.head = os.getppid()         # the head that started this program
        self.process = subprocess.Popen(command+['--port', str(port)], stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                        start_new_session=True)
        threading.Thread(target=follow_engine, args=(self.process.stdout,), daemon=True).start()

    def call(self, method, path, body=None, timeout=None, track=None):
        conn = http.client.HTTPConnection('127.0.0.1', self.port, timeout=timeout)
        conn.request(method, path, None if body is None else json.dumps(body).encode(), {'Content-Type': 'application/json'})
        if track is not None:           # before the answer: a non-streaming one begins only when it is complete
            track.append(conn.sock)
            with self.turn:
                self.upstream.add(conn.sock)
        return conn, conn.getresponse()

    def json(self, method, path, body=None, timeout=10):
        conn, response = self.call(method, path, body, timeout)
        try:
            data = response.read()
            if response.status != 200:
                raise ConnectionError(f'{path}: HTTP {response.status}')
            return json.loads(data)
        finally:
            conn.close()

    def acquire(self, cancelled=lambda: False):
        """FIFO turn on the engine; False if the client left or the engine is closing."""
        ticket = object()
        with self.turn:
            self.waiting.append(ticket)
            STATUS.update(queued=len(self.waiting)-1+int(self.busy))
            try:
                while not self.closing and not cancelled():
                    if not self.busy and self.waiting[0] is ticket:
                        self.busy = True
                        return True
                    self.turn.wait(.25)
                return False
            finally:
                self.waiting.remove(ticket)
                STATUS.update(queued=len(self.waiting))
                self.turn.notify_all()

    def release(self):
        with self.turn:
            self.busy = False
            self.turn.notify_all()

    def idle(self):
        with self.turn:
            return not self.busy and not self.waiting

    def interrupt(self):
        """Close the requests in flight: the engine drops each at the end of its pass."""
        with self.turn:
            sockets = list(self.upstream)
        for sock in sockets:
            try:
                sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
        return len(sockets)

    def settle(self, limit=SETTLE_LIMIT_S):
        """Wait until the engine stands between two passes with its slot free: the only
        state that can be saved. A pass that the cluster does not answer (a node or the
        head fell) never ends, so the engine is then asked to end it by itself from the
        model file (the lifeboat): only when it is blocked on the cluster, never while
        the cluster answers. True when the engine has settled."""
        started, asked = time.monotonic(), False
        blocked = lambda: STATUS.data['paused'] is not None or os.getppid() != self.head
        try:
            while time.monotonic()-started < limit and self.process.poll() is None:
                try:
                    if not any(slot.get('is_processing') for slot in self.json('GET', '/slots', timeout=4)):
                        return True
                except (OSError, ValueError, http.client.HTTPException):
                    pass
                if not asked and blocked() and time.monotonic()-started >= SETTLE_ASK_S:
                    asked = True
                    self.lifeboat.touch()
                    STATUS.event('warn', 'il cluster non risponde: il motore chiude da solo il passo in corso '
                                         '(legge gli esperti dal file del modello), poi la chat viene salvata')
                time.sleep(1 if asked else .3)
            return False
        finally:
            if asked:
                self.lifeboat.unlink(missing_ok=True)


class Chats:
    """Which chat the engine's slot holds, and its emergency copy on the SSD.

    A chat is named by a digest of its first user message; no text is stored
    outside the engine's own state file. Files are written only by save()."""

    def __init__(self, engine):
        self.engine, self.directory = engine, engine.state/'slot-cache'
        self.directory.mkdir(parents=True, exist_ok=True)
        self.index_path = self.directory/'index.json'
        try:
            self.index = json.loads(self.index_path.read_text())
        except (OSError, ValueError):
            self.index = {}
        self.loaded, self.dirty, self.seen = None, False, set()   # seen: chats this run already holds in RAM

    @staticmethod
    def key(body):
        first = next((m for m in body.get('messages', []) if isinstance(m, dict) and m.get('role') == 'user'), {})
        return hashlib.sha256(json.dumps(first, sort_keys=True).encode()).hexdigest()

    def tokens(self):
        try:
            slot = self.engine.json('GET', '/slots', timeout=4)[0]
            return (slot.get('n_prompt_tokens') or 0)+((slot.get('next_token') or [{}])[0].get('n_decoded') or 0)
        except (OSError, ValueError, IndexError, KeyError, http.client.HTTPException):
            return 0

    def before(self, key):
        """Engine turn held. Bring back this chat's emergency copy if RAM lost it."""
        entry = self.index.get(key)
        if key in self.seen or not entry or not (self.directory/entry['file']).is_file():
            return
        self.seen.add(key)
        started = time.monotonic()
        try:
            self.engine.json('POST', '/slots/0?action=restore', {'filename': entry['file']}, timeout=600)
            self.loaded, self.dirty = key, False
            STATUS.event('ok', f'chat ripristinata dal salvataggio di emergenza: {entry.get("tokens") or "?"} token, '
                               f'{entry["size"] >> 20} MiB in {time.monotonic()-started:.1f} s (nessuna rilettura)')
        except (OSError, ValueError, http.client.HTTPException) as error:
            STATUS.event('warn', f'ripristino non riuscito ({type(error).__name__}); il motore rilegge il prompt')

    def after(self, key):
        self.loaded, self.dirty = key, True
        self.seen.add(key)
        STATUS.update(chat=dict(loaded=True, dirty=True, tokens=self.tokens()))

    def save(self, reason, fallen=False):
        """Engine turn held and engine idle. The only SSD write of this program.
        fallen: the cluster is down and requests keep failing; the copy on disk is
        written again only when the chat has grown enough to be worth it."""
        if self.loaded is None or not self.dirty:
            return
        tokens = self.tokens()
        if tokens < MIN_SAVE_TOKENS:
            STATUS.event('info', f'chat di {tokens} token non salvata ({reason}): si rilegge in meno tempo')
            return
        entry = self.index.get(self.loaded)
        if fallen and entry and (self.directory/entry['file']).is_file() and 0 <= tokens-(entry.get('tokens') or 0) < MIN_SAVE_TOKENS:
            STATUS.event('info', f'chat già salvata a {entry.get("tokens")} token, ora {tokens}: non riscritta ({reason})')
            return
        name, started = f'chat-{self.loaded[:24]}.bin', time.monotonic()
        try:
            self.engine.json('POST', '/slots/0?action=save', {'filename': name+'.tmp'}, timeout=600)
            os.replace(self.directory/(name+'.tmp'), self.directory/name)
        except (OSError, ValueError, http.client.HTTPException) as error:
            (self.directory/(name+'.tmp')).unlink(missing_ok=True)
            STATUS.event('error', f'salvataggio di emergenza fallito ({type(error).__name__}): {reason}')
            return
        size = (self.directory/name).stat().st_size
        self.index[self.loaded] = dict(file=name, size=size, saved=time.time(), tokens=tokens)
        newest = sorted(self.index, key=lambda k: self.index[k].get('saved', 0), reverse=True)
        for old in newest[SAVED_CHATS:]:
            (self.directory/self.index.pop(old)['file']).unlink(missing_ok=True)
        temporary = self.index_path.with_suffix('.tmp')
        temporary.write_text(json.dumps(self.index))
        os.replace(temporary, self.index_path)
        self.dirty = False
        with STATUS.lock:
            STATUS.data['saves']['count'] += 1
            STATUS.data['saves']['bytes'] += size
            STATUS.data['chat'] = dict(loaded=True, dirty=False, tokens=tokens)
        STATUS.event('save', f'chat salvata su SSD ({reason}): {tokens} token, {size >> 20} MiB in {time.monotonic()-started:.1f} s')


class Prefix:
    """The harness's system prompt and tools, read once and kept ahead of every chat."""

    def __init__(self, engine):
        self.engine, self.path = engine, engine.state/'study-prefix-request.json'
        self.digest = self.digest_of(self.load())

    def load(self):
        try:
            return json.loads(self.path.read_text())
        except (OSError, ValueError):
            return None

    @staticmethod
    def digest_of(body):
        if not body:
            return None
        system = [m for m in body.get('messages', []) if m.get('role') in ('system', 'developer')]
        return hashlib.sha256(json.dumps([system, body.get('tools')], sort_keys=True).encode()).hexdigest()

    def warm(self):
        """Engine turn held, at start: one trial read of the tools."""
        body = self.load()
        if not body:
            STATUS.event('info', 'nessun prefisso strumenti salvato: verrà letto e ricordato alla prima chat')
            return
        started = time.monotonic()
        try:
            body.update(stream=False, max_tokens=1, max_completion_tokens=1, temperature=0, cache_prompt=True)
            body.pop('stream_options', None)
            usage = self.engine.json('POST', '/v1/chat/completions', body, timeout=1800).get('usage', {})
            tokens = usage.get('prompt_tokens', 0)
            STATUS.update(prefix=dict(tokens=tokens, seconds=round(time.monotonic()-started, 1), at=time.time()))
            STATUS.event('ok', f'strumenti dell\'harness letti una volta: {tokens} token in {time.monotonic()-started:.0f} s; ogni chat riparte da qui')
        except (OSError, ValueError, http.client.HTTPException) as error:
            STATUS.event('warn', f'lettura di prova degli strumenti non riuscita ({type(error).__name__})')

    def observe(self, body):
        """Remember a chat's prefix (never its conversation) for the next start."""
        messages, system = body.get('messages') or [], []
        for message in messages:
            if not isinstance(message, dict) or message.get('role') not in ('system', 'developer'):
                break
            system.append(message)
        if not system or not body.get('tools') or len(system) == len(messages):
            return
        prefix = {k: body[k] for k in WARM_KEYS if k in body}
        prefix['messages'] = system+[WARM_USER]
        digest = self.digest_of(prefix)
        data = json.dumps(prefix)
        if digest == self.digest or len(data) > 262144:
            return
        temporary = self.path.with_suffix('.tmp')
        temporary.write_text(data)
        os.replace(temporary, self.path)
        self.digest = digest
        STATUS.event('info', 'prompt di sistema/strumenti dell\'harness cambiati: prefisso aggiornato per il prossimo avvio')


# --- memory: stop cleanly before the Mac swaps -----------------------------------------------
def vm_counters():
    out = subprocess.run(['/usr/bin/vm_stat'], capture_output=True, text=True, timeout=5).stdout
    page = int(re.search(r'page size of (\d+) bytes', out)[1])
    count = lambda name: int(re.search(rf'^{name}:\s+(\d+)\.', out, re.M)[1])*page
    free = count('Pages free')+count('Pages inactive')+count('Pages speculative')+count('Pages purgeable')
    return free, count('Swapouts')


def watch_memory(server):
    total = int(subprocess.run(['/usr/sbin/sysctl', '-n', 'hw.memsize'], capture_output=True, text=True).stdout)
    baseline, warned = None, 0.0
    while not server.stopping.wait(5):
        try:
            free, swapouts = vm_counters()
            rss = int(subprocess.run(['/bin/ps', '-o', 'rss=', '-p', str(server.engine.process.pid)],
                                     capture_output=True, text=True, timeout=5).stdout or 0) << 10
        except (OSError, ValueError, TypeError, subprocess.SubprocessError):
            continue
        baseline = swapouts if baseline is None else baseline
        percent, new_swap = 100*free//total, max(swapouts-baseline, 0)
        STATUS.update(memory=dict(total_gb=round(total/2**30, 1), free_gb=round(free/2**30, 1), free_percent=percent,
                                  engine_gb=round(rss/2**30, 1), new_swap_mb=new_swap >> 20))
        reason = (f'memoria quasi esaurita ({percent}% libera)' if percent < FREE_STOP else
                  f'{new_swap >> 20} MiB di nuovo swap sul SSD' if new_swap >= SWAP_STOP_BYTES else None)
        if reason:
            STATUS.event('error', f'arresto protettivo: {reason}. La chat viene salvata, poi il motore si ferma per non usare lo swap')
            server.shutdown_engine(f'protezione memoria: {reason}')
            return
        if (percent < FREE_WARN or new_swap) and time.monotonic()-warned > 120:
            warned = time.monotonic()
            STATUS.event('warn', f'memoria: {percent}% libera, {new_swap >> 20} MiB di nuovo swap (arresto sotto {FREE_STOP}% o oltre {SWAP_STOP_BYTES >> 20} MiB)')


# --- HTTP ------------------------------------------------------------------------------------
class Handler(BaseHTTPRequestHandler):
    protocol_version = 'HTTP/1.1'

    def log_message(self, *_):
        pass

    def reply(self, status, body):
        data = json.dumps(body).encode()
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        engine = self.server.shared.engine
        try:
            if self.path == '/fronk/status':
                return self.reply(200, STATUS.snapshot())
            if self.path in ('/health', '/v1/models', '/props', '/slots'):
                return self.reply(200, engine.json('GET', self.path, timeout=3))
            return self.reply(404, {'error': 'unsupported path'})
        except (OSError, ValueError, http.client.HTTPException):
            return self.reply(503, {'error': 'engine loading or unavailable'})

    def do_POST(self):
        shared = self.server.shared
        if self.path == '/fronk/save':  # the console's "save now" button
            threading.Thread(target=shared.save_when_idle, args=('richiesto dalla console',), daemon=True).start()
            return self.reply(200, {'message': 'saving when the engine is idle'})
        if self.path not in POST_PATHS:
            return self.reply(404, {'error': 'unsupported path'})
        engine, chat = shared.engine, self.path in CHAT_PATHS
        held, conn, sent, key = False, None, False, None
        finished, cancelled, upstream = threading.Event(), threading.Event(), []

        def watch_client():  # a client that leaves frees the engine for the next request
            while not finished.wait(.25):
                try:
                    ready, _, _ = select.select([self.connection], [], [], 0)
                    if ready and not self.connection.recv(1, socket.MSG_PEEK):
                        cancelled.set()
                        for sock in upstream:
                            sock.shutdown(socket.SHUT_RDWR)
                        return
                except OSError:
                    return
        try:
            size = int(self.headers.get('Content-Length', '0'))
            if not 0 < size <= 32 << 20:
                return self.reply(413, {'error': 'invalid request size'})
            body = json.loads(self.rfile.read(size))
            if not isinstance(body, dict):
                raise ValueError('request must be an object')
            body.pop('id_slot', None)
            if chat and (engine.state/'gsq-no-vision').exists() and has_images(body):
                STATUS.event('warn', 'richiesta con immagini rifiutata: input immagini disattivato (state/gsq-no-vision)')
                return self.reply(400, {'error': {'type': 'invalid_request_error', 'code': 'images_off',
                    'message': 'input immagini disattivato su questa head (file state/gsq-no-vision): '
                               'togli il file e riavvia la head per riattivarlo'}})
            threading.Thread(target=watch_client, daemon=True).start()
            held = engine.acquire(cancelled.is_set)
            if not held:
                raise ConnectionError('client left or engine stopping')
            if chat:
                png_images(body)
                key = Chats.key(body)
                shared.chats.before(key)
                shared.prefix.observe(body)
                STATUS.update(request=dict(started=time.time(), phase='starting', path=self.path))
            conn, response = engine.call('POST', self.path, body, track=upstream)
            self.send_response(response.status)
            self.send_header('Content-Type', response.getheader('Content-Type', 'application/json'))
            self.send_header('Connection', 'close')
            self.end_headers()
            sent = self.close_connection = True
            while chunk := response.read1(16384):
                self.wfile.write(chunk)
                self.wfile.flush()
        except (OSError, ValueError, http.client.HTTPException) as error:
            if chat and held and not cancelled.is_set():
                STATUS.event('error', f'richiesta interrotta ({type(error).__name__}); il contesto letto resta nel motore')
            elif chat and held:
                STATUS.event('warn', 'il client ha chiuso la connessione; il contesto letto resta nel motore e viene riusato')
            if not sent and not cancelled.is_set():
                try:
                    self.reply(503, {'error': 'request interrupted or engine unavailable'})
                except OSError:
                    pass
            self.close_connection = True
        finally:
            finished.set()
            with engine.turn:
                engine.upstream.difference_update(upstream)
            if conn:
                conn.close()
            if held:
                if chat:
                    shared.chats.after(key)
                    with STATUS.lock:
                        STATUS.data['request'] = None
                        STATUS.data['totals']['requests'] += 1
                engine.release()


class Shared:
    def __init__(self, command, port, state):
        self.stopping, self.ending = threading.Event(), threading.Lock()
        self.engine = Engine(command, port+2, state)
        self.chats, self.prefix = Chats(self.engine), Prefix(self.engine)

    def ready(self):
        while not self.stopping.is_set() and self.engine.process.poll() is None:
            try:
                self.engine.json('GET', '/health', timeout=2)
                break
            except (OSError, ValueError, http.client.HTTPException):
                time.sleep(.5)
        else:
            return
        if self.engine.acquire(self.stopping.is_set):
            try:
                self.prefix.warm()
            finally:
                self.engine.release()
        STATUS.update(state='ready')
        STATUS.event('ok', 'motore pronto')

    def save_when_idle(self, reason, wait=0, stop=False, fallen=False):
        """Save the loaded chat. stop: a request in flight is closed first and the engine
        is brought between two passes, with its lifeboat when the cluster does not answer."""
        if stop and self.engine.interrupt():
            STATUS.event('warn', f'generazione interrotta per salvare la chat ({reason})')
        deadline = time.monotonic()+wait
        while not self.engine.idle() and time.monotonic() < deadline:
            time.sleep(.2)
        if not self.engine.idle():
            STATUS.event('warn', f'motore occupato: salvataggio non eseguito ({reason})')
            return False
        if not self.engine.acquire():
            return False
        try:
            if stop and not self.engine.settle():
                STATUS.event('error', f'il motore non ha chiuso il passo in corso: salvataggio non eseguito ({reason})')
                return False
            self.chats.save(reason, fallen)
            return True
        finally:
            self.engine.release()

    def rescue(self, seconds):
        """The cluster has not answered for a while: a node, a cable or the head fell.
        The chat is put in safety without waiting for it; the engine stays loaded."""
        with self.ending:
            if self.stopping.is_set() or STATUS.data['state'] == 'stopping':
                return
            reason = f'il cluster non risponde da {seconds} s'
            if self.save_when_idle(reason, wait=20, stop=True, fallen=True):
                STATUS.update(paused=None)
                STATUS.event('error', 'un pezzo del cluster è caduto: la chat è al sicuro (in RAM e, se abbastanza lunga, su SSD). '
                                      'Quando il pezzo è tornato, riavvia la head dalla console: la chat riprende senza rilettura')

    def shutdown_engine(self, reason):
        """Save the loaded chat, also in the middle of a generation, then stop everything."""
        STATUS.update(state='stopping')
        with self.ending:
            if not self.stopping.is_set():
                self.save_when_idle(reason, wait=20, stop=True)
                self.stopping.set()


def serve(shared, address, port):
    server = ThreadingHTTPServer((address, port), Handler)
    server.daemon_threads, server.shared = True, shared
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def main(command):
    port, socket_path = int(os.environ['FRONK_DS4_PORT']), Path(os.environ['FRONK_MOE_SOCKET'])
    state = socket_path.parent
    deadline = time.monotonic()+SETTLE_LIMIT_S+60   # an engine just stopped may still be saving its chat
    while True:
        with socket.socket() as probe:
            probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                probe.bind(('127.0.0.1', port))
                break
            except OSError:
                if time.monotonic() >= deadline:
                    raise SystemExit('fronk_engine: the chat port is still in use')
        time.sleep(.2)
    shared = Shared(command, port, state)
    servers = [serve(shared, '127.0.0.1', port)]
    try:                                 # the private cable to the machines that run the harness
        host, cable_port = (state/'gsq-listen').read_text().strip().rsplit(':', 1)
        servers.append(serve(shared, host, int(cable_port)))
        STATUS.event('info', f'API anche su {host}:{cable_port} (cavo privato)')
    except (OSError, ValueError) as error:
        if (state/'gsq-listen').exists():
            STATUS.event('warn', f'indirizzo del cavo privato non disponibile ({error})')
    parent = os.getppid()
    for number in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(number, lambda *_: threading.Thread(target=shared.shutdown_engine, args=('arresto della head',), daemon=True).start())
    threading.Thread(target=shared.ready, daemon=True).start()
    threading.Thread(target=watch_memory, args=(shared,), daemon=True).start()
    STATUS.event('info', 'un solo motore; le richieste successive attendono in coda')
    rescued = None                       # the pause already handled
    try:
        while not shared.stopping.wait(.5):
            if shared.engine.process.poll() is not None:
                STATUS.event('error', f'il motore è terminato (codice {shared.engine.process.returncode})')
                break
            paused = STATUS.data['paused']
            if paused and paused['since'] != rescued and time.time()-paused['since'] >= PAUSE_SAVE_S and STATUS.data['state'] != 'stopping':
                rescued = paused['since']
                threading.Thread(target=shared.rescue, args=(PAUSE_SAVE_S,), daemon=True).start()
            if os.getppid() != parent and STATUS.data['state'] != 'stopping':
                threading.Thread(target=shared.shutdown_engine, args=('la head è terminata',), daemon=True).start()
                STATUS.update(state='stopping')
    finally:
        shared.stopping.set()
        with shared.engine.turn:
            shared.engine.closing = True
            shared.engine.turn.notify_all()
        process = shared.engine.process
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=8)
            except subprocess.TimeoutExpired:
                process.kill()
        for server in servers:
            server.server_close()


if __name__ == '__main__':
    arguments = sys.argv[1:]
    if arguments[:1] == ['--']:
        arguments.pop(0)
    if not arguments:
        raise SystemExit('usage: fronk_engine.py -- ENGINE [arguments without --port]')
    main(arguments)
