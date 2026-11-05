"""fronk web console: run and configure the head from a browser.

    FRONK_SECRET=... python3 tools/fronk-web.py     # then open http://127.0.0.1:7730

Runs `./fronk head` as a child and sends it the same keys a terminal user
would ([p]roceed, [m]odified, [r]eport, [q]uit). Binds 127.0.0.1 only; every
POST needs the per-start token embedded in the page. The shared secret is
kept in this process's memory (environment or the form), never written.
"""
import fcntl, html, json, math, os, re, secrets, signal, subprocess, sys, threading, time, urllib.parse, urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fronk_chats, fronk_files, fronk_report  # noqa: E402 - local UI helpers

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONF = os.path.join(ROOT, "fronk.conf")
TOKEN = secrets.token_hex(16)
STATE = {"proc": None, "lines": [], "secret": os.environ.get("FRONK_SECRET", ""),
         "keys": "", "started": 0.0, "activity": "", "activity_t": 0.0, "phase": "IDLE", "phase_t": 0.0,
         "auto": True}  # auto: answer Proceed when the plan fits
FOLLOW_LOG = None
LOCK = threading.Lock()
PROMPT = re.compile(rb"\[[a-z]\][a-z]*( \([^)]*\))?: $")
EDITABLE = ("model", "burst", "ds4", "interfaces", "state-dir", "head-blocks")
# SSH aliases belong only to the development checkout. An upgraded install
# may still contain a copy of the old developer helper.
DEV_CLUSTER = os.path.isfile(os.path.join(ROOT, "tools", "cluster.sh")) and not os.path.isfile(
    os.path.join(ROOT, "..", "state", "install-role"))
NODES = ("shuttle", "imac", "m1pro") if DEV_CLUSTER else ()


def terminal_head_pid():
    state_dir = os.path.join(ROOT, conf_read().get("state-dir", "runs/state-head"))
    try:
        with open(os.path.join(state_dir, "head.lock")) as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                return lock.read().strip() or "unknown"
    except FileNotFoundError:
        pass
    return ""


def follow_terminal(path):
    """Only show Fronk output from the most recent head run in the log."""
    with open(path, errors="replace") as log:
        started = False
        while True:
            line = log.readline()
            if not line:
                time.sleep(.25)
                continue
            line = line.rstrip("\n")
            with LOCK:
                if line.startswith("FRONK checking unit manifest"):
                    started = True
                    STATE["lines"].clear()
                    STATE["started"] = time.time()
                if started:
                    STATE["lines"].append(line)
                    if line.startswith(("CONNECTION", "BENCH", "PLAN", "TIMING", "FRONK", "SELFTEST", "TEST", "REPORT", "SERVE", "fronk:")):
                        STATE["activity"], STATE["activity_t"] = line, time.time()


def installer_demo():
    """An open dashboard can show the demo launched by the installer on the head."""
    state_dir = os.path.join(ROOT, conf_read().get("state-dir", "runs/state-head"))
    path = os.path.join(state_dir, "install-demo.log.tmp")
    if not os.path.isfile(path) or not terminal_head_pid():
        return None
    try:
        with open(path, errors="replace") as log:
            lines = log.read().splitlines()
            elapsed = max(0, time.time() - os.fstat(log.fileno()).st_mtime)
    except FileNotFoundError:
        return None
    view = view_of(lines)
    activity = next((line for line in reversed(lines) if line.startswith(("BENCH", "CONNECTION", "PLAN", "DEMO", "TIMING", "REPORT"))), "")
    view.update(demo=True, observer=True, head_name=os.uname().nodename.split(".")[0], activity=activity,
                activity_age=int(elapsed), running=True, keys="", auto=False, elapsed=0, phase_age=0, failure="", lastline=activity)
    return view


def saved_secret():
    """fronk keeps the cluster secret in state-dir/cluster.secret (0600) after the first start."""
    state = conf_read().get("state-dir", "runs/state-head")
    return os.path.exists(os.path.join(ROOT, state, "cluster.secret"))


def conf_read():
    values = {}
    for line in open(CONF):
        parts = line.split()
        if len(parts) == 2 and not line.startswith("#"):
            values[parts[0]] = parts[1]
    return values


def conf_write(updates):
    out, seen = [], set()
    for line in open(CONF):
        parts = line.split()
        if len(parts) == 2 and parts[0] in updates and not line.startswith("#"):
            seen.add(parts[0])
            if updates[parts[0]]:
                out.append(f"{parts[0]} {updates[parts[0]]}\n")
            continue
        out.append(line)
    out += [f"{k} {v}\n" for k, v in updates.items() if k not in seen and v]
    open(CONF, "w").writelines(out)


def reader(proc):
    buf = b""
    while True:
        ch = proc.stdout.read(1)
        if not ch:
            break
        buf += ch
        # Prompts ("... [q]uit: ") end without a newline; flush them as soon as they appear.
        if ch == b"\n" or (buf.endswith(b": ") and PROMPT.search(buf)):
            line = buf.decode("utf-8", "replace").rstrip("\n")
            with LOCK:
                STATE["lines"].append(line)
                # The head reads one key per prompt; only the keys it offers are accepted.
                STATE["keys"] = "".join(re.findall(r"\[([a-z])\]", line)) if PROMPT.search(buf) else STATE["keys"]
                if re.match(r"(BENCH|TIMING|CONNECTION|DISCOVERY|PLAN|FRONK|SELFTEST|TEST|REPORT|SERVE|fronk:)", line):
                    STATE["activity"], STATE["activity_t"] = line.strip(), time.time()
                fits = any(l.startswith("CAPACITY resident=20/20") or
                           (l.startswith("CAPACITY resident=") and "planned=20/20" in l and "— fits" in l)
                           for l in STATE["lines"][-12:])
                if STATE["auto"] and "Proceed with this configuration" in line and fits:
                    proc.stdin.write(b"p\n"); proc.stdin.flush()
                    STATE["lines"].append("> p (automatic: all 20 units fit)"); STATE["keys"] = ""
            buf = b""
    with LOCK:
        STATE["lines"].append(f"[head exited with status {proc.wait()}]"); STATE["keys"] = ""


def ensure_nodes():
    """Start every node that is not running (a running one answers "already running")."""
    out = {}

    def one(n):
        r = subprocess.run(["tools/cluster.sh", "start", n, "configs/node.conf"], cwd=ROOT, capture_output=True, text=True)
        lines = (r.stdout + r.stderr).strip().splitlines()
        out[n] = lines[-1] if lines else f"exit {r.returncode}"
    threads = [threading.Thread(target=one, args=(n,)) for n in NODES]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return [f"[nodes] {n} start: {out[n]}" for n in NODES]


def start_head(refresh):
    with LOCK:
        if STATE["proc"] and STATE["proc"].poll() is None:
            return "head already running"
    started_nodes = ensure_nodes() if NODES else ["[nodes] Discovering nodes running their own service."]
    with LOCK:
        if STATE["proc"] and STATE["proc"].poll() is None:
            return "head already running"
        if not STATE["secret"] and not saved_secret():
            return "enter the cluster secret first"
        env = dict(os.environ, FRONK_BENCH_REFRESH="1" if refresh else "0")
        if STATE["secret"]:
            env["FRONK_SECRET"] = STATE["secret"]  # saved once to state-dir/cluster.secret by fronk
        env.pop("FRONK_ANSWER", None); env.pop("FRONK_ANSWER_AFTER", None)
        STATE["lines"] = started_nodes + ["$ ./fronk head" + (" (BENCH refresh)" if refresh else "")]
        STATE["keys"], STATE["started"], STATE["activity"], STATE["activity_t"] = "", time.time(), "", time.time()
        proc = subprocess.Popen(["./fronk", "head"], cwd=ROOT, env=env, stdin=subprocess.PIPE,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        STATE["proc"] = proc
    threading.Thread(target=reader, args=(proc,), daemon=True).start()
    return "head started"


def nodes(action):
    """Start, stop or list the nodes over SSH with tools/cluster.sh; output goes to the log."""
    if not NODES:
        return "Use the installer manager on each node to check its local service."
    if action != "status" and head_running():
        return "quit the head first: nodes cannot be started or stopped while it uses them"
    def run():
        for n in NODES:
            # stop: SIGTERM, and SIGKILL only if the node is still there after 4 s.
            cmd = ["tools/cluster.sh", action, n] + (["configs/node.conf"] if action == "start" else ["--force"] if action == "stop" else [])
            r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
            out = (r.stdout + r.stderr).strip().splitlines()
            with LOCK:
                STATE["lines"].append(f"[nodes] {n} {action}: " + (out[-1] if out else f"exit {r.returncode}"))
    threading.Thread(target=run, daemon=True).start()
    return f"nodes {action} sent"


def stop_head():
    """SIGTERM works at any moment; the head stops ds4 and closes its node sessions."""
    with LOCK:
        proc = STATE["proc"]
    if not proc or proc.poll() is not None:
        return "head is not running"
    proc.terminate()
    try:
        proc.wait(timeout=15)
    except subprocess.TimeoutExpired:
        proc.kill()
    return "head stopped"


def head_running():
    with LOCK:
        return bool(STATE["proc"] and STATE["proc"].poll() is None)


def send_key(key):
    with LOCK:
        proc = STATE["proc"]
        if not proc or proc.poll() is not None:
            return "head is not running"
        if key not in STATE["keys"]:
            return "the head is not asking for that now" if STATE["keys"] else "the head is working; wait for its question"
        STATE["keys"] = ""
        proc.stdin.write((key + "\n").encode()); proc.stdin.flush()
        STATE["lines"].append(f"> {key}")
    return "sent " + key


PHASES = ["CONNECTION", "DISCOVERY", "PLAN", "FRONK", "SELFTEST", "TEST", "REPORT"]


def blocks_of(ranges):
    out = []
    for part in ranges.split(","):
        if "-" in part and part[0].isdigit():
            a, b = part.split("-")
            out += list(range(int(a) // 32, (int(b) + 1) // 32))
    return out


def view_of(lines):
    """Phase model for the console, derived only from the head's own output."""
    v = {"phase": "IDLE", "waiting": "", "timing": {}, "nodes": {}, "owner": [None] * 20, "ssd": [],
         "verified": [], "selftest": "", "text": "", "report": {}, "capacity": "", "ram": "", "chat": "", "mtp": {}}
    profile = None
    for line in lines:
        s = line.strip()
        mtp = re.search(r"FRONK_TREE width=\d+.*", line)
        if mtp:
            v["mtp"] = {key: int(value) for key, value in re.findall(r"(\w+)=(-?\d+)", mtp.group())}
            if v["phase"] == "TEST":
                v["text"] += line[:mtp.start()]
            continue
        if s.startswith("FRONK_TREE_CHECK "):
            continue
        if line.startswith("FRONK checking") or line.startswith(("SERVE ds4 started", "SERVE Vulkan head started")):
            v["phase"] = "START"
        elif s.startswith("path node="):
            v["phase"] = "CONNECTION"
            f = dict(kv.split("=", 1) for kv in s.split() if "=" in kv)
            if f.get("chosen") == "yes":
                n = v["nodes"].setdefault(f["node"], {})
                n.update(address=f["address"], iface=f["via"], link=f["type"], rtt=float(f.get("p50_us", 0)))
        elif s.startswith("TIMING "):
            parts = s.split()
            v["timing"][parts[1]] = parts[2]
        elif line.startswith("# node "):
            v["phase"] = "DISCOVERY"; profile = v["nodes"].setdefault(line.split()[3], {})
        elif line.startswith("# head"):
            profile = v["nodes"].setdefault("head", {})
        elif not s:
            profile = None
        elif profile is not None and ":" in s and not s.startswith("#"):
            k, val = s.split(":", 1)
            if k in ("os", "arch", "cpu_model", "cores", "mem_total_bytes", "mem_cap_blocks", "multicast"):
                profile[k] = val
        elif s.startswith("BENCH node="):
            v["phase"] = "DISCOVERY"; v["nodes"].setdefault(s.split()[1][5:], {})["bench"] = "sweeping"
        elif s.startswith("PLAN dense head:"):
            v["dense"] = s.split(": ", 1)[1]
        elif s.startswith("PLAN nodes="):
            v["phase"] = "PLAN"; v["ram"] = s[5:]
        elif line.startswith("  ") and " link=" in line and " blocks=" in line:
            name = s.split()[0]
            m = re.search(r"neurons (\S+)\s+blocks=\s*(\d+)\s+cap=\s*(\d+)\s+([\d.]+) GB backend=(\S+) cached=(\d+)", s)
            if m:
                n = v["nodes"].setdefault(name, {})
                n.update(blocks=int(m.group(2)), cap=int(m.group(3)), gb=float(m.group(4)), backend=m.group(5), cached=int(m.group(6)))
                for b in blocks_of(m.group(1)):
                    v["owner"][b] = name
            elif name == "head-ssd":
                m = re.search(r"neurons (\S+)", s)
                v["ssd"] = blocks_of(m.group(1)) if m else []
                for b in v["ssd"]:
                    v["owner"][b] = "head-ssd"
        elif s.startswith("HYBRID node="):
            fields = dict(re.findall(r"([\w+]+)=([^\s]+)", s))
            name = fields.get("node", "")
            if name:
                v["nodes"].setdefault(name, {})["hybrid"] = fields
                mask = int(fields.get("ssd_mask", "0"))
                v["ssd"].extend(b for b in range(20) if mask & (1 << b))
        elif s.startswith("CAPACITY resident"):
            v["capacity"] = s[9:]
        elif "Proceed with this configuration" in s:
            v["waiting"] = "proceed"
        elif s.startswith("> "):
            v["waiting"] = ""
        elif s.startswith("CONFIGURATION ") or s.startswith("FRONK unit") or s.startswith("FRONK sent"):
            v["phase"] = "FRONK"
            m = re.match(r"FRONK unit (\d+) ->", s)
            if m:
                v["verified"].append(int(m.group(1)))
        elif s.startswith("SELFTEST"):
            v["phase"] = "SELFTEST"; v["selftest"] = s[9:]
        elif s.startswith(("TEST generation", "TEST pass ")):
            v["phase"] = "TEST"
        elif s.startswith("REPORT split "):
            continue  # The timing breakdown supplements the preceding node report.
        elif s.startswith("REPORT"):
            v["phase"] = "REPORT"
            f = dict(kv.split("=", 1) for kv in s.split() if "=" in kv)
            if "node" in f:
                v["nodes"].setdefault(f["node"], {})["report"] = {k: f.get(k, "0").replace("us", "") for k in ("compute", "network", "arrival", "last", "misses", "local_wait_us")}
            elif s.startswith("REPORT bottleneck"):
                v["report"]["bottleneck"] = f.get("bottleneck", ""); v["report"]["balanced"] = "not balanced" not in s
            else:
                v["report"].update(f)
                m = re.search(r"accepted=([\d.]+) tok/s", s)
                if m:
                    v["report"]["toks"] = m.group(1)
        elif s.startswith("Chat:"):
            v["chat"] = s; v["waiting"] = "after"
        elif v["phase"] == "TEST" and not s.startswith("SERVE"):
            v["text"] += line + "\n"
        if "cycle failed" in s or "[head exited" in s:
            v["waiting"] = "failed" if "failed" in s else "exited"
    return v


PROGRESS = re.compile(r"chat ctx=(\d+)\.\.(\d+):(\d+) (?:TOOLS )?(?:THINKING )?(.*)")


def chat_progress():
    """The current turn as ds4-server logs it: the prompt's reused prefix and new
    tokens, prefill progress every 32 tokens, decoding every 50 (state-dir/ds4.log)."""
    path = os.path.join(ROOT, conf_read().get("state-dir", "runs/state-head"), "ds4.log")
    try:
        with open(path, "rb") as f:
            base = max(0, os.path.getsize(path) - 16384); f.seek(base); tail = f.read().splitlines(keepends=True)
    except OSError:
        return {}
    v, at = {}, base
    for raw in tail:
        at += len(raw); line = raw.decode(errors="replace")
        # llama.cpp uses slot events for the same prompt and decode stages.
        prompt = re.search(r"new prompt, .*task.n_tokens = (\d+)", line)
        if prompt:
            v = {"turn": at, "phase": "prefill", "reused": 0, "new": int(prompt[1]), "read": 0, "rate": 0.0}
        if v and "prompt processing," in line:
            progress = re.search(r"n_tokens =\s*(\d+).*?/ ([\d.]+) tokens per second", line)
            if progress:
                v.update(read=int(progress[1]), rate=float(progress[2]))
        if v and "prompt processing done" in line:
            v.update(phase="decode", read=v["new"])
        if v and "stop processing: n_tokens" in line:
            v["phase"] = "done"
        m = PROGRESS.search(line)
        if not m:
            continue
        a, b, n, rest = int(m[1]), int(m[2]), int(m[3]), m[4]
        if rest.startswith("prompt start"):
            v = {"turn": at, "phase": "prefill", "reused": a, "new": n, "read": 0, "rate": 0.0}  # turn: where its line ends in the log
        elif not v:
            continue
        elif rest.startswith("prefill chunk"):
            p = re.match(r"prefill chunk (\d+)/(\d+).*avg=([\d.]+) t/s", rest)
            if p:
                v.update(read=int(p[1]), rate=float(p[3]))
        elif rest.startswith("prompt done"):
            v.update(phase="decode", read=v["new"])
        elif "decoding" in rest:
            g = re.match(r"gen=(\d+) .*avg=([\d.]+) t/s", rest)
            if g:
                v.update(phase="decode", gen=int(g[1]), decode_rate=float(g[2]))
        elif "finish=" in rest:
            v["phase"] = "done"
    return v


def chat_stream(messages, think, out, warm=False):
    """Relays ds4-server's SSE stream for a whole conversation to the browser.
    The page sends history trimmed to the configured model context."""
    port = int(conf_read().get("control-port", "7700")) + 10
    # No max_tokens: ds4 lets the answer use whatever context is left.
    request = {"model": "ds4", "temperature": 0.7, "think": bool(think),
               "stream": True, "stream_options": {"include_usage": True}, "messages": messages}
    if conf_read().get("ds4") == "tools/gpu-head/run.sh":
        request["chat_template_kwargs"] = {"enable_thinking": bool(think)}
    if warm:  # read ahead: ds4 reads the prompt into its cache; one token, discarded
        request["max_tokens"] = 1
    body = json.dumps(request).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", body,
                                 {"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=900) as r:
            for line in r:
                out.write(line); out.flush()
    except Exception as e:  # noqa: BLE001 - shown to the user verbatim
        out.write(b"data: " + json.dumps({"error": f"chat unavailable: {e}; start the head and wait for READY"}).encode() + b"\n\n")


WIKI = """
<h3>Start a node (every machine that lends memory)</h3>
<pre>make fronk
FRONK_SECRET='same secret as the head' ./fronk node --config configs/node.conf   # first time only; saved after
./fronk node --config configs/node.conf                                            # every later start</pre>
<p>Nodes need no roster: the head finds them on every wired or Wi-Fi interface. Linux needs <code>make</code> and <code>cc</code>; macOS needs the Xcode command line tools.</p>
<h3>Run the head (this Mac)</h3>
<pre>make fronk
FRONK_SECRET='...' ./fronk head</pre>
<p>Phases: CONNECTION (find, authenticate, measure links) → DISCOVERY (hardware, cached BENCH) → PLAN.
Answer <b>p</b> to proceed or <b>m</b> to re-run CONNECTION (e.g. after moving a cable). FRONK then sends only missing
units (SHA-256 verified), loads them resident, runs SELFTEST against the GGUF, starts ds4 and prints REPORT.
After REPORT: <b>m</b> re-plans, <b>r</b> reprints counters, <b>q</b> quits.</p>
<h3>Chat from the terminal</h3>
<pre>curl -s http://127.0.0.1:7710/v1/chat/completions -H 'Content-Type: application/json' \\
  -d '{"model":"ds4","messages":[{"role":"user","content":"Hello"}]}'</pre>
<h3>Useful switches</h3>
<table>
<tr><td><code>FRONK_BENCH_REFRESH=1 ./fronk head</code></td><td>re-measure every backend (otherwise cached per binary + hardware)</td></tr>
<tr><td><code>./fronk-cli -h NODE_IP --config fronk.conf INFO dataplane</code></td><td>inspect a node (redis-cli style)</td></tr>
<tr><td><code>./fronk-cli -h NODE_IP --config fronk.conf UNITS</code></td><td>units cached on a node</td></tr>
<tr><td><code>tools/cluster.sh check|push|build|start|stop NODE</code></td><td>developer helper over SSH (shuttle, imac, m1pro, hp)</td></tr>
</table>
<h3>fronk.conf keys (head)</h3>
<table>
<tr><td><code>model</code></td><td>local Q4 GGUF; units are sliced from it automatically</td></tr>
<tr><td><code>ds4</code></td><td>launcher for ds4-server (tools/ds4-head.sh); chat is on control-port+10</td></tr>
<tr><td><code>burst</code></td><td>tokens per barrier used for BENCH (2 with MTP)</td></tr>
<tr><td><code>interfaces</code></td><td><code>auto</code> or a comma list of local IPv4 addresses</td></tr>
<tr><td><code>password-env</code></td><td>environment variable holding the shared secret</td></tr>
<tr><td><code>head-blocks</code></td><td>expert blocks this Mac holds on its GPU (3.35 GB each); unset = free memory at plan time minus <code>head-reserve-gb</code></td></tr>
<tr><td><code>state-dir</code></td><td>manifest, head units (head-ssd), BENCH cache, ds4.log, moe.sock</td></tr>
</table>
<h3>Reading REPORT</h3>
<p><b>accepted tok/s</b> decode rate; <b>tau</b> deadline = barrier p99; per node <b>compute</b>, <b>network</b>
(arrival − compute), <b>last</b> = how often it finished last, <b>misses</b> = partial sums dropped by the deadline.
<b>head-ssd</b> appears only while nodes hold fewer than 20/20 blocks.</p>
<h3>Troubleshooting</h3>
<ul><li>A node is missing: check it runs <code>fronk node</code> with the same secret and that the cable/Wi-Fi is up, then press <b>m</b>.</li>
<li>Many deadline misses on one node: close heavy apps there; its compute column grows under load.</li>
<li>ds4 problems: <code>runs/state-head/ds4.log</code>.</li></ul>
"""

INSTALLED_GUIDE = """
<h3>On the head</h3>
<p>Open the Fronk Installer Manager and choose Join nodes. Enter the machine names in installation order.
Return to the head after each installation to verify the connection. Choose Configure model to use your GGUF.</p>
<h3>On a node</h3>
<p>Open the same manager on the named machine. Install the Node role and enter the head's pairing secret.
Reopen the manager and select Status to check the running node.</p>
<h3>Run the cluster</h3>
<p>Start head on the Run tab. Once the plan is ready, proceed to load the weights. Open Chat after Report.
For a terminal session, use the fronk-head launcher in your installation's bin folder.</p>
<h3>Your preferred client</h3>
<p>The default model API is <code>http://127.0.0.1:7710/v1</code> on the head.
Choose an OpenAI-compatible Chat Completions provider and get its model ID from
<code>/v1/models</code>. No API key is needed locally; never use the pairing code.
Read <code>docs/API.md</code> in your installed source for examples and a tested agent tool cycle.</p>
<h3>Share a result</h3>
<p>On the Run tab, Report → Save cluster report downloads hardware, allocation
and the test result. Review the file, add your cables and paste it into a GitHub
Cluster result issue. Nothing is uploaded. Read <code>docs/SHARE.md</code> for an example.</p>
"""

# Shown in both guides. Briefs are written for a model; the page's Copy button fills in the error.
HELP_GUIDE = f"""
<h3 id="help">If something stops</h3>
<p>The first error matters most: keep it, and note which computer showed it. Read
<code>{html.escape(os.path.join(ROOT, "docs", "TROUBLESHOOTING.md"))}</code>, or get help from a frontier model.
A coding agent opened in <code>{html.escape(ROOT)}</code> finds its instructions by itself; attach
<code>docs/llm/DEBUG.md</code> to a chat model. <code>docs/llm/README.md</code> has requests to copy, including
for setup, slow chats and new hardware. Keep pairing codes and private text out of what you share.</p>
"""

PAGE = os.path.join(ROOT, "tools", "fronk-web.html")  # read on every load, so edits show on reload
DEEP_CHAT = os.path.join(ROOT, "tools", "vendor", "deep-chat-2.5.1", "deepChat.bundle.js")
CHAT_PAGE = os.path.join(ROOT, "tools", "fronk-chat.html")
# ds4's memory by context size, from its own load-time plans on this Mac: 8.30 GiB
# at 10240 tokens, 9.02 GiB at 32768; the KV cache grows linearly in between.
DS4_FIXED_GIB, DS4_KV_GIB_PER_TOKEN = 7.97, 0.72 / 22528
CTX_SIZES = (8192, 16384, 32768, 49152, 65536, 98304, 131072)
BLOCK_GB = 3.347  # one expert block (FR_NATIVE_BLOCK_BYTES)


def ctx_file():  # read by tools/ds4-head.sh at the head's start
    return os.path.join(ROOT, conf_read().get("state-dir", "runs/state-head"), "ds4-ctx")


def context_sizes():
    """Each context size with the memory ds4 needs for it on this Mac, against the
    head's default 10 GB reserve for ds4: what fits, and what costs expert blocks
    (choosing more raises head-reserve-gb, so the head plans fewer blocks here)."""
    try:
        current = int(open(ctx_file()).read())
    except (OSError, ValueError):
        current = 32768
    if conf_read().get("ds4") == "tools/gpu-head/run.sh":
        return {"engine": "llama.cpp Vulkan", "current": current, "sizes": [{"ctx": n} for n in CTX_SIZES]}
    sizes = []
    for n in CTX_SIZES:
        gb = (DS4_FIXED_GIB + DS4_KV_GIB_PER_TOKEN * n) * 1.073741824
        over = max(0.0, gb - 10.0)
        sizes.append({"ctx": n, "gb": round(gb, 1), "extra_gb": round(over, 1), "blocks": math.ceil(over / BLOCK_GB)})
    fit = int((10.0 / 1.073741824 - DS4_FIXED_GIB) / DS4_KV_GIB_PER_TOKEN) // 1024 * 1024
    return {"current": current, "reserve_gb": 10.0, "fits_up_to": fit, "sizes": sizes}


OLD_CHATS = os.path.join(ROOT, "runs", "chats")  # first version's place; moved to fronk_chats.DIR at start


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def reply(self, code, body, kind="application/json"):
        data = body.encode() if isinstance(body, str) else body
        self.send_response(code)
        self.send_header("Content-Type", kind)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path == "/api/cluster-report":
            with LOCK:
                snapshot = view_of(list(STATE["lines"]))
            report = fronk_report.render(snapshot, fronk_report.local_hardware(),
                                         fronk_report.version_at(ROOT))
            return self.reply(200, report, "text/markdown; charset=utf-8")
        if self.path == "/":
            c = conf_read(); page = open(PAGE, encoding="utf-8").read().replace("%TOKEN%", TOKEN).replace("%WIKI%", (WIKI if NODES else INSTALLED_GUIDE) + HELP_GUIDE)
            page = page.replace("%NODE_CONTROLS%", "" if NODES else "hidden").replace("%NODE_START%", "starts the development nodes" if NODES else "discovers your installed nodes")
            page = page.replace("%SECRET_HINT%", "set from environment" if STATE["secret"] else "saved on this Mac" if saved_secret() else "required the first time")
            for k in ("model", "burst", "ds4", "interfaces", "head-blocks"):
                page = page.replace(f"%{k}%", html.escape(c.get(k, ""), quote=True))
            return self.reply(200, page, "text/html; charset=utf-8")
        if self.path in ("/chat", "/chat/"):
            page = open(CHAT_PAGE, encoding="utf-8").read().replace("%TOKEN%", TOKEN)
            return self.reply(200, page, "text/html; charset=utf-8")
        if self.path == "/api/model":  # ds4's context size, for the chat page's budgets
            port = int(conf_read().get("control-port", "7700")) + 10
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=3) as r:
                    model = json.load(r)["data"][0]
                    return self.reply(200, json.dumps({"ctx": model.get("context_length") or context_sizes()["current"]}))
            except Exception:  # noqa: BLE001 - no head: the page keeps its default
                return self.reply(503, json.dumps({"message": "ds4-server is not running"}))
        if self.path == "/api/context-size":
            return self.reply(200, json.dumps(context_sizes()))
        if self.path == "/api/chats":
            return self.reply(200, json.dumps(fronk_chats.listing()))
        if self.path.startswith("/api/search?"):
            q = urllib.parse.parse_qs(urllib.parse.urlsplit(self.path).query).get("q", [""])[0]
            return self.reply(200, json.dumps(fronk_chats.search(q)))
        m = re.match(r"/api/chats/([0-9a-z]+)$", self.path)
        if m:
            try:
                return self.reply(200, open(fronk_chats.path(m.group(1)), "rb").read())
            except (OSError, ValueError):
                return self.reply(404, json.dumps({"message": "no such chat"}))
        if self.path == "/vendor/deep-chat.js":  # the Chat view's component, served locally (tools/vendor)
            data = open(DEEP_CHAT, "rb").read()
            self.send_response(200)
            self.send_header("Content-Type", "text/javascript; charset=utf-8")
            self.send_header("Cache-Control", "max-age=86400")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            return self.wfile.write(data)
        if self.path == "/chat/progress":
            return self.reply(200, json.dumps(chat_progress()))
        if self.path == "/state":
            if not FOLLOW_LOG and not head_running():
                demo = installer_demo()
                if demo:
                    return self.reply(200, json.dumps(demo))
            with LOCK:
                lines = list(STATE["lines"]); now = time.time()
                v = view_of(lines)
                if v["phase"] == "TEST":
                    v["generation_progress"] = chat_progress()
                if v["phase"] != STATE["phase"]:
                    STATE["phase"], STATE["phase_t"] = v["phase"], now
                errors = [l for l in lines if re.match(r"(fronk|connection|discovery|selection|plane|serve): ", l)]
                live = {"observer": bool(FOLLOW_LOG), "head_name": os.uname().nodename.split(".")[0], "keys": STATE["keys"], "auto": STATE["auto"], "running": bool(terminal_head_pid()) if FOLLOW_LOG else bool(STATE["proc"] and STATE["proc"].poll() is None),
                        "elapsed": round(now - STATE["started"]) if STATE["started"] else 0,
                        "phase_age": round(now - STATE["phase_t"]) if STATE["phase_t"] else 0,
                        "activity": STATE["activity"], "activity_age": round(now - STATE["activity_t"]) if STATE["activity_t"] else 0,
                        "failure": next((l for l in reversed(errors) if "cycle" not in l), errors[-1] if errors else ""),
                        "lastline": next((l for l in reversed(lines) if l.strip() and not l.startswith("[nodes]")), "")}
            return self.reply(200, json.dumps(dict(v, **live)))
        m = re.match(r"/log\?since=(\d+)$", self.path)
        if m:
            with LOCK:
                since = int(m.group(1)); reset = since > len(STATE["lines"])  # a new head started
                since = 0 if reset else since; lines = STATE["lines"][since:]
                running = bool(STATE["proc"] and STATE["proc"].poll() is None)
                return self.reply(200, json.dumps({"lines": lines, "next": since + len(lines), "running": running, "reset": reset}))
        self.reply(404, "{}")

    def do_POST(self):
        if FOLLOW_LOG:
            return self.reply(409, json.dumps({"message": "This head is controlled by its terminal."}))
        if self.headers.get("X-Fronk-Token") != TOKEN:
            return self.reply(403, json.dumps({"message": "forbidden"}))
        if self.path == "/api/files":  # raw bytes; the name in X-File-Name (URL-encoded)
            size = int(self.headers.get("Content-Length", 0))
            if not 0 < size <= 200 * 1024 * 1024:
                return self.reply(413, json.dumps({"message": "files up to 200 MB"}))
            data = self.rfile.read(size)
            try:
                return self.reply(200, json.dumps(fronk_files.add(urllib.parse.unquote(self.headers.get("X-File-Name", "file")), data)))
            except Exception as e:  # noqa: BLE001 - the reason is shown next to the file
                return self.reply(422, json.dumps({"message": f"cannot read this file: {e}"}))
        try:
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
        except ValueError:
            return self.reply(400, json.dumps({"message": "request body is not JSON"}))
        if not isinstance(body, dict):
            return self.reply(400, json.dumps({"message": "request body must be a JSON object"}))
        if self.path == "/start":
            if head_running() or terminal_head_pid():
                return self.reply(409, json.dumps({"message": "A head or setup check is already running on this machine."}))
            hb = str(body.get("head_blocks", "")).strip()
            if hb and not (hb.isdigit() and int(hb) <= 20):
                return self.reply(200, json.dumps({"message": "blocks on this Mac must be 0-20"}))
            conf_write({"head-blocks": hb})  # empty removes the line: size from free memory
            return self.reply(200, json.dumps({"message": start_head(bool(body.get("refresh")))}))
        if self.path == "/auto":
            with LOCK:
                STATE["auto"] = bool(body.get("on"))
            return self.reply(200, json.dumps({"message": "plans that fit are accepted automatically" if STATE["auto"] else "you accept each plan"}))
        if self.path == "/stop":
            return self.reply(200, json.dumps({"message": stop_head()}))
        if self.path == "/nodes" and body.get("action") in ("start", "stop", "status"):
            return self.reply(200, json.dumps({"message": nodes(body["action"])}))
        if self.path == "/key" and body.get("key") in ("p", "m", "r", "q"):
            return self.reply(200, json.dumps({"message": send_key(body["key"])}))
        if self.path == "/settings":
            if body.get("secret"):
                STATE["secret"] = body.pop("secret")
            conf_write({k: str(v) for k, v in body.items() if k in EDITABLE and re.fullmatch(r"[^\s]*", str(v))})
            return self.reply(200, json.dumps({"message": "settings saved to fronk.conf"}))
        if self.path == "/api/context-size":  # saved for ds4-head.sh; a running head restarts with it
            n = int(body.get("ctx", 0)) if str(body.get("ctx", "")).isdigit() else 0
            if n not in CTX_SIZES:
                return self.reply(400, json.dumps({"message": "unknown context size"}))
            gb = (DS4_FIXED_GIB + DS4_KV_GIB_PER_TOKEN * n) * 1.073741824
            os.makedirs(os.path.dirname(ctx_file()), exist_ok=True)
            open(ctx_file(), "w").write(f"{n}\n")
            # Above the default 10 GB the head keeps more for ds4, so fewer expert blocks here.
            if conf_read().get("ds4") != "tools/gpu-head/run.sh":
                conf_write({"head-reserve-gb": f"{math.ceil(gb * 10) / 10 + 0.2:.1f}" if gb > 10.0 else ""})
            restarted = head_running()
            if restarted:
                stop_head(); start_head(False)
            return self.reply(200, json.dumps({"message": f"context {n} tokens" + (", head restarting" if restarted else ", used at the next start")}))
        if self.path == "/api/context":
            try:
                return self.reply(200, json.dumps(fronk_files.context_for([str(f) for f in body.get("files", [])][:20],
                                                  str(body.get("query", "")), max(300, min(250000, int(body.get("budget", 3000)))),
                                                  body.get("sent") if isinstance(body.get("sent"), dict) else None)))
            except (OSError, ValueError) as e:
                return self.reply(422, json.dumps({"message": str(e)}))
        m = re.match(r"/api/chats/([0-9a-z]+)(/delete|/attach)?$", self.path)
        if m:
            try:
                if m.group(2) == "/delete":  # moved to the chats folder's deleted/, not erased
                    fronk_chats.delete(m.group(1)); return self.reply(200, json.dumps({"message": "deleted"}))
                if m.group(2) == "/attach":  # a past chat, attached to the current one like a file
                    return self.reply(200, json.dumps(fronk_files.add(*fronk_chats.as_document(m.group(1)))))
                fronk_chats.save(m.group(1), body); return self.reply(200, json.dumps({"message": "saved"}))
            except (OSError, ValueError) as e:
                return self.reply(400, json.dumps({"message": str(e)}))
        if self.path == "/chat":
            messages = body.get("messages")
            if not isinstance(messages, list) or not messages or not all(
                    isinstance(m, dict) and m.get("role") in ("system", "user", "assistant") and isinstance(m.get("content"), str) for m in messages):
                return self.reply(400, json.dumps({"message": "messages: a list of {role, content}"}))
            self.send_response(200)  # HTTP/1.0: the stream ends when the connection closes
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            try:
                return chat_stream(messages, body.get("think"), self.wfile, bool(body.get("warm")))
            except (BrokenPipeError, ConnectionResetError):
                return None  # the page stopped reading (Stop or reload)
        self.reply(404, "{}")


if __name__ == "__main__":
    args = sys.argv[1:]
    if args[:1] == ["--replay"]:  # show a finished run's log, e.g. runs/real/run9.log
        STATE["lines"] = open(args[1], errors="replace").read().splitlines(); args = args[2:]
    if "--follow-log" in args:
        i = args.index("--follow-log")
        FOLLOW_LOG = args[i + 1]
        del args[i:i + 2]
    else:
        pid = terminal_head_pid()
        state_dir = os.path.join(ROOT, conf_read().get("state-dir", "runs/state-head"))
        candidate = os.path.join(state_dir, "terminal-head." + pid + ".log")
        if pid.isdigit() and os.path.isfile(candidate):
            FOLLOW_LOG = candidate
    if FOLLOW_LOG:
        if not os.path.isfile(FOLLOW_LOG):
            sys.exit("No terminal head log at " + FOLLOW_LOG)
        threading.Thread(target=follow_terminal, args=(FOLLOW_LOG,), daemon=True).start()
    launch_head = "--start" in args
    if launch_head:
        args.remove("--start")
    open_browser = "--open" in args
    if open_browser:
        args.remove("--open")
    port = int(args[0]) if args else 7730
    fronk_chats.migrate(OLD_CHATS)
    # Vision loads its text models on first use (~14 s); do it now, not on the first photo.
    threading.Thread(target=lambda: subprocess.run([fronk_files.extractor(), "warm"], capture_output=True, timeout=120), daemon=True).start()
    try:
        server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    except OSError as error:
        import errno
        if args or not open_browser or error.errno != errno.EADDRINUSE:
            raise
        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    print(f"fronk console on http://127.0.0.1:{server.server_port}", flush=True)
    if launch_head and not FOLLOW_LOG:
        print(start_head(False), flush=True)
    if open_browser:
        import webbrowser
        webbrowser.open(f"http://127.0.0.1:{server.server_port}/")
    # Stopping the console stops its head too, so nothing keeps running unseen.
    def on_term(*_):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, on_term)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        if head_running():
            print("stopping the head:", stop_head())
