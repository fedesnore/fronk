#!/usr/bin/env python3
"""The one yardstick: three fixed prompts (reading, prose, code) on the running head,
with the head's per-call timings (state file path-profile) summed per phase.
Run on the head: measure.py SOURCE.c [LABEL]"""
import json, os, re, sys, time, urllib.request
URL = 'http://127.0.0.1:7710'; LOG = os.path.expanduser('~/fronk-gsq-head/state/head.log')
code = ''.join(open(sys.argv[1]).readlines()[60:330]); label = sys.argv[2] if len(sys.argv) > 2 else ''
def chat(messages, limit, seed):
    body = dict(model='qwen3.8-flash-next', stream=False, temperature=0.8, cache_prompt=False, seed=seed, max_tokens=limit, messages=messages)
    r = urllib.request.Request(URL + '/v1/chat/completions', json.dumps(body).encode(), {'Content-Type': 'application/json'})
    with urllib.request.urlopen(r, timeout=3600) as f: return json.load(f).get('timings', {})
def swapouts():
    for line in os.popen('vm_stat').read().splitlines():
        if 'Swapouts' in line: return int(line.split()[-1].rstrip('.'))
    return 0
PATH = re.compile(r'FRONK_PATH T=(\d+) calls=(\d+) (.*)'); NODE = re.compile(r'FRONK_PATH_NODE T=(\d+) node=(\S+) answered=(\d+) arrival_us=([0-9.]+) last=(\d+)')
def bucket(t): return '1' if t == 1 else '2-4' if t <= 4 else '5-8' if t <= 8 else '9+'
def summary(text):
    agg, nodes = {}, {}
    for line in text.splitlines():
        m = PATH.search(line)
        if m:
            t, calls = int(m.group(1)), int(m.group(2)); a = agg.setdefault(bucket(t), {'calls': 0, 'tokens': 0})
            a['calls'] += calls; a['tokens'] += calls * t
            for k, v in re.findall(r'(\w+)=([0-9.]+)', m.group(3)): a[k] = a.get(k, 0.0) + float(v) * calls
            continue
        m = NODE.search(line)
        if m:
            n = nodes.setdefault((bucket(int(m.group(1))), m.group(2)), [0, 0.0, 0])
            n[0] += int(m.group(3)); n[1] += float(m.group(4)) * int(m.group(3)); n[2] += int(m.group(5))
    for b in ('1', '2-4', '5-8', '9+'):
        if b not in agg: continue
        a = agg[b]; c = a['calls']; g = lambda k: a.get(k, 0.0) / c
        uses = '  '.join('%s %.1f' % (k, g(k)) for k in ('slices', 'cache0', 'cache1', 'cache2') if k in a)
        print('   %-3s tokens (%.1f avg, %5d calls): prepare %.2f ms, wait %.2f, of which reading replies %.2f in %.1f datagrams; local %.2f; uses a call: %s' % (
            b, a['tokens'] / c, c, g('prepare_us') / 1e3, g('wait_us') / 1e3, g('rx_us') / 1e3, g('datagrams'), g('local_compute_us') / 1e3, uses))
        who = ['%s %.2f ms (last %d%%)' % (name, v[1] / v[0] / 1e3, round(100 * v[2] / c)) for (bb, name), v in sorted(nodes.items()) if bb == b and v[0]]
        print('        answers whole after: ' + '; '.join(who))
print('== %s %s' % (time.strftime('%H:%M'), label)); s0 = swapouts()
for phase in ('read', 'prose', 'code'):
    at = os.path.getsize(LOG)
    if phase == 'read':
        t = chat([dict(role='user', content='Leggi questo codice e rispondi solo OK.\n```c\n' + code + '```')], 1, 3)
        print('reading: %d tokens at %.1f tok/s' % (t.get('prompt_n', 0), t.get('prompt_per_second', 0)))
    else:
        prompt = 'Scrivi un testo di circa 400 parole sulla storia dei fari e di chi li custodiva.' if phase == 'prose' else 'Scrivi in Python un parser di file INI con sezioni, commenti e valori tra virgolette, con i test.'
        t = chat([dict(role='user', content=prompt)], 420, 5 if phase == 'prose' else 7)
        n, acc, dn = t.get('predicted_n', 0), t.get('draft_n_accepted') or 0, t.get('draft_n') or 0; passes = max(n - acc, 1)
        print('%s: %d tokens at %.1f tok/s; %d passes of width %.2f, %.0f ms a pass; drafts %d/%d' % (phase, n, t.get('predicted_per_second', 0), passes, 1 + dn / passes, t.get('predicted_ms', 0) / passes, acc, dn))
    time.sleep(0.5)
    with open(LOG, errors='replace') as f: f.seek(at); summary(f.read())
chat([dict(role='user', content='Rispondi solo: ok')], 8, 1)   # a tiny chat last: a later stop has nothing large to save
print('swap-outs during the run: %d pages' % (swapouts() - s0))
