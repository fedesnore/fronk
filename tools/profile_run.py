#!/usr/bin/env python3
"""Two plain workloads for the head's per-call profile: a long read, then prose and
code written. Run on the head: profile_run.py SOURCE.c PHASE (read|prose|code)."""
import json, sys, time, urllib.request

URL = 'http://127.0.0.1:7710'
code = ''.join(open(sys.argv[1]).readlines()[60:330])
phase = sys.argv[2]


def chat(messages, limit, seed):
    body = dict(model='qwen3.8-flash-next', stream=False, temperature=0.8, cache_prompt=False, seed=seed,
                max_tokens=limit, messages=messages)
    request = urllib.request.Request(URL + '/v1/chat/completions', json.dumps(body).encode(), {'Content-Type': 'application/json'})
    t0 = time.time()
    with urllib.request.urlopen(request, timeout=3600) as response:
        t = json.load(response).get('timings', {})
    return t, time.time() - t0


if phase == 'read':
    t, wall = chat([dict(role='user', content='Leggi questo codice e rispondi solo OK.\n```c\n' + code + '```')], 1, 3)
    print('read: %d tokens at %.1f tok/s' % (t.get('prompt_n', 0), t.get('prompt_per_second', 0)))
elif phase == 'prose':
    t, wall = chat([dict(role='user', content='Scrivi un testo di circa 400 parole sulla storia dei fari e di chi li custodiva.')], 420, 5)
    print('prose: %d tokens at %.1f tok/s, drafts %s/%s' % (t.get('predicted_n', 0), t.get('predicted_per_second', 0), t.get('draft_n_accepted'), t.get('draft_n')))
else:
    t, wall = chat([dict(role='user', content='Scrivi in Python un parser di file INI con sezioni, commenti e valori tra virgolette, con i test.')], 420, 7)
    print('code: %d tokens at %.1f tok/s, drafts %s/%s' % (t.get('predicted_n', 0), t.get('predicted_per_second', 0), t.get('draft_n_accepted'), t.get('draft_n')))
