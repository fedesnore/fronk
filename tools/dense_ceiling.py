#!/usr/bin/env python3
"""The engine alone, with the expert hook returning zeros (FRONK_MOE_LOOPBACK=zero):
what the head costs when the experts cost nothing. The text is wrong by design.
Run on the head with the head stopped: dense_ceiling.py SOURCE.c PORT"""
import json, sys, time, urllib.request

port = sys.argv[2]
code = ''.join(open(sys.argv[1]).readlines()[100:330])


def post(body):
    request = urllib.request.Request('http://127.0.0.1:%s/completion' % port, json.dumps(body).encode(),
                                     {'Content-Type': 'application/json'})
    with urllib.request.urlopen(request, timeout=3600) as response:
        return json.load(response).get('timings', {})


base = dict(temperature=0, cache_prompt=False)
t = post(dict(base, prompt='Explain this C code:\n' + code, n_predict=1))
print('reading: %d tokens at %.1f tok/s (%.1f ms a token)' % (t.get('prompt_n', 0), t.get('prompt_per_second', 0), t.get('prompt_ms', 0) / max(t.get('prompt_n', 1), 1)))
for label, extra in (('one token a pass, no drafts', {'speculative.n_max': 0}), ('drafts of 7 (MTP)', {})):
    t = post(dict(base, prompt='The three laws of thermodynamics are', n_predict=160, **extra))
    n, ms = t.get('predicted_n', 0), t.get('predicted_ms', 0)
    print('writing, %s: %d tokens at %.1f tok/s (%.1f ms a token); drafts accepted %s of %s' % (
        label, n, t.get('predicted_per_second', 0), ms / max(n, 1), t.get('draft_n_accepted'), t.get('draft_n')))
