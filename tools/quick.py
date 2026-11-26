#!/usr/bin/env python3
"""The quick yardstick: prose and code only, each setting on the same requests, no restart.
   quick.py [N_MAX:P_MIN[:TEMP[:P_CHAIN]] ...]     e.g. quick.py 12:0.5:0 12:0.15:0:0.3
Each setting: two prose and two code requests of TOKENS tokens, twice, fixed seeds, settings in turn. Per phase:
tok/s, passes, tokens a pass, ms a pass, drafted and accepted tokens. About a minute a
setting. The engine's own launch values are the defaults of every request."""
import json, os, sys, time, urllib.request
URL = 'http://127.0.0.1:7710/v1/chat/completions'; TOKENS = int(os.environ.get('TOKENS', '200'))
# With TEMP 0 the text is the same for every setting: passes and time compare exactly.
PROSE = ['Racconta in modo dettagliato la storia di un faro su un\'isola del nord e della famiglia che lo custodiva.',
         'Spiega a un amico perché il cielo al tramonto diventa rosso, con esempi e senza formule.',
         'Explain how TCP congestion control works, step by step, for a junior engineer.']
CODE = ['Scrivi in C una tabella hash con indirizzamento aperto: inserimento, ricerca, cancellazione e ridimensionamento.',
        'Write a Python class for an LRU cache with get, put, and a method that returns the keys in order of use.']
def run(prompt, seed, n_max, p_min, temp, chain=None):
    body = dict(model='qwen3.8-flash-next', stream=False, temperature=temp, seed=seed, max_tokens=TOKENS, cache_prompt=False,
                messages=[dict(role='user', content=prompt)])
    if n_max: body['speculative.n_max'] = n_max
    if p_min is not None: body['speculative.p_min'] = p_min
    if chain is not None: body['speculative.p_chain'] = chain
    r = urllib.request.Request(URL, json.dumps(body).encode(), {'Content-Type': 'application/json'})
    with urllib.request.urlopen(r, timeout=900) as f:
        return json.load(f).get('timings', {})
def main():
    settings = sys.argv[1:] or ['0:-']
    rounds = int(os.environ.get('ROUNDS', '2'))
    acc = {}   # (setting, phase) -> [tokens, ms, drafted, accepted]
    # Settings take turns on every prompt, in a rotating order: the expert cache learns
    # the prompts as they repeat, and every setting gets the same share of that.
    for r in range(rounds):
        for phase, prompts in (('prose', PROSE), ('code', CODE)):
            for i, prompt in enumerate(prompts):
                order = settings[(r + i) % len(settings):] + settings[:(r + i) % len(settings)]
                for s in order:
                    parts = s.split(':')
                    n_max = int(parts[0]); p_min = None if len(parts) < 2 or parts[1] == '-' else float(parts[1])
                    temp = float(parts[2]) if len(parts) > 2 else 0.8
                    chain = float(parts[3]) if len(parts) > 3 else None
                    t = run(prompt, 11 + 7 * r + i, n_max, p_min, temp, chain)
                    e = acc.setdefault((s, phase), [0, 0, 0, 0])
                    e[0] += t.get('predicted_n', 0); e[1] += t.get('predicted_ms', 0)
                    e[2] += t.get('draft_n', 0); e[3] += t.get('draft_n_accepted', 0)
    print('%-14s %-6s %7s %7s %8s %8s %12s' % ('setting', 'phase', 'tok/s', 'passes', 'tok/pass', 'ms/pass', 'accepted'))
    for s in settings:
        for phase in ('prose', 'code'):
            n, ms, drafted, accepted = acc[(s, phase)]
            passes = max(1, n - accepted)
            print('%-14s %-6s %7.1f %7d %8.2f %8.0f %6d/%-5d' % (s, phase, n / (ms / 1000), passes, n / passes, ms / passes, accepted, drafted), flush=True)

if __name__ == '__main__':
    main()
