#!/usr/bin/env python3
"""fronk_bench.py [--url http://127.0.0.1:7710] [--read N] [--write N] [--label TEXT]

Measure the cluster's two speeds on a fixed workload: prompt reading (a fresh
N-word text, never cached) and reply writing (temperature 0). Prints one JSON
line per measurement from the engine's own timings; nothing is saved."""
import argparse, json, os, random, subprocess, time, urllib.request

WORDS = ('casa libro tempo strada mare luce vento pietra fiume monte parola numero forza energia sistema '
         'calcolo modello nodo rete memoria lettura scrittura velocità misura errore prova risultato '
         'metodo funzione valore campo spazio punto linea piano volume massa carica onda fase').split()


def post(url, body, timeout=3600):
    request = urllib.request.Request(url + '/v1/chat/completions', json.dumps(body).encode(),
                                     {'Content-Type': 'application/json'})
    started = time.monotonic()
    with urllib.request.urlopen(request, timeout=timeout) as response:
        result = json.load(response)
    return result, time.monotonic() - started


def memory():
    out = subprocess.run(['vm_stat'], capture_output=True, text=True).stdout
    page = int(out.split('page size of ')[1].split()[0])
    get = lambda name: int(out.split(name + ':')[1].split('.')[0]) * page / 2**30
    swap = subprocess.run(['sysctl', '-n', 'vm.swapusage'], capture_output=True, text=True).stdout
    return dict(free_gb=round(get('Pages free') + get('Pages inactive') + get('Pages speculative'), 2),
                wired_gb=round(get('Pages wired down'), 2), compressed_gb=round(get('Pages occupied by compressor'), 2),
                swap_used=swap.split('used = ')[1].split()[0])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--url', default='http://127.0.0.1:7710')
    parser.add_argument('--read', type=int, default=900)
    parser.add_argument('--write', type=int, default=200)
    parser.add_argument('--label', default='')
    parser.add_argument('--draft', type=int, help='draft tokens per pass for this request (speculative.n_max)')
    parser.add_argument('--temp', type=float, default=0.0, help='sampling temperature (the harness sends none: the engine default, 0.8)')
    parser.add_argument('--seed', help='repeat exactly the same prompts (to test chat reuse)')
    parser.add_argument('--kind', default='prose', choices=('prose', 'code', 'copy'))
    args = parser.parse_args()
    rng = random.Random(args.seed or os.urandom(8))
    text = ' '.join(rng.choice(WORDS) for _ in range(args.read))
    TEXT = ('Il motore a quattro tempi trasforma l\'energia chimica del carburante in lavoro meccanico attraverso '
            'quattro fasi che si ripetono in sequenza. Nella fase di aspirazione il pistone scende e la valvola di '
            'aspirazione si apre, lasciando entrare nel cilindro la miscela di aria e carburante. Nella fase di '
            'compressione entrambe le valvole sono chiuse e il pistone risale, comprimendo la miscela in un volume '
            'molto piccolo e aumentandone temperatura e pressione. Nella fase di espansione la candela accende la '
            'miscela: la combustione fa crescere rapidamente la pressione e spinge il pistone verso il basso, ed e\' '
            'questa la sola fase in cui il motore produce lavoro utile. Nella fase di scarico la valvola di scarico '
            'si apre e il pistone, risalendo, espelle i gas combusti. Il moto alterno del pistone viene trasformato '
            'in moto rotatorio dall\'albero a gomiti tramite la biella, mentre l\'albero a camme, sincronizzato con '
            'l\'albero a gomiti, comanda l\'apertura e la chiusura delle valvole. Un volano immagazzina energia '
            'durante l\'espansione e la restituisce nelle altre tre fasi, rendendo regolare la rotazione del motore. '
            'Il rendimento del motore dipende dal rapporto di compressione, dalla qualita\' della combustione e '
            'dagli attriti interni, e nei motori moderni viene migliorato con l\'iniezione diretta e la fasatura '
            'variabile delle valvole.')
    base = dict(model='qwen3.8-flash-next', stream=False, temperature=args.temp, cache_prompt=True, seed=7)
    if args.draft:
        base['speculative.n_max'] = args.draft
    copy_ask = ('Riscrivi il testo seguente per intero, identico, sostituendo soltanto la parola "motore" con '
                '"propulsore". Scrivi solo il testo.\n\n' + TEXT)
    ask = ('Spiega in italiano, in modo discorsivo e dettagliato, come funziona un motore a quattro tempi.' if args.kind == 'prose' else
           'Scrivi in Python una classe Matrix completa con __init__, __add__, __sub__, __mul__, transpose, determinant e __repr__, con docstring. Solo codice.')
    if args.read:
        result, wall = post(args.url, dict(base, max_tokens=1, messages=[
            {'role': 'user', 'content': f'Prova {rng.random()}. Leggi e rispondi OK.\n{text}'}]))
        t = result.get('timings', {})
        print(json.dumps(dict(label=args.label, test='read', tokens=t.get('prompt_n'), tok_s=round(t.get('prompt_per_second', 0), 2),
                              wall_s=round(wall, 1), memory=memory())), flush=True)
    if args.write:
        result, wall = post(args.url, dict(base, max_tokens=args.write, messages=[
            {'role': 'user', 'content': f'Prova {rng.random()}. {copy_ask if args.kind == "copy" else ask}'}]))
        t = result.get('timings', {})
        print(json.dumps(dict(label=args.label, test='write', tokens=t.get('predicted_n'), tok_s=round(t.get('predicted_per_second', 0), 2),
                              kind=args.kind, draft_n=t.get('draft_n'), draft_accepted=t.get('draft_n_accepted'), wall_s=round(wall, 1),
                              memory=memory())), flush=True)


if __name__ == '__main__':
    main()
