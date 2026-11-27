#!/usr/bin/env python3
"""A five-turn chat that changes subject (C code, Python, Italian prose, English
mail, back to Python): the kind of work whose routing the cache has to follow.
Run on the head: python3 chat_trace.py SOURCE.c [URL]. Prints one line per turn."""
import json, sys, time, urllib.request

URL = sys.argv[2] if len(sys.argv) > 2 else 'http://127.0.0.1:7710'
code = ''.join(open(sys.argv[1]).readlines()[183:272])

STORIA = """La bottega di via dei Fabbri aprì nel 1921, quando il nonno di Marta tornò dalla
Francia con una cassa di attrezzi e l'idea di riparare orologi da torre. Per quarant'anni
lavorò su ingranaggi grandi come ruote di carro, salendo sui campanili di mezza provincia
con una lanterna e un quaderno dove annotava ogni dente consumato. Il figlio, che non amava
le altezze, spostò l'attività sugli strumenti di misura: bilance, manometri, contatori
dell'acqua. Fu lui a comprare il primo tornio di precisione, pagato a rate per sei anni, e a
insegnare il mestiere a una decina di ragazzi del quartiere. Negli anni Ottanta arrivarono
i componenti elettronici e la bottega rischiò di chiudere: riparare costava più che
sostituire. Marta, che aveva studiato fisica, decise di puntare su ciò che le fabbriche non
facevano più: pezzi unici, tarature certificate, restauri per musei. Oggi il laboratorio
ha cinque persone, un forno per la tempra, due macchine a controllo numerico e ancora il
quaderno del nonno, che viene consultato quando un campanile della zona smette di suonare.
Il problema di oggi non è il lavoro, che non manca, ma trovare chi voglia impararlo: servono
anni per riconoscere a orecchio un asse fuori centro, e nessuna scuola lo insegna. Per questo
Marta ha proposto al Comune un corso serale gratuito, tenuto in bottega, con dieci posti
all'anno e un piccolo compenso per chi resta almeno due stagioni."""

TURNS = [
    ("Ecco una parte di un programma C:\n```c\n" + code + "```\nSpiega in italiano, in circa 200 parole, "
     "che cosa fa receive_loop e perché tiene solo la richiesta più recente.", 350),
    ("Ora scrivi in Python una classe che ricompone una richiesta dai suoi frammenti come fa quel codice "
     "(bitmap dei frammenti arrivati, scarto dei duplicati e di quelli in ritardo), con tre test.", 450),
    (STORIA + "\n\nRiassumi questo testo in cinque punti brevi.", 260),
    ("Write a short email in English to a colleague who maintains the network code: explain the three main "
     "risks of sending a large UDP datagram that the IP layer has to fragment, and what you would measure first.", 260),
    ("Torna alla classe Python di prima: aggiungi un contatore dei frammenti duplicati e riscrivi tutta la classe.", 450),
]

messages = [dict(role='system', content='Sei un assistente tecnico preciso e conciso.')]
total_read = total_written = 0; started = time.time()
for n, (text, limit) in enumerate(TURNS, 1):
    messages.append(dict(role='user', content=text))
    body = dict(model='qwen3.8-flash-next', stream=False, temperature=0.8, cache_prompt=True, seed=11,
                max_tokens=limit, messages=messages)
    request = urllib.request.Request(URL + '/v1/chat/completions', json.dumps(body).encode(),
                                     {'Content-Type': 'application/json'})
    t0 = time.time()
    with urllib.request.urlopen(request, timeout=3600) as response:
        result = json.load(response)
    t = result.get('timings', {}); answer = result['choices'][0]['message'].get('content') or ''
    messages.append(dict(role='assistant', content=answer))
    total_read += t.get('prompt_n', 0); total_written += t.get('predicted_n', 0)
    print(json.dumps(dict(turn=n, read=t.get('prompt_n'), read_tok_s=round(t.get('prompt_per_second', 0), 1),
                          written=t.get('predicted_n'), write_tok_s=round(t.get('predicted_per_second', 0), 1),
                          draft_n=t.get('draft_n'), draft_accepted=t.get('draft_n_accepted'),
                          wall_s=round(time.time() - t0, 1), answer_start=answer[:70].replace('\n', ' '))), flush=True)
print(json.dumps(dict(turns=len(TURNS), read=total_read, written=total_written, wall_s=round(time.time() - started, 1))))
