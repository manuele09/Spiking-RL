# L1 — stand & balance con spinte (Go2, MuJoCo CPU)

Configurazione D: reward rilassato (w_pose 0.05, w_action_rate 0.002, penalità di caduta −1), spinte in training 0.3–1.2 m/s ogni 1–2 s,
std iniziale 0.3, 2.5M passi, PPO 8 env, actor 2×128 (SNN: LIF, encoding diretto, readout lineare mediato su T), critic ANN 2×128.
Valutazione: 30 episodi da 10 s per intensità, spinta di intensità fissa ogni 1.5–3 s, azioni deterministiche, seed di valutazione 50000+.

## Sopravvivenza (episodi arrivati al time-out)

| Actor | 0.5 m/s | 1.0 m/s | 1.5 m/s | 2.0 m/s | 2.5 m/s |
|---|---|---|---|---|---|
| Azione zero (solo PD) | 30/30 | 15/30 | 0/30 | 0/30 | 0/30 |
| ANN seed 0 | 30/30 | 28/30 | 1/30 | 0/30 | 0/30 |
| ANN seed 1 | 30/30 | 28/30 | 0/30 | 0/30 | 0/30 |
| SNN T=4 (seed 0) | 30/30 | 26/30 | 0/30 | 0/30 | 0/30 |
| SNN T=1 (seed 0) | 30/30 | 20/30 | 1/30 | 0/30 | 0/30 |

Firing rate a fine training (strato 1 / strato 2): SNN T=4 0.26 / 0.49; SNN T=1 0.28 / 0.46.

## Cronologia dei tentativi ANN (stesso protocollo di valutazione)
- v0 (reward di piano, std 1.0, spinte ≤ 2.0 m/s, 2M passi): 3/30 a 1.0 m/s, peggio dell'azione zero.
- B (std 0.3, spinte ≤ 1.5): 4/30. C (spinte ogni 1–2 s): 9/30.
- D (reward rilassato + penalità di caduta): 28/30. La causa più plausibile è che le penalità su posa/azioni impedissero il passo di recupero (ipotesi non isolata con ablation: D cambia tre cose insieme).

## Cosa NON si può concludere
- 1 seed per SNN, 2 per ANN; n=30 episodi per punto (IC binomiale al 95% a 28/30: circa 78–99%, a 20/30: 47–83%).
- SNN T=4 vs ANN a 1.0 m/s (26 vs 28/30): differenza dentro il rumore. SNN T=1 (20/30) è plausibilmente peggiore, ma serve più di 1 seed.
- Nessun limite superiore a 1.2 m/s per nessuno: lo stesso setup non generalizza a spinte oltre il range di training.
- Nessuna misura di energia/SynOps; nessuna robustezza ad altro (attrito, carico, latenza).

## Conto operazioni/energia (Appendice B, `python -m spiking_rl.eval.synops`), un episodio di test con spinta 1.0 m/s
Convenzione 45 nm: 0.9 pJ per AC, 4.6 pJ per MAC. NON sono misure; ignorano aggiornamento dei neuroni, memoria e traffico.

| Actor | MAC | AC (spike × fan-out) | Energia stimata |
|---|---|---|---|
| ANN 45→128→128→12 | 23 680 | 0 | 109 nJ |
| SNN T=4 (conservativo: primo strato contato T volte) | 23 040 | 15 457 | 120 nJ (1.1× l'ANN: peggiore) |
| SNN T=4 (ottimizzato: corrente W·obs calcolata una sola volta) | 5 760 | 15 457 | 40 nJ (2.7× meglio dell'ANN) |

Spike medi per inferenza per strato: 101 e 213 (firing rate 0.2 e 0.49). Il secondo strato è molto attivo, e per questo il risparmio è modesto
con full accounting. Coerente con la letteratura (Yan et al. 2409.08290: il vantaggio esiste solo con firing rate molto bassi).
