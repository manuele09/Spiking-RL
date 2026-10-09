# E1 — SNN-PPO vs ANN-PPO su task classici (L0-c)

Eseguito su CPU (4 core), 3 seed (0,1,2), PPO del repo (`spiking_rl/algos/ppo.py`), actor 2x128, 8 env.
Valutazione deterministica a fine training (20 episodi, normalizzazione osservazioni congelata).
CartPole: 100k passi. Pendulum: 300k passi. SNN: LIF, encoding diretto, decoder a readout lineare mediato su T, pendenza surrogate schedulata 0.5→2.

| Task | Actor | Ritorno medio ± std (3 seed) | Per seed |
|---|---|---|---|
| CartPole-v1 | ANN | 500.0 ± 0.0 | 500, 500, 500 |
| CartPole-v1 | SNN T=1 | 462.0 ± 53.7 | 500, 500, 386 |
| CartPole-v1 | SNN T=4 | 500.0 ± 0.0 | 500, 500, 500 |
| Pendulum-v1 | ANN | −139.4 ± 12.7 | −129.4, −157.3, −131.4 |
| Pendulum-v1 | SNN T=1 | −125.4 ± 4.9 | −132.3, −121.6, −122.2 |
| Pendulum-v1 | SNN T=4 | −120.9 ± 1.1 | −119.7, −120.6, −122.3 |

Lettura: con questo setup la pipeline SNN-PPO funziona (criterio L0-c: SNN ≥ 90% dell'ANN) senza trucchi particolari oltre all'init rate-aware.
Con soli 3 seed e task semplici NON si può dedurre che l'SNN sia migliore dell'ANN su Pendulum (la differenza è nel rumore di seed
dell'ANN); l'unico segnale è che T=1 è meno stabile su CartPole (1 seed su 3 a 386). Nota: il campionamento di CartPole e Pendulum non
misura energia né robustezza; serve solo a validare la pipeline prima del Go2.
