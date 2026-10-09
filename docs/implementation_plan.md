# Piano implementativo e di test — policy spiking per la locomozione del Unitree Go2

Versione 0.1 — 2026-10-09. Responsabile tecnico: (da assegnare). Documenti collegati: `literature_review.md` (rassegna), `sources_verification.md` (cosa è stato verificato e cosa è stato corretto), `scripts/` (smoke test eseguibili).

> **Come leggere.** Ogni affermazione numerica è di uno di tre tipi: **[V]** verificata su fonte/codice (vedi `sources_verification.md`), **[stima]** nostra stima ingegneristica, **[da verificare]** da controllare prima di dipenderci. Nessun comando qui riportato è stato eseguito su GPU: i comandi marcati ✅ sono stati eseguiti nel container (CPU) e funzionano; gli altri sono ricostruiti dai README verificati.

## Indice

0. [Sintesi esecutiva e decisioni chiave](#0-sintesi-esecutiva-e-decisioni-chiave)
1. [Curriculum di compiti a livelli](#1-curriculum-di-compiti-a-livelli)
2. [Architetture da testare](#2-architetture-da-testare)
3. [Setup passo passo](#3-setup-passo-passo)
4. [Protocollo sperimentale](#4-protocollo-sperimentale)
5. [Troubleshooting e piani B](#5-troubleshooting-e-piani-b)
6. [Rischi, decisioni aperte, milestone, pubblicabilità](#6-rischi-decisioni-aperte-milestone-pubblicabilità)
7. [Checklist finale e primo sprint](#7-checklist-finale-e-primo-sprint)
- [Appendice A — Specifica del "core SNN"](#appendice-a--specifica-del-core-snn)
- [Appendice B — Metodologia energia/SynOps](#appendice-b--metodologia-energiasynops)
- [Appendice C — Metriche e definizioni](#appendice-c--metriche-e-definizioni)

---

## 0. Sintesi esecutiva e decisioni chiave

**Obiettivo finale.** Go2 reale che trotta seguendo comandi (vx, vy, ωz), robusto a spinte, con una policy *spiking appresa*. Per quanto trovato (e con una riserva, vedi sotto), non esiste in letteratura.

**Riserva di novità da sciogliere subito (task S1 del primo sprint).** Jiang et al. è uscito a **ICASSP 2025** [V]; una pagina ResearchGate associata contiene testo su un deploy reale su **Go1** che sembra appartenere a un altro lavoro, ma va letto il PDF IEEE finale prima di scrivere "mai fatto su quadrupede reale". Il nostro claim robusto resta comunque: *Go2 + comando (vx,vy,ωz) + spinte + student spiking ricorrente senza history encoder + studio sistematico di robustezza SNN vs ANN + (opz.) sim2real*. Inoltre la tesi `Kostantinoskanell/thesis` [V] (non peer-reviewed, senza licenza) ha già fatto in **simulazione** PPO→belly-flop e distillazione+DAgger su Go2: dobbiamo andare oltre (ricorrenza, robustezza misurata, reale).

**Decisioni chiave raccomandate (dettagli in §3):**

| Decisione | Scelta | Motivo breve |
|---|---|---|
| Simulatore principale per baseline/teacher | **Isaac Lab** (≥2.3, Isaac Sim 5.1) su **GPU NVIDIA ≥16 GB** | Config Go2 verificate (`Isaac-Velocity-Flat/Rough-Unitree-Go2-v0`), rsl_rl con Distillation integrata, `unitree_rl_lab` per deploy [V] |
| Fallback senza Isaac (e sviluppo nel container CPU) | **MuJoCo + PyTorch** (menagerie `unitree_go2`, env Gymnasium vettorizzato nostro) per L0–L2; **MuJoCo Playground/MJX** con env Go2 derivato da `go1/joystick.py` e `RSLRLBraxWrapper` se c'è una GPU ma non Isaac | Stessa libreria RL (rsl_rl) e stesso codice SNN PyTorch in entrambi i casi [V] |
| Sim2sim | **MuJoCo** (menagerie) con il nostro harness + **`unitree_mujoco`** (DDS identico al robot) | Fedele al deploy [V] |
| Libreria SNN | **LIF custom in PyTorch** (Appendice A, `scripts/smoke_lif_surrogate.py` ✅) come default; snnTorch/SpikingJelly solo se il micro-benchmark su GPU target dà >1.5× | Controllo totale su stato, reset per `dones`, export; SpikingJelly è in transizione V2 [V] |
| Algoritmo | PPO (rsl_rl) per ANN; per SNN: **PPO diretto come primo tentativo, distillazione da teacher privilegiato + DAgger come percorso principale** | Due lavori indipendenti riportano PPO-SNN più lento/instabile su quadrupedi [V] |
| Hardware robot | **Go2 Edu** (sviluppo secondario completo) | Air/Pro non consentono il controllo low-level [V sito Unitree] |
| Neuromorfico | Opzionale, dopo il sim2real; Loihi 2 via INRC come co-processore off-board | Lava-DL è archiviato, accesso via proposta [V] |

**Percorso a gradini.** L0 sanity → L1 balance con spinte → L2 balance + altezza/assetto → L3 trotto a velocità fissa → L4 comando completo → L5 robustezza → L6 terreni (leggeri) → L7 sim2sim → L8 sim2real → L9 neuromorfico (opz.). Ogni livello ha go/no-go numerici; l'ANN deve passare un livello **prima** che l'SNN lo tenti.

---

## 1. Curriculum di compiti a livelli

### 1.0 Convenzioni comuni a tutti i livelli (Go2)

Fissate una volta e usate identiche in Isaac Lab, MuJoCo e sul robot.

| Elemento | Valore | Fonte |
|---|---|---|
| Frequenza policy / fisica | 50 Hz (dt 0.02) / 200 Hz (dt 0.005, decimation 4) | Isaac Lab [V] |
| Posa di default (hip, thigh, calf) | **Isaac Lab:** (±0.1, 0.8 ant. / 1.0 post., −1.5); **menagerie `home`:** (0, 0.9, −1.8), z=0.27 | [V] — scegliere **una** posa (raccomandato: quella Isaac Lab, portandola nel MJCF come keyframe) e non cambiarla più |
| Azione | 12 offset di posizione: `q_des = q_default + 0.25·a`; rsl_rl non clippa `a` di default (`clip_actions=None` [V]), Playground clippa a ±1 [V]: noi clippiamo `a` a ±4 (⇒ offset ±1 rad) in tutti i simulatori e sul robot | Isaac Lab Go2 scale 0.25 [V] |
| PD | Kp 25, Kd 0.5, coppia satura 23.5 Nm (Isaac) — nel MJCF menagerie i limiti sono ±23.7 (hip/thigh) e ±45.4 (knee) [V]; **usare 23.5 ovunque** per coerenza con il robot | [V] |
| Osservazioni actor (45-d) | ω_base (3), gravità proiettata (3), comando (3), q−q_default (12), q̇ (12), ultima azione (12). **Niente velocità lineare** nell'actor (il Go2 non la misura) | [V] Genesis/unitree_rl_gym; **in Isaac Lab main la lin_vel è nella policy e va rimossa → gruppo `critic` separato** [V] |
| Osservazioni critic (privilegiate) | actor + v_base (3) + attrito, massa aggiunta, offset COM, Kp/Kd scale, forza esterna corrente, contatti piedi (4), eventuale height scan | nostro |
| Scale/rumore osservazioni | rumore uniforme come Isaac Lab (ω ±0.2, grav ±0.05, q ±0.01, q̇ ±1.5) [V]; normalizzazione empirica delle osservazioni (running mean/std) congelata al deploy | nostro |
| Terminazioni | contatto base; \|roll\| o \|pitch\| > 1.0 rad; time-out 20 s | Isaac + nostro |
| Seed | 5 seed per risultati principali, 3 per ablation | §4 |

### 1.1 Tabella del curriculum

| Liv. | Nome | Scopo | Simulatore | Go/no-go (ANN) | SNN passa se |
|---|---|---|---|---|---|
| L0 | Sanity pipeline | validare LIF, PPO-SNN, env Go2 | CPU (Gymnasium Pendulum/CartPole + MuJoCo) | — | vedi §1.2 |
| L1 | Stand & balance con spinte | equilibrio statico/di recupero | MuJoCo CPU **o** Isaac | sopravvivenza ≥ 97 % a spinte ≤ 1.0 m/s | ≥ 95 % e ≤ 1.2× tempo di recupero dell'ANN |
| L2 | Balance + altezza/assetto comandati | tracking continuo senza locomozione | idem | RMSE h ≤ 1.5 cm, pitch ≤ 2° | RMSE ≤ 1.3× ANN |
| L3 | Trotto avanti a velocità fissa | gait | Isaac (o MJX) | \|e_vx\| ≤ 0.10 m/s, cadute ≤ 2 % | ≤ 0.12 m/s, cadute ≤ 3 % |
| L4 | Comando (vx, vy, ωz) | obiettivo funzionale | Isaac (o MJX) | RMSE vx/vy ≤ 0.12, ωz ≤ 0.20 rad/s; cadute ≤ 2 % | ≤ 1.25× ANN su ogni metrica, cadute ≤ 3 % |
| L5 | Robustezza | spinte 1.5 m/s, attrito 0.3, +5 kg, latenza 40 ms, rumore ×2, dropout | Isaac + MuJoCo | nessuna condizione con cadute > 10 % | report comparativo (anche negativo) |
| L6 | Terreni leggeri | ghiaia/pendii ≤ 15°, gradini ≤ 5 cm | Isaac rough | cadute ≤ 5 % | ≤ 8 % |
| L7 | Sim2sim | gap tra motori di fisica | MuJoCo + unitree_mujoco | degradazione metriche ≤ 20 %, cadute ≤ 3 % | idem |
| L8 | Sim2real | Go2 Edu | reale | ANN cammina e regge spinte moderate | SNN idem (claim principale) |
| L9 | Neuromorfico (opz.) | Loihi 2 co-processore | reale + Loihi | — | tracking entro 1.3× della GPU |

### 1.2 L0 — Sanity della pipeline (CPU, 1–2 giorni)

**L0-a Core SNN (✅ eseguito).** `python scripts/smoke_lif_surrogate.py` deve stampare `RISULTATO: OK`: firing rate per strato in (0.05, 0.6), gradiente al primo strato > 0, reset per maschera `dones` corretto. Risultato nel container: rate [0.215, 0.217], grad 7.4e-3, T=4 batch 4096 hidden 256 → 112 ms fwd+bwd su 4 CPU (bench). **Nota:** con l'init di default di `nn.Linear` il secondo strato era morto (rate 0.0): l'init "rate-aware" dell'Appendice A è obbligatoria.

**L0-b Go2 in MuJoCo (✅ eseguito).** `python scripts/smoke_go2_mujoco.py`: nq=19, nv=18, nu=12; PD Kp 25/Kd 0.5 sulla posa `home` → base a 0.208 m (parte da 0.27: **il PD a Kp 25 cede ~6 cm sotto gravità, pitch −0.10 rad**: normale, la compensazione è compito della policy); ~10.9 k physics step/s per env su CPU → ~2.7 k policy step/s per env. Con 16–64 env vettorizzati su 4 core [stima] 5–10 k policy step/s: sufficiente per L1/L2 (≤ 20 M step), non per L3+ (≥ 150 M step).

**L0-c SNN-PPO su task classico.** Pendulum-v1 e CartPole-v1 (Gymnasium) con lo stesso codice SNN che userete per il Go2 (actor spiking + critic ANN), PPO (CleanRL-style o SB3 con policy custom).
- Input: obs normalizzate; T ∈ {1, 4}; 2×128 LIF.
- Successo: SNN raggiunge ≥ 90 % del ritorno dell'ANN MLP entro 2× i campioni dell'ANN, su 3 seed. Pendulum: ANN ≈ −150 ± 50 in ~200 k step [stima]; CartPole: 500/500.
- Se fallisce: §5.1 prima di toccare il Go2.

**L0-d Baseline Isaac Lab (se GPU disponibile).** `Isaac-Velocity-Flat-Unitree-Go2-v0`, 300 iterazioni, deve dare un Go2 che cammina (reward di tracking > 0.8·max). È il test del setup Isaac, non ancora la baseline del paper (ha lin_vel nelle obs e spinte off).

### 1.3 L1 — Stand & balance con spinte

| Voce | Definizione |
|---|---|
| Task | Il robot parte in piedi nella posa di default (con rumore ±0.1 rad sui giunti, ±5° su roll/pitch) e deve restare in piedi; comando = 0. |
| Osservazioni | come §1.0 con comando fisso a 0 (manteniamo i 3 numeri per non cambiare l'interfaccia). |
| Azioni | 12 offset, scale 0.25. |
| Reward (pesi iniziali) | orientamento piatto −2.5·‖g_xy‖²; altezza base −20·(h−h_ref)² con h_ref = altezza della posa di default (≈0.30–0.32 m in Isaac [da misurare in sim]); lin_vel_xy −1·‖v_xy‖²; lin_vel_z −2; ang_vel −0.05·‖ω‖²; posa −0.5·‖q−q_def‖²; coppie −2e-4; action rate −0.01; acc. giunti −2.5e-7; penalità contatti non-piede −1; bonus sopravvivenza +0.5/step (×dt). |
| Terminazioni | contatto base/coscia con il suolo; \|roll\|,\|pitch\| > 1 rad; h < 0.15 m; time-out 10 s. |
| Randomizzazioni | attrito 0.5–1.25; massa base −1…+3 kg; COM ±3 cm; Kp/Kd ±10 %; ritardo azione 0–1 step (0–20 ms); rumore obs come §1.0. |
| Disturbi | `push_by_setting_velocity` su x e y con \|Δv\| uniforme in [0.3, 1.0] m/s ogni 3–6 s (più frequente che in Isaac Lab, dove è 10–15 s e **disabilitato** nel config Go2 [V]); nel 20 % degli episodi anche impulsi di forza 50–150 N per 0.1–0.3 s. |
| Metriche | tasso di sopravvivenza a 10 s; tempo di recupero (da spinta a ‖v_xy‖<0.1 m/s e ‖g_xy‖<0.05); max inclinazione dopo spinta; coppia media; firing rate per strato (SNN); SynOps/inferenza. |
| Go/no-go ANN | sopravvivenza ≥ 97 % su 500 episodi con spinte fino a 1.0 m/s; recupero medio ≤ 1.0 s. |
| SNN passa se | sopravvivenza ≥ 95 % e recupero ≤ 1.2× ANN, con **lo stesso budget di campioni ×2**. |
| Criterio per L2 | ANN e almeno un'architettura SNN passano; curve e tabella archiviate (§4). |

Perché L1 prima di tutto: separa "il decoder SNN produce azioni abbastanza precise e stabili" dal problema del gait; è anche il primo task sensato su robot reale (L8-ii) e l'unica cosa che un SNN+RL ha già fatto su un quadrupede reale (progetto PuppyPi, [S] nella rassegna).

### 1.4 L2 — Balance + altezza e assetto comandati

| Voce | Definizione |
|---|---|
| Task | come L1 ma con comando (h_cmd, pitch_cmd, roll_cmd) ricampionato ogni 3–5 s: h ∈ [0.22, 0.34] m, pitch ∈ ±0.25 rad, roll ∈ ±0.15 rad; comando di velocità sempre 0. |
| Osservazioni | §1.0 con il vettore comando esteso a 6 (vx, vy, ωz, h, pitch, roll) — **da qui in poi l'interfaccia resta a 6 comandi**, con h/pitch/roll fissati ai default nei livelli successivi (o rimossi: decisione D4, §6). |
| Reward | L1 con tracking: +1.0·exp(−(h−h_cmd)²/0.0025) +0.5·exp(−(θ−θ_cmd)²/0.01). |
| Metriche | RMSE altezza (cm), RMSE pitch/roll (°), tempo di assestamento al cambio comando, sopravvivenza alle spinte. |
| Go/no-go ANN | RMSE h ≤ 1.5 cm, pitch ≤ 2°, roll ≤ 2°; spinte come L1 ≥ 95 %. |
| SNN passa se | ogni RMSE ≤ 1.3× ANN. |

Motivazione: è il test più diretto della **precisione di decodifica** (rate vs membrana, T) senza la complessità del gait. Se la membrana con T=1 non raggiunge 1.5 cm, lo scoprite qui, non a L4.

### 1.5 L3 — Trotto avanti a velocità fissa

| Voce | Definizione |
|---|---|
| Task | vx_cmd = 0.5 m/s costante (vy = ωz = 0) su piano; episodi 20 s. Poi vx ∈ {0.3, 0.5, 0.8}. |
| Reward | Isaac Lab Go2 flat [V]: track_lin 1.5 (σ²=0.25), track_ang 0.75, lin_vel_z −2, ang_vel_xy −0.05, flat_orientation −2.5, torques −2e-4, dof_acc −2.5e-7, action_rate −0.01, feet_air_time 0.25 (soglia 0.5 s), + **penalità slittamento piedi −0.1** e **feet_clearance −0.5** (da Playground Go1 [V]) per avere un trotto pulito; terminazione contatto base. |
| Randomizzazioni e spinte | L1 + spinte 10–15 s (Isaac default, riattivate). |
| Metriche | errore medio e RMSE di vx; tasso di cadute; duty factor e simmetria del gait (fase relativa diagonale ≈ 0.5); altezza di passo; CoT = P_mecc/(m·g·v); jitter azioni (‖Δa‖ medio); firing rate; SynOps. |
| Go/no-go ANN | \|e_vx\| medio ≤ 0.10 m/s; cadute ≤ 2 % su 500 ep; gait diagonale (fase 0.5 ± 0.1). |
| SNN passa se | \|e_vx\| ≤ 0.12, cadute ≤ 3 %, jitter ≤ 1.5× ANN. |

### 1.6 L4 — Comando completo (vx, vy, ωz)

| Voce | Definizione |
|---|---|
| Task | comandi ricampionati ogni 10 s: vx ∈ [−1.0, 1.0], vy ∈ [−0.6, 0.6], ωz ∈ [−1.0, 1.0] rad/s; 10 % di episodi "stand still" (0,0,0); modalità heading opzionale off. (Isaac usa vy ±1.0 [V]; ±0.6 è più realistico per il Go2 e per il reale: decisione D5.) |
| Reward | L3. Aggiungere `stand_still` −1 (movimento giunti con comando nullo, da Playground [V]). |
| Metriche | RMSE vx, vy, ωz per fascia di comando; cadute; tempo di risposta al cambio comando (90 %); CoT; firing rate; SynOps. |
| Go/no-go ANN | RMSE vx ≤ 0.12, vy ≤ 0.12, ωz ≤ 0.20; cadute ≤ 2 %; risposta ≤ 1.0 s. |
| SNN passa se | ≤ 1.25× ANN su ogni RMSE; cadute ≤ 3 %. |
| Criterio per L5 | tabella per fascia di comando (anche vy e ωz, i più deboli per SNN secondo la tesi Go2 [V]) con 5 seed. |

### 1.7 L5 — Robustezza (studio SNN vs ANN a parità di reward)

Protocollo di valutazione (nessun training nuovo, salvo la variante "trained-with-disturbances"):

| Asse | Livelli di test | Metrica |
|---|---|---|
| Spinte laterali/frontali | Δv ∈ {0.5, 1.0, 1.5, 2.0} m/s, a vx_cmd ∈ {0, 0.5} | % recuperi, tempo di recupero |
| Attrito | μ ∈ {0.3, 0.5, 0.8, 1.25} | cadute, RMSE |
| Carico | +2, +5, +8 kg sulla base (COM spostato 5 cm) | cadute, RMSE, altezza |
| Latenza | 0, 20, 40, 60 ms | cadute, RMSE |
| Rumore obs | ×1, ×2, ×4 del livello di training | RMSE |
| Dropout sensori | 10 % dei canali q̇ a zero per 0.2 s ogni 2 s; IMU bias 2° | cadute |
| Perturbazioni avversarie | PGD sulle obs (ε = 0.05 normalizzato) [solo in sim] | RMSE |
| Guasto attuatore | Kp −30 % su una gamba | cadute |

Selezione delle policy "a parità di reward": per ogni coppia ANN/SNN scegliere i checkpoint con reward di training entro ±3 % l'uno dall'altro (altrimenti il confronto è confuso dalla qualità del training). Go/no-go ANN: nessuna condizione "nominale+1 livello" con cadute > 10 %. **Risultato pubblicabile in ogni caso**, incluso "l'SNN non è più robusta" (la tesi Go2 ha trovato esattamente questo con rumore LiDAR [V]).

### 1.8 L6 — Terreni leggeri (opzionale per l'obiettivo, consigliato per la robustezza reale)

Isaac Lab `Rough-Unitree-Go2` con curriculum ridotto: pendii ≤ 15°, gradini ≤ 5 cm, ghiaia. Senza height scan (propriocettivo puro, come il deploy). Go/no-go: cadute ≤ 5 % (ANN), ≤ 8 % (SNN). È il livello in cui la **memoria** (GRU vs SNN ricorrente) conta di più: è l'arena per l'architettura N1 (§2).

### 1.9 L7 — Sim2sim (MuJoCo)

1. Harness nostro su menagerie `scene.xml` (stessa posa, Kp/Kd, scale, normalizzazione, obs) — eseguire L4/L5 con la stessa policy. Go/no-go: metriche entro +20 % rispetto a Isaac, cadute ≤ 3 %.
2. `unitree_mujoco` (C++) + controller Python/C++ che parla DDS `LowCmd/LowState` [V]: testa **l'intera catena di deploy** (parsing IMU, ordine giunti, segni, unità). Go/no-go: il robot simulato cammina con il controller "di produzione" per 60 s con comandi da joystick.
3. Test di latenza: iniettare 10/20/40 ms e confrontare con L5.

### 1.10 L8 — Sim2real su Go2 Edu

Fasi obbligatorie, ognuna con criterio di stop:
- **L8-0 Misure sul robot**: latenza DDS round-trip (ms), frequenza LowState (Hz), ordine giunti/segni, Kp/Kd efficaci (step response su un giunto in aria), offset di zero. Serve per impostare la randomizzazione della latenza (20 ms è un dato Go1 [NV]).
- **L8-i Replay open-loop** con il robot appeso: inviare q_des registrate dal sim; verificare tracking di posizione (RMSE ≤ 0.05 rad) e assenza di saturazioni.
- **L8-ii L1/L2 sul robot** su tappeto di gomma, con imbragatura allentata: stand con spinte manuali leggere (≤ 0.5 m/s stimati), poi altezza/assetto. Criterio: 10 prove senza caduta.
- **L8-iii L3** a 0.3 m/s, poi 0.5, poi L4 con joystick; spinte via asta con cella di carico [da procurare] o impulsi di forza misurati.
- Ordine: **ANN prima**, poi SNN con lo **stesso** controller; qualsiasi fallimento ANN è un problema di deploy, non di SNN.
- Sicurezza: e-stop su timeout DDS, limiti di coppia 80 % all'inizio, zona delimitata, due persone.

### 1.11 L9 — Neuromorfico (opzionale)

Loihi 2 via INRC (vLab SSH) [V]: (i) quantizzazione int8 pesi / int16 stato in sim con training QAT; (ii) replay offline di osservazioni registrate dal Go2 → confronto azioni (RMSE); (iii) anello chiuso in sim2sim con Loihi come co-processore; (iv) reale. Stewart et al. hanno mostrato 4.2 ms/inferenza ma tracking peggiore per la quantizzazione [V]: il go/no-go è "RMSE tracking ≤ 1.3× della GPU".

---

## 2. Architetture da testare

### 2.1 Convenzioni di descrizione
- Ingresso 45-d (o 48 con comandi estesi), uscita 12-d (media della gaussiana; log-std è un parametro separato non-spiking, come in PopSAN).
- **T** = passi interni della SNN per ogni passo di controllo (50 Hz). T=4 ⇒ la rete "gira" a 200 Hz.
- Critic sempre ANN MLP [512,256,128] con osservazioni privilegiate (scartato al deploy).
- Costo in "unità ANN" = tempo di training rispetto alla baseline A0 nello stesso ambiente [stima].

### 2.2 (a) Architetture esistenti in letteratura

| ID | Nome | Schema | Training | Costo | Rischi | Ruolo |
|---|---|---|---|---|---|---|
| **A0** | ANN MLP (baseline obbligatoria) | 45→512→256→128→12 ELU (flat: 128×3) | PPO rsl_rl, hyper Isaac Go2 [V] | 1× (flat 300 iter ≈ 10–20 min su RTX 4090 [stima]; rough 1500 iter ≈ 1–3 h [stima]) | nessuno | riferimento, teacher-non-privilegiato |
| **A1** | ANN ricorrente (GRU) + teacher privilegiato (RMA/Lee-style) | teacher: A0 + obs privilegiate; student: GRU 256 → MLP, obs propriocettive | PPO teacher, poi `Distillation` rsl_rl (student agisce, teacher etichetta, TBPTT 15) [V] | 1.5–2× | drift student su terreni difficili | baseline "con memoria" contro cui confrontare N1 |
| **A2** | PopSAN-style actor | encoding a popolazione (10 neuroni gaussiani per dim → 450), 2×256 LIF, decodifica rate → lineare float; T=5 | PPO diretto, surrogate rettangolare/arctan | 2–4× | convergenza lenta (Jiang, tesi) [V]; decodifica float = costo "full accounting" | riproduce lo stato dell'arte |
| **A3** | Fully spiking, decodifica a membrana (ILC-SAN/Jiang-like) | encoding diretto (corrente) 45→256→256 LIF → 12 neuroni non-spiking (membrana integrata su T); T ∈ {1,2,4} | PPO diretto, pendenza surrogate schedulata (2510.24461) [V] | 2–3× | T=1 può non bastare per precisione (L2 lo scopre); neuroni morti | candidato deploy "senza float" |
| **A4** | SNN student feedforward distillato | A3 come student; teacher A0/A1 privilegiato | `Distillation` rsl_rl + schedule DAgger nostro (β da 0.5 → 0) [da implementare: rsl_rl non mixa [V]] | 1× teacher + 1× student | copia errori del teacher | percorso principale se A2/A3 non convergono |
| **A5** | Conversione ANN→SNN + CRPI | A0 convertito, T=8–32 | nessun RL; CRPI (2601.21778, ICML 2026) [V] | 0.2× | T grande = latenza; errori correlati | controllo negativo/positivo per il claim "serve il training spiking" |

### 2.3 (b) Architetture nuove proposte

Per ciascuna: schema, training, costo, rischio, perché potrebbe essere nuova, verifiche di novità da rifare **prima** di investire (una ricerca mirata su arXiv/Scholar con le stringhe indicate), baseline necessarie per un claim onesto.

#### N1 — Student spiking ricorrente senza history encoder (contributo principale, rischio medio)
- **Schema.** 45 → 256 LIF → **256 ALIF ricorrente** (W_rec sparsa 20 %, soglia adattiva con τ_adapt appresa 0.2–2 s) → 12 neuroni di membrana. **Lo stato di membrana/adattamento NON viene resettato tra passi di controllo** (carry), solo a fine episodio (maschera `dones`, già nel core ✅). T=2 interni. La memoria a lungo termine (attrito, carico, latenza) deve emergere nelle soglie adattive, sostituendo la GRU/1D-CNN di RMA.
- **Training.** Teacher A1-privilegiato (o A0 con obs privilegiate) → `Distillation` rsl_rl con TBPTT (`gradient_length` 15–50) [V] + mixing DAgger β nostro; fine-tuning PPO opzionale con lr 1e-4. Stage 2: aggiungere una loss ausiliaria di **ricostruzione delle extrinsics** (attrito, massa, latenza) da una testa lineare sullo stato nascosto — misura esplicita della memoria.
- **Costo.** 2× A1 [stima]. BPTT su 50 passi × T=2 × 4096 env: memoria ~ 4–8 GB [stima].
- **Rischi.** Student non eguaglia il teacher su L5/L6 (drift); esplosione di gradiente nella ricorrenza (→ §5.2); i test di L5 potrebbero mostrare che la memoria non serve su piano (allora il claim si sposta su L6).
- **Perché nuova.** La rassegna non ha trovato SNN usate come *history/adaptation encoder* per locomozione; Jiang usa un adaptation module separato [V]; la tesi Go2 usa student feedforward [V].
- **Verifiche di novità da rifare.** Stringhe: "spiking" + ("history encoder" | "adaptation module" | "RMA") + ("quadruped" | "legged"); "adaptive threshold" + "locomotion" + "spiking"; controllare Xiao et al. 2025 (*Control and Decision*) e la versione ICASSP di Jiang.
- **Baseline per un claim onesto.** A1 (GRU student) con **stesso teacher e stesso budget**; N1 senza carry (= A4) ; N1 con LIF non adattivi; ANN MLP con stack di 50 obs (RMA-encoder). Ablation sulla loss di extrinsics.

#### N2 — CPG spiking appreso + residuo spiking (rischio medio)
- **Schema.** Due moduli: (i) **CPG spiking**: 4 popolazioni di 32 LIF con accoppiamenti laterali appresi (matrice 4×4 di pesi inibitori/eccitatori) e **frequenza modulata dal comando** (corrente tonica = f(‖v_cmd‖)); uscita: fase per gamba via lettura a membrana (sin/cos); (ii) **residuo**: SNN A3 che riceve obs + fasi CPG e produce correzioni Δq (scale 0.15). Il PD riceve q_CPG(fase) + Δq con q_CPG una traiettoria di piede parametrica (altezza passo 8 cm) come in CPG-RL [V 2212.14400 solo rassegna].
- **Training.** PPO end-to-end sui pesi di accoppiamento + residuo (gradienti attraverso le membrane); in alternativa CPG fissato a mano e solo residuo appreso (variante "N2-fixed", più sicura).
- **Costo.** 1.5× A3. **Rischi.** Il CPG spiking può bloccarsi (fase ferma) a comando nullo — serve un "gate" di stand-still; vicino a Han & Sengupta 2605.09595 (CPG non spiking + residuo + EP) [V]: la novità è "tutto spiking e appreso con PPO + comando vy/ωz + spinte".
- **Verifiche di novità.** "spiking CPG" + "residual" + "reinforcement learning" + "quadruped"; leggere 2605.09595 e 2312.15805 per intero.
- **Baseline.** CPG-RL non spiking (oscillatori di Hopf) con stesso residuo; N2-fixed; A3.
- **Analisi extra.** Robustezza a spinte durante il ciclo (fase di risposta), Pareto CoT vs spike/passo, transizione di gait a diverse velocità.

#### N3 — Neuroni multi-scala temporale eterogenei (add-on a basso rischio)
- **Schema.** In A3/N1 ogni neurone ha τ_mem, τ_syn e τ_adapt **appresi e inizializzati da una distribuzione log-uniforme (5 ms – 2 s)** invece di un τ per strato. Stesso costo di A3.
- **Training.** Identico a A3/N1; parametrizzazione τ = softplus con clamp.
- **Perché nuova.** Huebotter et al. [V] mostrano l'importanza di τ appresi e soglie adattive su un braccio model-based; nessun lavoro trovato lo fa in RL locomotorio con analisi dell'eterogeneità risultante.
- **Verifiche.** "heterogeneous time constants" + "spiking" + "reinforcement learning"/"locomotion" (Perez-Nieves et al. 2021 è il riferimento su task supervisionati [NV]).
- **Baseline.** A3/N1 con τ omogeneo per strato; ANN GRU. **Analisi.** istogramma dei τ dopo il training vs condizione (attrito/carico), ablation per gruppo di τ.

#### N4 — Riflesso spinale spiking veloce sotto la policy lenta (rischio alto)
- **Schema.** Policy SNN a 50 Hz produce q_des; uno strato "spinale" di 12×16 LIF a **200–500 Hz** riceve (q, q̇, stima contatto da corrente/coppia) e modula il PD: Δτ = W·spike (clip ±5 Nm) o scala Kp locale. Idea: risposta a spinte/scivolamenti entro 5 ms invece di 20 ms.
- **Training.** PPO congiunto con due scale temporali (lo strato spinale vede 4–10 passi per azione della policy) — oppure appreso **offline** come regressore di un controllore di riflesso (stance stiffening) e poi fine-tuned.
- **Costo.** 3× A3 (simulazione a decimazione 1). **Rischi.** Instabilità del PD modulato; nessuna evidenza che a 50 Hz il budget sia il collo di bottiglia su piano; difficile da deployare (richiede loop a 500 Hz sul robot: il DDS Go2 fornisce LowState a 500 Hz [da verificare]).
- **Verifiche.** "spiking reflex" + "legged"/"quadruped"; "spinal" + "spiking" + "robot". **Baseline.** Policy a 100 Hz senza riflesso; riflesso ANN. **Analisi.** Tempo di risposta alla spinta (ms) e forza massima tollerata.

#### N5 — Testa di adattamento plastica a tre fattori su base congelata (rischio alto)
- **Schema.** N1 congelata + uno strato di pesi W_plast (256×64) aggiornati online con regola e-prop/tre fattori: ΔW = η · e_trace(pre,post) · M, con M = segnale di sorpresa (errore di predizione di una testa che predice q̇_{t+1}). Nessun RL online.
- **Training.** Meta-ottimizzazione di η e delle forme delle tracce in sim (come SMA [V]); valutazione su cambi di attrito/carico in L5.
- **Rischi.** SMA esiste già (simile); la tesi Go2 ha trovato R-STDP inutile o dannoso [V]. Da fare **solo** se N1 mostra limiti chiari di adattamento in L5/L6.
- **Verifiche.** Leggere SMA per intero e cercare "three-factor" + "locomotion" + 2024–2026. **Baseline.** N1 senza plasticità; RMA (A1).

#### N6 — Propriocezione event-based + stimatore spiking di velocità (rischio medio-alto, opzionale)
- **Schema.** Encoder delta: spike quando |Δobs| > θ (per canale), con refresh periodico ogni 0.5 s; stimatore spiking ricorrente che ricostruisce v_base (supervisionato dal sim) e la fornisce all'actor come in Concurrent Estimator [V rassegna].
- **Rischi.** A robot fermo l'informazione svanisce (refresh obbligatorio); il guadagno è solo sull'input (poche SynOps). Utile soprattutto per L9.
- **Verifiche.** "event-based proprioception"/"delta modulation" + "legged". **Baseline.** Stimatore ANN; actor senza stimatore.

### 2.4 Priorità e ordine
1. A0 → A1 (baseline e teacher) → A3 e A2 (PPO diretto, 2 tentativi max) → A4 (distillazione) → **N1** (+N3 come variante) → L5 robustezza → N2 → (N4/N5/N6 solo con risultati e tempo).
2. Alto rischio dichiarato: **N4, N5**; medio-alto: N6; medio: N1, N2; basso: N3.

---

## 3. Setup passo passo

### 3.1 Confronto simulatori e raccomandazione

| Criterio | Isaac Lab (Isaac Sim 5.1) | MuJoCo Playground (MJX/Warp) | Genesis | MuJoCo CPU + PyTorch | mjlab |
|---|---|---|---|---|---|
| Go2 pronto | **sì**, flat/rough [V] | **no** (solo Go1 [V]) → env da derivare + `go2_mjx.xml` di menagerie [V] | sì, esempio minimale [V] | modello sì (menagerie), env da scrivere | no (G1) [V] |
| GPU | obbligatoria, ≥16 GB VRAM, Ubuntu 22.04, Py 3.11 [V] | obbligatoria (JAX CUDA 12) [V] | consigliata | **no** | obbligatoria |
| RL stack | rsl_rl + Distillation integrata [V] | brax PPO (JAX) **o** rsl_rl via `RSLRLBraxWrapper` [V] | rsl_rl ≥5 [V] | qualsiasi (SB3/CleanRL/rsl_rl) | rsl_rl |
| Deploy | `unitree_rl_lab` → `unitree_mujoco` → sdk2 [V] | nessuno | nessuno | `unitree_mujoco` | — |
| Maturità Go2 | alta | media (Go1 maturo, Go2 no) | bassa | — | bassa |
| Nel container attuale | **no** (niente GPU) | no | no (lento) | **sì** ✅ | no |

**Raccomandazione.** Principale: **Isaac Lab**. Sim2sim: **MuJoCo**. Fallback principale se Isaac non è disponibile: **MuJoCo Playground + env Go2 derivato da `go1/joystick.py`** (sostituendo il modello con `unitree_go2/go2_mjx.xml`, Kp/Kd/posa/obs come §1.0) con **rsl_rl via `learning/train_rsl_rl.py`/`RSLRLBraxWrapper`** così il codice SNN PyTorch è identico; `pert_config.enable=True` per le spinte [V]. Nel container CPU: MuJoCo + PyTorch per L0–L2 e per tutti gli unit test.

### 3.2 Repo da clonare (fissare i commit al primo setup)

| Repo | URL | Licenza | Tag/commit da fissare | Uso |
|---|---|---|---|---|
| mujoco_menagerie | https://github.com/google-deepmind/mujoco_menagerie | BSD-3 (modello Go2) [V] | `0059d43` (2026-10-07) ✅ clonato sparse in `third_party/` | modello Go2 |
| IsaacLab | https://github.com/isaac-sim/IsaacLab | BSD-3 | tag **v2.3.x** (quello richiesto da unitree_rl_lab [V]) oppure 3.0.0-beta2 [da decidere: D2] | training principale |
| rsl_rl | https://github.com/leggedrobotics/rsl_rl | [da verificare, presumibilmente BSD-3] | PyPI `rsl-rl-lib==5.5.1` [V] (Isaac Lab installa la sua: non mescolare) | PPO/Distillation |
| unitree_rl_lab | https://github.com/unitreerobotics/unitree_rl_lab | Apache-2.0 [V] | HEAD al setup | task Go2 + deploy C++ |
| unitree_mujoco | https://github.com/unitreerobotics/unitree_mujoco | BSD-3 [V] | HEAD | sim2sim DDS |
| unitree_sdk2 / unitree_sdk2_python | https://github.com/unitreerobotics/unitree_sdk2 , …/unitree_sdk2_python | BSD-3 [V] | HEAD; cyclonedds 0.10.2 [V] | robot |
| mujoco_playground | https://github.com/google-deepmind/mujoco_playground | Apache-2.0 [V] | HEAD (install da sorgente con `uv` [V]) | fallback GPU |
| pop-spiking-deep-rl | https://github.com/combra-lab/pop-spiking-deep-rl | MIT [V] | — | riferimento PopSAN (codice vecchio, non riusare direttamente) |
| Kostantinoskanell/thesis | https://github.com/Kostantinoskanell/thesis | **nessuna licenza** [V] | — | solo lettura, **non copiare codice** |

### 3.3 Dipendenze (due ambienti)

**Ambiente A — `spk-cpu` (container, ✅ creato in `.venv`)**: Python 3.13, `torch==2.14.1+cpu`, `mujoco==3.15.0`, `numpy`; poi `gymnasium`, `hydra-core`, `tensorboard`, `wandb` (opz.), `snntorch==1.0.0` e `spikingjelly==0.0.0.0.14` **solo per il benchmark**.

```bash
cd /home/user/Spiking-RL
python3 -m venv .venv && . .venv/bin/activate                      # ✅
pip install --index-url https://download.pytorch.org/whl/cpu torch   # ✅
pip install mujoco numpy                                             # ✅
pip install gymnasium hydra-core tensorboard pytest                  # da fare (S3)
pip install snntorch spikingjelly                                    # opzionale, benchmark
```

**Ambiente B — `spk-gpu` (workstation/VM con GPU)**: Ubuntu 22.04, driver ≥ 580, Python 3.11, Isaac Sim 5.1 (pip) + Isaac Lab, poi `pip install -e .` del nostro pacchetto. Comandi (dal README Isaac Lab/unitree_rl_lab [V]; **non eseguiti qui**):

```bash
# Isaac Lab (percorso pip di Isaac Sim, guida "Installation using Isaac Sim Pip Package")
conda create -n env_isaaclab python=3.11 && conda activate env_isaaclab
pip install torch --index-url https://download.pytorch.org/whl/cu128      # [da verificare la cu-version richiesta da Isaac Sim 5.1]
pip install "isaacsim[all,extscache]==5.1.0" --extra-index-url https://pypi.nvidia.com   # [da verificare sintassi esatta nella guida]
git clone https://github.com/isaac-sim/IsaacLab.git && cd IsaacLab && git checkout <TAG_SCELTO>
./isaaclab.sh --install rsl_rl
# test
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py --task Isaac-Velocity-Flat-Unitree-Go2-v0 --headless --max_iterations 50
# unitree_rl_lab (fuori dalla cartella IsaacLab)
git clone https://github.com/unitreerobotics/unitree_rl_lab.git && cd unitree_rl_lab && ./unitree_rl_lab.sh -i && ./unitree_rl_lab.sh -l
```

Fallback GPU senza Isaac (Playground):
```bash
git clone https://github.com/google-deepmind/mujoco_playground && cd mujoco_playground
uv venv --python 3.12 && source .venv/bin/activate
uv pip install -U "jax[cuda12]" && uv --no-config sync --all-extras      # [V README]
python -c "import jax; print(jax.default_backend())"                       # deve stampare gpu
python learning/train_jax_ppo.py --env_name Go1JoystickFlatTerrain          # smoke [da verificare il nome esatto dell'env]
```

### 3.4 Struttura della repo proposta

```
Spiking-RL/
├── docs/                      # questo piano, rassegna, verifica fonti, report esperimenti
├── scripts/                   # smoke test e benchmark ✅ (3 script)
├── spiking_rl/                # pacchetto (da creare nello sprint 1)
│   ├── snn/                   # core SNN: neuroni (LIF/ALIF/multi-τ), surrogate, encoders, decoders, init
│   ├── models/                # actor/critic per rsl_rl: ANN MLP, GRU, SNN (A2–A5, N1–N3)
│   ├── algos/                 # estensioni: DAgger-mixing per Distillation, scheduler surrogate, PPO patch
│   ├── envs/
│   │   ├── mujoco_go2/        # env Gymnasium/VecEnv CPU (L0–L2, sim2sim harness)
│   │   ├── isaaclab_go2/      # cfg che ereditano UnitreeGo2{Flat,Rough}EnvCfg (critic group, push on, L1/L2 task)
│   │   └── playground_go2/    # (fallback) env MJX derivato da go1/joystick.py
│   ├── eval/                  # protocollo L5, metriche gait, SynOps/energia, statistiche
│   └── deploy/                # export (TorchScript/ONNX), controller DDS, logging reale
├── configs/                   # Hydra: env/, model/, algo/, exp/ (un file per esperimento E-xx)
├── tests/                     # pytest: neuroni, reset per dones, encoders, env obs ordering, export round-trip
├── third_party/               # cloni (gitignored) con commit fissati in docs/sources_verification.md
├── logs/ runs/ wandb/         # gitignored
└── requirements-cpu.txt / requirements-gpu.txt
```

### 3.5 Libreria SNN: scelta e micro-benchmark

Criteri: (1) stato esplicito e resettabile per env (`dones`) — necessario per il carry di N1 e per rsl_rl; (2) nessun kernel custom (export su robot/NIR); (3) velocità su batch 4096 × T=4; (4) stabilità dell'API.
- **Custom PyTorch** (Appendice A): soddisfa 1–2–4; velocità ✅ 112 ms fwd+bwd su 4 CPU (batch 4096, 2×256, T=4) — su GPU attesa < 5 ms [stima].
- **snnTorch 1.0.0** [V]: API semplice, `snn.Leaky`; stato gestito a mano (equivalente al custom).
- **SpikingJelly** [V]: stabile 0.0.0.0.14, **V2 rc con torch ≥ 2.6/Py ≥ 3.11 e Triton**; multi-step fuso più veloce per T grandi, ma API in transizione.
- **Spyx/SNNAX (JAX)**: solo se si sceglie il percorso brax-PPO puro in Playground.

Comando: `python scripts/bench_snn_libs.py --batch 4096 --T 4 --iters 20` **sulla GPU target**. Regola: si resta sul custom a meno che un'altra libreria non sia > 1.5× più veloce.

### 3.6 Config, logging, seed, riproducibilità
- **Hydra** (`configs/`), un file `exp/E-xx.yaml` per esperimento con `seed`, `env`, `model`, `algo`; la config risolta e il `git rev-parse HEAD` salvati nella cartella del run.
- **Logger**: TensorBoard sempre; **wandb** opzionale (`logger=wandb` in rsl_rl [V]); progetto `spiking-go2`, tag `L{livello}/{arch}/{seed}`.
- **Seed**: `torch`, `numpy`, env; Isaac Lab/PhysX non è bit-riproducibile su GPU [stima generale]: la riproducibilità è statistica (5 seed).
- **Metriche da loggare per ogni run SNN**: firing rate per strato (media/percentili), % neuroni silenti (<1 %) e saturi (>90 %), SynOps/inferenza, norma dei gradienti per strato, pendenza surrogate corrente, T.
- **Checkpoint**: ogni 50 iterazioni + "best by eval"; export TorchScript del solo actor con normalizzazione inclusa.

---

## 4. Protocollo sperimentale

Tempi: [stima] su una RTX 4090/L40S con 4096 env; raddoppiare su GPU meno potenti. "Seed" = run indipendenti.

| # | Esperimento | Input | Output atteso | Criterio di successo | Tempo/GPU | Seed | Grafici/tabelle |
|---|---|---|---|---|---|---|---|
| E0 | Smoke CPU (L0-a/b) ✅ | scripts | OK | OK | 1 min CPU | 1 | — |
| E1 | SNN-PPO Pendulum/CartPole (L0-c) | A0 vs A3 (T=1,4) | curve di ritorno | §1.2 | 1 h CPU | 3 | ritorno vs step |
| E2 | Setup Isaac + baseline ufficiale Go2 flat (L0-d) | `Isaac-Velocity-Flat-Unitree-Go2-v0` | video + reward | cammina | 20 min | 1 | — |
| E3 | **Baseline ANN A0 nostra** L1→L4 (obs senza lin_vel, critic asimmetrico, spinte on) | cfg §1 | policy per livello | go/no-go §1 | 4×(0.5–3 h) | 5 | tabelle per livello; tracking per fascia |
| E4 | Baseline A1 (teacher privilegiato + GRU student) L4/L6 | E3 teacher | student | ≥ 95 % del teacher | 3 h + 3 h | 5 | teacher vs student |
| E5 | Benchmark librerie SNN (GPU) | `bench_snn_libs.py` | ms per fwd+bwd | scelta | 10 min | — | tabella |
| E6 | A3 PPO diretto, L1 (T ∈ {1,2,4}; surrogate fissa vs schedulata) | cfg | curve | SNN ≥ 95 % ANN | 6 × 1 h | 3 | ritorno vs iter; firing rate per strato |
| E7 | A2 PopSAN PPO diretto, L1 | cfg | curve | idem | 2 h | 3 | come E6 |
| E8 | A3/A2 L2 (precisione decodifica) | migliori di E6/E7 | RMSE h/pitch | §1.4 | 4 h | 3 | RMSE vs T |
| E9 | A3 PPO diretto L3 (max 2 tentativi di tuning) | cfg | gait o belly-flop | §1.5 | 2 × 3 h | 3 | — |
| E10 | **A4 distillazione** L3→L4 (teacher E3/E4) con e senza DAgger-mixing | cfg | student SNN | §1.6 | 2 × 4 h | 5 | errore student vs teacher; RMSE per comando |
| E11 | **N1** L4 (+ ablation: no-carry, no-ALIF, no-loss-extrinsics) | cfg | student ricorrente | §1.6 | 4 × 6 h | 5 (3 ablation) | tabella ablation; decodifica delle extrinsics (R²) |
| E12 | N3 su A3 e N1 | cfg | τ appresi | ≥ parità | 2 × 6 h | 3 | istogrammi τ |
| E13 | **L5 robustezza** per tutte le policy a parità di reward (A0, A1, A4, N1, ±N2) | checkpoint | matrice condizioni × policy | — | 1 h/policy (eval) | 5 | heatmap cadute; curve recupero vs spinta; CI |
| E14 | N2 CPG spiking L3/L4 | cfg | gait | §1.5–1.6 | 2 × 6 h | 3 | Pareto CoT vs spike/passo |
| E15 | L6 terreni: A1 vs N1 | cfg | cadute | §1.8 | 2 × 8 h | 3 | cadute per tipo di terreno |
| E16 | SynOps/energia (App. B) per tutte le policy | checkpoint + rollout | tabella | — | 30 min | 5 | SynOps e "energia stimata" con e senza full accounting |
| E17 | Sim2sim MuJoCo + unitree_mujoco (L7) | checkpoint | metriche | §1.9 | 2 h | 5 | Isaac vs MuJoCo |
| E18 | Sim2real (L8-0…iii) | robot | log | §1.10 | 3–5 giorni | n/a | video, tracking reale, spinte |
| E19 | (opz.) Loihi 2 (L9) | checkpoint | RMSE | §1.11 | settimane | — | GPU vs Loihi |

**Regole.**
1. Nessun esperimento SNN su un livello prima che E3 lo abbia passato.
2. Ogni risultato riportato come media ± std su ≥ 3 seed, con IC 95 % bootstrap (10 000 ricampionamenti); confronto SNN vs ANN con test di **Welch** (o Mann-Whitney se n piccolo/non normale) e correzione di **Holm** sulle famiglie di confronti; riportare anche l'effect size (Cliff's δ). Opzionale: IQM in stile rliable.
3. Per L5 la "parità di reward" è un vincolo di selezione dei checkpoint (±3 %), e va dichiarata nel paper.
4. Energia: solo **SynOps misurate** + conversione a pJ dichiarata (App. B); mai "misure" senza hardware.

---

## 5. Troubleshooting e piani B

Ordine di diagnosi: guardare sempre prima (1) firing rate per strato, (2) norma dei gradienti per strato, (3) KL/entropia PPO, (4) video.

### 5.1 L'SNN non converge con PPO (L0-c o L1)
1. **Strato morto/saturo** (rate ≈ 0 o ≈ 1): init rate-aware (App. A) — nel nostro smoke test è bastata; soglia 1.0 con input normalizzati; bias iniziale 0; se saturo ridurre std init di 2×. Penalità L2 sul rate verso 0.1–0.2 (peso 1e-3).
2. **Pendenza surrogate**: provare α ∈ {0.5, 1, 2} e schedule da bassa ad alta (2510.24461 [V]: shallower/schedulata → 2.1×).
3. **T**: aumentare a 4–8 (precisione) solo dopo aver sistemato 1–2; T=1 è un ANN binarizzato, aspettatevi gap.
4. **lr**: 3e-4 → 1e-4 (metà dell'ANN); `desired_kl` 0.01 → 0.005; grad clip 0.5.
5. **Normalizzazione**: obs normalizzate (running stats) obbligatorie; niente BatchNorm; eventuale "membrane norm" per strato identica in train/deploy.
6. **Decodifica**: passare da rate+lineare a membrana integrata (o viceversa) e verificare l'ampiezza delle azioni (std iniziale delle azioni ≈ 0.5–1 come ANN).
7. Se dopo 2 tentativi (E6/E9) non si arriva al go/no-go: **passare ad A4 (distillazione)**. È il percorso previsto, non un fallimento.

### 5.2 Gradienti esplodono (N1/N3 ricorrenti)
- TBPTT più corto (`gradient_length` 15), grad clip 0.5, τ_adapt clampato ≤ 2 s, W_rec inizializzata con raggio spettrale 0.5, detach dello stato ogni `gradient_length` passi (rsl_rl lo fa [V]).
- Se la ricorrenza non aiuta su piano (E11 ablation ≈ no-carry): spostare il claim su L5/L6 dove serve memoria; se neanche lì: riportare come risultato negativo + analisi delle extrinsics decodificate.

### 5.3 Il robot cade / belly-flop / resta fermo
- Belly-flop (minimo locale, visto nella tesi Go2 [V]): terminazione su contatto cosce/base, penalità altezza, bonus sopravvivenza, curriculum di comando (0.3 → 0.8 m/s), **inizializzare da teacher** (A4).
- Resta fermo con comando: aumentare peso tracking (1.5 → 2.0), ridurre penalità coppie; verificare che `stand_still` non sia attiva con comando ≠ 0.
- Cade subito: controllare segni/ordine giunti e scale azione; eseguire `smoke_go2_mujoco.py` equivalente nell'env.

### 5.4 Trotto a saltelli / jitter / pacing
- Jitter: aumentare action_rate (−0.01 → −0.03), penalità acc. giunti; per SNN aumentare T o usare filtro passa-basso sull'uscita (dichiarato); misurare ‖Δa‖.
- Saltelli (bound): feet_air_time soglia 0.5 s → 0.3; aggiungere penalità su lin_vel_z e **reward di fase diagonale** (o N2).
- Pacing: penalità contatti laterali contemporanei; randomizzare l'init della fase.

### 5.5 Tracking vy/ωz scarso (tipico SNN)
- Verificare che la precisione di decodifica a L2 sia ok; aumentare la popolazione di uscita per dimensione (ILC-SAN [V]); pesare di più track_ang (0.75 → 1.0); comandi più piccoli in curriculum; DAgger con più campioni a vy/ωz alti.

### 5.6 Training lento o fuori memoria con 4096 env
- Profilare: se la SNN domina (atteso con T=4 e 2 strati da 256: 4× le FLOP di un MLP), ridurre T a 2, usare `torch.compile` sul core, mixed precision (bf16 per le correnti, fp32 per le membrane), o SpikingJelly multi-step (E5).
- Memoria BPTT: `gradient_length` 15; 2048 env × 48 step invece di 4096 × 24.
- Isaac Lab: `--headless`, disattivare sensori non necessari (height scanner su flat).

### 5.7 Isaac Lab non disponibile / instabile
- Percorso Playground/MJX (§3.1) con env Go2 derivato; stessi reward/obs; rsl_rl via wrapper.
- Se neanche GPU: MuJoCo CPU per L1/L2 e per lo sviluppo dell'SNN; L3+ richiede GPU (D1).

### 5.8 Sim2sim/sim2real gap
- Ordine: (1) ordine/segni giunti e unità IMU (quaternione, frame), (2) latenza misurata vs randomizzata, (3) Kp/Kd effettivi (step response), (4) attrito piedi, (5) **actuator net** da dati reali (non ne esiste uno pubblico per Go2 [NV]) — raccogliere 5 min di dati con chirp su giunti e allenare un MLP coppia = f(q_err, q̇ storia); (6) Dao & Fern 2604.11090 [V] per adattare il simulatore con < 5 min di dati.
- SNN specifico: congelare le running stats; verificare che il carry di stato sia identico (nessun reset implicito nel controller); confrontare azioni sim vs reale su obs registrate (RMSE < 0.05 rad).

### 5.9 Robot reale: policy ANN ok, SNN no
- Replay offline delle obs reali nella SNN su PC: se le azioni divergono da quelle in sim → problema numerico (fp16, quantizzazione); se coincidono → problema di timing (T interni oltre il budget 20 ms? misurare).

---

## 6. Rischi, decisioni aperte, milestone, pubblicabilità

### 6.1 Cosa serve dall'utente

| Necessità | Perché | Stato |
|---|---|---|
| **GPU NVIDIA ≥ 16 GB VRAM** (workstation o VM cloud; Ubuntu 22.04) | Isaac Lab/Isaac Sim e anche MJX/Genesis. Il container attuale non ha GPU ✅ verificato | **bloccante per L3+** |
| **Go2 Edu** (versione, firmware, modulo Orin?) | Air/Pro non consentono sviluppo secondario [V]; X parziale | da chiarire |
| Area di test, tappeto, imbragatura, e-stop, 2 persone | sicurezza L8 | — |
| Accesso INRC (PI permanente, proposta di progetto) | L9 | opzionale |
| Decisione sul venue target | definisce l'ampiezza (robotica vs neuromorfico) | — |

### 6.2 Decisioni da prendere

- **D1** GPU: quale (consigliato RTX 4090/L40S/A6000 ≥ 24 GB per 4096 env con BPTT) e dove.
- **D2** Versione Isaac Lab: **2.3.x** (compatibile con `unitree_rl_lab` oggi [V]) vs 3.0 beta (più recente, API in movimento). Consigliato 2.3.x.
- **D3** Posa di default unica (Isaac vs menagerie `home`): consigliato Isaac e portarla nel MJCF.
- **D4** Interfaccia comando: 3 (vx,vy,ωz) o 6 (con h, pitch, roll) da L2 in poi. Consigliato 6 con default per non cambiare la rete tra livelli.
- **D5** Range vy: ±0.6 (consigliato) vs ±1.0 (Isaac).
- **D6** Contributo principale: N1 (consigliato) vs N2.
- **D7** Fare L9 (Loihi 2)? Solo se c'è un PI INRC e tempo dopo L8.

### 6.3 Milestone (date relative a T0 = GPU disponibile; sprint di 2 settimane)

| Milestone | Quando | Contenuto |
|---|---|---|
| M0 | T0 − 1 sett. | Sprint 1 (§7) completato nel container: core SNN, env MuJoCo, E1 |
| M1 | T0 + 2 sett. | E2–E5: baseline ANN L1–L4 passate, teacher pronto, libreria scelta |
| M2 | T0 + 5 sett. | E6–E10: SNN su L1/L2; verdetto PPO diretto vs distillazione; A4 su L4 |
| M3 | T0 + 8 sett. | E11–E12: N1 su L4 con ablation |
| M4 | T0 + 10 sett. | E13, E16: studio di robustezza + SynOps → **primo manoscritto (sim)** |
| M5 | T0 + 12 sett. | E17: sim2sim; controller DDS pronto |
| M6 | T0 + 14–16 sett. | E18: sim2real L8-0…iii (serve il robot) → **manoscritto principale** |
| M7 | opz. | E14/E15 (N2, terreni), E19 (Loihi) |

### 6.4 Pubblicabilità per livello
- Dopo M2–M4 (solo sim): "Spiking student policies for Go2 velocity tracking: a systematic robustness and energy study vs ANN" — pubblicabile anche con risultati negativi (robustezza non superiore; energia solo 1.0–1.5× con full accounting, come la tesi [V]). Venue: IROS/ICRA workshop, Frontiers in Neurorobotics, Neuromorphic Computing and Engineering.
- Con N1 positivo su L5/L6: "Recurrent spiking student replaces history encoder" — ICRA/IROS/CoRL.
- Dopo M6: primo trotto comandato e robusto a spinte con SNN appresa su Go2 reale — RA-L/ICRA/Science Robotics (short) a seconda della solidità; va fatto il controllo di novità S1.
- L9: NICE/ICONS; con hardware reale a gambe sarebbe il primo [V rassegna].

### 6.5 Rischi principali

| Rischio | Prob. | Impatto | Mitigazione |
|---|---|---|---|
| Nessuna GPU per mesi | media | blocca L3+ | fallback MuJoCo CPU per L1/L2 e sviluppo; noleggio cloud a ore |
| Jiang ICASSP ha un deploy reale su Go1 | media | riduce il claim | S1; riposizionare su Go2+comandi+spinte+ricorrenza |
| Risparmio energetico nullo con decodifica float | alta | claim energetico debole | A3 (membrana) e full accounting onesto; N6 |
| SNN meno robusta dell'ANN | media | — | è comunque un risultato (E13) |
| Sim2real fallisce per latenza/attuatori | media | blocca M6 | L8-0 misure, actuator net, Dao & Fern |
| API rsl_rl/Isaac cambiano | alta | tempo | pin di versione, test di integrazione |

---

## 7. Checklist finale e primo sprint

### 7.1 Checklist "pronti a partire"
- [x] Rassegna letta; fonti chiave verificate (`sources_verification.md`).
- [x] Container: venv con torch CPU + mujoco; Go2 menagerie clonato; smoke test OK.
- [ ] S1 controllo di novità su Jiang ICASSP 2025 (PDF IEEE) e Xiao et al. 2025.
- [ ] D1–D7 decise (§6.2).
- [ ] GPU disponibile; Isaac Lab installato; E2 eseguito.
- [ ] Go2 Edu confermato e accessibile.

### 7.2 Primo sprint (10 task, in ordine; tutti eseguibili nel container tranne S9–S10)

| # | Task | Comando / artefatto | Fatto quando |
|---|---|---|---|
| S1 | Controllo di novità: leggere PDF IEEE di Jiang (DOI 10.1109/ICASSP49660.2025.10890793) e cercare "spiking" + "Go2"/"Go1" + "real" 2024–2026; aggiornare `sources_verification.md` | WebSearch/biblioteca | nota scritta con verdetto |
| S2 | Smoke test ✅ | `. .venv/bin/activate && python scripts/smoke_lif_surrogate.py && python scripts/smoke_go2_mujoco.py` | `RISULTATO: OK`, `PD hold: OK` |
| S3 | Dipendenze CPU + scheletro pacchetto | `pip install gymnasium hydra-core tensorboard pytest`; creare `spiking_rl/{snn,models,algos,envs,eval,deploy}` con `__init__.py`; `requirements-cpu.txt` | `pytest` gira (anche vuoto) |
| S4 | Core SNN come modulo + test | spostare le classi di `scripts/smoke_lif_surrogate.py` in `spiking_rl/snn/{neurons,surrogate,init}.py`; aggiungere ALIF, multi-τ, encoder a popolazione, decoder a membrana; `tests/test_neurons.py` (rate, grad, reset per dones, carry) | test verdi |
| S5 | Env Go2 MuJoCo vettorizzato (L1) | `spiking_rl/envs/mujoco_go2/env.py` (obs §1.0, PD, spinte, randomizzazioni, terminazioni, reward L1); test: obs shape 45, ordine giunti = Isaac, `smoke` 1000 step × 16 env | ≥ 5 k policy step/s su 4 core [stima da misurare] |
| S6 | PPO di riferimento su CPU | CleanRL-style PPO (o SB3) con actor intercambiabile ANN/SNN; E1 su Pendulum/CartPole | criterio §1.2 |
| S7 | E3-L1 su CPU con ANN (16–64 env, 10 M step) | `python -m spiking_rl.train exp=L1_ann seed=0` | go/no-go L1 ANN |
| S8 | E6-L1 su CPU con A3 (T=1,4) e A2 | `exp=L1_snn_a3 T=4` | SNN passa L1 o diagnosi §5.1 documentata |
| S9 | (GPU) Isaac Lab setup + E2 + benchmark E5 | §3.3 | video Go2 che cammina; tabella bench |
| S10 | (GPU) Cfg Isaac nostre: `UnitreeGo2*EnvCfg` derivate con gruppo `critic`, `push_robot` riattivato, comandi §1.6; E3-L3/L4 ANN | `./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py --task SpkGo2-Flat-v0 --headless` | go/no-go L3/L4 ANN |

---

## Appendice A — Specifica del "core SNN"

Implementazione di riferimento: `scripts/smoke_lif_surrogate.py` ✅ (PyTorch puro).

- **Neurone LIF** (corrente istantanea): `u_t = β·u_{t-1} + W x_t + b`, spike `s_t = H(u_t − θ)`, reset hard `u_t ← u_t·(1 − s_t)`; θ = 1; β = σ(w_β) appreso per neurone (init da τ=2 passi).
- **ALIF** (N1): soglia `θ_t = 1 + γ·a_t`, `a_t = ρ·a_{t-1} + s_{t-1}`, ρ = exp(−dt/τ_adapt) con τ_adapt appreso in [0.1, 2] s.
- **Multi-τ** (N3): β e ρ per neurone, init log-uniforme.
- **Surrogate**: arctan `dS/du = (α/2)/(1 + (π/2·α·u)²)`, α modificabile a runtime (`set_alpha`) per lo scheduling (es. α: 0.5 → 2 lineare nelle prime 30 % iterazioni, da validare in E6).
- **Init rate-aware** (obbligatoria): `std(W) = 1/√(fan_in · E[x²])` con E[x²] = 1 per obs normalizzate, ≈ 0.15 per ingressi spiking; bias 0. Senza, il secondo strato è morto (osservato ✅).
- **Encoding**: diretto (obs normalizzata iniettata a ogni passo interno) di default; popolazione gaussiana (A2) come opzione.
- **Decoding**: 12 neuroni non-spiking, membrana integrata e mediata su T (A3/N1) oppure rate → lineare (A2). log-std separata.
- **Stato**: buffer `(num_env, n)`; `reset(dones)` azzera solo gli env terminati; `carry=True` mantiene lo stato tra passi di controllo (N1), `carry=False` lo azzera a ogni passo (A2/A3).
- **Integrazione rsl_rl**: il modello actor deve esporre la stessa interfaccia del modello ricorrente di rsl_rl (`reset(dones)`, `detach_hidden_state`, forward in modalità rollout e batch con `masks`) — **[da verificare]** sui sorgenti della versione pinnata (`rsl_rl/modules/rnn.py` è il riferimento [V]).

## Appendice B — Metodologia energia/SynOps

- **SynOps** per inferenza = Σ_strati (numero di spike in ingresso allo strato × fan-out). Per l'ANN: MAC = Σ fan_in × fan_out.
- Conversione dichiarata (45 nm, Horowitz 2014, convenzione usata in Jiang [V] e nella maggior parte della letteratura SNN): **0.9 pJ per AC** (spike), **4.6 pJ per MAC**. Riportare sempre anche le operazioni **non spiking**: encoder (prodotto obs×W del primo strato: MAC), decoder lineare (A2: MAC), normalizzazione → "full accounting" (la tesi Go2 mostra 10.4× su SynOps ma 2.4×/1.04× full accounting [V]).
- Limiti da dichiarare nel paper: (1) non sono misure; (2) ignorano memoria, traffico e overhead del controllore; (3) su GPU/Jetson un'SNN **non** risparmia energia (esecuzione densa); (4) solo hardware neuromorfico (L9) dà numeri reali — Stewart et al. riportano 0.013 J vs 0.217 J per inferenza *ma* includono la potenza statica dell'intero sistema [V].
- Grafico: Pareto "RMSE tracking vs SynOps" e "cadute sotto spinta vs SynOps" per tutte le policy (E16 + E13).

## Appendice C — Metriche e definizioni

| Metrica | Definizione |
|---|---|
| RMSE vx/vy/ωz | sulla finestra dopo 1 s dal cambio comando, episodi senza caduta |
| Caduta | terminazione per contatto base/cosce o \|roll\|,\|pitch\| > 1 rad |
| Recupero da spinta | tempo da Δv a (‖v_xy − v_cmd‖ < 0.1 m/s ∧ ‖g_xy‖ < 0.05) per 0.5 s; fallito se caduta entro 3 s |
| Jitter | media di ‖a_t − a_{t−1}‖₂ |
| CoT | Σ|τ·q̇|·dt / (m·g·d) sulla distanza percorsa |
| Fase diagonale | differenza di fase di contatto tra FL e RR (0.5 = trotto perfetto) |
| Firing rate | spike/(neuroni·T) per strato; silente < 1 %, saturo > 90 % |
| SynOps | App. B |
