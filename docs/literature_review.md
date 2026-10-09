# Spiking RL per la locomozione del Unitree Go2 — rassegna della letteratura

Data: 2026-10-09. Sintesi di quattro ricerche parallele (SNN per locomozione, RL standard per Go2, architetture SNN in RL, hardware neuromorfico e lacune).

**Affidabilità.** La ricerca è stata fatta via web search/fetch, non su un indice bibliografico esaustivo. Legenda: **[V]** pagina/abstract/README letto; **[S]** visto solo in snippet di ricerca; **[M]** da memoria, non verificato. Prima di citare qualunque voce [S]/[M] in un paper, controllare la fonte. Le cifre di energia sono quasi sempre *stime* (conteggio operazioni a 45 nm), non misure su silicio.

---

## 1. Risposta alla domanda chiave

**Nessuno, per quanto trovato, ha pubblicato una policy SNN appresa che trotti in modo robusto, con comando (vx, vy, yaw) e resistenza a disturbi, su un Go2 reale (o su un quadrupede di complessità equivalente).**

| Cosa esiste | Limite |
|---|---|
| Jiang et al., *Fully Spiking Neural Network for Legged Robots* ([arXiv 2310.05022](https://arxiv.org/abs/2310.05022)) — A1, Cassie, MIT Humanoid, Isaac Gym, PPO + RMA/AMP, actor stile PopSAN con LIF; tracking di velocità su terreni difficili; robusto a rumore sui comandi (σ=0.3) [V] | **Solo simulazione**; energia stimata (−96/−82/−59% per T=1/2/3); convergenza più lenta dell'ANN con PPO |
| Wang et al., *Artificial plateau neurons…*, Nat. Commun. 17:5801 (2026) — **Go2 reale** guidato da oscillatori spiking [V] | Hardware analogico custom, **nessun RL**, nessun comando vx/vy/yaw; anche bloccate, solo 8 motori attivi; due velocità (0.15 e 0.4 m/s) |
| Zhang et al., *ES-Parkour* ([arXiv 2503.09985](https://arxiv.org/abs/2503.09985)) — SNN con distillazione da teacher ANN privilegiato [V] | Solo simulazione, camera a eventi simulata |
| Tesi non peer-reviewed su Go2 simulato (repo `Kostantinoskanell/thesis`) [V README] | PPO diretto fallisce (minimo locale "belly-flop"); funziona con distillazione + DAgger; risparmio SynOps 10× ma solo 1.04× nel miglior setting locomotorio |
| Han & Sengupta, CPG spiking con astrociti, R-STDP ([arXiv 2312.15805](https://arxiv.org/abs/2312.15805)); versione con PPO adattato/equilibrium propagation su terreno irregolare ([arXiv 2605.09595](https://arxiv.org/abs/2605.09595)) [V abstract] | Solo simulazione |
| Esapodi con CPG spiking su hardware reale (Loihi 1, SpiNNaker, FPGA, Arduino) | CPG progettati a mano, senza controllo in anello chiuso della velocità né recupero da spinte |

Non ho trovato: nessuna policy locomotoria SNN su Loihi/Loihi 2/SpiNNaker2 su robot a gambe reale; nessun test di spinte su hardware reale; nessuna misura di potenza on-robot di un controllore SNN appreso. L'unico "SNN+RL su quadrupede reale" trovato è un progetto studentesco su PuppyPi (hobbistico) in cui l'SNN stava in piedi ma non camminava.

**Conseguenza:** il tuo obiettivo (trotto robusto, comandato, a spinte, SNN, Go2 reale) è davvero aperto.

---

## 2. Stato dell'arte del RL standard per Go2 (la baseline da battere)

**Stack consigliato:** Isaac Lab + RSL-RL (o `unitree_rl_gym`/legged_gym con Isaac Gym, ormai in sola manutenzione). Alternative: MuJoCo Playground (JAX/MJX; ha Go1 joystick, nessun Go2 nel paper), Genesis (esempio Go2 minimale), mjlab. Il deploy usa `unitree_sdk2` (DDS) con il robot in debug mode. `unitree_rl_lab` (Isaac Lab, Apache-2.0) supporta Go2.

**Lavori chiave** (tutti [V] salvo dove indicato):
- Rudin et al., CoRL 2021 ([2109.11978](https://arxiv.org/abs/2109.11978)): PPO massivamente parallelo; 4096 env × 24 step, 50 Hz, curriculum di terreni; ANYmal cammina in minuti.
- Lee et al., Sci. Robotics 2020 ([2010.11251](https://arxiv.org/abs/2010.11251)): teacher privilegiato → student TCN.
- RMA ([2107.04034](https://arxiv.org/abs/2107.04034)): policy base + extrinsics encoder, adaptation module 1D-CNN su 50 passi di storia.
- Walk These Ways ([2212.03238](https://arxiv.org/abs/2212.03238)): Go1, comando di gait (frequenza, fasi, altezza…) con clock; actuator network + latenza 20 ms; trained su terreno piano.
- Concurrent state estimator ([2202.05481](https://arxiv.org/abs/2202.05481)), DreamWaQ ([2301.10602](https://arxiv.org/abs/2301.10602)): stima della velocità del corpo, actor-critic asimmetrico.
- CPG-RL / Visual CPG-RL ([2212.14400](https://arxiv.org/abs/2212.14400)); DayDreamer ([2206.14176](https://arxiv.org/abs/2206.14176), [S]); DiffuseLoco ([2404.19264](https://arxiv.org/abs/2404.19264)); policy generaliste (URMA [2409.06366](https://arxiv.org/abs/2409.06366), LocoFormer [2509.23745](https://arxiv.org/abs/2509.23745)) — ricerca, non baseline pratiche per Go2.
- Go2 specifico: Dao & Fern ([2604.11090](https://arxiv.org/abs/2604.11090)), sim-to-real da <5 min di dati reali.

### Ricetta baseline per Go2 (sintesi; numeri letti dai config Isaac Lab / unitree_rl_gym)
- **Osservazioni actor (45-d, solo propriocettive):** vel. angolare ×0.25, gravità proiettata, comando (vx,vy,yaw), `q−q_default`, `q̇×0.05`, ultima azione; rumore come Isaac Lab. Velocità lineare di base **solo nel critic** (il Go2 reale non la misura).
- **Azioni:** 12 offset di posizione, `q_des = q_default + 0.25·a`; PD Kp≈20–25, Kd≈0.5; policy a 50 Hz (dt 0.005, decimation 4).
- **Reward:** tracking lineare esp. +1.5, yaw +0.75 (σ²=0.25); lin_vel_z −2; ang_vel_xy −0.05; orientamento piatto −2.5; coppie −2e-4; acc. giunti −2.5e-7; action rate −0.01; feet air time ≈ +0.25; terminazione su contatto base.
- **Randomizzazione:** attrito 0.4–1.25, massa −1…+3 kg, Kp/Kd ±10%, forza motore ±10%, offset giunti ±0.02 rad, ritardo azione 0–20 ms, **spinte 0.5–1 m/s ogni 10–15 s (nel task Go2 di Isaac Lab sono disattivate: vanno abilitate)**.
- **PPO:** 4096 env, rete [512,256,128] ELU, lr 1e-3 KL-adattivo (0.01), clip 0.2, 5 epoche, 4 minibatch, γ 0.99, λ 0.95, entropia 0.01, ~1500 iterazioni.
- **Sim-to-real:** sim2sim in MuJoCo, stessi Kp/Kd/pose/scale, misurare la latenza del *proprio* Go2 (20 ms è il dato Go1), actuator net se il gap persiste. Non ho trovato un dataset/actuator net open per Go2.

Con questa ricetta, comando di velocità + robustezza a spinte con una MLP è un problema risolto: la baseline ANN è il riferimento obbligato per qualsiasi claim SNN.

---

## 3. Architetture SNN tipiche in RL a controllo continuo

- **Neuroni:** LIF (default quasi universale, reset hard), PLIF, ALIF/soglie adattive e costanti di tempo apprese (importanti in Huebotter et al. [2509.05356](https://arxiv.org/abs/2509.05356)).
- **Encoding:** population coding con campi recettivi gaussiani (PopSAN, [2010.09635](https://arxiv.org/abs/2010.09635)) oppure encoding diretto/analogico (corrente iniettata nel primo strato).
- **Decoding:** conteggi/rate di popolazione + layer lineare float (PopSAN), oppure potenziale di membrana di neuroni di uscita non-spiking (ILC-SAN [2401.05444](https://arxiv.org/abs/2401.05444), DSQN) → actor completamente spiking.
- **Training:** surrogate gradient/BPTT dominante; una pendenza *poco ripida/schedulata* dà un guadagno di 2.1× in RL ([2510.24461](https://arxiv.org/abs/2510.24461)). ANN→SNN in controllo continuo soffre di errori correlati (CRPI, [2601.21778](https://arxiv.org/abs/2601.21778)). e-prop/regole a tre fattori (lf-cs [2402.10069](https://arxiv.org/abs/2402.10069); Synaptic Motor Adaptation [2306.01906](https://arxiv.org/abs/2306.01906)) per policy ricorrenti/adattive; STDP e neuroevoluzione solo per CPG piccoli.
- **Schema ibrido standard:** actor spiking + critic ANN (scartato al deploy).
- **Off-policy:** la target network confligge con la dinamica spiking → Proxy Target ([2505.24161](https://arxiv.org/abs/2505.24161)). PPO non ha target network.
- **Parallelismo GPU:** SpikeRL ([2502.17496](https://arxiv.org/abs/2502.17496)); SpikeGym [S] riporta SNN-PPO veloce con shallow net, con difficoltà in profondità su Ant. Nessun lavoro trovato che misuri lo scaling con ≥4096 env.
- **Spiking SSM/transformer per policy di controllo:** non trovato (Spiking Decision Transformer [2508.21505] e Decision SpikeFormer sono offline/classic control).
- **Librerie:** SpikingJelly (kernel CuPy fusi, multi-step), snnTorch (semplice, serve `torch.compile`), Norse, Spyx/SNNAX (JAX, adatte a MJX/Brax), Lava-dl/Rockpool/sinabs (orientate all'hardware), Nengo (CPG a mano). Nessun benchmark indipendente confronta tutte per RL parallelo: **fare un test noi** su GPU target. Un layer LIF custom sono poche righe.

### Raccomandazioni di design per un actor SNN + PPO parallelo
1. Actor spiking, critic ANN asimmetrico (input privilegiati).
2. LIF con leak appreso per strato; aggiungere soglie adattive/ALIF solo se serve memoria.
3. Encoding diretto della propriocezione (poi confrontare population coding); decoding da potenziale di membrana mediato su T, deviazione standard della gaussiana come parametro separato non-spiking.
4. T piccolo (1–4); surrogate con pendenza schedulata.
5. Evitare batch norm con minibatch PPO; preferire normalizzazione per strato/membrana, identica in train e deploy.
6. lr più basso dell'ANN, gradient clipping, normalizzazione delle osservazioni, log dei firing rate per strato (0 o 1 = strato morto/saturo).
7. Se PPO da zero non converge (come in tesi Go2 e in Jiang et al.): **distillazione da teacher ANN** (anche privilegiato) + DAgger.
8. Memoria: BPTT limitato ai T passi interni; stato di membrana resettato a fine episodio; considerare di portarlo tra passi di controllo (può sostituire un GRU/history encoder).

---

## 4. Hardware neuromorfico

- **Loihi 2:** interi, pesi 8 bit, attivazioni 16 bit; mapping pratico via Sigma-Delta Neural Networks (SDNN) con Lava/Lava-DL/NIR. Stewart et al. ([2512.03911](https://arxiv.org/abs/2512.03911)): policy PPO 12×64×64×6 su Loihi 2, ~4.2 ms, 0.013 J vs 0.217 J su GPU, **ma tracking peggiore** (RMSE 0.225 vs 0.142 m) per la quantizzazione; solo simulazione (Astrobee). Mangalore et al. ([2401.14885](https://arxiv.org/abs/2401.14885)): QP neuromorfico per MPC di ANYmal, ≥100× meno energia-ritardo, <10 ms [abstract].
- **Robot reali su Loihi:** drone con visione+controllo neuromorfici (Paredes-Vallés et al., Sci. Robotics 2024); braccio con controllo adattivo; hexapod CPG (Loihi 1). Nessuno a gambe con policy appresa.
- **SpiNNaker2:** Q-network 8 bit su task classici (Arfa et al., [2507.23562](https://arxiv.org/abs/2507.23562)); nessun robot a gambe. **Speck/Xylo/Akida/Innatera:** nessun controllo motorio trovato.
- **Fattibilità su Go2 (giudizio nostro, non da fonti):** a 50 Hz il budget è 20 ms, compatibile con ~4 ms su Loihi 2. Ma Loihi 2 richiede accesso INRC e sarebbe un co-processore esterno via Ethernet, non onboard. Piano realistico: (1) policy SNN su Jetson/PC per validare sim-to-real e robustezza (nessun guadagno energetico, GPU è densa); (2) poi replay offline e infine anello chiuso su Loihi 2 con policy congelata, dichiarando "co-processore off-board".

---

## 5. Lacune e idee di contributo (novità × fattibilità)

Mappa delle lacune [dalle ricerche]: policy SNN appresa su robot reale; Loihi 2 a gambe; SNN come history/adaptation encoder (nessun lavoro trovato); distillazione teacher privilegiato → student SNN per locomozione (non trovata); stima di stato spiking (non trovata); analisi sistematica di robustezza SNN vs ANN (solo rumore gaussiano sui comandi); training hardware-aware per locomozione; spiking diffusion/transformer per locomozione (solo manipolazione/navigazione).

1. **Teacher privilegiato ANN → student SNN ricorrente per Go2, con modulo spiking ricorrente al posto di history encoder/RMA.** Alta novità, alta fattibilità. Rischio: student non eguaglia il teacher su terreni difficili. Primo esperimento: student LIF con population coding + 2 strati ricorrenti, DAgger; confronto con student GRU su tracking, recupero da spinte, spike rate.
2. **Studio sistematico di robustezza SNN vs ANN** (spinte, carico, attrito, latenza, dropout osservazioni, attacchi PGD) con politiche PPO a parità di reward. Novità medio-alta, fattibilità molto alta; anche un risultato nullo è pubblicabile.
3. **Policy SNN con quantizzazione hardware-aware (int8/16) e gate sim2sim → sim2real**, verificata prima con emulazione numerica Loihi 2. Novità medio-alta, fattibilità media.
4. **Prima policy a gambe in anello chiuso su Loihi 2 (o SpiNNaker2) come co-processore.** Novità molto alta, fattibilità medio-bassa (accesso HW, I/O). Primo passo: replay di osservazioni Go2 registrate.
5. **Testa di adattamento spiking plastica (tre fattori) su policy base congelata, confrontata con RMA e SMA.** Novità media, fattibilità media.
6. **Prior CPG + residuo spiking con reward energetico (Pareto CoT vs spike/passo).** Novità media (vicino a [2605.09595]), fattibilità alta.
7. **Propriocezione event-based (delta coding) e stimatore spiking di velocità/contatti** in sostituzione della velocità privilegiata. Novità alta, fattibilità media (stato perso a robot fermo → refresh periodico).
8. **Livello riflesso spinale spiking veloce sotto la policy lenta.** Novità alta, fattibilità medio-bassa.

---

## 6. Piano proposto (da discutere)

0. **Baseline ANN** Go2 in Isaac Lab con la ricetta di §2 (vx/vy/yaw, spinte, randomizzazione). Servono come riferimento e come teacher.
1. **Actor SNN diretto con PPO** (§3) e, se non converge, **distillazione + DAgger**. Metriche: errore di tracking vx/vy/yaw, successo al recupero da spinte, spike rate/SynOps, confronto ANN abbinato.
2. **Contributo principale**: idea 1 (student SNN ricorrente senza history encoder) + idea 2 (studio di robustezza).
3. **Sim2sim → sim2real su Go2** con policy SNN su Jetson/PC; poi (opzionale) Loihi 2.

## 7. Voci non verificate / da controllare
- Venue di Jiang et al. (alcuni citano ICASSP 2025), ICRA 2023 per DreamWaQ, RA-L 2022 per Concurrent Training.
- "Hybrid-Coding", "Spiking-PPO" come metodo nominato e un benchmark "Spiking RL MuJoCo" dedicato: non trovati come tali.
- Xiao, Han, Wang, *Control and Decision* 2025 (SNN RL + CPG quadrupede): visto solo come riferimento.
- Alcuni ID arXiv 26xx.xxxxx provengono dal tool di fetch; ricontrollare.
- Zanatta et al., Sci. Rep. 14:30648 (2024): letto solo da snippet.
- Costi/tempi di training e star GitHub sono approssimativi.
