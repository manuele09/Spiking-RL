# Controllo di novità mirato (2026-10-10)

Complementa `literature_review.md` e `sources_verification.md` (non ripete ciò che c'è).

**Livello di lettura.** FT = full text letto; ABS = abstract/landing page; SNIP = solo snippet di ricerca. La ricerca web è non esaustiva: "non trovato" significa "non trovato con ~15 query", non "non esiste". Il tool di ricerca ha restituito molti risultati ripetuti; la copertura di Google Scholar, Semantic Scholar e dei proceedings IEEE è nulla (la pagina IEEE Xplore non si è caricata).

---

## Q1. Jiang et al. (2310.05022 / ICASSP 2025): deploy su quadrupede reale?

- **Fully Spiking Neural Network for Legged Robots** — X. Jiang, Q. Zhang, J. Sun, J. Cao, J. Ma, R. Xu; arXiv v1 ott. 2023, v3 set. 2024; ICASSP 2025 (DOI 10.1109/ICASSP49660.2025.10890793, già in `sources_verification.md`). <https://arxiv.org/html/2310.05022v3>
- **Letto: FT della v3 HTML.** Solo simulazione in Isaac Gym: A1, Cassie, MIT Humanoid. Nessuna occorrenza di Go1, hardware, real-world, deployment, sim-to-real; nessuna sezione "future work". Energia stimata analiticamente (MAC 4.6 pJ, AC 0.9 pJ, float32, 45 nm). Limiti dichiarati: gradiente surrogato impreciso, convergenza più lenta dell'ANN. Robustezza: solo rumore gaussiano sui *comandi* (σ=0.1/0.2/0.3), ANN cade a 0.3.
- **Non letto:** il PDF IEEE finale (pagina non caricata). Non ho trovato alcuna versione giornale/estesa degli stessi autori. La pagina ResearchGate citata nel precedente controllo non è stata risolta: il testo "Go1 reale, AMP" resta più plausibilmente di un altro lavoro, ma **non posso escludere** differenze tra v3 arXiv e versione ICASSP.
- **Lavoro correlato degli stessi autori: ES-Parkour** (Zhang, Cao, Sun, Shao, Han, Zhao, Guo, Xu; arXiv 2503.09985, mar. 2025; venue non determinata). <https://arxiv.org/pdf/2503.09985> — **FT** (testo estratto dal PDF). SNN IF, T=4, spiking ResNet-18 + **GRU (non spiking)** per fondere propriocezione ed eventi, actor MLP spiking [512,256,128]; **distillazione ANN teacher → SNN student** con MSE (poi interazione on-policy dello student); 32 env, Isaac Gym, camera a eventi *simulata*; energia analitica MAC/AC con −88.3%. Le conclusioni parlano di validare "in simulazione per la fattibilità su robot reali": **solo simulazione**, nessun hardware.
- **Verdetto Q1: non trovato (nessun deploy reale in Jiang et al. v3).** Residuo di rischio: versione ICASSP non letta.

## Q2. Studio sistematico di robustezza SNN vs ANN in locomozione / RL

| Lavoro | Letto | Contenuto | Verdetto |
|---|---|---|---|
| Jiang et al. 2310.05022 | FT | Solo rumore sui comandi, 1 tipo di perturbazione, una policy per robot | parzialmente |
| Repo tesi Go2 (Kostantinoskanell/thesis) | README (da verifica precedente) | Ipotesi di robustezza al rumore *refutata* in navigazione (LiDAR σ=0.8: MLP 38%, SNN 18%); non peer-reviewed | parzialmente |
| Han & Sengupta, 2605.09595 (Neuromorphic RL for Quadruped Locomotion on Uneven Terrain) | HTML primi ~100k caratteri su ~149k | A1 in MuJoCo, **solo simulazione**; **CPG Hopf continuo, rete a equilibrium propagation, non spiking** (correzione al precedente "CPG spiking"); confronto con PPO-BPTT su terreni/velocità/durezze, 500 episodi × 5 seed, domain randomization con spinte in training; nessun test sistematico di robustezza SNN | parzialmente (rigore statistico, ma non SNN e non robustezza) |
| SpikeVLA, ICML 2026 (arXiv 2606.27807; Song, Nie, Teng et al.) | ABS + snippet | Afferma che la codifica Laplaciana migliora stabilità e robustezza al rumore in locomozione quadrupede; è un VLA, dettagli non verificati | parzialmente / da leggere |
| Robustezza avversaria SNN in visione (Sharmin et al. 1905.02704; ICLR 2026 "Robust SNNs Against Adversarial Attacks" 2602.20548; arXiv 2512.22522 "robustezza sovrastimata") | ABS/SNIP | Solo classificazione. Messaggio utile: la robustezza apparente degli SNN è spesso un artefatto di gradienti mascherati (attacchi ben costruiti la annullano) | rilevante come metodologia, non come risposta |
| QC-SANE (TNNLS 2023), NoisySAN 2403.04162, SANSAC 2608.22729 (ICLR 2026 workshop) | ABS/SNIP | RL continuo con attori spiking su MuJoCo; prestazioni, esplorazione; **nessuna** analisi di robustezza | non trovato (sul tema) |

- **Verdetto Q2: non trovato** uno studio sistematico SNN vs ANN su locomozione a gambe con spinte, carico, attrito, latenza, rumore sulle osservazioni, attacchi avversari. Le evidenze esistenti sono aneddotiche (un tipo di rumore) e **contraddittorie** (SNN più robusto sui comandi in Jiang; più fragile nel rumore sensoriale nella tesi). La letteratura di visione suggerisce di usare attacchi adattivi (BPDA/EOT, trasferimento da surrogato ANN) per non dichiarare robustezza fittizia.

## Q3. Policy SNN ricorrente (memoria intrinseca) al posto di history encoder / RMA

- **Synaptic Motor Adaptation (SMA)** — S. Schmidgall, J. Hays; arXiv 2306.01906 (giu. 2023), venue non dichiarata. ABS. Regola a tre fattori meta-ottimizzata che approssima l'embedding privilegiato in stile RMA, quadrupede; sim/reale non dichiarato nell'abstract. **Verdetto: parzialmente** (adattamento plastico spiking in stile RMA, ma nessuna policy con soglie adattive/ALIF che sostituisca l'adaptation module, nessun Go2, nessuna distillazione teacher-student).
- **ES-Parkour** (sopra): usa distillazione teacher privilegiato → student SNN, ma la memoria è affidata a un **GRU non spiking**. Quindi "teacher-student verso SNN": già fatto (parkour, sim); "memoria spiking": no. **Verdetto: parzialmente.** *Questo corregge la rassegna, che dava "distillazione teacher privilegiato → student SNN per locomozione" come non trovata: esiste in ES-Parkour e nella tesi Go2.*
- **lf-cs** (2402.10069), **Huebotter et al.** (2509.05356, ALIF/costanti di tempo apprese su braccio), **A Spiking Neural Architecture for Coordinating Arm and Locomotor Control** (Steffen, Simone, Damberger, DeWolf, Ly, Eliasmith; arXiv 2606.11034, giu. 2026, NEF/Nengo, H1 simulato, approssima una policy ANN di riferimento, niente robustezza né energia misurata; ABS): nessuno sostituisce un history encoder in locomozione.
- Ricerche specifiche "spiking" + RMA / rapid motor adaptation / state-space / history encoder: **nessun risultato pertinente** (solo lavori ANN come SleepWalking 2608.30883, DWAQ, che sono l'alternativa non spiking da citare).
- **Verdetto Q3: non trovato** (SNN ricorrente intrinseca come sostituto dichiarato di history encoder/RMA in locomozione a gambe). Rischio da monitorare: SMA e ES-Parkour sono vicini concettualmente.

## Q4. SNN appresa su quadrupede reale con comando di velocità (2025–2026)

- Ricerche su spiking + Unitree/Go1/Go2/A1/ANYmal + sim-to-real, Loihi/SpiNNaker/Jetson, venue (ICRA/IROS/CoRL/RA-L/Sci. Robotics/Nat. MI): **nessun lavoro trovato**.
- Vicini ma non equivalenti:
  - Wang et al., Nat. Commun. 17:5801 (2026): Go2 reale, oscillatori su hardware analogico, nessun RL né comando (già in rassegna, ABS).
  - Arena, Cannizzo, Li Noce (2025): rete spiking ispirata al cervello di Drosophila per navigazione visiva con Go2 **simulato in Gazebo**, non RL, non velocity tracking (SNIP; autori e venue da verificare; potrebbe essere lavoro del vostro gruppo: controllare la citazione).
  - ICANN 2025 (Springer, doi 10.1007/978-3-032-04555-3_25): CPG spiking con feedback sensoriale per cambio di andatura (SNIP).
  - Han & Sengupta 2312.15805 (CPG spiking astrociti, simulato) e 2605.09595 (sim, non spiking).
  - Tesi Pittsburgh (CPG su Loihi/Arduino), non peer-reviewed.
- **Verdetto Q4: non trovato.** Nessuna evidenza di hardware neuromorfico con policy appresa a gambe. Limite: la ricerca non copre i proceedings di ICRA/IROS 2026 in modo indicizzato.

## Q5. Energia SNN vs ANN con "full accounting"

- **Jiang et al. e ES-Parkour (FT):** conteggio analitico MAC 4.6 pJ / AC 0.9 pJ float32 a 45 nm; il primo strato (encoder) è trattato come MAC; nessuna misura, nessun costo di memoria, nessun costo del decoder float. Risparmi dichiarati: −59…−96% (Jiang), −88.3% (ES-Parkour, SNN confrontato su vision encoder).
- **Yan, Bai, Tang, Wong, "Reconsidering the Energy Efficiency of SNN Inference from Analytical Perspectives"** (arXiv 2409.08290; IEEE TCAD) — ABS: includendo memoria e movimento dati e confrontando con ANN quantizzate a ⌈log₂(T+1)⌉ bit, il vantaggio SNN vale solo in regimi specifici (con T=5, spike rate < ~5.7% per battere la QNN equivalente).
- **"Are SNNs really more energy-efficient than ANNs? An in-depth hardware-aware study"** (Dampfhöffer et al., IEEE TETCI 2022, HAL cea-03852141; autori da verificare; SNIP/ABS): break-even solo a 0.15–1.38 spike per sinapsi per inferenza; i modelli IF battono i LIF.
- Un lavoro sul calcolo spaziale (arXiv 2505.11418, SNIP) mostra la sensibilità al rapporto AC/MAC (~2/3 invece di 0.9/4.6).
- **Lavori "più onesti" nel dominio locomozione/controllo:** (a) la tesi Go2: miglior setting locomotorio 1.04× con full accounting (encoder/decoder float), 0.57× a T=8; navigazione 10.4× SynOps ma 2.4× full accounting (non peer-reviewed, README); (b) Stewart et al. 2512.03911 (Astrobee su Loihi 2: 0.013 J vs 0.217 J misurati, ma tracking peggiore; già in rassegna).
- **Verdetto Q5: parzialmente.** Esistono analisi generali oneste (visione) e un caso Go2 non revisionato; **non esiste** un confronto energetico peer-reviewed su locomozione a gambe con full accounting (encoder/decoder float, accessi in memoria, sensibilità a rapporto AC/MAC e a T).

---

## Claim possibili e rischio di anticipazione

| Claim | Rischio | Motivazione |
|---|---|---|
| Prima SNN appresa su Go2 **reale** con comando (vx,vy,yaw) in anello chiuso | **Basso-medio** | Nulla trovato (Q4). Rischio residuo: preprint 2026 non indicizzati, lavori cinesi, versione ICASSP non letta. Formulare come "per quanto ne sappiamo" e rifare la ricerca prima della submission. |
| Primo studio **sistematico** di robustezza SNN vs ANN in locomozione | **Basso-medio** | Esistono solo test su un tipo di rumore (Jiang) e un risultato negativo non peer-reviewed. Rischio che 2606.27807 (SpikeVLA) o un nuovo preprint lo copra in parte. Rafforzare con attacchi adattivi e multi-seed (come richiesto dalla letteratura di visione). |
| Prima policy SNN **ricorrente intrinseca** (ALIF/costanti apprese) che sostituisce history encoder/RMA in locomozione | **Medio** | Non trovata (Q3), ma SMA (adattamento plastico spiking) e ES-Parkour (teacher-student, GRU non spiking) sono vicini. Evitare "primo a distillare in SNN"; limitare a "memoria nei neuroni al posto di GRU/adaptation module". |
| Distillazione teacher privilegiato → student SNN in locomozione | **Alto** | Già fatto in ES-Parkour (parkour, sim) e nella tesi Go2. Non rivendicarlo come novità. |
| Policy SNN a gambe su Loihi 2 / SpiNNaker2 in anello chiuso | **Basso** (novità) ma **fattibilità bassa** | Nulla trovato; richiede INRC. |
| Risparmio energetico SNN su locomozione | **Rischio di claim non difendibile** | Con full accounting il risparmio può essere ~1× (tesi Go2). Se rivendicato, farlo solo con conteggio completo e analisi di sensibilità (Yan et al.), oppure come risultato negativo/onesto. |
| Primo confronto energetico full-accounting su locomozione a gambe | **Medio-basso** | Non trovato peer-reviewed; vale come contributo metodologico. |

## Limiti di questo controllo
- FT letti: Jiang v3 HTML, ES-Parkour PDF, 2605.09595 (parziale, ~2/3). Tutto il resto è ABS o SNIP.
- Non letti: ICASSP finale, SpikeVLA, SMA (full text), Dampfhöffer, Xiao et al. 2025, Arena et al. 2025 (autori/venue da confermare).
- Le date arXiv 26xx provengono dal tool di ricerca e vanno ricontrollate prima di citarle.
