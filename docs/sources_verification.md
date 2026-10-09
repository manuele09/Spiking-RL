# Verifica delle fonti (2026-10-09)

Stato di verifica delle voci di `literature_review.md` da cui dipende una scelta di design del piano.
Legenda: **[V-full]** testo completo/HTML letto; **[V-abs]** abstract/landing page letta; **[V-code]** file di codice/README letto dal repo; **[NV]** non verificato; **[CORR]** discrepanza corretta rispetto alla rassegna.

## 1. Lavori SNN per locomozione

| Voce | Stato | Cosa risulta | Note / correzioni |
|---|---|---|---|
| Jiang et al., *Fully Spiking Neural Network for Legged Robots*, arXiv 2310.05022 | [V-full] v3 HTML + [V-abs] IEEE | **ICASSP 2025**, DOI 10.1109/ICASSP49660.2025.10890793. A1, Cassie, MIT Humanoid in Isaac Gym, 500 Hz; PopSAN-like (pop. encoding per dimensione, LIF current-based, reset hard), "temporal shrinking" multi-stadio con classificatori ausiliari; energia T=1/2/3: 3.44/15.54/35.49 ×10⁻⁶ mJ vs ANN 86.27 (−96.0/−82.0/−58.9 %); convergenza PPO più lenta dell'ANN; rumore su comandi σ=0.1–0.3, ANN cade a 0.3. | [CORR] venue confermata ICASSP 2025. **Attenzione:** una pagina ResearchGate associata al titolo contiene testo su "Go1 reale, AMP, zero-shot su terreno difficile" che sembra appartenere a un altro lavoro (PTRL). **Prima di scrivere "nessun SNN appreso su quadrupede reale" va letto il PDF IEEE finale**: se la versione ICASSP ha un esperimento reale su Go1, il nostro claim si restringe a "Go2 + comando vx/vy/yaw + spinte + student ricorrente". |
| Wang et al., *Artificial plateau neurons with in-situ spike-malleability for rhythmic quadrupedal locomotion*, Nat. Commun. 17:5801 (2026), DOI 10.1038/s41467-026-72428-2 | [V-abs] (search + PMC) | Neurone PG-TS (plateau gate + threshold switch), ~141 pJ/spike; 4 oscillatori → giunti thigh/calf anteriori, spike → segnale sinusoidale via filtro gaussiano → PD; trotto su **Go2 reale**; Isaac Gym per simulazione. Nessun RL, nessun comando di velocità. | Conferma la rassegna. |
| ES-Parkour, arXiv 2503.09985 | [V-abs] | SNN + camera a eventi per parkour quadrupede, energia 11.7 % dell'ANN. Metodo di training (distillazione) e reale/simulato **non deducibili dall'abstract**. | [NV] il dettaglio "distillazione da teacher privilegiato" e "solo simulazione" non è confermato: leggere il PDF. |
| Repo `Kostantinoskanell/thesis` | [V-code] README | Go2 in Isaac Lab (locomozione) e MuJoCo (navigazione); snntorch + LIF/ALIF propri, PopSAN, R-STDP; **PPO diretto → belly-flop**, funziona con **distillazione + DAgger**; locomozione: miglior setting T=5 + penalità firing = **1.04×** più economico dell'MLP con "full accounting", T=8 = 0.57× (peggio); navigazione: 10.4× su SynOps ma 2.4× full accounting; **ipotesi di robustezza al rumore refutata** (LiDAR σ=0.8: MLP 38 %, SNN 18 % successo); R-STDP non aiuta. **Nessuna licenza** nel repo → non riusare codice. | Conferma e rafforza: la robustezza SNN **non va assunta**, va misurata; il risparmio energetico con decodifica float può essere nullo. |
| Han & Sengupta, arXiv 2312.15805 (CPG spiking + astrociti, R-STDP) | [V-abs] v4 (set. 2025) | Quadrupede simulato, trotto su piano, 23.3× meno potenza di calcolo. | OK. |
| Han & Sengupta, arXiv 2605.09595 | [V-abs] | *Neuromorphic RL for Quadruped Locomotion Control on Uneven Terrain*: Equilibrium Propagation come PPO locale ("output nudging", clipping bilaterale), CPG + residuo di postura, A1 12-DoF, 4.3× meno memoria GPU di BPTT. | [CORR] **il CPG non è dichiarato spiking** nell'abstract; "EP-based PPO" sì. Simulazione vs reale non deducibile. |
| Xiao, Han, Wang, *Control and Decision* 2025 (SNN-RL + CPG, quadrupede) | [NV] | Visto solo come citazione (in cinese). | Da controllare se ha esperimenti reali. |
| Zanatta et al., *Exploring spiking neural networks for deep RL in robotic tasks*, Sci. Rep. 2024 | [V-abs] (search) | Benchmark SNN-RL su Ant in Isaac Gym/MuJoCo, non hardware. | [CORR] l'ID corretto è **s41598-024-77779-8**, non "14:30648". |

## 2. RL standard per Go2 (baseline)

| Voce | Stato | Cosa risulta | Note |
|---|---|---|---|
| Isaac Lab `velocity_env_cfg.py` (main) | [V-code] | Obs policy: base_lin_vel, base_ang_vel, projected_gravity, commands, joint_pos_rel, joint_vel_rel, actions, height_scan (rumore uniforme per termine; nessuna scala). Azione joint_pos scale 0.5 (base). Eventi: attrito statico 0.8 (fisso!), massa base ±5 kg, COM ±5 cm, **push_robot ogni 10–15 s con vx,vy ∈ ±0.5 m/s**. Reward: track_lin 1.0, track_ang 0.5, lin_vel_z −2, ang_vel_xy −0.05, torques −1e-5, dof_acc −2.5e-7, action_rate −0.01, feet_air_time 0.125, undesired_contacts −1. Terminazioni: time_out, base_contact. dt 0.005, decimation 4, episodio 20 s. Comandi: vx,vy,ωz ∈ ±1, heading. | **[CORR]** nel branch main la velocità lineare di base è **nelle osservazioni della policy** (non solo critic) e non c'è gruppo "critic" separato: va aggiunto da noi (asimmetrico). I valori "×0.25, ×0.05" della rassegna vengono da legged_gym/unitree_rl_gym, non da Isaac Lab. |
| Isaac Lab `config/go2/rough_env_cfg.py` | [V-code] | **`push_robot = None`** (spinte disabilitate), massa base −1…+3 kg, scale azione 0.25, track_lin 1.5, track_ang 0.75, torques −2e-4, feet_air_time 0.01, undesired_contacts rimosso, base_com rimosso. | Conferma: spinte da riabilitare. |
| Isaac Lab `config/go2/flat_env_cfg.py` | [V-code] | Eredita Rough; flat_orientation_l2 −2.5, feet_air_time 0.25; terreno piano, niente height scan. | OK. |
| Isaac Lab `UNITREE_GO2_CFG` | [V-code] | DCMotorCfg: Kp 25, Kd 0.5, effort/saturation 23.5 Nm, vel. limit 30 rad/s; posa: hip ±0.1, thigh 0.8 (ant.) / 1.0 (post.), calf −1.5. | OK. |
| Isaac Lab `go2/agents/rsl_rl_ppo_cfg.py` | [V-code] | Rough: 24 step/env, 1500 iter, [512,256,128] ELU, lr 1e-3 adattivo KL 0.01, clip 0.2, entropia 0.01, 5 epoche, 4 minibatch, γ 0.99, λ 0.95, grad clip 1.0. Flat: 300 iter, [128,128,128]. | OK. |
| Isaac Lab gym id | [V-code] | `Isaac-Velocity-Flat-Unitree-Go2-v0`, `Isaac-Velocity-Rough-Unitree-Go2-v0` (+ `-Play-v0`). | OK. |
| Isaac Lab requisiti | [V-abs] docs | Ubuntu 22.04 / Win 11, ≥32 GB RAM, **GPU NVIDIA ≥16 GB VRAM**, driver ≥580.65, Python 3.11 (Isaac Sim 5.x); Isaac Sim 5.1.0 raccomandato; ultima tag Isaac Lab **3.0.0-beta2** (pagina incoerente: badge 6.1.0 per Isaac Sim). Installazione pip di Isaac Sim disponibile. | **Il container attuale non ha GPU** (`nvidia-smi` assente, 4 CPU, 15 GB RAM, Python 3.13): Isaac Lab non è eseguibile qui. |
| `unitree_rl_lab` | [V-code] README | Go2, H1, G1-29dof; badge Isaac Lab 2.3.0 / Isaac Sim 5.1.0; Apache-2.0; sim2sim via `unitree_mujoco`; deploy C++ con unitree_sdk2 (esempio G1). | OK. |
| `unitree_rl_gym` | [V-code] README | Go2/H1/H1_2/G1 in Isaac Gym; `deploy/deploy_mujoco/deploy_mujoco.py`, `deploy/deploy_real/deploy_real.py`; BSD-3. | Isaac Gym è deprecato: usare solo come riferimento per deploy. |
| `unitree_mujoco` | [V-code] README | Simulatore MuJoCo con DDS identico al robot (LowCmd/LowState, dominio 1 su `lo`); Go2 supportato; C++ raccomandato; BSD-3. | Base per L7 sim2sim "fedele al deploy". |
| `unitree_sdk2_python` | [V-code] README | cyclonedds 0.10.2; `example/low_level/lowlevel_control.py <iface>`; **spegnere sport_mode dall'app** prima del low-level. BSD-3. | OK. |
| Unitree Go2 (sito ufficiale) | [V-abs] | ~15 kg; 70×31×40 cm; coppia max ~45 Nm (giunto maggiore); **sviluppo secondario: non disponibile su Air/Pro, parziale su X, completo su Edu**; Edu con modulo Orin opzionale. | **Serve un Go2 Edu** (o X, da chiarire) per il controllo low-level. |
| `mujoco_menagerie/unitree_go2` | [V-code] clone sparse, commit `0059d43` (2026-10-07) | `go2.xml`, `scene.xml`, `go2_mjx.xml`, `scene_mjx.xml`; BSD-3; keyframe `home` z=0.27, giunti (0, 0.9, −1.8)×4; motori a coppia ctrlrange ±23.7 (hip) / ±45.43 (knee); armature 0.01, frictionloss 0.2. | Il PD va fatto in Python (gli attuatori sono `motor`). |
| MuJoCo Playground | [V-code] README + `go1/joystick.py` + `learning/` | Solo **Go1** (nessun Go2) fra i quadrupedi; env Go1 joystick: ctrl_dt 0.02, sim_dt 0.004, Kp 35, Kd 0.5, action_scale 0.5, obs 48 (con linvel rumorosa), privileged 123; spinte (`pert_config`, velocity_kick 0–3 m/s) **disabilitate di default**; `learning/train_jax_ppo.py` e **`learning/train_rsl_rl.py`**; wrapper `mujoco_playground/_src/wrapper_torch.py::RSLRLBraxWrapper` (DLPack, obs `state`/`privileged_state`). Apache-2.0. GPU + JAX CUDA12 richiesti; Python ≥3.10. | Nome PyPI `playground` (0.2.0) non verificato sulla pagina PyPI (errore di caricamento); il README consiglia l'install da sorgente con `uv`. |
| Genesis | [V-abs] docs | `examples/locomotion/go2_{env,train,eval}.py`, obs 45, 4096 env, rsl-rl-lib ≥5.0; reward semplificata (6 termini); "esempio minimale". | Non maturo come baseline; utile come terzo simulatore. |
| mjlab | [V-code] README | Isaac Lab API + MuJoCo Warp; **solo G1** nel README (Go1 citato nei docs secondo una ricerca, Go2 no); richiede GPU NVIDIA; Apache-2.0; PyPI `mjlab` 1.6.0. | Opzione futura, non ora. |
| rsl_rl | [V-code] | `rsl-rl-lib` 5.5.1 su PyPI; `Distillation` (student agisce, teacher etichetta, loss MSE/Huber, TBPTT `gradient_length`, supporta student ricorrenti, **nessun mixing DAgger β**); moduli `mlp.py`, `rnn.py` (classe `RNN` GRU/LSTM con `reset(dones)` e `detach_hidden_state`), `cnn.py`, `normalization.py`; `VecEnv.get_observations() -> TensorDict`, `step -> (TensorDict, rew, done, extras)`. Isaac Lab `RslRlOnPolicyRunnerCfg` ha `actor`/`critic: RslRlMLPModelCfg` e `obs_groups` obbligatorio (es. `{"actor": ["policy"], "critic": ["policy","privileged"]}`). | [NV] il punto esatto in cui agganciare una rete custom (classe modello che incapsula `MLP`) non è stato letto: da verificare su `rsl_rl/models/` o `rsl_rl/modules/__init__.py` della versione pinnata. |

## 3. Architetture / training SNN

| Voce | Stato | Cosa risulta |
|---|---|---|
| PopSAN, arXiv 2010.09635 | [V-abs] + [V-code] repo `combra-lab/pop-spiking-deep-rl` | CoRL 2020; PPO/DDPG/TD3/SAC; Loihi 140× meno energia di Jetson TX2; repo **MIT**, codice Python 3.5–3.8 (vecchio) con cartella PPO. |
| ILC-SAN, arXiv 2401.05444 | [V-abs] | Azione dal **potenziale di membrana di neuroni non-spiking** di uscita, una popolazione per dimensione, connessioni intra-strato. Venue non indicata. |
| Van den Berghe et al., arXiv 2510.24461 | [V-abs] | Pendenze del surrogate meno ripide/schedulate → **2.1×** in RL; usa una **policy privilegiata guida** per il bootstrap; task: drone reale. Baseline BC/TD3BC. |
| CRPI, arXiv 2601.21778 | [V-abs] | **ICML 2026**; conversione ANN→SNN in controllo continuo soffre di errori temporalmente correlati; CRPI riporta il potenziale residuo tra passi. |
| Proxy Target, arXiv 2505.24161 | [V-abs] | **NeurIPS 2025**; problema della target network con SNN off-policy; +32 %. Non riguarda PPO. |
| Huebotter et al., arXiv 2509.05356 | [V-abs] | Braccio (Panda) model-based; ablation: inizializzazione, **costanti di tempo apprese, soglie adattive**, compressione latente. Non è RL model-free. |
| SMA, arXiv 2306.01906 | [V-abs] | Regola a tre fattori meta-ottimizzata, quadrupede, simile a RMA; reale/simulato non dichiarato. |
| lf-cs, arXiv 2402.10069 | [V-abs] | PPO "bio-plausibile" per reti spiking ricorrenti; nessun task robotico nell'abstract. |
| SpikeRL, arXiv 2502.17496 | [V-abs] | Framework distribuito (NCCL, mixed precision), PopSAN-like; 4.26× più veloce del precedente. Non riguarda env paralleli su GPU. |
| Stewart et al., arXiv 2512.03911 | [V-full] HTML | NICE 2026 (sottomesso); Astrobee, 12×64×64×6 ReLU PPO → SDNN; Loihi 2 4.2 ms vs GPU 4.9 ms; 0.013 J vs 0.217 J totali per inferenza (0.008 vs 0.069 J dinamici); RMSE posizione 0.225 vs 0.142 m; Lava-DL 0.5.0, Lava 0.9.0, NxKernel 0.4.1; Isaac Lab 2.2.0. |
| Lava-DL | [V-code] | **Repo archiviato (read-only)**, Intel non garantisce supporto; install da sorgente/conda; BSD-3. `lava-nc`/`lava-dl` non su PyPI. |
| INRC / Loihi 2 | [V-abs] | Accesso via proposta di progetto (PI permanente di università/lab), cloud vLab via SSH o sistema in prestito ≤1 anno; RFP chiusa per fondi. |

## 4. Librerie SNN (PyPI, 2026-10-09)

| Pacchetto | Versione | Note |
|---|---|---|
| `spikingjelly` | 0.0.0.0.14 stabile; **V2 (2.0.0rc) in pre-release**: torch ≥2.6, Python ≥3.11, backend Triton opzionale, niente CuPy | API in transizione: pinnare la versione. |
| `snntorch` | 1.0.0 (2026-06-29), MIT | `snn.Leaky` ecc. |
| `spyx` | 1.0.0 (JAX) | Pagina PyPI non letta. |
| `norse` | 1.1.0 | — |
| `nir` | 1.0.8 | Formato di scambio per hardware neuromorfico. |
| `torch` | 2.14.1 (CPU wheel per Py 3.13 installata nel container) | — |
| `mujoco`/`mujoco-mjx`/`mujoco-warp` | 3.15.0 | — |
| `rsl-rl-lib` | 5.5.1 | — |
| `genesis-world` | 1.4.3 | — |
| `brax` | 0.14.2 | — |
| `hydra-core` 1.3.7, `wandb` 0.30.0, `tensorboard` 2.21.0, `gymnasium` 1.4.0, `stable-baselines3` 2.9.0 | — | — |

## 5. Non verificato (resta [NV])

- DreamWaQ (ICRA 2023?), Concurrent Training (RA-L 2022?), Walk These Ways (latenza 20 ms, actuator net): non riletti; non determinano scelte di design.
- "Hybrid-Coding", "Spiking-PPO" come metodi nominati: non cercati di nuovo.
- Dettagli di SpikeGym e dello scaling SNN-PPO con ≥4096 env: nessuna fonte trovata; il nostro micro-benchmark (`scripts/bench_snn_libs.py`) lo misura.
- Tempi di training citati nel piano sono **stime** e marcati come tali.
