# CLAUDE.md — Spiking-RL (policy spiking + RL per la locomozione del Unitree Go2)

Leggi questo file per intero prima di fare qualsiasi cosa. Lo stato completo è qui; i dettagli in `docs/`.

## Obiettivo e preferenze dell'utente
- **Obiettivo finale:** Go2 (versione **Edu**, controllo low-level possibile) che trotta con comandi (vx, vy, yaw), robusto a spinte, con una policy **spiking appresa**. Vogliono un contributo innovativo. Si parte da compiti semplici (balancing) e si sale a gradini (L0…L9, vedi `docs/implementation_plan.md`).
- L'utente parla **italiano**; rispondi in italiano. Vuole risultati **onesti**: dichiara limiti (pochi seed, un solo robot simulato, stime vs misure), distingui "letto full text" da "visto abstract". Non fare claim di novità senza controllo.
- Non creare pull request se non richiesto. Sviluppo sul branch `claude/magical-fermat-3ptegv`, commit frequenti con push (`git push -u origin <branch>`). Hook di stop richiede working tree pulito e pubblicato.
- L'utente NON tornerà sulla sessione precedente: tutto il contesto vive in questo file e in `docs/`.

## Stato al 2026-10-10
**Fatto (tutto su CPU, MuJoCo, container senza GPU):**
- Rassegna bibliografica (`docs/literature_review.md`, verifica fonti `docs/sources_verification.md`, controllo novità `docs/novelty_check.md`).
- Piano implementativo completo scritto da un agente (`docs/implementation_plan.md`): curriculum L0–L9, architetture A0–A5 (esistenti) e N1–N6 (nuove), setup, protocollo esperimenti, troubleshooting. **L1 è stato modificato** rispetto al piano originale (vedi sotto).
- Core SNN (`spiking_rl/snn`): LIF e ALIF con surrogate arctan, init "rate-aware" (obbligatoria, senza il 2° strato è morto), reset per `dones`, encoder diretto e a popolazione.
- Actor/critic intercambiabili (`spiking_rl/models/actors.py`), PPO di riferimento in stile CleanRL (`spiking_rl/algos/ppo.py`), CLI `python -m spiking_rl.train`.
- **E1** (SNN-PPO vs ANN-PPO su CartPole/Pendulum, 3 seed): pipeline validata (`docs/results_E1.md`).
- Ambiente **Go2Balance-v0** (MuJoCo, `spiking_rl/envs/mujoco_go2/env.py`): obs 45-d, azione 12 offset, PD Kp 25/Kd 0.5, 50 Hz, pose di default Isaac Lab, spinte, randomizzazioni, parametri per i test di robustezza.
- **L1** (stand & balance con spinte): baseline ANN e SNN T=4/T=1 con la **config D** (`docs/results_L1.md`; checkpoint in `results/l1/`).
- Strumenti: `spiking_rl/eval/{push_curve,robustness,synops}.py`, `scripts/render_policy.py` (video), `scripts/render_go2.py`.

**Risultati L1 chiave (30 episodi/punto, spinta fissa ogni 1.5–3 s, sopravvivenza a 10 s):**

| Actor | 0.5 m/s | 1.0 m/s | 1.5 m/s |
|---|---|---|---|
| Azione zero (solo PD) | 30/30 | 15/30 | 0/30 |
| ANN (3 seed) | 30/30 ×3 | 28, 28, 24 | ~0 |
| SNN T=4 (3 seed) | 30/30 ×3 | 26, 21, 21 | 0 |
| SNN T=1 (1 seed) | 30/30 | 20 | 1 |

Lettura onesta: l'SNN impara a recuperare ma sembra un po' peggiore della ANN (differenza non provata con 3 seed). Nessun run regge spinte > ~1.2 m/s (range di training). Energia (stima 45 nm, `synops.py`): SNN T=4 vs ANN 1.1× peggio (conservativo) o 2.7× meglio (corrente del 1° strato calcolata una volta), con firing rate 0.26/0.49.

**Lezione importante (non ripetere l'errore):** con il reward del piano originale la ANN NON imparava (3–9/30 a 1.0 m/s, peggio dell'azione zero). Ha funzionato solo con la **config D**: `w_pose 0.05` (era 0.5), `w_action_rate 0.002` (era 0.01), `term_penalty 1.0`, spinte 0.3–1.2 m/s ogni 1–2 s, `init_log_std -1.2`. Le penalità alte su posa/azioni impediscono il passo di recupero. D cambia 3 cose insieme: non sappiamo quale sia decisiva (ablation non fatta). Inoltre la policy a azione zero sopravvive già a 29/30 con spinte ≤ 1.0 m/s ogni 3–6 s: per questo il criterio L1 è diventato una **curva di sopravvivenza vs intensità della spinta**.

## Architettura SNN attuale (actor)
obs(45, normalizzate) → corrente diretta ripetuta T volte → Linear 45→128 → LIF → Linear 128→128 → LIF → Linear 128→12 (readout float sugli spike, media su T) = media della gaussiana; std appresa e separata (init e^-1.2). LIF: u←β·u+Wx, spike a u≥1, reset hard, β=sigmoid(w) per neurone (init 0.5). Surrogate arctan con pendenza 0.5→2.0 nel primo 30% del training. **Nessuna memoria tra passi di controllo** (carry=False), nessuna ALIF nei run. ~24k parametri (come l'ANN 128×128 ELU). Critic: MLP 128×128 ANN.

## Cosa NON è stato fatto / aperto
- Più seed (servono ≥5 per configurazione): mancano T=1 seed 1 (interrotto a 805k passi) e seed 2.
- Test di robustezza (`robustness.py`, ~1 h a run su CPU con 20 episodi/condizione) e conto SynOps su tutti i run: solo smoke test fatti, nessun risultato utilizzabile.
- L2 (altezza/assetto comandati), L3+ (trotto) — richiedono GPU.
- Architetture nuove (memoria spiking/ALIF al posto dell'history encoder: N1), distillazione teacher→student.
- Controllo novità: nessun lavoro trovato che anticipi (a) studio sistematico robustezza SNN vs ANN in locomozione, (b) SNN ricorrente al posto dell'history encoder, (c) SNN appresa su Go2 reale con comando di velocità — ma la ricerca non è esaustiva. La distillazione teacher→student SNN è già fatta (ES-Parkour). Han&Sengupta 2605.09595 NON è spiking. Residuo: il PDF IEEE finale di Jiang et al. non era leggibile (rischio "deploy Go1 reale").

## Prossimi passi consigliati
1. **GPU via SSH (funzionante dal 2026-10-10):** PC remoto `miniworkstation` (IP Tailscale 100.71.145.63, hostname `lenovo`), utente non privilegiato `claude-gpu` (niente sudo, non nel gruppo docker, home `/home/claude-gpu`; sul PC ci sono altri utenti: lavora solo nella tua home). GPU **RTX A2000 12 GB** (driver 595.84, nvcc presente, Python 3.12.3, ~78 GB liberi): sotto i 16 GB di Isaac Lab, quindi probabile MJX/MuJoCo Playground. Variabili d'ambiente: `CLAUDE_GPU_SSH_KEY_B64`, `CLAUDE_GPU_HOSTKEY`, `CLAUDE_GPU_HOST`, `CLAUDE_GPU_PORT`, `CLAUDE_GPU_USER`, `CLAUDE_TS_AUTHKEY`. **Ad ogni sessione nuova:** `bash scripts/tailscale_up.sh` (entra nella tailnet in modalità userspace, proxy SOCKS5 localhost:1055; il demone va rilanciato ogni volta), poi `bash scripts/gpu_ssh_check.sh`. Serve `apt-get update && apt-get install -y openssh-client netcat-openbsd`. Non stampare mai chiavi; non chiedere chiavi in chat. Installa sul PC solo in userspace (venv/pip --user, niente sudo).
2. Con la GPU: scegliere Isaac Lab (GPU ≥16 GB NVIDIA) vs MJX/MuJoCo Playground (più leggero) in base al computer; portare l'env di balancing; rifare baseline ANN con ≥5 seed; poi L2, L3, L4 come da piano. Stima del guadagno: ~30× in passi/s (stima, da misurare).
3. Senza GPU: completare i seed, eseguire robustness/synops su tutti i run, scrivere un resoconto onesto.
4. Idea di contributo principale (piano §2.3): N1 = student spiking ricorrente (ALIF/carry) senza history encoder, distillato da teacher privilegiato + studio sistematico di robustezza SNN vs ANN. Prima del lavoro lungo, rifare il controllo di novità.

## Setup del container (si perde ad ogni nuova sessione; i run non committati sono persi)
```bash
python3 -m venv .venv && . .venv/bin/activate
pip install --index-url https://download.pytorch.org/whl/cpu torch==2.14.1
pip install -r requirements-cpu.txt imageio imageio-ffmpeg pillow
mkdir -p third_party && cd third_party && git clone --filter=blob:none --sparse --depth 1 https://github.com/google-deepmind/mujoco_menagerie.git && cd mujoco_menagerie && git sparse-checkout set unitree_go2 && cd ../..
apt-get update -q && apt-get install -y -q libosmesa6 libegl1   # solo per il rendering video
python -m pytest -q tests          # 10 test, tutti verdi
```
- Comandi: `scripts/run_l1.sh <ann|snn> <T> <seed>` (config D, ~40–60 min/run su 4 core), `python -m spiking_rl.eval.push_curve <run_dir>`, `…robustness`, `…synops`, `PYTHONPATH=. python scripts/render_policy.py --out X.mp4 --run <run_dir> --compare-zero --push 1.0`.
- I checkpoint committati sono in `results/l1/` e `results/e1/`; per usare gli script di valutazione copiali in `runs/` (ignorata da git).
- `runs/`, `third_party/`, `*.pt` sono in `.gitignore` (i checkpoint di `results/` sono stati aggiunti con `git add -f`).

## Trappole note
- **Processi in background:** i processi staccati (`nohup`, `setsid`) NON sopravvivono tra un turno e l'altro. Usa `run_in_background` del tool Bash; il limite massimo è 2 h, poi viene ucciso (un batch lungo va spezzato in run singoli che saltano ciò che è finito: `run_l1.sh` lo fa).
- **Gymnasium 1.x:** con `AutoresetMode.SAME_STEP` le info degli episodi sono in `info["final_info"]["episode"]` come dict di array (maschera `_r`), non una lista; l'obs finale dei time-limit è in `info["final_obs"]`.
- **Ordine dei giunti:** MJCF menagerie = (FL,FR,RL,RR)×(hip,thigh,calf); Isaac Lab = per tipo di giunto. Serve una permutazione per sim2sim/deploy.
- **Posa di default:** Isaac Lab (±0.1, 0.8 ant./1.0 post., −1.5), usata dappertutto. Quota di riferimento della base (piedi a terra, senza flessione PD): 0.3235 m.
- Il PD a Kp 25 cede ~6 cm sotto gravità: lo compensa la policy.
- Velocità: ~900–1500 passi/s in totale con 8 env async su 4 core; il collo di bottiglia è l'env MuJoCo in Python, non la rete. Una GPU da sola non lo accelera: serve un simulatore su GPU.
- `torch.load` richiede `weights_only=False` per i nostri checkpoint (contengono dict con numpy).
- Il valutatore di robustezza con CPU occupata dal training è molto lento (2 episodi × 19 condizioni = 6 min).
