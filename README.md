# Spiking-RL — policy spiking per la locomozione del Unitree Go2

Stato: **fase di pianificazione**. Nessun training implementato.

- `docs/literature_review.md` — rassegna bibliografica.
- `docs/sources_verification.md` — cosa della rassegna è stato verificato/corretto (2026-10-09).
- `docs/implementation_plan.md` — **piano implementativo e di test** (curriculum L0–L9, architetture, setup, protocollo, troubleshooting, sprint 1).
- `scripts/` — smoke test eseguibili su CPU:
  - `smoke_lif_surrogate.py` — core SNN (LIF + surrogate gradient + reset per `dones`);
  - `smoke_go2_mujoco.py` — Go2 di mujoco_menagerie con PD e spinta;
  - `bench_snn_libs.py` — micro-benchmark custom vs snntorch vs spikingjelly.

Setup minimo (CPU):
```bash
python3 -m venv .venv && . .venv/bin/activate
pip install --index-url https://download.pytorch.org/whl/cpu torch==2.14.1
pip install -r requirements-cpu.txt
mkdir -p third_party && cd third_party && git clone --filter=blob:none --sparse --depth 1 https://github.com/google-deepmind/mujoco_menagerie.git && cd mujoco_menagerie && git sparse-checkout set unitree_go2 && cd ../..
python scripts/smoke_lif_surrogate.py && python scripts/smoke_go2_mujoco.py
```
