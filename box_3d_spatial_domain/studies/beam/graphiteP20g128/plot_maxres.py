"""AMEn Maxres per sweep for the graphite P20 g128 beam runs run_*/beam.log, one curve per run."""
import re
from pathlib import Path

import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent

fig, ax = plt.subplots(figsize=(7, 4.5))
for d in sorted(HERE.glob("run_*/")):
    log = (d / "beam.log").read_text()
    maxres = [float(m) for m in re.findall(r"Maxres\s+([0-9.eE+-]+)", log)]
    true_res = re.search(r"true residual ([0-9.eE+-]+)", log)
    label = d.name[4:] + (f" (true res. {float(true_res.group(1)):.1e})" if true_res else "")
    ax.semilogy(range(1, len(maxres) + 1), maxres, "o-", ms=4, label=label)
    print(f"{d.name}: {len(maxres)} sweeps, Maxres {' '.join(f'{m:.2e}' for m in maxres)}"
          + (f", true residual {true_res.group(1)}" if true_res else ""))
ax.axhline(1e-4, color="k", ls=":", lw=0.8, label="1e-4")
ax.set_xlabel("sweep"); ax.set_ylabel("Maxres"); ax.set_ylim(1e-4, 10)
ax.set_title("Graphite P20 g128 beam, AMEn Maxres per sweep")
ax.grid(alpha=0.4, which="both"); ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(HERE / "maxres_runs.png", dpi=150)
