"""AMEn Maxres per sweep for the beam P0 runs c<c>/beam.log, one curve per scattering ratio c."""
import re
from pathlib import Path

import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent

fig, ax = plt.subplots(figsize=(7, 4.5))
for d in sorted(HERE.glob("c*/")):
    log = (d / "beam.log").read_text()
    maxres = [float(m) for m in re.findall(r"Maxres\s+([0-9.eE+-]+)", log)]
    ranks = re.findall(r"Solution rank \[([^\]]*)\]", log)
    true_res = re.search(r"true residual ([0-9.eE+-]+)", log)
    label = f"c = {d.name[1:]}" + (f" (true res. {float(true_res.group(1)):.1e})" if true_res else " (running)")
    ax.semilogy(range(1, len(maxres) + 1), maxres, "o-", ms=4, label=label)
    print(f"{d.name}: {len(maxres)} sweeps, Maxres {' '.join(f'{m:.2e}' for m in maxres)}")
    if ranks:
        print(f"    final ranks [{ranks[-1].strip()}]" + (f", true residual {true_res.group(1)}" if true_res else ""))
ax.axhline(1e-6, color="k", ls=":", lw=0.8, label="eps = 1e-6")
ax.set_xlabel("sweep"); ax.set_ylabel("Maxres")
ax.set_title("Beam P0, AMEn kickrank 2: Maxres per sweep vs scattering ratio")
ax.grid(alpha=0.4, which="both"); ax.legend()
fig.tight_layout()
fig.savefig(HERE / "maxres_vs_c.png", dpi=150)
