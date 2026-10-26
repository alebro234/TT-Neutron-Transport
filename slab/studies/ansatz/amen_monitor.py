"""Monitoring helpers for the ansatz studies: log tee, capture of the C++ AMEn output (printed on file descriptor 1,
invisible to sys.stdout), per-sweep parsing and reports."""
import ctypes
import os
import re
import sys
from contextlib import contextmanager

import numpy as np

_libc = ctypes.CDLL(None)


class Tee:
    """Write to the terminal and to the run log at the same time."""
    def __init__(self, *streams):
        self.streams = streams

    def write(self, s):
        for st in self.streams:
            st.write(s)
            st.flush()

    def flush(self):
        for st in self.streams:
            st.flush()


def start_log(path):
    """Tee stdout and stderr to `path`."""
    log = open(path, 'w')
    sys.stdout = Tee(sys.__stdout__, log)
    sys.stderr = Tee(sys.__stderr__, log)
    return log


@contextmanager
def capture_fd1(path):
    """Redirect file descriptor 1 (where the C++ AMEn prints) to `path`; Python prints inside go there too.
    The captured text is returned in the yielded dict under 'text'."""
    out = {}
    sys.stdout.flush()
    saved_py, saved_fd = sys.stdout, os.dup(1)
    with open(path, 'w+b') as f:
        os.dup2(f.fileno(), 1)
        sys.stdout = sys.__stdout__
        try:
            yield out
        finally:
            sys.__stdout__.flush()
            _libc.fflush(None)
            os.dup2(saved_fd, 1)
            os.close(saved_fd)
            sys.stdout = saved_py
            f.seek(0)
            out['text'] = f.read().decode(errors='replace')


def parse_amen(text):
    """Per-sweep history from the C++ AMEn verbose output: solution ranks, max residual, time [s]; plus the number
    of local GMRES / direct solves and the GMRES iteration counts."""
    sweeps = []
    for block in re.split(r'Starting sweep ', text)[1:]:
        m_r = re.search(r'Solution rank \[ ([\d ]+)\]', block)
        m_res = re.search(r'Maxres ([0-9.eE+-]+|nan|inf)', block)
        m_t = re.search(r'\nTime ([0-9.eE+-]+) ms', block)
        if m_r and m_res:
            sweeps.append(dict(ranks=[int(x) for x in m_r.group(1).split()], maxres=float(m_res.group(1)),
                               time=float(m_t.group(1)) / 1e3 if m_t else np.nan))
    n_local = len(re.findall(r'Choosing iterative solver', text)), len(re.findall(r'Choosing direct solver', text))
    gmres_its = [int(x) for x in re.findall(r'after (\d+) iterations', text)]
    return sweeps, n_local, gmres_its


def report_amen(tag, sweeps, n_local, gmres_its, wall):
    print(f"\n[{tag}] AMEn history ({len(sweeps)} sweeps, wall {wall:.2f} s; local solves: "
          f"{n_local[0]} GMRES ({sum(gmres_its)} its total), {n_local[1]} direct)")
    print("  sweep | max res    | max rank | ranks                          | time [s]")
    for k, s in enumerate(sweeps):
        print(f"  {k + 1:5d} | {s['maxres']:.3e}  | {max(s['ranks']):8d} | {str(s['ranks']):30s} | {s['time']:.2f}")


def tt_entries(T):
    return sum(c.numel() for c in T.cores)


def rel_res(A, x, F):
    return float((A @ x - F).norm() / F.norm())
