"""
AI3403 Multi-Agent Systems - Assignment 3, Problem 1
Formation Control: N=20 agents spell L -> O -> K -> E -> S -> H
Graph: Erdos-Renyi (N=20, p=0.3)
"""

import numpy as np
import networkx as nx
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import PillowWriter
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment

np.random.seed(42)

# ─── Graph ─────────────────────────────────────────────────────────────────────
N = 20
p = 0.3

while True:
    G = nx.erdos_renyi_graph(N, p, seed=42)
    if nx.is_connected(G):
        break

A_adj = nx.to_numpy_array(G)
print(f"Graph: {N} nodes, {G.number_of_edges()} edges, connected={nx.is_connected(G)}")

# ─── Letter definitions (all centred at origin after sampling) ─────────────────

def _sample_segs(segs, n):
    segs  = [(np.array(s, float), np.array(e, float)) for s, e in segs]
    lens  = np.array([np.linalg.norm(e - s) for s, e in segs])
    total = lens.sum()
    pts   = []
    for (s, e), L in zip(segs, lens):
        k = max(2, int(round(n * L / total)))
        for t in np.linspace(0, 1, k, endpoint=False):
            pts.append(s + t * (e - s))
    pts = np.array(pts)
    if len(pts) >= n:
        idx = np.round(np.linspace(0, len(pts) - 1, n)).astype(int)
        return pts[idx]
    while len(pts) < n:
        pts = np.vstack([pts, pts[-1:]])
    return pts[:n]


def letter_L(n=20):
    segs = [
        ([0, 4], [0, 0]),   # vertical
        ([0, 0], [2, 0]),   # horizontal bottom
    ]
    return _sample_segs(segs, n)


def letter_O(n=20):
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.column_stack([1 + np.cos(t), 2 + 2 * np.sin(t)])


def letter_K(n=20):
    segs = [
        ([0, 0], [0, 4]),    # left vertical
        ([0, 2], [2, 4]),    # upper diagonal
        ([0, 2], [2, 0]),    # lower diagonal
    ]
    return _sample_segs(segs, n)


def letter_E(n=20):
    segs = [
        ([0, 4], [0, 0]),       # left vertical
        ([0, 4], [2, 4]),       # top bar
        ([0, 2], [1.5, 2]),     # middle bar
        ([0, 0], [2, 0]),       # bottom bar
    ]
    return _sample_segs(segs, n)


def letter_S(n=20):
    # Two half-circles: top arc (right→left) + bottom arc (left→right)
    top_n    = n // 2
    bot_n    = n - top_n
    t_top    = np.linspace(0, np.pi, top_n, endpoint=False)
    t_bot    = np.linspace(np.pi, 2 * np.pi, bot_n, endpoint=False)
    top_pts  = np.column_stack([1 + np.cos(t_top),  3 + np.sin(t_top)])
    bot_pts  = np.column_stack([1 + np.cos(t_bot),  1 + np.sin(t_bot)])
    return np.vstack([top_pts, bot_pts])


def letter_H(n=20):
    segs = [
        ([0, 0], [0, 4]),   # left vertical
        ([0, 2], [2, 2]),   # crossbar
        ([2, 0], [2, 4]),   # right vertical
    ]
    return _sample_segs(segs, n)


LETTERS = ['L', 'O', 'K', 'E', 'S', 'H']
FUNCS   = {'L': letter_L, 'O': letter_O, 'K': letter_K,
           'E': letter_E, 'S': letter_S, 'H': letter_H}

# ─── Formation control simulation ──────────────────────────────────────────────

alpha1 = 0.04   # consensus weight
alpha2 = 0.08   # direct attraction weight
T      = 200    # steps per letter
SKIP   = 4      # save every SKIP-th frame

x0 = np.random.uniform(-3, 3, (N, 2))   # random initial positions

frames_pos   = []
frames_label = []

prev_x = x0.copy()

for letter in LETTERS:
    raw = FUNCS[letter](N)
    raw -= raw.mean(axis=0)                          # centre at origin

    _, col = linear_sum_assignment(cdist(prev_x, raw))
    target = raw[col]

    xi = prev_x.copy()
    for step in range(T):
        u = np.zeros((N, 2))
        for i in range(N):
            for j in range(N):
                if A_adj[i, j] == 1:
                    u[i] -= alpha1 * ((xi[i] - xi[j]) - (target[i] - target[j]))
            u[i] -= alpha2 * (xi[i] - target[i])
        xi = xi + u
        if step % SKIP == 0:
            frames_pos.append(xi.copy())
            frames_label.append(letter)

    prev_x = xi.copy()

print(f"Total frames: {len(frames_pos)}")

# ─── Animation ─────────────────────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(6, 6))
ax.set_xlim(-3.5, 3.5);  ax.set_ylim(-3.5, 3.5)
ax.set_aspect('equal')
ax.set_facecolor('#1a1a2e')
ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
for sp in ax.spines.values():
    sp.set_visible(False)
fig.patch.set_facecolor('#1a1a2e')

edge_artists = []
for i in range(N):
    for j in range(i + 1, N):
        if A_adj[i, j] == 1:
            ln, = ax.plot([], [], color='#4a90d9', alpha=0.2, lw=0.8, zorder=1)
            edge_artists.append((i, j, ln))

scat  = ax.scatter([], [], c='#00d4ff', s=80, zorder=3, edgecolors='white', linewidths=0.5)
ttl   = ax.set_title('', fontsize=22, fontweight='bold', color='white', pad=12)


def init_anim():
    scat.set_offsets(np.empty((0, 2)))
    ttl.set_text('')
    for _, _, ln in edge_artists:
        ln.set_data([], [])
    return [scat, ttl] + [ln for _, _, ln in edge_artists]


def update_anim(k):
    pos = frames_pos[k]
    scat.set_offsets(pos)
    ttl.set_text(f'Letter : {frames_label[k]}')
    for i, j, ln in edge_artists:
        ln.set_data([pos[i, 0], pos[j, 0]], [pos[i, 1], pos[j, 1]])
    return [scat, ttl] + [ln for _, _, ln in edge_artists]


ani = animation.FuncAnimation(
    fig, update_anim, frames=len(frames_pos),
    init_func=init_anim, blit=True, interval=60
)

ani.save('lokesh_formation.gif', writer=PillowWriter(fps=18), dpi=110)
plt.close(fig)
print("Saved: lokesh_formation.gif")
