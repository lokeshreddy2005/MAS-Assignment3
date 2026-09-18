"""
AI3403 Multi-Agent Systems - Assignment 3, Problem 1
Formation Control: N=20 agents spell L -> O -> K -> E -> S -> H
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

# ─── Parameters ────────────────────────────────────────────────────────────────
N          = 20
p_er       = 0.3
alpha1     = 0.02    # consensus formation weight
alpha2     = 0.02    # direct attraction weight
STEPS_PER  = 80      # steps per letter  (total = 80 × 6 = 480)
SAVE_EVERY = 2       # save every 2nd frame to keep GIF size down

LETTERS = ['L', 'O', 'K', 'E', 'S', 'H']
TOTAL   = STEPS_PER * len(LETTERS)   # 480

# ─── Connected Erdos-Renyi graph ───────────────────────────────────────────────
while True:
    G = nx.erdos_renyi_graph(N, p_er, seed=42)
    if nx.is_connected(G):
        break

A_adj = nx.to_numpy_array(G)
print(f"Graph: {N} nodes, {G.number_of_edges()} edges, connected={nx.is_connected(G)}")

# ─── Letter target positions  (bounding box roughly [0.5, 4] × [0.5, 4.5]) ───

def _sample_segs(segs, n):
    """Sample exactly n points uniformly along a sequence of line segments."""
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
    segs = [([0.5, 4.5], [0.5, 0.5]),   # vertical
            ([0.5, 0.5], [3.5, 0.5])]    # horizontal
    return _sample_segs(segs, n)


def letter_O(n=20):
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.column_stack([2.0 + 1.6 * np.cos(t),
                            2.5 + 2.0 * np.sin(t)])


def letter_K(n=20):
    segs = [([0.5, 0.5], [0.5, 4.5]),   # left vertical
            ([0.5, 2.5], [3.5, 4.5]),    # upper arm
            ([0.5, 2.5], [3.5, 0.5])]    # lower arm
    return _sample_segs(segs, n)


def letter_E(n=20):
    segs = [([0.5, 4.5], [0.5, 0.5]),   # left vertical
            ([0.5, 4.5], [3.5, 4.5]),    # top bar
            ([0.5, 2.5], [2.5, 2.5]),    # middle bar
            ([0.5, 0.5], [3.5, 0.5])]    # bottom bar
    return _sample_segs(segs, n)


def letter_S(n=20):
    n_top = n // 2
    n_bot = n - n_top
    t_top = np.linspace(0,      np.pi, n_top, endpoint=False)
    t_bot = np.linspace(np.pi, 2*np.pi, n_bot, endpoint=False)
    top = np.column_stack([2.0 + 1.5 * np.cos(t_top), 3.5 + 1.0 * np.sin(t_top)])
    bot = np.column_stack([2.0 + 1.5 * np.cos(t_bot), 1.5 + 1.0 * np.sin(t_bot)])
    return np.vstack([top, bot])


def letter_H(n=20):
    segs = [([0.5, 0.5], [0.5, 4.5]),   # left vertical
            ([0.5, 2.5], [3.5, 2.5]),    # crossbar
            ([3.5, 0.5], [3.5, 4.5])]    # right vertical
    return _sample_segs(segs, n)


FUNCS = {'L': letter_L, 'O': letter_O, 'K': letter_K,
         'E': letter_E, 'S': letter_S, 'H': letter_H}

# ─── Simulation ────────────────────────────────────────────────────────────────
xi = np.random.uniform(0.5, 4.0, (N, 2))   # random initial positions

all_pos  = []   # saved positions
all_step = []   # global step index
all_ltr  = []   # current letter

global_step = 0

for letter in LETTERS:
    target_raw = FUNCS[letter](N)

    # Hungarian assignment: minimise total travel distance
    _, col = linear_sum_assignment(cdist(xi, target_raw))
    target = target_raw[col]

    for step in range(STEPS_PER):
        global_step += 1
        u = np.zeros((N, 2))
        for i in range(N):
            for j in range(N):
                if A_adj[i, j] == 1:
                    # consensus formation term
                    u[i] -= alpha1 * ((xi[i] - xi[j]) - (target[i] - target[j]))
            # direct attraction term
            u[i] -= alpha2 * (xi[i] - target[i])
        xi = xi + u

        if global_step % SAVE_EVERY == 0:
            all_pos.append(xi.copy())
            all_step.append(global_step)
            all_ltr.append(letter)

print(f"Total animation frames: {len(all_pos)}")

# ─── Build animation (style matching reference) ────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 6))
fig.patch.set_facecolor('white')
ax.set_facecolor('white')

ax.set_xlim(-0.2, 5.0)
ax.set_ylim(-0.2, 5.5)
ax.set_xlabel('X coordinate', fontsize=11)
ax.set_ylabel('Y coordinate', fontsize=11)
ax.grid(True, linestyle='--', linewidth=0.6, alpha=0.6, color='gray')
ax.tick_params(labelsize=9)

# Pre-create edge line artists
edge_artists = []
for i in range(N):
    for j in range(i + 1, N):
        if A_adj[i, j] == 1:
            ln, = ax.plot([], [], color='gray', alpha=0.3, lw=0.8, zorder=1)
            edge_artists.append((i, j, ln))

scat = ax.scatter([], [], c='blue', s=120, zorder=3, label=f'Agents (N = {N})')
ttl  = ax.set_title('', fontsize=13, fontweight='bold', color='black')
ax.legend(loc='upper right', fontsize=10, framealpha=0.9)


def init_anim():
    scat.set_offsets(np.empty((0, 2)))
    ttl.set_text('')
    for _, _, ln in edge_artists:
        ln.set_data([], [])
    return [scat, ttl] + [ln for _, _, ln in edge_artists]


def update_anim(k):
    pos = all_pos[k]
    scat.set_offsets(pos)
    ttl.set_text(
        f'Target Formation:Letter {all_ltr[k]} (Step {all_step[k]}/{TOTAL})')
    for i, j, ln in edge_artists:
        ln.set_data([pos[i, 0], pos[j, 0]], [pos[i, 1], pos[j, 1]])
    return [scat, ttl] + [ln for _, _, ln in edge_artists]


ani = animation.FuncAnimation(
    fig, update_anim, frames=len(all_pos),
    init_func=init_anim, blit=True, interval=80
)

ani.save('lokesh_formation.gif', writer=PillowWriter(fps=15), dpi=100)
plt.close(fig)
print("Saved: lokesh_formation.gif")
