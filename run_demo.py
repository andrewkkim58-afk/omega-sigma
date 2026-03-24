"""
Ω-Σ Engine: Variance-Dominated Cascade Arrest (Final Demo)
Author: Andrew Kim, Emerald Research Group
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)

# =========================================================
# 1. HYPERPARAMETERS
# =========================================================
N_SHELLS = 16
NU = 2e-5
C_TRANS = 40.0
DT = 5e-5
STEPS = 30000
VAR_LIMIT = 2e8

os.makedirs("data", exist_ok=True)

weights = 4.0 ** np.arange(N_SHELLS, dtype=np.float64)
shells = np.arange(N_SHELLS)

# =========================================================
# 2. CORE FUNCTIONS
# =========================================================
def compute_variance(a):
    A_total = np.sum(a) + 1e-12
    p = a / A_total
    mu = np.sum(p * weights)
    return np.sum(p * (weights - mu)**2)

def tail_mass(a, q0=10):
    return np.sum(a[q0:])

# =========================================================
# 3. SIMULATION
# =========================================================
def simulate_cascade(apply_omega_sigma_constraint=False):
    a = np.array([np.exp(-0.8 * abs(q - 2)) for q in range(N_SHELLS)], dtype=np.float64)

    history_a = []
    history_v = []
    history_tail = []

    for step in range(STEPS):
        variance = compute_variance(a)

        if step % 25 == 0:
            history_a.append(a.copy())
            history_v.append(variance)
            history_tail.append(tail_mass(a))

        # -------------------------------
        # Forward-biased transport
        # -------------------------------
        transport = np.zeros(N_SHELLS)

        for q in range(N_SHELLS):
            inflow = C_TRANS * (a[q-1]**1.2) if q > 0 else 0.0
            outflow = C_TRANS * (a[q]**1.2)
            transport[q] = inflow - outflow

        # -------------------------------
        # Low-shell forcing (CRITICAL FIX)
        # -------------------------------
        forcing = np.zeros(N_SHELLS)
        forcing[1] = 8.0
        forcing[2] = 4.0

        # -------------------------------
        # Ω-Σ Controller
        # -------------------------------
        effective_nu = NU
        if apply_omega_sigma_constraint:
           ratio = variance / VAR_LIMIT
           effective_nu = NU * (1.0 + 40.0 / (1.0 + np.exp(-8.0 * (ratio - 1.0)))) 
        
        # -------------------------------
        # Dissipation
        # -------------------------------
        dissipation = -effective_nu * weights * a

        # -------------------------------
        # Update
        # -------------------------------
        a = a + DT * (forcing + transport + dissipation)

        # positivity only (NO hard clipping)
        a = np.maximum(a, 1e-16)

    return (
        np.array(history_a),
        np.array(history_v),
        np.array(history_tail)
    )

# =========================================================
# 4. EXECUTION
# =========================================================
if __name__ == "__main__":
    print("Initializing Ω-Σ Dyadic Shell Simulation...")

    base_a, base_v, base_tail = simulate_cascade(False)
    cons_a, cons_v, cons_tail = simulate_cascade(True)

    time_axis = np.arange(len(base_v)) * DT * 25

    # =====================================================
    # 🔥 CORRECT METRICS (THIS IS THE REAL FIX)
    # =====================================================

    peak_base = np.max(base_v)
    peak_cons = np.max(cons_v)

    peak_reduction = 100 * (peak_base - peak_cons) / peak_base if peak_base > 0 else 0

    tail_base = np.max(base_tail)
    tail_cons = np.max(cons_tail)

    tail_reduction = 100 * (tail_base - tail_cons) / tail_base if tail_base > 0 else 0

    int_tail_base = np.trapezoid(base_tail, time_axis)
    int_tail_cons = np.trapezoid(cons_tail, time_axis)

    int_tail_reduction = 100 * (int_tail_base - int_tail_cons) / int_tail_base if int_tail_base > 0 else 0

    final_var_reduction = 100 * (base_v[-1] - cons_v[-1]) / base_v[-1] if base_v[-1] > 0 else 0

    print("\n" + "="*55)
    print("Ω-Σ ENGINE: SIMULATION COMPLETE")
    print("="*55)

    print(f"Peak variance (baseline):      {peak_base:,.2f}")
    print(f"Peak variance (Ω-Σ):           {peak_cons:,.2f}")
    print(f"Peak suppression:              {peak_reduction:.2f}%\n")

    print(f"Peak tail mass (baseline):     {tail_base:.6f}")
    print(f"Peak tail mass (Ω-Σ):          {tail_cons:.6f}")
    print(f"Tail suppression:              {tail_reduction:.2f}%\n")

    print(f"Integrated tail suppression:   {int_tail_reduction:.2f}%")
    print(f"Final variance reduction:      {final_var_reduction:.2f}%\n")

    # =====================================================
    # 5. PLOTS
    # =====================================================
    plt.style.use("dark_background")

    # Variance
    plt.figure(figsize=(8, 4))
    plt.plot(time_axis, base_v, lw=2, label="Unconstrained")
    plt.plot(time_axis, cons_v, lw=2, label="Ω-Σ Controlled")
    plt.yscale("log")
    plt.title("Spectral Variance History")
    plt.xlabel("Time")
    plt.ylabel("Variance")
    plt.legend()
    plt.tight_layout()
    plt.savefig("data/variance_history.png", dpi=300)
    plt.close()

    # Tail mass
    plt.figure(figsize=(8, 4))
    plt.plot(time_axis, base_tail, lw=2, label="Unconstrained")
    plt.plot(time_axis, cons_tail, lw=2, label="Ω-Σ Controlled")
    plt.title("High-Shell Tail Mass")
    plt.xlabel("Time")
    plt.ylabel("Tail Mass (q ≥ 10)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("data/tail_mass.png", dpi=300)
    plt.close()

    # Final spectrum
    plt.figure(figsize=(8, 4))
    plt.plot(shells, base_a[-1], marker='o', ls='--', label="Baseline")
    plt.plot(shells, cons_a[-1], marker='s', lw=2, label="Ω-Σ")
    plt.yscale("log")
    plt.title("Final Spectral Distribution")
    plt.xlabel("Shell Index q")
    plt.ylabel("Energy a_q")
    plt.legend()
    plt.tight_layout()
    plt.savefig("data/final_spectrum.png", dpi=300)
    plt.close()
    
    # =====================================================
    # 6. ANIMATION
    # =====================================================
    print("Rendering GIF animation (this takes ~10 seconds)...")

    # Freeze the final frame so viewers can inspect the terminal state
    hold_frames = 75
    base_a_anim = np.concatenate([base_a, np.repeat(base_a[-1][None, :], hold_frames, axis=0)], axis=0)
    cons_a_anim = np.concatenate([cons_a, np.repeat(cons_a[-1][None, :], hold_frames, axis=0)], axis=0)

    fig4, (ax_base, ax_cons) = plt.subplots(1, 2, figsize=(12, 5))
    fig4.patch.set_facecolor("black")

    def init_ax(ax, title, color):
        ax.set_title(title, fontweight='bold')
        ax.set_facecolor("black")
        ax.set_ylim(1e-16, 10.0)
        ax.set_xlim(-0.6, N_SHELLS - 0.4)
        ax.set_yscale('log')
        ax.set_xlabel("Shell Index $q$")
        ax.set_ylabel("Energy $a_q$")
        ax.grid(True, which="both", axis="y", alpha=0.15, linestyle="--")
        bars = ax.bar(shells, np.full(N_SHELLS, 1e-16), color=color, alpha=0.85, width=0.8)
        return bars

    bars_base = init_ax(ax_base, "Baseline (Hallucination)", '#81D4FA')  # Cyan
    bars_cons = init_ax(ax_cons, "Ω-Σ Engine (Stable)", '#FFF59D')       # Yellow

    time_text = fig4.text(
        0.5, 0.02, "",
        ha="center", va="bottom",
        color="white", fontsize=11
    )

    def update(frame):
        for bar, h in zip(bars_base, base_a_anim[frame]):
            bar.set_height(max(float(h), 1e-16))

        for bar, h in zip(bars_cons, cons_a_anim[frame]):
            bar.set_height(max(float(h), 1e-16))

        if frame < len(base_a):
            t_val = frame * DT * 25.0
            time_text.set_text(f"t = {t_val:.4f}")
        else:
            time_text.set_text("Final state (held)")

        return list(bars_base) + list(bars_cons) + [time_text]

    ani = animation.FuncAnimation(
        fig4,
        update,
        frames=len(base_a_anim),
        interval=25,
        blit=True,
        repeat=False
    )

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    ani.save("data/cascade_demo.gif", writer="pillow", fps=24)
    plt.close(fig4)
    
    print("GIF rendered successfully! Check the /data folder.")