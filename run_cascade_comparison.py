"""
Omega-Sigma Variational Intelligence: Cascade Comparison
Author: Andrew Kim, Emerald Research Group
Target: JAX/PyTorch Autoregressive Rollout Constraint Proof-of-Concept

This script simulates the dyadic shell model of fluid turbulence.
It compares an unconstrained cascade (which hallucinates/blows up) 
against an Omega-Sigma constrained cascade (which enforces variance-induced subcriticality).
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

# =========================================================
# HYPERPARAMETERS & INITIALIZATION
# =========================================================
N_SHELLS = 16          # Number of dyadic frequency shells
NU = 1e-3              # Kinematic viscosity
C_TRANS = 15.0         # Nonlinear transport coefficient
DT = 1e-4              # Integration time step
STEPS = 50000          # Total simulation steps
VAR_LIMIT = 5e4        # The Omega-Sigma Variance Barrier

# Create data directory for CSV outputs
os.makedirs("data", exist_ok=True)

# =========================================================
# CORE MATHEMATICAL ENGINE
# =========================================================
def compute_variance(a):
    """Computes the spectral variance of the dyadic shell distribution."""
    A_total = np.sum(a) + 1e-12
    p = a / A_total
    weights = 4.0 ** np.arange(N_SHELLS)
    mu = np.sum(p * weights)
    variance = np.sum(p * (weights - mu)**2)
    return A_total, variance

def simulate_cascade(apply_omega_sigma_constraint=False):
    """
    Integrates the shell model ODE:
    da_q/dt = -nu * 4^q * a_q + c * (a_{q-1}^1.5 - a_q^1.5)
    """
    # Initial energy injection at q=2
    a = np.array([np.exp(-np.abs(q - 2)) for q in range(N_SHELLS)])
    
    history = []
    
    for step in range(STEPS):
        A_total, variance = compute_variance(a)
        
        # Log data every 100 steps
        if step % 100 == 0:
            history.append({
                'time': step * DT,
                'energy_total': A_total,
                'spectral_variance': variance,
                'a_array': a.copy()
            })
            
        # Compute nonlinear forward transport (T_{q-1 -> q} - T_{q -> q+1})
        transport = np.zeros(N_SHELLS)
        for q in range(N_SHELLS):
            inflow = (a[q-1]**1.5) if q > 0 else 0.0
            outflow = a[q]**1.5
            transport[q] = C_TRANS * (inflow - outflow)
            
        # The Omega-Sigma Constraint (Subcritical Projection)
        # If variance breaches the limit, we mathematically force the transport
        # into the viscous subcritical regime (simulating the loss penalty update)
        effective_nu = NU
        if apply_omega_sigma_constraint and variance > VAR_LIMIT:
            effective_nu = NU * 50.0  # Variance Penalty activates

        # Euler Integration Step
        dissipation = -effective_nu * (4.0 ** np.arange(N_SHELLS)) * a
        da_dt = dissipation + transport
        
        a = a + DT * da_dt
        a = np.maximum(a, 0.0) # Energy cannot be negative

    return pd.DataFrame(history)

# =========================================================
# EXECUTION & PLOTTING
# =========================================================
if __name__ == "__main__":
    print("Initializing Omega-Sigma Dyadic Shell Simulation...")
    
    # Run both simulations
    print("Running Unconstrained Baseline...")
    df_baseline = simulate_cascade(apply_omega_sigma_constraint=False)
    
    print("Running Omega-Sigma Constrained Model...")
    df_constrained = simulate_cascade(apply_omega_sigma_constraint=True)
    
    # Export the CSV (for the academic appendix requirement)
    df_constrained[['time', 'energy_total', 'spectral_variance']].to_csv("data/run_summary.csv", index=False)
    print("Data exported to data/run_summary.csv")

    # --- PLOT 1: Spectral Variance History ---
    plt.style.use('dark_background')
    fig, ax1 = plt.subplots(figsize=(10, 5))
    
    ax1.plot(df_baseline['time'], df_baseline['spectral_variance'], color='#FF3366', linewidth=2, label='Unconstrained (Hallucination)')
    ax1.plot(df_constrained['time'], df_constrained['spectral_variance'], color='#00FFCC', linewidth=2, label='$\Omega$--$\Sigma$ Constrained (Stable)')
    
    ax1.set_title("Spectral Variance History (Suppression of Extremes)", fontsize=14, fontweight='bold')
    ax1.set_xlabel("Simulation Time", fontsize=12)
    ax1.set_ylabel("Variance $Var(2^{2q})$", fontsize=12)
    ax1.set_yscale('log')
    ax1.grid(color='#333333', linestyle='--', alpha=0.7)
    ax1.legend(loc='upper left', frameon=False)
    plt.tight_layout()
    plt.savefig("data/variance_history.png", dpi=300)
    
    # --- PLOT 2: Exponential Viscous Tail (Final State) ---
    fig, ax2 = plt.subplots(figsize=(10, 5))
    
    final_a_base = df_baseline['a_array'].iloc[-1]
    final_a_const = df_constrained['a_array'].iloc[-1]
    shells = np.arange(N_SHELLS)
    
    ax2.plot(shells, final_a_base, marker='o', color='#FF3366', linestyle='--', label='Unconstrained Drift')
    ax2.plot(shells, final_a_const, marker='s', color='#00FFCC', linewidth=2, label='$\Omega$--$\Sigma$ Exponential Tail')
    
    ax2.set_title("Final Spectral Distribution (Subcritical Viscous Tail)", fontsize=14, fontweight='bold')
    ax2.set_xlabel("Shell Index $q$", fontsize=12)
    ax2.set_ylabel("Shell Energy $a_q$", fontsize=12)
    ax2.set_yscale('log')
    ax2.grid(color='#333333', linestyle='--', alpha=0.7)
    ax2.legend(loc='upper right', frameon=False)
    plt.tight_layout()
    plt.savefig("data/exponential_tail.png", dpi=300)
    
    print("Simulation complete. Plots saved to /data directory.")