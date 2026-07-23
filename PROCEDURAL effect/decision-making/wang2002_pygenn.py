"""
Probabilistic decision making by slow reverberation in cortical circuits.
X-J Wang, Neuron 2002.

http://dx.doi.org/10.1016/S0896-6273(02)01092-9

Converted from Brian2 to PyGeNN 5.2.0.

Architecture:
  - Custom neuron models with built-in synaptic traces (sAMPA, sNMDA, sGABA etc.)
  - spike-triggered trace increments in reset_code
  - Poisson background + stimulus input via sim_code random draws
  - Host-side batched computation of population weighted sums (W matrix)
    to amortize GPU-CPU transfer overhead (~every 0.4ms instead of every timestep)
"""

import numpy as np
import pygenn
from pygenn import GeNNModel, VarLocation, init_var
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
from time import perf_counter

# ============================================================================
# Custom Neuron Models
# ============================================================================

# Unified neuron model (used for both E and I).
# Differentiated by: reset_code behaviour and rate_stim variable.

wang2002_neuron = pygenn.create_neuron_model(
    "Wang2002",

    # ---- parameters (immutable after build) ----
    params=[
        "C", "TauM", "Vrest", "Vreset", "Vthresh", "TauRefrac",
        "TauAMPA", "TauX", "TauNMDA", "Alpha", "TauGABA",
        "V_E", "V_I",
        "a_nmda", "b_nmda",
        "gAMPA_ext", "gAMPA", "gNMDA", "gGABA",
        "rate_ext",               # background Poisson rate (kHz)
    ],

    # ---- variables (mutable, read-write during sim) ----
    vars=[
        ("V",          "scalar", pygenn.VarAccess.READ_WRITE),
        ("RefracTime", "scalar", pygenn.VarAccess.READ_WRITE),
        # pre-synaptic traces (decay + spike increment)
        ("sAMPA_ext", "scalar", pygenn.VarAccess.READ_WRITE),
        ("sAMPA",     "scalar", pygenn.VarAccess.READ_WRITE),
        ("x_nmda",    "scalar", pygenn.VarAccess.READ_WRITE),
        ("sNMDA",     "scalar", pygenn.VarAccess.READ_WRITE),
        ("sGABA",     "scalar", pygenn.VarAccess.READ_WRITE),
        # per-neuron stimulus rate — updated at stim onset/offset from host
        ("rate_stim", "scalar", pygenn.VarAccess.READ_ONLY),
        # post-synaptic weighted sums — pushed from host every ~0.4 ms
        ("S_AMPA",    "scalar", pygenn.VarAccess.READ_ONLY),
        ("S_NMDA",    "scalar", pygenn.VarAccess.READ_ONLY),
        ("S_GABA",    "scalar", pygenn.VarAccess.READ_ONLY),
        # neuron type flag: 0 = excitatory, 1 = inhibitory (set at init)
        ("is_inhib",  "scalar", pygenn.VarAccess.READ_ONLY),
    ],

    sim_code="""
    // --- Poisson background ---
    if (gennrand_uniform() < rate_ext * dt) {
        sAMPA_ext += 1.0f;
    }
    // --- Stimulus Poisson (per-neuron, selective populations only) ---
    if (rate_stim > 0.0f && gennrand_uniform() < rate_stim * dt) {
        sAMPA_ext += 1.0f;
    }

    // --- LIF dynamics ---
    if (RefracTime <= 0.0f) {
        // leak  [nA] : gL = C / TauM
        const scalar I_leak = (C / TauM) * (V - Vrest);

        // synaptic currents [nA] — conductance-based
        const scalar I_AMPA_ext = gAMPA_ext * sAMPA_ext * (V - V_E);
        const scalar I_AMPA     = gAMPA * S_AMPA * (V - V_E);
        // NMDA with voltage-dependent Mg2+ block
        const scalar I_NMDA     = gNMDA * S_NMDA * (V - V_E)
                                  / (1.0f + exp(-a_nmda * V) / b_nmda);
        const scalar I_GABA     = gGABA * S_GABA * (V - V_I);

        const scalar Isyn = I_AMPA_ext + I_AMPA + I_NMDA + I_GABA;

        V += (-I_leak - Isyn) / C * dt;
    }
    else {
        RefracTime -= dt;
    }

    // --- synaptic trace decay ---
    sAMPA_ext -= sAMPA_ext / TauAMPA * dt;
    sAMPA     -= sAMPA     / TauAMPA * dt;
    x_nmda    -= x_nmda    / TauX    * dt;
    sNMDA     += (-sNMDA / TauNMDA + Alpha * x_nmda * (1.0f - sNMDA)) * dt;
    sGABA     -= sGABA     / TauGABA * dt;
    """,

    threshold_condition_code="(RefracTime <= 0.0f) && (V >= Vthresh)",

    reset_code="""
    V = Vreset;
    RefracTime = TauRefrac;
    // excitatory: AMPA + NMDA traces; inhibitory: GABA trace
    if (is_inhib < 0.5f) {
        sAMPA  += 1.0f;
        x_nmda += 1.0f;
    } else {
        sGABA += 1.0f;
    }
    """,
)


# ============================================================================
# Parameters — identical to Brian2 version
# ============================================================================

modelparams = {
    # Common LIF
    "V_L":    -70.0,    # mV
    "Vth":    -50.0,    # mV
    "Vreset": -55.0,    # mV

    # E
    "gE":         0.025,    # muS  (= 25 nS)
    "tau_m_E":    20.0,     # ms
    "tau_ref_E":  2.0,      # ms

    # I
    "gI":         0.020,    # muS  (= 20 nS)
    "tau_m_I":    10.0,     # ms
    "tau_ref_I":  1.0,      # ms

    # reversal potentials (mV)
    "V_E":  0.0,
    "V_I": -70.0,

    # NMDA nonlinearity
    "a": 0.062,    # mV^-1
    "b": 3.57,

    # synaptic time constants (ms)
    "tauAMPA":  2.0,
    "tau_x":    2.0,
    "tauNMDA": 100.0,
    "alpha":     0.5,    # ms^-1  (= 0.5 kHz)
    "tauGABA":   5.0,

    # external conductances (muS, unscaled)
    "gAMPA_ext_E": 0.0021,    # = 2.1 nS
    "gAMPA_ext_I": 0.00162,   # = 1.62 nS

    # recurrent conductances (muS, unscaled — will be divided by N)
    "gAMPA_E_raw": 0.080,     # = 80 nS
    "gNMDA_E_raw": 0.264,     # = 264 nS
    "gGABA_E_raw": 0.520,     # = 520 nS
    "gAMPA_I_raw": 0.064,     # = 64 nS
    "gNMDA_I_raw": 0.208,     # = 208 nS
    "gGABA_I_raw": 0.400,     # = 400 nS

    # background noise (kHz)
    "nu_ext": 2.4,

    # neuron counts
    "N_E": 1600,
    "N_I":  400,

    # fraction of selective neurons
    "fsel": 0.15,

    # Hebb-strengthened weight
    "wp": 1.7,
}


# ============================================================================
# Stimulus
# ============================================================================

class Stimulus:
    """Time-varying Poisson rate for selective populations."""
    def __init__(self, Ton, Toff, mu0, coh):
        self.Ton  = Ton     # ms
        self.Toff = Toff    # ms
        self.mu0  = mu0     # kHz
        self.set_coh(coh)

    def set_coh(self, coh):
        self.pos_rate = self.mu0 * (1.0 + coh / 100.0)
        self.neg_rate = self.mu0 * (1.0 - coh / 100.0)

    def get_rate(self, t, pop):
        """Return rate (kHz) for selective population 1 or 2 at time t (ms)."""
        if self.Ton <= t < self.Toff:
            return self.pos_rate if pop == 1 else self.neg_rate
        return 0.0


# ============================================================================
# Weighted-sum computation (host side, batched)
# ============================================================================

def compute_weighted_sums(sAMPA_E, sNMDA_E, sGABA_I, N0, N1, N2, W):
    """
    Population-level weighted sums (W · Σ_traces).

    Returns (S_AMPA_E0, S_AMPA_E1, S_AMPA_E2,
             S_NMDA_E0, S_NMDA_E1, S_NMDA_E2,
             S_AMPA_I,  S_NMDA_I,  S_GABA).
    """
    sum_ampa = np.array([
        sAMPA_E[:N0].sum(),
        sAMPA_E[N0:N0+N1].sum(),
        sAMPA_E[N0+N1:N0+N1+N2].sum(),
    ])
    sum_nmda = np.array([
        sNMDA_E[:N0].sum(),
        sNMDA_E[N0:N0+N1].sum(),
        sNMDA_E[N0+N1:N0+N1+N2].sum(),
    ])
    S_ampa = W.dot(sum_ampa)
    S_nmda = W.dot(sum_nmda)
    S_gaba = sGABA_I.sum()

    return (S_ampa[0], S_ampa[1], S_ampa[2],
            S_nmda[0], S_nmda[1], S_nmda[2],
            S_ampa[0], S_nmda[0],   # I receives same as non-selective E0
            S_gaba)


# ============================================================================
# Simulation class
# ============================================================================

class Wang2002Sim:
    """PyGeNN simulation of Wang 2002 decision-making model."""

    def __init__(self, params, stimparams, dt_ms=0.02):
        self.params = params.copy()
        self.dt     = dt_ms

        N_E  = params["N_E"]
        N_I  = params["N_I"]
        fsel = params["fsel"]
        wp   = params["wp"]

        # subpopulation sizes
        self.N1 = int(fsel * N_E)
        self.N2 = self.N1
        self.N0 = N_E - self.N1 - self.N2

        # weight matrix
        wm = (1.0 - wp * fsel) / (1.0 - fsel)
        self.W = np.array([
            [1.0, 1.0, 1.0],
            [ wm,  wp,  wm],
            [ wm,  wm,  wp],
        ], dtype=np.float64)

        # ---- scale conductances by neuron counts ----
        p = self.params
        p["gAMPA_E"] = p["gAMPA_E_raw"] / N_E
        p["gNMDA_E"] = p["gNMDA_E_raw"] / N_E
        p["gGABA_E"] = p["gGABA_E_raw"] / N_I
        p["gAMPA_I"] = p["gAMPA_I_raw"] / N_E
        p["gNMDA_I"] = p["gNMDA_I_raw"] / N_E
        p["gGABA_I"] = p["gGABA_I_raw"] / N_I

        # capacitance
        C_E = p["gE"] * p["tau_m_E"]   # nF = muS·ms
        C_I = p["gI"] * p["tau_m_I"]

        # ---- neuron parameter dictionaries ----
        self.e_params = {
            "C": C_E, "TauM": p["tau_m_E"],
            "Vrest": p["V_L"], "Vreset": p["Vreset"],
            "Vthresh": p["Vth"], "TauRefrac": p["tau_ref_E"],
            "TauAMPA": p["tauAMPA"], "TauX": p["tau_x"],
            "TauNMDA": p["tauNMDA"], "Alpha": p["alpha"],
            "TauGABA": p["tauGABA"],
            "V_E": p["V_E"], "V_I": p["V_I"],
            "a_nmda": p["a"], "b_nmda": p["b"],
            "gAMPA_ext": p["gAMPA_ext_E"],
            "gAMPA": p["gAMPA_E"],
            "gNMDA": p["gNMDA_E"],
            "gGABA": p["gGABA_E"],
            "rate_ext": p["nu_ext"],
        }
        self.i_params = {
            "C": C_I, "TauM": p["tau_m_I"],
            "Vrest": p["V_L"], "Vreset": p["Vreset"],
            "Vthresh": p["Vth"], "TauRefrac": p["tau_ref_I"],
            "TauAMPA": p["tauAMPA"], "TauX": p["tau_x"],
            "TauNMDA": p["tauNMDA"], "Alpha": p["alpha"],
            "TauGABA": p["tauGABA"],
            "V_E": p["V_E"], "V_I": p["V_I"],
            "a_nmda": p["a"], "b_nmda": p["b"],
            "gAMPA_ext": p["gAMPA_ext_I"],
            "gAMPA": p["gAMPA_I"],
            "gNMDA": p["gNMDA_I"],
            "gGABA": p["gGABA_I"],
            "rate_ext": p["nu_ext"],
        }

        # ---- variable initialisation ----
        self.var_init = {
            "V": init_var("Uniform", {"min": p["Vreset"], "max": p["Vth"]}),
            "RefracTime": 0.0,
            "sAMPA_ext": 0.0, "sAMPA": 0.0,
            "x_nmda": 0.0, "sNMDA": 0.0, "sGABA": 0.0,
            "rate_stim": 0.0,
            "S_AMPA": 0.0, "S_NMDA": 0.0, "S_GABA": 0.0,
            "is_inhib": 0.0,
        }

        # ---- stimulus ----
        self.stimulus = Stimulus(
            stimparams["Ton"], stimparams["Toff"],
            stimparams["mu0"], stimparams["coh"],
        )

        # ---- host-side buffers (allocated once) ----
        self._buf_sAMPA_E  = np.empty(N_E, dtype=np.float32)
        self._buf_sNMDA_E  = np.empty(N_E, dtype=np.float32)
        self._buf_sGABA_I  = np.empty(N_I, dtype=np.float32)
        self._buf_S_AMPA   = np.empty(N_E, dtype=np.float32)
        self._buf_S_NMDA   = np.empty(N_E, dtype=np.float32)
        self._buf_S_GABA   = np.empty(N_E, dtype=np.float32)
        self._buf_stim     = np.empty(N_E, dtype=np.float32)

    # ------------------------------------------------------------------
    def build(self, model_name="wang2002_pygenn", rng_seed=1):
        np.random.seed(rng_seed)

        model = GeNNModel("float", model_name)
        model.dt = self.dt
        model.fuse_postsynaptic_models = False
        model.default_var_location = VarLocation.HOST_DEVICE
        model.default_sparse_connectivity_location = VarLocation.HOST_DEVICE

        # ---- populations ----
        pop_E = model.add_neuron_population(
            "E", self.params["N_E"],
            wang2002_neuron, self.e_params, self.var_init,
        )
        pop_I = model.add_neuron_population(
            "I", self.params["N_I"],
            wang2002_neuron, self.i_params, self.var_init,
        )

        # spike recording
        pop_E.spike_recording_enabled = True
        pop_I.spike_recording_enabled = True

        self.model  = model
        self.pop_E  = pop_E
        self.pop_I  = pop_I

        print(f"Model '{model_name}': "
              f"E={self.params['N_E']} (N0={self.N0}, N1={self.N1}, N2={self.N2}), "
              f"I={self.params['N_I']}")
        print(f"W = \n{self.W}")
        print(f"wp={self.params['wp']}, wm={self.W[1,0]:.4f}")

    # ------------------------------------------------------------------
    def _push_stim_rates(self, t):
        """Push per-neuron stimulus rates to GPU (called at stim boundaries)."""
        rate1 = self.stimulus.get_rate(t, 1)
        rate2 = self.stimulus.get_rate(t, 2)

        buf = self._buf_stim
        buf[:self.N0] = 0.0
        buf[self.N0:self.N0+self.N1] = rate1
        buf[self.N0+self.N1:self.N0+self.N1+self.N2] = rate2
        self.pop_E.vars["rate_stim"].view[:] = buf
        self.pop_E.vars["rate_stim"].push_to_device()
        # I never receives stimulus
        # (rate_stim for I stays at init value 0)

    # ------------------------------------------------------------------
    def _push_weighted_sums(self):
        """Pull synaptic traces, compute weighted sums, push back."""
        N_E, N_I = self.params["N_E"], self.params["N_I"]
        N0, N1, N2 = self.N0, self.N1, self.N2

        # pull
        self.pop_E.vars["sAMPA"].pull_from_device()
        self._buf_sAMPA_E[:] = self.pop_E.vars["sAMPA"].view

        self.pop_E.vars["sNMDA"].pull_from_device()
        self._buf_sNMDA_E[:] = self.pop_E.vars["sNMDA"].view

        self.pop_I.vars["sGABA"].pull_from_device()
        self._buf_sGABA_I[:] = self.pop_I.vars["sGABA"].view

        # compute
        (a0, a1, a2, n0, n1, n2, aI, nI, g) = compute_weighted_sums(
            self._buf_sAMPA_E, self._buf_sNMDA_E, self._buf_sGABA_I,
            N0, N1, N2, self.W,
        )

        # push to E
        ba = self._buf_S_AMPA;  ba[:N0] = a0;  ba[N0:N0+N1] = a1;  ba[N0+N1:] = a2
        self.pop_E.vars["S_AMPA"].view[:] = ba
        self.pop_E.vars["S_AMPA"].push_to_device()

        bn = self._buf_S_NMDA;  bn[:N0] = n0;  bn[N0:N0+N1] = n1;  bn[N0+N1:] = n2
        self.pop_E.vars["S_NMDA"].view[:] = bn
        self.pop_E.vars["S_NMDA"].push_to_device()

        bg = self._buf_S_GABA;  bg[:] = g
        self.pop_E.vars["S_GABA"].view[:] = bg
        self.pop_E.vars["S_GABA"].push_to_device()

        # push to I
        self._buf_S_AMPA[:N_I] = aI
        self.pop_I.vars["S_AMPA"].view[:] = self._buf_S_AMPA[:N_I]
        self.pop_I.vars["S_AMPA"].push_to_device()

        self._buf_S_NMDA[:N_I] = nI
        self.pop_I.vars["S_NMDA"].view[:] = self._buf_S_NMDA[:N_I]
        self.pop_I.vars["S_NMDA"].push_to_device()

        self._buf_S_GABA[:N_I] = g
        self.pop_I.vars["S_GABA"].view[:] = self._buf_S_GABA[:N_I]
        self.pop_I.vars["S_GABA"].push_to_device()

    # ------------------------------------------------------------------
    def run(self, T_ms, rng_seed=1, report_interval=200.0,
            sum_update_steps=20):
        """
        Run simulation.

        Parameters
        ----------
        T_ms : float
            Total simulation time (ms).
        rng_seed : int
            Random seed for numpy (GPU RNG uses its own seed).
        report_interval : float
            Progress report interval (ms).
        sum_update_steps : int
            Number of timesteps between weighted-sum updates.
            Default 20 → every 0.4 ms (at dt=0.02 ms).
        """
        duration_steps = int(round(T_ms / self.dt))
        self.model.build()
        self.model.load(num_recording_timesteps=duration_steps)

        # mark inhibitory neurons (must be after load())
        self.pop_I.vars["is_inhib"].view[:] = 1.0
        self.pop_I.vars["is_inhib"].push_to_device()
        # E neurons default to is_inhib=0 (set via var_init)

        # initial stimulus state
        self._push_stim_rates(0.0)

        print(f"Simulating {T_ms} ms  (dt={self.dt} ms, "
              f"sum update every {sum_update_steps} steps = "
              f"{sum_update_steps * self.dt:.2f} ms) ...")

        t_start = perf_counter()
        next_report = report_interval
        step = 0

        # track stimulus state so we only push at boundaries
        stim_on = False
        Ton  = self.stimulus.Ton
        Toff = self.stimulus.Toff

        while self.model.t < T_ms:
            self.model.step_time()
            step += 1
            t = self.model.t

            # ---- stimulus boundary detection (trigger ONCE per edge) ----
            is_in_window = (Ton <= t < Toff)
            if not stim_on and is_in_window:
                stim_on = True
                self._push_stim_rates(t)
                print(f"  Stimulus ON  at t = {t:.1f} ms")
            elif stim_on and not is_in_window:
                stim_on = False
                self._push_stim_rates(t)
                print(f"  Stimulus OFF at t = {t:.1f} ms")

            # ---- batched weighted-sum update ----
            if step % sum_update_steps == 0:
                self._push_weighted_sums()

            # ---- progress ----
            if t >= next_report:
                pct = 100.0 * t / T_ms
                elapsed = perf_counter() - t_start
                print(f"  {pct:5.1f}%  ({t:.0f}/{T_ms} ms)  "
                      f"elapsed {elapsed:.1f}s")
                next_report += report_interval

        t_end = perf_counter()
        print(f"Simulation done.  Wall time: {t_end - t_start:.2f} s")

        # ---- pull spikes ----
        self.model.pull_recording_buffers_from_device()
        self.spike_times_E, self.spike_ids_E = self.pop_E.spike_recording_data[0]
        self.spike_times_I, self.spike_ids_I = self.pop_I.spike_recording_data[0]

        n_spikes_E = len(self.spike_times_E)
        n_spikes_I = len(self.spike_times_I)
        rate_E = n_spikes_E / self.params["N_E"] / (T_ms * 0.001)
        rate_I = n_spikes_I / self.params["N_I"] / (T_ms * 0.001)
        print(f"E: {n_spikes_E} spikes, avg rate {rate_E:.1f} Hz")
        print(f"I: {n_spikes_I} spikes, avg rate {rate_I:.1f} Hz")

        # ---- final weighted-sum push (to flush last batch) ----
        self._push_weighted_sums()

    # ------------------------------------------------------------------
    def save_spikes(self, file_e="spikesE.txt", file_i="spikesI.txt"):
        """Save spike times in Brian2-compatible format (neuron_id, time_s)."""
        tE_s = self.spike_times_E * 0.001   # ms → s
        tI_s = self.spike_times_I * 0.001

        print(f"Saving E spikes → {file_e}")
        np.savetxt(file_e,
                   np.column_stack((self.spike_ids_E, tE_s)),
                   fmt="%-9d %25.18e",
                   header="{:<8} {:<25}".format("Neuron", "Time (s)"))

        print(f"Saving I spikes → {file_i}")
        np.savetxt(file_i,
                   np.column_stack((self.spike_ids_I, tI_s)),
                   fmt="%-9d %25.18e",
                   header="{:<8} {:<25}".format("Neuron", "Time (s)"))

    # ------------------------------------------------------------------
    def plot_raster(self, filename="wang2002_pygenn_raster.pdf"):
        """Spike raster with subpopulation boundaries."""
        fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

        tE_s = self.spike_times_E * 0.001
        axes[0].scatter(tE_s, self.spike_ids_E, s=0.8, c="black", marker=".",
                        rasterized=True)
        axes[0].set_ylabel("Neuron index (E)")
        axes[0].set_title("Wang 2002 — Decision-Making Model (PyGeNN)")

        N0, N1 = self.N0, self.N1
        for y, lbl, c in [(N0, f"N0={N0}", "blue"),
                          (N0+N1, f"N1={N1}", "red"),
                          (N0+N1+self.N2, f"N2={self.N2}", "orange")]:
            axes[0].axhline(y=y, color=c, ls="--", alpha=0.4)
            axes[0].text(0.01, y+5, lbl, color=c, fontsize=7, va="bottom")

        tI_s = self.spike_times_I * 0.001
        axes[1].scatter(tI_s, self.spike_ids_I, s=0.8, c="red", marker=".",
                        rasterized=True)
        axes[1].set_ylabel("Neuron index (I)")
        axes[1].set_xlabel("Time (s)")

        plt.tight_layout()
        plt.savefig(filename, dpi=150)
        print(f"Raster → {filename}")
        plt.close()

    # ------------------------------------------------------------------
    def plot_firing_rates(self, filename="wang2002_pygenn_rates.pdf",
                          bin_ms=5.0):
        """Population firing rates over time."""
        t_end = max(self.spike_times_E.max() if len(self.spike_times_E) else 0,
                    self.spike_times_I.max() if len(self.spike_times_I) else 0)
        bins = np.arange(0, t_end + bin_ms, bin_ms)
        t_centers = (bins[:-1] + bins[1:]) * 0.0005   # ms → s

        fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

        # --- E subpopulation rates ---
        N0, N1, N2 = self.N0, self.N1, self.N2
        for lbl, lo, hi, c in [("Non-sel E0", 0, N0, "gray"),
                                ("Sel-1 E1", N0, N0+N1, "blue"),
                                ("Sel-2 E2", N0+N1, N0+N1+N2, "red")]:
            m = (self.spike_ids_E >= lo) & (self.spike_ids_E < hi)
            if m.any():
                h, _ = np.histogram(self.spike_times_E[m], bins)
                axes[0].plot(t_centers, h / (hi-lo) / (bin_ms*0.001),
                             label=lbl, color=c, lw=1)

        axes[0].set_ylabel("Rate (Hz)")
        axes[0].set_title("E Subpopulation Firing Rates")
        axes[0].legend(fontsize=7)

        # --- I rate ---
        if len(self.spike_times_I):
            hI, _ = np.histogram(self.spike_times_I, bins)
            axes[1].plot(t_centers, hI / self.params["N_I"] / (bin_ms*0.001),
                         color="green", lw=1)
        axes[1].set_ylabel("Rate (Hz)")
        axes[1].set_title("Inhibitory Population")

        # --- population averages ---
        hE, _ = np.histogram(self.spike_times_E, bins)
        axes[2].plot(t_centers, hE / self.params["N_E"] / (bin_ms*0.001),
                     label="E avg", color="black", lw=1)
        if len(self.spike_times_I):
            axes[2].plot(t_centers, hI / self.params["N_I"] / (bin_ms*0.001),
                         label="I avg", color="green", lw=1)
        axes[2].set_ylabel("Rate (Hz)")
        axes[2].set_title("Population-Averaged Rates")
        axes[2].set_xlabel("Time (s)")
        axes[2].legend()

        plt.tight_layout()
        plt.savefig(filename, dpi=150)
        print(f"Rates → {filename}")
        plt.close()


# ============================================================================
# __main__
# ============================================================================

if __name__ == "__main__":
    stimparams = {
        "Ton":  500.0,     # ms
        "Toff": 1500.0,    # ms
        "mu0":  0.040,     # kHz = 40 Hz
        "coh":  51.2,      # percent coherence
    }

    dt_ms = 0.02
    T_ms  = 2000.0

    sim = Wang2002Sim(modelparams, stimparams, dt_ms=dt_ms)
    sim.build(model_name="wang2002_pygenn", rng_seed=4)
    sim.run(T_ms, rng_seed=4, report_interval=200.0, sum_update_steps=20)

    sim.save_spikes("spikesE.txt", "spikesI.txt")
    sim.plot_raster("wang2002_pygenn_raster.pdf")
    sim.plot_firing_rates("wang2002_pygenn_rates.pdf")
