import os
import matplotlib.pyplot as plt
from scipy.signal import welch

def truncate_psd(f, psd, f_max=100):
    if f is None or psd is None:
        return None, None
    mask = f <= f_max
    return f[mask], psd[mask]

def plot_psd(rate_smoothed, time_bins, model_name, sample_bin, suffix, 
             area, layer=None, pop=None):
    stim_start = stim_end = None
    if model_name and "_stim" in model_name:
        try:
            stim_start = float(model_name.split("_start")[1].split("s")[0]) * 1000
            stim_end = float(model_name.split("_end")[1].split("s")[0]) * 1000
        except Exception as e:
            print(f"Error parsing stim time: {e}")

    fs = 1 / sample_bin * 1000
    if stim_start and stim_end:
        t_range = time_bins[:-1]
        stim_mask = (t_range >= stim_start) & (t_range < stim_end)
        nostim_mask = ~stim_mask

        if stim_mask.sum() > 10:
            f_stim, psd_stim = welch(rate_smoothed[stim_mask], fs=fs, nperseg=256)
        else:
            f_stim, psd_stim = None, None

        if nostim_mask.sum() > 10:
            f_nostim, psd_nostim = welch(rate_smoothed[nostim_mask], fs=fs, nperseg=256)
        else:
            f_nostim, psd_nostim = None, None
    else:
        f_nostim, psd_nostim = welch(rate_smoothed, fs=fs, nperseg=256)
        f_stim, psd_stim = None, None

    f_nostim, psd_nostim = truncate_psd(f_nostim, psd_nostim)
    f_stim, psd_stim = truncate_psd(f_stim, psd_stim)

    fig_psd, axs_psd = plt.subplots(1, 2 if psd_stim is not None else 1, figsize=(10, 4))
    if psd_stim is not None:
        axs_psd[0].plot(f_nostim, psd_nostim, color="black", label="No Stim")
        axs_psd[0].set_title("No Stim")
        axs_psd[1].plot(f_stim, psd_stim, color="orange", label="Stim")
        axs_psd[1].set_title("Stim")
        for ax in axs_psd:
            ax.set_xlabel("Frequency (Hz)")
            ax.set_ylabel("Power")
            ax.grid(True)
    else:
        axs_psd.plot(f_nostim, psd_nostim, color="black", label="No Stim")
        axs_psd.set_xlabel("Frequency (Hz)")
        axs_psd.set_ylabel("Power")
        axs_psd.set_title("No Stim")
        axs_psd.grid(True)
    if layer == None and pop != None:
        fig_psd.suptitle(f"{area} - {pop} Psd - {model_name}" if model_name else f"{area} - {pop} Psd")
        fig_psd.tight_layout(rect=[0, 0, 1, 0.95])
        os.makedirs(f"psd/{area}/pop", exist_ok=True)
        fig_psd.savefig(f"psd/{area}/pop/{pop}_{suffix}.png")
    elif layer != None and pop == None:
        fig_psd.suptitle(f"{area} - Layer {layer} Psd - {model_name}" if model_name else f"{area} - Layer {layer} Psd")
        fig_psd.tight_layout(rect=[0, 0, 1, 0.95])
        os.makedirs(f"psd/{area}/layer", exist_ok=True)
        fig_psd.savefig(f"psd/{area}/layer/{layer}_{suffix}.png")
    elif layer == None and pop == None:
        fig_psd.suptitle(f"{area} Psd - {model_name}" if model_name else f"{area} Psd")
        fig_psd.tight_layout(rect=[0, 0, 1, 0.95])
        os.makedirs(f"psd/{area}/", exist_ok=True)
        fig_psd.savefig(f"psd/{area}/psd_{suffix}.png")
    plt.close(fig_psd)