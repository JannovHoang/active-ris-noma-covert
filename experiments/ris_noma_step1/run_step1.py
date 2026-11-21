"""Chạy từ thư mục dự án: python run_step1.py --config configs/demo_k2.json."""
from pathlib import Path
import argparse
import csv
import json
import platform
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Không cần GUI; hình luôn được lưu vào ổ đĩa.
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, FormatStrFormatter

from ris_noma import __version__
from ris_noma.config import load_config
from ris_noma.random_streams import make_rng, STREAM
from ris_noma.geometry import sample_layout
from ris_noma.channels import build_bank, public_order
from ris_noma.metrics import evaluate_candidate


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def save_bank(path, layout, bank):
    np.savez_compressed(path, h_ar=bank.h_ar, h_ru=bank.h_ru, phase=bank.phase,
                        h_tilde=bank.h_tilde, g_scalar_tilde=bank.g_scalar_tilde,
                        signal_gain_tilde=bank.signal_gain_tilde,
                        noise_gain_tilde=bank.noise_gain_tilde,
                        loss_ar=layout.loss_ar, loss_ru=layout.loss_ru,
                        public_xy=layout.public_xy)


def plots(output, c, layout, bank, order, samples):
    label = "DEMO - illustrative parameters" if c["profile"] == "demo" else "Baseline configuration - fixed candidate"
    fig, ax = plt.subplots(figsize=(7.8, 5))
    for name in ["alice", "ris", "bob", "willie", "public_center"]:
        xy = c["positions"][name]
        ax.scatter(*xy, s=60, marker="s" if name == "ris" else "o")
        ax.annotate(name, xy, xytext=(-6, 6) if name == "ris" else (5, 6),
                    ha="right" if name == "ris" else "left", textcoords="offset points")
    center = c["positions"]["public_center"]
    width = c["public_half_width_m"]
    ax.add_patch(plt.Rectangle(np.asarray(center) - width, 2 * width, 2 * width, fill=False, linestyle="--", color="gray"))
    for rank, original in enumerate(order, start=1):
        xy = layout.public_xy[original]
        ax.scatter(*xy, marker="^", s=75)
        ax.annotate(f"C{rank} (original {original})", xy, xytext=(-7, -13), ha="right", textcoords="offset points")
    ax.set(xlabel="x (m)", ylabel="y (m)", title=label)
    all_xy = np.vstack([layout.receiver_xy, c["positions"]["alice"], c["positions"]["ris"]])
    ax.set_xlim(all_xy[:, 0].min() - 5, all_xy[:, 0].max() + 15)
    ax.set_ylim(all_xy[:, 1].min() - 7, all_xy[:, 1].max() + 8)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=.25)
    fig.tight_layout()
    fig.savefig(output / "layout.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].hist(np.abs(bank.h_ar).ravel() ** 2, bins=55, density=True, alpha=.8)
    axes[0].axvline(1, color="crimson", linestyle="--", label="theoretical mean = 1")
    axes[0].set(xlabel="Normalized |h_ar|^2", ylabel="Density", title=f"Rician KR={c['rician_k']}")
    axes[0].legend(fontsize=8)
    axes[1].hist(samples["bob_rate"], bins=40, alpha=.8)
    axes[1].set(xlabel="Bob instantaneous rate (bit/s/Hz)", ylabel="Count", title="Fixed candidate, label bank")
    axes[1].xaxis.set_major_locator(MaxNLocator(4))
    axes[1].xaxis.set_major_formatter(FormatStrFormatter("%.4f"))
    fig.suptitle(label, fontsize=10)
    fig.tight_layout()
    fig.savefig(output / "channel_and_rate.png", dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/demo_k2.json")
    parser.add_argument("--output", default="outputs/step1_k2")
    args = parser.parse_args()
    c = load_config(args.config)
    layout = sample_layout(c, make_rng(c["root_seed"], c["layout_id"], "geometry"))
    label = build_bank(c, layout, make_rng(c["root_seed"], c["layout_id"], "label"))
    evaluation = build_bank(c, layout, make_rng(c["root_seed"], c["layout_id"], "evaluation"))
    order = public_order(label)  # Tuyệt đối không sắp lại bằng bank evaluation.
    reference = c["reference_allocation"]
    summary, samples = evaluate_candidate(c, layout, label, order, reference["mb"], reference["public"], reference["s"])
    eval_summary, _ = evaluate_candidate(c, layout, evaluation, order, reference["mb"], reference["public"], reference["s"])
    bob_aligned = np.sum(np.abs(label.h_ar) * np.abs(label.h_ru[:, 0, :]), axis=1)
    phase_error = float(np.max(np.abs(label.h_tilde[:, 0] - bob_aligned)) / np.max(bob_aligned))
    if phase_error > 1e-12:
        raise ArithmeticError("Pha Bob không khớp (3.5)")
    if np.array_equal(label.h_ar, evaluation.h_ar):
        raise ArithmeticError("Hai bank fading trùng nhau")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "config_used.json", c)
    metadata = {
        "step": 1, "profile": c["profile"], "optimized": False,
        "scope": "fixed_candidate_rate_outage_and_willie_powers; DEP/PSO/GNN not implemented",
        "root_seed": c["root_seed"], "layout_id": c["layout_id"], "role_codes": STREAM,
        "rng": "NumPy default_rng with SeedSequence([root_seed,layout_id,role_code])",
        "receiver_order_raw_arrays": ["Bob", *[f"public_original_{i}" for i in range(c["n_public"])], "Willie"],
        "public_original_ids_in_sic_order": order.tolist(),
        "ordering_metric": "mean signal_gain_tilde on label bank; fixed on evaluation",
        "layout_eligible_by_bob_mean_gain_rule": summary["bob_strongest_mean_gain_this_bank"],
        "geometry_rejection_rule": "reject if any modeled link < min_link_distance_m; no automatic resampling",
        "public_xy_original_order": layout.public_xy.tolist(),
        "distance_ar_m": layout.distance_ar_m, "distance_ru_m": layout.distance_ru_m.tolist(),
        "loss_ar": layout.loss_ar, "loss_ru": layout.loss_ru.tolist(),
        "h_ar_sample_mean_power": label.mean_h_ar_power,
        "h_ru_sample_mean_power": np.mean(np.abs(label.h_ru) ** 2, axis=(0, 2)).tolist(),
        "bob_phase_relative_error": phase_error,
        "source_equations_v4": ["2.12", "3.2", "3.4", "4.1-4.5", "5.3-5.7", "6.2-6.3", "8.1-8.3", "9.6-9.10", "12.1-12.8"],
        "versions": {"python": platform.python_version(), "numpy": np.__version__, "matplotlib": matplotlib.__version__, "project": __version__},
    }
    write_json(output / "metadata.json", metadata)
    write_json(output / "metrics_label.json", summary)
    write_json(output / "metrics_evaluation.json", eval_summary)
    with (output / "rates_label.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["realization", "bob_rate_bit_s_hz", *[f"C{i+1}_rate_bit_s_hz" for i in range(c["n_public"])], "bob_outage"])
        writer.writerows([i, samples["bob_rate"][i], *samples["public_rates"][i], int(samples["bob_outage"][i])] for i in range(c["n_fading"]))
    save_bank(output / "label_bank.npz", layout, label)
    save_bank(output / "evaluation_bank.npz", layout, evaluation)
    plots(output, c, layout, label, order, samples)
    print("PROFILE:", c["profile"].upper(), "| fixed candidate; DEP not evaluated")
    print("K / N / M:", c["n_public"], c["n_ris"], c["n_fading"])
    print("SIC order (original public IDs):", order.tolist())
    print("E[|h_ar|^2] sample:", round(label.mean_h_ar_power, 6))
    print("Bob phase relative error:", f"{phase_error:.3e}")
    print("Bob mean rate [label / evaluation]:", round(summary["bob_mean_rate_bit_s_hz"], 6), "/", round(eval_summary["bob_mean_rate_bit_s_hz"], 6))
    print("Public mean rates [label]:", np.round(summary["public_mean_rate_bit_s_hz"], 6).tolist())
    print("Bob outage [label / evaluation]:", summary["bob_outage_estimate"], "/", eval_summary["bob_outage_estimate"])
    print("Bob mean-gain ordering rule:", metadata["layout_eligible_by_bob_mean_gain_rule"])
    print("Full feasibility:", summary["full_feasibility"])
    print("Saved to:", output.resolve())
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, KeyError) as error:
        print(f"CONFIG ERROR: {error}", file=sys.stderr)
        raise SystemExit(2)
