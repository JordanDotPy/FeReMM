import io
import os
from collections import defaultdict
from dataclasses import dataclass, field

import matplotlib as mpl
import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from PIL import Image, ImageDraw, ImageFont
from sklearn.decomposition import PCA


@dataclass
class FeReMMPlot:
    policy_reward: list = field(default_factory=list)
    policy_entropy: list = field(default_factory=list)
    policy_loss: list = field(default_factory=list)

    bad_client_prediction: list = field(default_factory=list)
    client_validation_trajectory: dict = field(default_factory=dict)
    client_training_trajectory: dict = field(default_factory=dict)
    client_class_distribution: dict = field(default_factory=dict)
    client_distance: defaultdict = field(default_factory=lambda: defaultdict(list))
    client_image_features: dict = field(default_factory=dict)

    val_client_loss: defaultdict = field(default_factory=lambda: defaultdict(list))
    val_client_acc: defaultdict = field(default_factory=lambda: defaultdict(list))
    val_client_auc: defaultdict = field(default_factory=lambda: defaultdict(list))
    delta_acc: list = field(default_factory=list)
    delta_auc: list = field(default_factory=list)
    delta_loss: list = field(default_factory=list)

    test_loss: list = field(default_factory=list)
    test_acc: list = field(default_factory=list)
    test_auc: list = field(default_factory=list)
    test_pr: list = field(default_factory=list)
    test_re: list = field(default_factory=list)


@dataclass
class CentralizedPlot:
    training_loss: list = field(default_factory=list)
    training_acc: list = field(default_factory=list)

    validation_loss: list = field(default_factory=list)
    validation_acc: list = field(default_factory=list)
    validation_auc: list = field(default_factory=list)

    test_pr: list = field(default_factory=list)
    test_re: list = field(default_factory=list)


@dataclass
class FederatedPlot:
    training_loss: defaultdict = field(default_factory=lambda: defaultdict(list))
    training_acc: defaultdict = field(default_factory=lambda: defaultdict(list))

    val_client_loss: defaultdict = field(default_factory=lambda: defaultdict(list))
    val_client_acc: defaultdict = field(default_factory=lambda: defaultdict(list))
    val_client_auc: defaultdict = field(default_factory=lambda: defaultdict(list))

    testing_loss: list = field(default_factory=list)
    testing_acc: list = field(default_factory=list)
    testing_auc: list = field(default_factory=list)
    test_pr: list = field(default_factory=list)
    test_re: list = field(default_factory=list)

    client_validation_trajectory: dict = field(default_factory=dict)
    client_training_trajectory: dict = field(default_factory=dict)
    client_class_distribution: dict = field(default_factory=dict)
    client_distance: defaultdict = field(default_factory=lambda: defaultdict(list))


def _out(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return path


def _save(fig, path):
    fig.savefig(_out(path))
    plt.close(fig)


def simple_plot(data, title, y):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(list(range(len(data))), data)
    ax.set_title(title)
    ax.set_xlabel("Round")
    ax.set_ylabel(y)
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    return fig


def client_plot(data, title, x, y, y_range=None, shape_legend=False):
    fig, ax = plt.subplots(figsize=(10, 6))
    for client_id, d in data.items():
        ax.plot(list(range(len(d))), d, label=f"C{client_id}")

    ax.set_title(title)
    if x: ax.set_xlabel(x)
    if y: ax.set_ylabel(y)
    if "loss" not in y.lower():
        ax.set_ylim(0.0, 1.0)
    if y_range is not None:
        ax.set_ylim(y_range[0], y_range[1])

    if shape_legend:
        plt.xticks(np.arange(0, 20, 5), fontsize=24)
        plt.yticks(fontsize=24)
        ax.legend(loc="upper left", ncol=2, frameon=False, handlelength=1.2, columnspacing=0.6,
                  handletextpad=0.3, prop={"size": 24, "weight": "bold"})
    else:
        ax.legend()
    ax.grid(True)
    fig.tight_layout()
    return fig


def prre_plot(pr, re):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(pr, re)
    ax.set_xlabel("Class Threshold")
    ax.set_ylabel("Score")
    ax.set_title("Object-Level Detection Performance")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    return fig


def auc_plot(auc):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(list(range(len(auc))), auc, label="Validation AUC")
    ax.set_title("Validation AUC")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("AUC Score")
    ax.set_ylim(0.0, 1.0)
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    return fig


def badclient_plot(pred):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(list(range(5)), pred, label='Avg. Predicted')
    ax.set_title("Average Poisioned Client Prediction")
    ax.set_xlabel("Round")
    ax.set_ylabel("Poisioned Client Frequency")
    ax.grid(True)
    fig.tight_layout()
    return fig


def centralized_plot(train_metric, val_metric, metric_name):
    fig, ax = plt.subplots(figsize=(10, 6))
    rounds = list(range(len(train_metric)))
    ax.plot(rounds, train_metric, label=f"Training {metric_name}")
    ax.plot(rounds, val_metric, label=f"Validation {metric_name}")
    ax.set_title("Training Convergence")
    ax.set_xlabel("Epoch")
    ax.set_ylabel(f"{metric_name} Score")
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    return fig


def _round_plot(values, title, ylabel, ylim=None):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_title(title)
    ax.plot(list(range(len(values))), values)
    ax.set_xlabel("Round")
    ax.set_ylabel(ylabel)
    if ylim: ax.set_ylim(*ylim)
    ax.grid(True)
    fig.tight_layout()
    return fig


def federated_plot(plot):
    return (_round_plot(plot.testing_loss, "Global Model Test Convergence", "Loss"),
            _round_plot(plot.testing_acc, "Global Model Accuracy Score", "Accuracy", (0.0, 1.0)),
            _round_plot(plot.testing_auc, "Global Model AUC Score", "AUC", (0.0, 1.0)))


def save_centralized_plots(plotter, model_name, dataset):
    out = f"evaluation/plots/centralized/{model_name}/{dataset}"
    _save(centralized_plot(plotter.training_acc, plotter.validation_acc, "ACC"), f"{out}/acc.png")
    _save(centralized_plot(plotter.training_loss, plotter.validation_loss, "Loss"), f"{out}/loss.png")
    _save(auc_plot(plotter.validation_auc), f"{out}/auc.png")


def save_federated_plots(plotter, framework, dataset, model_name, num_clients, bc):
    out = f"evaluation/plots/{framework}/{model_name}/{dataset}"
    prefix = f"{out}/{num_clients}{bc}"

    if plotter.test_pr is not None and plotter.test_re is not None:
        _save(prre_plot(plotter.test_pr, plotter.test_re), f"{prefix}prc_plot.png")

    loss, acc, auc = federated_plot(plotter)
    _save(loss, f"{prefix}loss.png")
    _save(acc, f"{prefix}acc.png")
    _save(auc, f"{prefix}auc.png")
    _save(client_distance_plot(plotter.client_distance, *distance_ylim([plotter.client_distance])), f"{prefix}total_param_distances.png")
    _save(client_plot(plotter.val_client_loss, "Round", "Client Validation Loss", "Loss"), f"{prefix}val_loss.png")
    _save(client_plot(plotter.val_client_acc, "Round", "Client Validation ACC", "ACC"), f"{prefix}val_acc.png")
    _save(client_plot(plotter.val_client_auc, "Round", "Client Validation AUC", "AUC"), f"{prefix}val_auc.png")


def save_feremm_plots(plotter, dataset, model_name, num_clients, bc, layer_names):
    out = f"evaluation/plots/feremm/{model_name}/{dataset}"
    prefix = f"{out}/{num_clients}{bc}"

    _save(simple_plot(plotter.policy_reward, "Policy Rewards", "Reward"), f"{prefix}policy_reward_plot.png")
    _save(simple_plot(plotter.test_loss, "Globa Model Testing Loss", "Loss"), f"{prefix}loss_plot.png")
    _save(client_plot(plotter.val_client_loss, "Round", "Client Validation Loss", "Loss"), f"{prefix}val_loss_plot.png")
    _save(client_distance_plot(plotter.client_distance, *distance_ylim([plotter.client_distance])), f"{prefix}total_param_distances.png")

    if bc != "":
        _save(badclient_plot(plotter.bad_client_prediction), f"{prefix}bad_merge_plot.png")

    if plotter.test_pr is not None and plotter.test_re is not None:
        _save(prre_plot(plotter.test_pr, plotter.test_re), f"{prefix}prc_plot.png")

    _save(simple_plot(plotter.delta_acc, "Global Model Change in ACC After Merge", "Delta ACC"), f"{prefix}delta_acc_plot.png")
    _save(simple_plot(plotter.delta_auc, "Global Model Change in AUC After Merge", "Delta AUC"), f"{prefix}delta_auc_plot.png")
    _save(simple_plot(plotter.delta_loss, "Global model Change in Loss After Merge", "Delta Loss"), f"{prefix}delta_loss_plot.png")
    _save(simple_plot(plotter.test_acc, "Global Model Testing ACC", "ACC"), f"{prefix}acc_plot.png")
    _save(simple_plot(plotter.test_auc, "Global Model Testing AUC", "AUC"), f"{prefix}auc_plot.png")
    _save(client_plot(plotter.val_client_acc, "Round", "Client Validation ACC", "ACC"), f"{prefix}val_acc_plot.png")
    _save(client_plot(plotter.val_client_auc, "Round", "Client Validation AUC", "AUC"), f"{prefix}val_auc_plot.png")


def _load(path):
    return torch.load(path, weights_only=False)


def framework_prre_plot(dataset, num_clients, model, bc=""):
    folder = f"evaluation/saved_data/{model}/{dataset}/"
    curves = [("centralized", "", "orange"), ("fedavg", f"{num_clients}{bc}", "blue"),
              ("fedprox", f"{num_clients}{bc}", "green"), ("feremm", f"{num_clients}{bc}", "red")]

    fig, ax = plt.subplots(figsize=(10, 6))
    for label, prefix, color in curves:
        pr = np.array(_load(f"{folder}{prefix}{label}_pr.pt")).squeeze().reshape(-1)
        re = np.array(_load(f"{folder}{prefix}{label}_re.pt")).squeeze().reshape(-1)
        ax.plot(re, pr, color=color, label=label)

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    path = f"evaluation/plots/{model}_{dataset}_{num_clients}prre.png"
    plt.savefig(_out(path))
    print(f"File Saved: {path}")


def client_class_distribution(dataset, num_clients):
    df = _load(f"evaluation/saved_data/{dataset}_{num_clients}class_distribution.pt")
    fig, ax = plt.subplots(figsize=(12, 8))
    im = ax.imshow(df.values, aspect="auto", cmap="viridis", vmin=0.0, vmax=df.values.max())
    fig.colorbar(im, ax=ax, label="Class % Present in Dataset")
    ax.set_xticks(range(len(df.columns)))
    ax.set_xticklabels(df.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels([f"{idx}" for idx in df.index])
    ax.set_xlabel("Class")
    ax.set_title("Client Training Dataset Class Distribution")
    fig.tight_layout()
    path = f"evaluation/plots/{dataset}_{num_clients}class_distribution.png"
    plt.savefig(_out(path), dpi=300, bbox_inches="tight")
    print(f"File Saved: {path}")


def plot_client_image_stats(dataset, num_clients):
    stats_df = _load(f"evaluation/saved_data/{dataset}_{num_clients}image_features.pt")
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(stats_df))
    width = 0.20
    bars = [("mean_intensity", "Mean Intensity"), ("std_intensity", "Intensity Std"),
            ("mean_image_contrast", "Mean Image Contrast"), ("std_image_contrast", "Std Image Contrast")]
    for i, (column, label) in enumerate(bars):
        ax.bar(x + (i - 1.5) * width, stats_df[column], width, label=label)

    ax.set_xticks(x)
    ax.set_xticklabels([f"Client {c}" for c in stats_df["client"]])
    ax.set_xlabel("Client")
    ax.set_ylabel("Normalized Value")
    ax.set_title("Per-Client Image Feature Statistics")
    ax.legend(fontsize=7, bbox_to_anchor=(1.05, 0.5), loc="upper left")
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    path = f"evaluation/plots/{dataset}_{num_clients}image_stats.png"
    fig.savefig(_out(path), dpi=300, bbox_inches="tight")
    print(f"File Saved: {path}")


def policy_differences(model, num_clients, dataset, bc=""):
    folder = f"evaluation/saved_data/{model}/{dataset}/{num_clients}{bc}feremm_"
    data = {
        "REINFORCE-PPO": ([_load(folder + "normal_entropy.pt"), _load(folder + "normal_loss.pt")], plt.cm.Reds),
        "REINFORCE": ([_load(folder + "u2_entropy.pt"), _load(folder + "u2_loss.pt")], plt.cm.Blues),
    }
    labels = ["Entropy", "Loss"]

    fig, ax = plt.subplots(figsize=(10, 7))
    fig.subplots_adjust(right=0.78)
    for update, (series, cmap) in data.items():
        for i, dp in enumerate(series):
            if isinstance(dp[0], torch.Tensor):
                dp = [z.item() for z in dp]
            ax.plot(list(range(len(dp))), dp, color=cmap(0.35 + 0.60 * (i / len(series))),
                    linewidth=2, alpha=0.85, label=f"{update} - {labels[i]}")

    ax.set_xlabel("Action")
    ax.legend(fontsize=7, bbox_to_anchor=(1.05, 0.5), loc="upper left")
    path = f"evaluation/plots/{model}_{dataset}_{num_clients}policy_differences.png"
    plt.savefig(_out(path))
    print(f"File Saved: {path}")


def build_logit_trajectory_matrix(all_histories):
    vectors = []
    metadata = []
    for framework_name, logits_history in all_histories.items():
        for round_id in sorted(logits_history.keys()):
            for client_id in logits_history[round_id]:
                logits = logits_history[round_id][client_id]
                if isinstance(logits, torch.Tensor):
                    logits = logits.detach().cpu().reshape(-1).numpy()
                else:
                    logits = np.asarray(logits).reshape(-1)
                vectors.append(logits)
                metadata.append({"framework": framework_name, "round": round_id, "client": client_id})

    trajectory_df = pd.DataFrame(metadata)
    Z = PCA(n_components=2).fit_transform(np.stack(vectors))
    trajectory_df["pc1"] = Z[:, 0]
    trajectory_df["pc2"] = Z[:, 1]
    return trajectory_df


def plot_logit_trajectories(framework,
                            model,
                            dataset,
                            num_clients,
                            t,
                            x_min,
                            x_max,
                            y_min,
                            y_max,
                            bc=""):
    history = {framework: _load(f"evaluation/saved_data/{model}/{dataset}/{num_clients}{bc}{framework}_{t.lower()}_trajectory.pt")}
    trajectory_df = build_logit_trajectory_matrix(history)

    fig, ax = plt.subplots(figsize=(10, 7))
    cmap = plt.cm.turbo
    norm = mpl.colors.Normalize(vmin=trajectory_df["round"].min(), vmax=trajectory_df["round"].max())
    for _, group in trajectory_df.groupby("client"):
        group = group.sort_values("round")
        ax.scatter(group["pc1"], group["pc2"], s=55, c=group["round"], cmap=cmap, norm=norm, edgecolors='black')

    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)

    sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=ax)

    fig.tight_layout()
    path = f"evaluation/plots/{model}_{dataset}_{framework}_{num_clients}{t.lower()}_trajectory.png"
    plt.savefig(_out(path), dpi=300, bbox_inches="tight")
    print(f"File Saved: {path}")
    plt.close(fig)


def trajectory_zoom_window(data, zoom_width=4, skip_rounds=3):
    curves = np.array([data[c] for c in sorted(data)])
    moves = np.diff(curves, axis=1).std(axis=0)
    start = max(range(skip_rounds, len(moves) - zoom_width + 2), key=lambda i: moves[i:i + zoom_width - 1].mean())
    return start, start + zoom_width - 1


def distance_xlim(n_rounds):
    return -0.05 * (n_rounds - 1), 1.05 * (n_rounds - 1)


def distance_ylim(datasets,
                  inset_bounds=(0.42, 0.60, 0.55, 0.35),
                  margin=0.04):
    curves = np.concatenate([np.array([d[c] for c in sorted(d)]) for d in datasets])
    n = curves.shape[1]
    x_lo, x_hi = distance_xlim(n)
    under_inset = np.arange(n) >= x_lo + inset_bounds[0] * (x_hi - x_lo)
    pad = margin * (curves.max() - curves.min())
    y_min = curves.min() - pad
    top_under_inset = curves[:, under_inset].max() + pad
    y_max = max(curves.max() + pad, y_min + (top_under_inset - y_min) / inset_bounds[1])
    return y_min, y_max


def client_distance_plot(data,
                         y_min=0.0014,
                         y_max=0.004,
                         zoom=True,
                         zoom_window=None,
                         zoom_width=4,
                         inset_bounds=(0.42, 0.60, 0.55, 0.35)):
    clients = sorted(data)
    curves = np.array([data[c] for c in clients])
    rounds = np.arange(curves.shape[1])

    fig, ax = plt.subplots(figsize=(10, 6))
    lines = [ax.plot(rounds, d, label=f"C{c}")[0] for c, d in zip(clients, curves)]
    ax.set_xlim(*distance_xlim(len(rounds)))
    ax.set_ylim(y_min, y_max)
    ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(nbins=5))
    plt.xticks(np.arange(0, 20, 5), fontsize=24)
    plt.yticks(fontsize=24)
    ax.grid(True)

    if zoom:
        x0, x1 = zoom_window if zoom_window is not None else trajectory_zoom_window(data, zoom_width)
        seg = curves[:, x0:x1 + 1]
        pad = 0.1 * (seg.max() - seg.min()) or 0.1 * seg.max()

        axins = ax.inset_axes(list(inset_bounds))
        for line, d in zip(lines, curves):
            axins.plot(rounds, d, color=line.get_color())
        axins.set_xlim(x0, x1)
        axins.set_ylim(seg.min() - pad, seg.max() + pad)
        axins.set_xticks([])
        axins.set_yticks([])
        ax.indicate_inset_zoom(axins, edgecolor="gray")

    fig.tight_layout()
    return fig


def make_distance_figure(model="resnet18", nc=10):
    for ds in DATASETS:
        folder = f"evaluation/saved_data/{model}/{ds}/{nc}"
        data = {fw: _load(f"{folder}{fw}_total_param_distances.pt") for fw in FRAMEWORKS}
        y_min, y_max = distance_ylim(list(data.values()))
        window = trajectory_zoom_window(data["feremm"])
        for fw, d in data.items():
            _save(client_distance_plot(d, y_min=y_min, y_max=y_max, zoom_window=window),
                  f"evaluation/plots/{fw}/{model}/{ds}/{nc}total_param_distances.png")
    create_distance_collage(model, nc)


FRAMEWORKS = ["fedavg", "fedprox", "feremm"]
DATASETS = ["breast_mnist", "retina_mnist", "derma_mnist", "tissue_mnist", "oct_mnist"]


def shared_legend_image(labels,
                        width,
                        fontsize=22):
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    handles = [Line2D([0], [0], color=colors[i % len(colors)], lw=3) for i in range(len(labels))]
    fig = plt.figure(figsize=(width / 100, 1.0), dpi=100)
    fig.legend(handles, labels, loc="center", ncol=len(labels), frameon=False,
               prop={"size": fontsize, "weight": "bold"}, handlelength=1.5, columnspacing=1.2)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def build_collage(image_paths,
                  output_path,
                  title,
                  x_title,
                  y_title,
                  legend_labels=None):
    col_labels = ["BreastMNIST", "RetinaMNIST", "DermaMNIST", "TissueMNIST", "OCTMNIST"]
    row_labels = ["FedAvg", "FedProx", "FeReMM"]
    cols, rows = 5, 3
    target_width = 500
    padding = 30
    main_title_height = 80
    top_label_height = 70
    bottom_axis_height = 80
    left_axis_width = 80
    left_label_width = 140

    font_main = ImageFont.truetype("DejaVuSans-Bold.ttf", 30)
    font_title = ImageFont.truetype("DejaVuSans-Bold.ttf", 25)
    font_axis = ImageFont.truetype("DejaVuSans-Bold.ttf", 24)

    resized = []
    for p in image_paths:
        img = Image.open(p).convert("RGB")
        resized.append(img.resize((target_width, int(img.height * (target_width / img.width)))))
    cell_width = target_width
    cell_height = max(img.height for img in resized)

    legend_img = shared_legend_image(legend_labels, cols * cell_width) if legend_labels else None

    grid_left = left_axis_width + left_label_width + padding
    grid_top = main_title_height + top_label_height + padding
    grid_width = cols * cell_width + (cols - 1) * padding
    grid_height = rows * cell_height + (rows - 1) * padding

    canvas_width = grid_left + grid_width + padding
    canvas_height = grid_top + grid_height + padding + bottom_axis_height
    if legend_img:
        canvas_height += legend_img.height + padding
    canvas = Image.new("RGB", (canvas_width, canvas_height), "white")
    draw = ImageDraw.Draw(canvas)

    def text_size(text, font):
        box = draw.textbbox((0, 0), text, font=font)
        return box[2] - box[0], box[3] - box[1]

    w, _ = text_size(title, font_main)
    draw.text(((canvas_width - w) / 2, 60), title, fill="black", font=font_main)
    for col, name in enumerate(col_labels):
        w, _ = text_size(name, font_title)
        draw.text((grid_left + col * (cell_width + padding) + (cell_width - w) / 2, main_title_height + 55), name, fill="black", font=font_title)
    for row, name in enumerate(row_labels):
        _, h = text_size(name, font_title)
        draw.text((left_axis_width + 25, grid_top + row * (cell_height + padding) + (cell_height - h) / 2), name, fill="black", font=font_title)

    for idx, img in enumerate(resized):
        row, col = divmod(idx, cols)
        canvas.paste(img, (grid_left + col * (cell_width + padding), grid_top + row * (cell_height + padding)))

    w, _ = text_size(x_title, font_axis)
    x_title_y = grid_top + grid_height + padding
    draw.text(((canvas_width - w) / 2, x_title_y), x_title, fill="black", font=font_axis)
    if legend_img:
        canvas.paste(legend_img, (int(grid_left + (grid_width - legend_img.width) / 2), x_title_y + bottom_axis_height - padding))

    w, h = text_size(y_title, font_axis)
    y_label = Image.new("RGBA", (w + 20, h + 20), (255, 255, 255, 0))
    ImageDraw.Draw(y_label).text((10, 10), y_title, fill="black", font=font_axis)
    y_label = y_label.rotate(90, expand=True)
    canvas.paste(y_label, (10, int(grid_top + grid_height / 2 - y_label.height / 2)), y_label)

    canvas.save(_out(output_path), dpi=(300, 300))
    print(f"File Saved: {output_path}")


def create_collage(mt, xt, yt, nc=""):
    paths = [f"evaluation/plots/resnet18_{ds}_{fw}_{nc}training_trajectory.png" for fw in FRAMEWORKS for ds in DATASETS]
    build_collage(paths, f"evaluation/plots/resnet18_{nc}training_collage.png", mt, xt, yt)


def create_distance_collage(model="resnet18", nc=10, num_clients_in_legend=10):
    paths = [f"evaluation/plots/{fw}/{model}/{ds}/{nc}total_param_distances.png" for fw in FRAMEWORKS for ds in DATASETS]
    build_collage(paths, f"evaluation/plots/{model}_{nc}param_collage.png",
                  "Client Model vs. Global Model Parameter Distances", "Communication Round",
                  "Total Parameter Distance (L2-Distance)", legend_labels=[f"C{i}" for i in range(num_clients_in_legend)])


if __name__ == "__main__":
    make_distance_figure("resnet18", 10)
