import torch
import torch.nn as nn
from parser_check import Parser
from load_model import load_federated
from preprocessing import seed_everything, set_device
from logs.logger import FederatedLogs
from visualization import FederatedPlot, save_federated_plots
from federated import FederatedLearning
from save_results import save_federated_data
from dataloaders.dataclass import DataClass


if __name__ == "__main__":
    seed_everything()

    parse = Parser("fedavg")
    model_name, dataset, num_clients, bad_client = parse.clean_args
    device = set_device()
    print('Current device:', device)
    print('Number of Devices: ', torch.cuda.device_count())

    gm, cms, cos, layers = load_federated(model_name, dataset, num_clients, "fedavg", device)

    log_dir = f"logs/fedavg/{model_name}/{dataset}/"
    sbc = "" if not bad_client else str(bad_client)
    logger = FederatedLogs(log_dir, str(num_clients), sbc)
    plotter = FederatedPlot()

    dc = DataClass(dataset, "fedavg", num_clients)
    cdls, val_dl, test_dl = dc.execute()
    loss_fn = nn.BCEWithLogitsLoss() if "breast" in dataset else nn.CrossEntropyLoss()
    te = 25 if dataset == "oct_mnist" and model_name == "resnet50" else 5

    fl = FederatedLearning(gm, cms, cos, model_name, num_clients, dataset, "fedavg", bad_client, layers, te, device)
    fl._execute(cdls, val_dl, test_dl, loss_fn, plotter, logger)

    if parse.save_data:
        save_federated_data(fl, plotter, sbc)
    if parse.save_plots:
        save_federated_plots(plotter, "fedavg", dataset, model_name, num_clients, sbc)
