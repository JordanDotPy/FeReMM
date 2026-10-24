import torch
import torch.nn as nn
from parser_check import Parser
from load_model import load_federated
from preprocessing import seed_everything, set_device
from logs.logger import FeReMMLogs
from visualization import FeReMMPlot
from feremm import FeReMM
from save_results import save_feremm_data
from visualization import save_feremm_plots
from dataloaders.dataclass import DataClass


if __name__ == "__main__":
    seed_everything()

    parse = Parser("feremm")
    model_name, dataset, num_clients, bad_client = parse.clean_args
    device = set_device()
    print('Current device:', device)
    print('Number of Devices: ', torch.cuda.device_count())

    cms, cos, p, po, gm, go, layers = load_federated(model_name, dataset, num_clients, "feremm", device)

    log_dir = f"logs/feremm/{model_name}/{dataset}/"
    sbc = "" if not bad_client else str(bad_client)
    logger = FeReMMLogs(log_dir, str(num_clients), sbc)
    plotter = FeReMMPlot()

    dc = DataClass(dataset, "feremm", num_clients, bad_client=bad_client)
    cdls, val_dl, test_dl = dc.execute()
    loss_fn = nn.BCEWithLogitsLoss() if "breast" in dataset else nn.CrossEntropyLoss()
    te = 25 if dataset == "oct_mnist" and model_name == "resnet50" else 5

    fr = FeReMM(gm, go, cms, cos, model_name, p, po, layers,
                dataset, num_clients, device, logger, te, bad_client=bad_client)

    fr._execute(cdls, val_dl, test_dl, loss_fn, plotter)

    if parse.save_data:
        save_feremm_data(fr, plotter, cdls, sbc)
    if parse.save_plots:
        save_feremm_plots(plotter, dataset, model_name, num_clients, sbc, layers)
