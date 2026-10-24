from parser_check import Parser
from dataloaders.dataclass import DataClass
from preprocessing import seed_everything, set_device
from logs.logger import CentralizedLogs
from visualization import CentralizedPlot, save_centralized_plots
from load_model import load_model
from save_results import save_centralized_data
from train import engine_class
import torch
import torch.nn as nn


if __name__ == "__main__":
    seed_everything()

    parse = Parser("centralized")
    model_name, dataset = parse.clean_args
    device = set_device()
    print('Current device:', device)
    print('Number of Devices: ', torch.cuda.device_count())

    model, optimizer = load_model(model_name, dataset, device)

    log_dir = f"logs/centralized/{model_name}/{dataset}/"
    logger = CentralizedLogs(log_dir)
    plotter = CentralizedPlot()

    dc = DataClass(dataset, "centralized")
    train_dl, val_dl, test_dl = dc.execute()

    loss_fn = nn.BCEWithLogitsLoss() if "breast" in dataset else nn.CrossEntropyLoss()
    model, train_logits = engine_class.train(model, optimizer, loss_fn, train_dl,
                                             val_dl, logger, plotter, 100, device)

    _, _, _, pr, re, _ = engine_class.evaluate(model, loss_fn, test_dl, device, log=logger)

    plotter.test_pr = pr
    plotter.test_re = re

    if parse.save_data:
        save_centralized_data(plotter, model_name, dataset)
    if parse.save_plots:
        save_centralized_plots(plotter, model_name, dataset)

