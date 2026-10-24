from dataclasses import dataclass
import logging
import os


def setup_logger(mode, log_dir, num_clients="", bad_clients=""):
    os.makedirs(log_dir, exist_ok=True)

    if mode.lower() == "train":
        log_file = os.path.join(log_dir, num_clients + bad_clients + "training.log")
    elif mode.lower() == "test":
        log_file = os.path.join(log_dir, num_clients + bad_clients + "testing.log")
    elif mode.lower() == "validate":
        log_file = os.path.join(log_dir, num_clients + bad_clients + "validate.log")
    elif mode.lower() == "merge":
        log_file = os.path.join(log_dir, num_clients + bad_clients + "merging.log")
    elif mode.lower() == "policy":
        log_file = os.path.join(log_dir, num_clients + bad_clients + "policy.log")
    elif mode.lower() == "reward":
        log_file = os.path.join(log_dir, num_clients + bad_clients + "reward.log")
    else:
        log_file = os.path.join(log_dir, num_clients + bad_clients + "default.log")

    logger = logging.getLogger(f"logger_{mode}")
    logger.setLevel(logging.INFO)

    if logger.hasHandlers():
        logger.handlers.clear()

    file_handler = logging.FileHandler(log_file, mode="w")
    file_handler.setLevel(logging.INFO)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


@dataclass
class CentralizedLogs:
    def __init__(self, directory):
        self.train = setup_logger('train', directory)
        self.validate = setup_logger('validate', directory)
        self.test = setup_logger('test', directory)

    def log_train_class(self, ep, acc, loss, *args):
        self.train.info(f"======== Epoch {ep} Training Metrics ========")
        self.train.info(f"  Loss: {loss}")
        self.train.info(f"  ACC: {acc}")

    def log_validate_class(self, ep, acc, auc, loss, *args):
        self.validate.info(f"======== Epoch {ep} Validation Metrics ========")
        self.validate.info(f"  Loss: {loss}")
        self.validate.info(f"  ACC: {acc}")
        self.validate.info(f"  AUC: {auc}")

    def log_test_class(self, acc, auc, pr_auc, pr, re, f1, loss, *args):
        self.test.info(f"======== Testing Metrics ========")
        self.test.info(f"  Loss: {loss}")
        self.test.info(f"  ACC: {acc}")
        self.test.info(f"  AUC: {auc}")
        self.test.info(f"  Precision-Recall AUC: {pr_auc}")
        self.test.info(f"  Best-F1: {f1}")
        self.test.info(f"  Precision @ Best-F1: {pr}")
        self.test.info(f"  Recall @ Best-F1: {re}")


@dataclass
class FederatedLogs:
    def __init__(self, directory, num_clients, bad_clients=""):
        self.train = setup_logger('train', directory, num_clients, bad_clients)
        self.validate = setup_logger('validate', directory, num_clients, bad_clients)
        self.test = setup_logger('test', directory, num_clients, bad_clients)

    def log_train_class(self, ep, acc, loss, cid, round):
        self.train.info(f"======== Client {cid} Epoch {ep} Training Metrics Global Round {round} ========")
        self.train.info(f"  Loss: {loss}")
        self.train.info(f"  ACC: {acc}")

    def log_validate_class(self, ep, acc, auc, loss, cid, round):
        self.validate.info(f"======== Client {cid} Epoch {ep} Validation Metrics Global Round {round} ========")
        self.validate.info(f"  Loss: {loss}")
        self.validate.info(f"  ACC: {acc}")
        self.validate.info(f"  AUC: {auc}")

    def log_test_class(self, acc, auc, pr_auc, pr, re, f1, loss, round):
        self.test.info(f"======== Global Model Testing Metrics Global Round {round} ========")
        self.test.info(f"  Loss: {loss}")
        self.test.info(f"  ACC: {acc}")
        self.test.info(f"  ROC-AUC: {auc}")
        self.test.info(f"  Precision-Recall AUC: {pr_auc}")
        self.test.info(f"  Best-F1: {f1}")
        self.test.info(f"  Precision @ Best-F1: {pr}")
        self.test.info(f"  Recall @ Best-F1: {re}")


@dataclass
class FeReMMLogs:
    def __init__(self, directory, num_clients, bad_clients=""):
        self.train = setup_logger('train', directory, num_clients, bad_clients)
        self.validate = setup_logger('validate', directory, num_clients, bad_clients)
        self.test = setup_logger('test', directory, num_clients, bad_clients)
        self.merge = setup_logger('merge', directory, num_clients, bad_clients)
        self.policy = setup_logger('policy', directory, num_clients, bad_clients)
        self.reward = setup_logger('reward', directory, num_clients, bad_clients)

    def log_merges(self, merges, round):
        self.merge.info(f"======== Merges Withinin Global Round {round} ========")
        self.merge.info(f"  GM Aggregation: {merges}")

    def log_policy(self, loss, round):
        self.policy.info(f"======== Policy Loss Global Round {round} ========")
        self.policy.info(f"  Policy Loss: {loss}")

    def log_rewards(self, rewards, round):
        self.reward.info(f"======== Policy Reward Global Round {round} ========")
        self.reward.info(f"  Policy Reward: {rewards}")

    def log_train_class(self, ep, cid, acc, loss, round):
        self.train.info(f"======== Client {cid} Epoch {ep} Training Metrics Global Round {round} ========")
        self.train.info(f"  Loss: {loss}")
        self.train.info(f"  ACC: {acc}")

    def log_validate_class(self, id, acc, loss, round, t):
        self.validate.info(f"======== Model {id} {t} Validation Metrics Global Round {round} ========")
        self.validate.info(f"  Loss: {loss}")
        self.validate.info(f"  ACC: {acc}")

    def log_test_class(self, acc, auc, pr_auc, pr, re, f1, loss, round):
        self.test.info(f"======== Global Model Testing Metrics Global Round {round} ========")
        self.test.info(f"  Loss: {loss}")
        self.test.info(f"  ACC: {acc}")
        self.test.info(f"  ROC-AUC: {auc}")
        self.test.info(f"  Precision-Recall AUC: {pr_auc}")
        self.test.info(f"  Best-F1: {f1}")
        self.test.info(f"  Precision @ Best-F1: {pr}")
        self.test.info(f"  Recall @ Best-F1: {re}")

