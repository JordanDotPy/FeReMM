import sys
import argparse


class Parser():
    def __init__(self, framework):
        self.framework = framework
        self.parser = argparse.ArgumentParser()

        if self.framework == "centralized":
            self.parser_centralized()
        elif self.framework in ["fedavg", "fedprox", "feremm"]:
            self.parser_federated()
        else:
            sys.exit("PARSER ERROR: Invalid Framework Name...")

        self.parser.add_argument("--save_data", action="store_true", help="save the run's .pt result files to evaluation/saved_data/")
        self.parser.add_argument("--save_plots", action="store_true", help="save the run's plots to evaluation/plots/")

        args = self.parser.parse_args()
        self.save_data = args.save_data
        self.save_plots = args.save_plots

        if self.framework == "centralized":
            self.clean_args = tuple(self.parser_check(args))
        else:
            self.clean_args = tuple(self.parser_check_federated(args))


    def parser_federated(self):
        self.parser.add_argument("--dataset", type=str, default="")
        self.parser.add_argument("--model_name", type=str, default="")
        self.parser.add_argument("--num_clients", type=int, default=0)
        self.parser.add_argument("--bad_clients", type=int, default=0)


    def parser_centralized(self):
        self.parser.add_argument("--model_name", type=str, default="")
        self.parser.add_argument("--dataset", type=str, default="")


    def parser_check(self, args):
        if args.dataset.lower() not in ['retina_mnist', 'breast_mnist', 'tissue_mnist', 'derma_mnist', 'oct_mnist']:
            sys.exit("Invalid Dataset, choose from: retina_mnist, breast_mnist, tissue_mnist, derma_mnist, oct_mnist")

        if args.model_name.lower() not in ['resnet50', 'resnet18']:
            sys.exit("Invalid Model Type Given, exiting...")

        return args.model_name.lower(), args.dataset.lower()


    def parser_check_federated(self, args):
        model_name, dataset = self.parser_check(args)

        if args.num_clients < 3:
            sys.exit("Invalid Number of Clients, pick a number >= 3.")

        if not 0 <= args.bad_clients < args.num_clients:
            sys.exit("Invalid Number of Bad Clients, pick a number from 0 to num_clients - 1.")
        return model_name, dataset, args.num_clients, args.bad_clients




