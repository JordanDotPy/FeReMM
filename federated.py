import torch
import torch.nn as nn
import feremm_utils as fu
from train import engine_class, engine_proxclass


class FederatedLearning():
    def __init__(self, global_model, client_models, client_optimizers,
                 model_name, num_clients, dataset, framework, bad_client,
                 layers, te, device):

        self.num_clients = num_clients
        self.bad_client = bad_client
        self.dataset = dataset
        self.gm = global_model
        self.cms = client_models
        self.cos = client_optimizers
        self.model_name = model_name
        self.framework = framework
        self.device = device
        self.global_iterations = 20
        self.epochs = te
        self.layers = layers

    def _execute(self, cdls, vdl, test_dl, loss_fn, plotter, log):
        for i in range(self.global_iterations):
            client_data_sizes = []

            global_params = {
                name: param.detach().clone().to(self.device)
                for name, param in self.gm.named_parameters()
            }
            global_state = self.gm.state_dict()

            plotter.client_validation_trajectory[i] = {}
            plotter.client_training_trajectory[i] = {}
            for id, dl in zip(self.cms.keys(), cdls):
                if self.framework == "fedavg":
                    self.cms[id], train_logits = engine_class.train(self.cms[id], self.cos[id], loss_fn, dl, vdl, cid=id, round=i,
                                                                    log=log, plot=plotter, device=self.device, epochs=self.epochs)
                else:
                     self.cms[id], train_logits = engine_proxclass.train(self.cms[id], self.cos[id], loss_fn, dl, vdl, log=log, plot=plotter, cid=id,
                                                                         round=i, device=self.device, global_params=global_params, epochs=self.epochs)

                client_data_sizes.append(len(dl.dataset))
                plotter.client_training_trajectory[i][id] = train_logits

            # Implement Scale Attack within Bad Clients
            if self.bad_client:
                scale_attack_flag = [0, 0, 0, 1] * 5
                if scale_attack_flag[i]:
                    b = list(self.cms.keys())[-self.bad_client:]
                    for id in b:
                        self.cms[id] = fu.scaling_attack(self.gm, self.cms[id], self.num_clients)
                    # Used to only plot bad client model logit trajectories, ensure integrity of RNG state is secure.
                    cpu_rng_state = torch.get_rng_state()
                    cuda_rng_state = torch.cuda.get_rng_state_all()
                    for id in b:
                        _, _, _, _, _, val_logit = engine_class.evaluate(self.cms[id], loss_fn, vdl, self.device)
                        plotter.client_validation_trajectory[i][id] = val_logit
                    torch.set_rng_state(cpu_rng_state)
                    torch.cuda.set_rng_state_all(cuda_rng_state)

            # Distance between locally trained clients and the pre-merging global model
            for id in self.cms.keys():
                plotter.client_distance[id].append(fu.compute_total_parameter_distance(self.cms[id], self.gm))

            client_states = [m.state_dict() for m in self.cms.values()]
            total_samples = sum(client_data_sizes)
            weights = [n / total_samples for n in client_data_sizes]

            self._model_aggregation(client_states, global_state, weights)

            if self.framework == "fedavg":
                # Used to only plot global model logit trajectories, ensure integrity of RNG state is secure.
                cpu_rng_state = torch.get_rng_state()
                cuda_rng_state = torch.cuda.get_rng_state_all()
                _, _, _, _, _, val_logit = engine_class.evaluate(self.gm, loss_fn, vdl, self.device)
                plotter.client_validation_trajectory[i]["gm"] = val_logit
                torch.set_rng_state(cpu_rng_state)
                torch.cuda.set_rng_state_all(cuda_rng_state)

                _, _, _, pr, re, _ = engine_class.evaluate(self.gm, loss_fn, test_dl, device=self.device,
                                                           log=log, plot=plotter, round=i)

            else:
                # Used to only plot global model logit trajectories, ensure integrity of RNG state is secure.
                cpu_rng_state = torch.get_rng_state()
                cuda_rng_state = torch.cuda.get_rng_state_all()
                _, _, _, _, _, val_logit = engine_proxclass.evaluate(self.gm, loss_fn, vdl, self.device)
                plotter.client_validation_trajectory[i]["gm"] = val_logit
                torch.set_rng_state(cpu_rng_state)
                torch.cuda.set_rng_state_all(cuda_rng_state)

                _, _, _, pr, re, _ = engine_proxclass.evaluate(self.gm, loss_fn, test_dl, device=self.device,
                                                               log=log, plot=plotter, round=i)

            self._distribute_global_model(global_state)
            plotter.test_pr = pr
            plotter.test_re = re

            # Reset Client Optimizers to new merged model version
            for id, client in self.cms.items():
                self.cos[id] = torch.optim.AdamW(client.parameters(), lr=1e-5, weight_decay=1e-4)


    def _model_aggregation(self, client_states, global_state, weights):
        with torch.no_grad():
            for key in client_states[0].keys():
                if not client_states[0][key].dtype.is_floating_point:
                    continue
                avg = sum(client_states[i][key] * weights[i]
                          for i in range(len(client_states)))
                global_state[key] = avg
            self.gm.load_state_dict(global_state)


    def _distribute_global_model(self, global_state):
        for id, client_model in self.cms.items():
            client_model.load_state_dict(global_state)
