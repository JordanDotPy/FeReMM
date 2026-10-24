import torch
import torch.nn as nn
from train import engine_feremmclass as efc
import model_layers as ml
import feremm_utils as fu
import feremm_policy as fp
from collections import defaultdict
import copy


class FeReMM():
    def __init__(self, global_model, global_optimizer, client_models, client_optimizers, model_name,
                 policy_model, policy_optimizer, layers, dataset, num_clients, device, log, te, bad_client=False):

        self.dataset = dataset
        self.model_name = model_name
        self.num_clients = num_clients
        self.bad_client = bad_client
        self.gm = global_model
        self.go = global_optimizer
        self.cms = client_models
        self.cos = client_optimizers
        self.policy = policy_model
        self.po = policy_optimizer
        self.layers = layers
        self.device = device
        self.epochs = te
        self.num_actions = 20
        self.global_iterations = 20
        self.logger = log


    def _aggregate_models(self, data, round):
        merge_idx = data.policy_outputs["merge_idx"]
        print(f"MERGE IDX: {merge_idx}")
        mc = [data.policy_inputs["indexes"][id].item() for id, x in enumerate(merge_idx) if x == 1]
        print(f"MERGE_PREDICTION: {mc}")
        candidate_states = [data.model_states[id] for id in mc]
        candidate_datasizes = [data.data_sizes[id] for id in mc]

        if candidate_states and candidate_datasizes:
            self.gm = fu.merge_clients(self.gm, candidate_states, candidate_datasizes)
        return mc


    def _update_policy(self, data, round):
        policy_logits = data.policy_outputs["logits"]
        merge_action = data.policy_outputs["merge_idx"]

        policy_loss, policy_entropy = fp.update_policy(self.policy, self.po,
                                                       data.policy_inputs,
                                                       policy_logits,
                                                       merge_action,
                                                       data.policy_reward)
        self.logger.log_policy(policy_loss, round)
        return policy_loss, policy_entropy


    def _execute(self, cdls, vdl, tdl, loss_fn, plotter):
        for round in range(self.global_iterations):

            data = fu.FeReMMStruct()
            plotter.client_validation_trajectory[round] = {}
            plotter.client_training_trajectory[round] = {}

            for id, dl in zip(self.cms.keys(), cdls):
                 self.cms[id], train_logits = efc.train(self.cms[id], dl, self.cos[id], loss_fn, log=self.logger,
                                                        device=self.device, epochs=self.epochs,
                                                        cid=id, round=round, struct=data)
                 data.data_sizes[id] = len(dl.dataset)
                 plotter.client_training_trajectory[round][id] = train_logits

            data.model_states["gm"] = copy.deepcopy(self.gm.state_dict())
            loss_before, acc_before, auc_before, grads, _ = efc.validate(self.gm, vdl, self.go, loss_fn, self.device, "gm",
                                                                         layer=self.layers, log=self.logger, round=round,
                                                                         struct=data, hook="premerge")
            data.acc_before["gm"] = acc_before
            data.auc_before["gm"] = auc_before
            data.loss_before["gm"] = loss_before

            data.loss_before["gm"] = loss_before
            data.grad_tracker["gm"]["premerge"] = grads
            fu.average_activations(data, "gm", "premerge")

            # implement Scaling Attack within Bad Clients
            if self.bad_client:
                scale_attack_flag = [0, 0, 0, 1] * 5
                if scale_attack_flag[round] == 1:
                    print(f"===== SCALE ATTACK ROUND =====")
                    b = list(self.cms.keys())[-self.bad_client:]
                    for id in b:
                        print(f"SCALING ATTACK ON CLIENT {id}")
                        self.cms[id] = fu.scaling_attack(self.gm, self.cms[id], self.num_clients)

            for id in self.cms.keys():
                data.model_states[id] = {
                    key: value.detach().clone().to(self.device)
                    for key, value in self.cms[id].state_dict().items()
                }

            for id, client in self.cms.items():
                val_loss, val_acc, val_auc, grads, val_logits = efc.validate(client, vdl, self.cos[id], loss_fn, self.device, id, layer=self.layers,
                                                                             log=self.logger, round=round, struct=data, hook="premerge")
                data.acc_before[id] = val_acc
                data.auc_before[id] = val_auc

                plotter.val_client_acc[id].append(val_acc)
                plotter.val_client_auc[id].append(val_auc)

                data.loss_before[id] = val_loss
                plotter.val_client_loss[id].append(val_loss)
                if val_logits is not None: plotter.client_validation_trajectory[round][id] = val_logits

                data.grad_tracker[id]["premerge"] = grads
                fu.average_activations(data, id, "premerge")

            act_cos_results, act_kl_results = fu.activation_similarity(data.act_tracker)
            grad_cos_results, grad_mag_results = fu.gradient_similarity(data.grad_tracker)

            fp.create_state(data, act_cos_results, act_kl_results, grad_cos_results, grad_mag_results,
                            data.acc_before, data.loss_before, self.cms.keys(), self.layers,
                            self.device)
            data.client_updates = fu.compute_client_update(self.gm, self.cms, data.client_updates, self.layers)

            bcp = 0
            for _ in range(self.num_actions):
                # Randomly Shuffle State and Action Space
                perm = torch.randperm(data.policy_inputs["indexes"].shape[0], device=self.device, dtype=torch.long)
                data.policy_inputs["indexes"] = data.policy_inputs["indexes"][perm]
                print(f"SHUFFLE ORDER: {data.policy_inputs['indexes']}")
                data.policy_inputs["grad"] = data.policy_inputs["grad"][perm, :, :]
                data.policy_inputs["act"] = data.policy_inputs["act"][perm, :, :]
                data.policy_inputs["metric"] = data.policy_inputs["metric"][perm, :]
                data.policy_inputs["loss"] = data.policy_inputs["loss"][perm, :]

                # Policy Forwardpass
                data.policy_outputs["logits"], data.policy_outputs["merge_idx"] = self.policy(data.policy_inputs, self.device)
                current_action = self._aggregate_models(data, round)

                if self.bad_client and scale_attack_flag[round] == 1:
                    for bid in list(self.cms.keys())[-self.bad_client:]:
                        if bid in current_action:
                            bcp += 1

                # Run another validation round to see differences before and after merging
                data.act_tracker["gm"]["postmerge"] = defaultdict(list)
                loss_after, acc_after, auc_after, grads, val_logits = efc.validate(self.gm, vdl, self.go, loss_fn, self.device, "gm", layer=self.layers,
                                                                                   log=self.logger, round=round, struct=data, hook="postmerge")
                data.acc_after = acc_after
                data.auc_after = auc_after
                data.loss_after = loss_after

                data.grad_tracker["gm"]["postmerge"] = grads
                fu.average_activations(data, "gm", "postmerge")

                data.policy_reward = fp.compute_reward(data, current_action, self.device)

                current_reward = data.policy_reward.mean().item()
                current_eval = fp.compute_eval(data)
                if data.best_action["action"] is None or current_eval > data.best_action["eval"]:
                    data.best_action["action"] = current_action
                    data.best_action["reward"] = current_reward
                    data.best_action["eval"] = current_eval
                    data.best_action["state_dict"] = copy.deepcopy(self.gm.state_dict())
                    data.best_action["acc"] = data.acc_after
                    data.best_action["auc"] = data.auc_after
                    data.best_action["loss"] = data.loss_after
                    plotter.client_validation_trajectory[round]["gm"] = val_logits

                pol_loss, pol_ent = self._update_policy(data, round)
                plotter.policy_entropy.append(pol_ent)
                plotter.policy_loss.append(pol_loss)
                plotter.delta_acc.append(data.acc_after - data.acc_before["gm"])
                plotter.delta_auc.append(data.auc_after - data.auc_before["gm"])
                plotter.delta_loss.append(data.loss_before["gm"] - data.loss_after)
                plotter.policy_reward.append(current_reward)
                self.gm.load_state_dict(data.model_states["gm"])

            if self.bad_client and scale_attack_flag[round] == 1:
                plotter.bad_client_prediction.append(bcp / self.num_actions)
            self.logger.log_merges(data.best_action["action"], round)
            self.logger.log_rewards(data.best_action["reward"], round)

            for id in self.cms.keys():
                plotter.client_distance[id].append(fu.compute_total_parameter_distance(self.cms[id], self.gm))

            # Load best merge into global model
            self.gm.load_state_dict(data.best_action["state_dict"])

            # Run Testing for Final Evaluation
            loss, acc, auc, pr, re = efc.evaluate(self.gm, tdl, loss_fn, self.device,
                                                  log=self.logger, round=round)

            plotter.test_loss.append(loss)
            plotter.test_acc.append(acc)
            plotter.test_auc.append(auc)
            plotter.test_pr = pr
            plotter.test_re = re

            # Reset Model & Optimizers to new merged model
            for id, client in self.cms.items():
                client.load_state_dict(self.gm.state_dict())
                self.cos[id] = torch.optim.AdamW(client.parameters(), lr=1e-5, weight_decay=1e-4)

