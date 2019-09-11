from collections import defaultdict
from math import ceil

import numpy as np
import torch

import pytorch_privacy.utils.torch_nest_utils as nest
from pytorch_privacy.analysis import QueryWithLedger
from pytorch_privacy.dp_query import GaussianDPQuery
from .wrapper_optimizer import WrapperOptimizer
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class DPOptimizer(WrapperOptimizer):
    """
        Optimiser wrapper for doing differential privacy on any gradient based PyTorch optimiser.
    """
    def __init__(self,
                 optimizer,
                 model,
                 loss_per_example,
                 dp_sum_query,
                 num_microbatches=None):
        """
        Wraps a PyTorch optimiser with a given model and loss function. To do differential privacy, we need to take per
        example gradients, so the loss function must be per example (i.e. a vector).
        :param optimizer: A base PyTorch to use for the optimisation
        :param model: An instance of nn.Module that computes the outputs y_hat from an input x
        :param loss_per_example: A vector loss function that takes y_true and y_hat to comupte the loss
        :param dp_sum_query: An instance of a DP query to track privacy loss with
        :param num_microbatches: A number of microbatches to split the data into. If None, do not use microbatching
        """
        self.optimizer = optimizer
        self.model = model
        self.loss_per_example = loss_per_example
        self.dp_sum_query = dp_sum_query
        self.num_microbatches = num_microbatches
        self._summary_value = 0

        self._global_parameters = self.dp_sum_query.initial_global_state()
        self._derived_records_data = defaultdict(list)

    def fit_batch(self, x: torch.Tensor, y: torch.Tensor):
        """
        Perform a single gradient optimisation step of the model with differentially private gradients
        :param x: The model inputs
        :param y: The model outputs
        :return: The total loss of (x, y) at this step
        """
        # Compute the per datapoint loss
        loss = self.loss_per_example(self.model(x), y)

        # Grab the param groups to be optimised form the optimiser
        param_groups = self.optimizer.param_groups

        # Get the correct shape gradient tensors to then set to the initial
        # state of the sample. Often all zero for zero gradients.
        sample_state = self.dp_sum_query.initial_sample_state(
            nest.parameters_to_tensor_groups(param_groups, 'data')
        )

        # Get the parameters for doing the dp query on this sample of data
        sample_params = self.dp_sum_query.derive_sample_params(self._global_parameters)

        # compute the appropriate microbatch size
        microbatch_size = 1 if self.num_microbatches is None else ceil(loss.shape[0] / self.num_microbatches)

        # Chunk the losses into microbatches
        microbatches_losses = loss.split(microbatch_size, dim=0)

        def process_microbatch(losses, sample_state):
            # Zero out the current gradients on parameters
            self.optimizer.zero_grad()
            # Take the mean of the losses, so that we compute the mean gradient
            microbatch_loss = losses.mean(dim=0)
            # Compute the gradients for this microbacth
            microbatch_loss.backward(retain_graph=True)
            # Extract the gradients from the parameters of interest
            record = self.get_grads(param_groups)
            # Accumulate the gradients onto the current sample stack, applying what ever DP operations are required
            sample_state = self.dp_sum_query.accumulate_record(sample_params, sample_state, record)
            # Gather any information of interest from the query
            derived_record_data = self.dp_sum_query.get_record_derived_data()
            return sample_state, derived_record_data

        self._derived_records_data = defaultdict(list)

        # For each microbatch, process the gradients
        for losses in microbatches_losses:
            sample_state, derived_record_data = process_microbatch(losses,
                                                                   sample_state)  # accumulate up the clipped microbatch gradients

            for k, v in derived_record_data.items():
                self._derived_records_data[k].append(v)

        self._derived_records_data = dict(self._derived_records_data)

        for k, v in self._derived_records_data.items():
            # summarise statistics instead
            self._derived_records_data[k] = np.percentile(np.array(v), [10.0, 30.0, 50.0, 70.0, 90.0])
            if k == "l2_norm:":
                p_clip = np.mean(
                    np.array(v) > self._global_parameters.l2_norm_clip.detach().numpy())
                self._summary_value = {"percentage_clipped": p_clip}

        # Finish the DP query, usually by adding noise to the accumulated gradient information
        final_grads, _ = self.dp_sum_query.get_noised_result(sample_state, self._global_parameters)

        # for k, v in self.model.named_parameters():
        #     logger.debug(f"{k} mean_grad {torch.sqrt(torch.mean(v.grad.data ** 2))}")

        # Put the DP gradients onto the parameters
        self.apply_grads(param_groups, grads=final_grads)

        # Finally, take the optimisation step with the DP gradients
        self.optimizer.step()
        self.optimizer.zero_grad()

        # return the loss at the start (for efficiency purposes)
        return torch.sum(loss).detach().numpy()

    def apply_grads(self, param_groups, grads):
        for param_group, grad_group in zip(param_groups, grads):
            for p, grad in zip(param_group['params'], grad_group):
                p.grad.data = grad

    def get_grads(self, param_groups):
        grads = []

        for group in param_groups:
            group_grads = []
            for p in group['params']:
                group_grads.append(p.grad.data.clone().detach())
            grads.append(group_grads)

        return grads

    def get_logged_statistics(self):
        return self._derived_records_data

    def get_step_summary(self):
        return self._summary_value


class DPGaussianOptimizer(DPOptimizer):
    """ Specific Gaussian mechanism optimizer for L2 clipping and noise privacy """

    def __init__(self,
                 l2_norm_clip,
                 noise_multiplier,
                 ledger=None,
                 *args,
                 **kwargs):
        dp_sum_query = GaussianDPQuery(l2_norm_clip, l2_norm_clip * noise_multiplier)

        if ledger:
            dp_sum_query = QueryWithLedger(dp_sum_query, ledger=ledger)

        super().__init__(
            dp_sum_query=dp_sum_query,
            *args,
            **kwargs
        )

    @property
    def ledger(self):
        return self.dp_sum_query.ledger
