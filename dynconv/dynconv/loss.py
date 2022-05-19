import utils.logger as logger
import math
import torch
import torch.nn as nn


class SparseBoundary:
    def __init__(self, sparse_target, num_epochs, strategy, valid_range=0.33, static_range=0.1):
        self.sparsity_target = sparse_target
        self.num_epochs = num_epochs
        self.strategy = strategy
        self.valid_range = valid_range
        self.static_range = static_range
        assert self.strategy in ["static", "wider", "narrower", "static_range"]

    def update(self, meta):
        if self.strategy == "static":
            return self.sparsity_target, self.sparsity_target
        elif self.strategy == "static_range":
            return min((self.sparsity_target + self.static_range), 1), \
                   max((self.sparsity_target + self.static_range), 0)
        else:
            p = meta['epoch'] / (self.valid_range*self.num_epochs)
            if self.strategy == "wider":
                progress = math.cos(min(max(p, 0), 1) * (math.pi / 2)) ** 2
            elif self.strategy == "narrower":
                progress = math.sin(min(max(p, 0), 1) * (math.pi / 2)) ** 2
            upper_bound = (1 - progress*(1-self.sparsity_target))
            lower_bound = progress*self.sparsity_target
            return upper_bound, lower_bound


class SparsityCriterion(nn.Module):
    ''' 
    Defines the sparsity loss, consisting of two parts:
    - network loss: MSE between computational budget used for whole network and target 
    - block loss: sparsity (percentage of used FLOPS between 0 and 1) in a block must lie between upper and lower bound. 
    This loss is annealed.
    '''

    def __init__(self, sparsity_target, unlimited_lower=False, layer_loss_method="flops", **kwargs):
        super(SparsityCriterion, self).__init__()
        self.sparsity_target = sparsity_target
        self.bound = SparseBoundary(sparsity_target, **kwargs)
        self.lower_unlimited = unlimited_lower
        self.layer_loss_method = layer_loss_method

    def calculate_layer_ratio(self, m_dil, m):
        c = m_dil.active_positions * m_dil.flops_per_position + m.active_positions * m.flops_per_position
        t = m_dil.total_positions * m_dil.flops_per_position + m.total_positions * m.flops_per_position
        if self.layer_loss_method == "flops":
            try:
                layer_perc = c / t
            except RuntimeError:
                layer_perc = torch.true_divide(c, t)
        elif self.layer_loss_method == "later_mask":
            layer_perc = m.hard.sum()/m.hard.numel()
        elif self.layer_loss_method == "front_mask":
            layer_perc = m_dil.hard.sum()/m_dil.hard.numel()
        else:
            raise NotImplementedError(self.layer_loss_method)
        return layer_perc, c, t

    def forward(self, meta):
        upper_bound, lower_bound = self.bound.update(meta)
        layer_percents, mask_percents = [], []
        loss_block = torch.tensor(.0).to(device=meta['device'])
        cost, total = torch.tensor(.0).to(device=meta['device']), torch.tensor(.0).to(device=meta['device'])

        for i, mask in enumerate(meta['masks']):
            m_dil, m = mask['dilate'], mask['std']
            mask_percents.append(m.hard.sum()/m.hard.numel())

            # c = m_dil.active_positions * m_dil.flops_per_position + \
            #     m.active_positions * m.flops_per_position
            # t = m_dil.total_positions * m_dil.flops_per_position + \
            #     m.total_positions * m.flops_per_position
            #
            # try:
            #     layer_perc = c / t
            # except RuntimeError:
            #     layer_perc = torch.true_divide(c, t)
            layer_perc, c, t = self.calculate_layer_ratio(m_dil, m)

            layer_percents.append(layer_perc)
            # logger.add('layer_perc_'+str(i), layer_perc.item())
            assert layer_perc >= 0 and layer_perc <= 1, layer_perc
            loss_block += max(0, layer_perc - upper_bound)**2  # upper bound
            if not self.lower_unlimited:
                loss_block += max(0, lower_bound - layer_perc)**2  # lower bound

            cost += c
            total += t

        perc = cost/total
        assert perc >= 0 and perc <= 1, perc
        loss_block /= len(meta['masks'])
        loss_network = (perc - self.sparsity_target)**2

        # logger.add('upper_bound', upper_bound)
        # logger.add('lower_bound', lower_bound)
        # logger.add('cost_perc', perc.item())
        # logger.add('loss_sp_block', loss_block.item())
        # logger.add('loss_sp_network', loss_network.item())
        return loss_network, loss_block, layer_percents
