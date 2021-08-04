import torch
import torch.nn as nn
import torch.nn.functional as F
from math import pi, log
from torch.utils import data
from torch import tensor


class MultivariateGaussianMDN(nn.Module):
    """
    For a documented version of this code, see: 
    https://github.com/mackelab/pyknos/blob/main/pyknos/mdn/mdn.py
    """

    def __init__(
        self,
        features,
        hidden_net,
        num_components,
        hidden_features,
    ):

        super().__init__()

        self._features = features
        self._num_components = num_components
        
        self._hidden_net = hidden_net
        self._logits_layer = nn.Linear(hidden_features, num_components)
        self._means_layer = nn.Linear(hidden_features, num_components * features)
        self._unconstrained_diagonal_layer = nn.Linear(
            hidden_features, num_components * features
        )

    def get_mixture_components(
        self, context
    ):
        h = self._hidden_net(context)

        logits = self._logits_layer(h)
        logits = logits - torch.logsumexp(logits, dim=1).unsqueeze(1)
        means = self._means_layer(h).view(-1, self._num_components, self._features)

        log_variances = self._unconstrained_diagonal_layer(h).view(
            -1, self._num_components, self._features
        )
        variances = torch.exp(log_variances)

        return logits, means, variances

def mog_log_prob(theta, logits, means, variances):

    _, _, theta_dim = means.size()
    theta = theta.view(-1, 1, theta_dim)

    log_cov_det = -0.5*torch.log(torch.prod(variances, dim=2))

    a = logits
    b = -(theta_dim / 2.0) * log(2 * pi)
    c = log_cov_det
    d1 = theta.expand_as(means) - means
    precisions = 1.0 / variances
    exponent = torch.sum(d1 * precisions * d1, dim=2)
    exponent = tensor(-0.5) * exponent

    return torch.logsumexp(a + b + c + exponent, dim=-1)

def mog_sample(logits, means, variances):

    coefficients = F.softmax(logits, dim=-1)

    choices = torch.multinomial(
        coefficients, num_samples=1, replacement=True
    ).view(-1)

    # Select first batch-position.
    chosen_means = means[0, choices, :]
    chosen_variances = variances[0, choices, :]

    _, _, output_dim = means.shape
    standard_normal_samples = torch.randn(output_dim, 1)
    zero_mean_samples = standard_normal_samples * torch.sqrt(chosen_variances)
    samples = chosen_means + zero_mean_samples

    return samples