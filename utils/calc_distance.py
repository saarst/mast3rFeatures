import numpy as np
import os
from scipy import linalg
import torch

def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).

    Stable version by Dougal J. Sutherland.

    Params:
    -- mu1   : Numpy array containing the activations of a layer of the
               inception net (like returned by the function 'get_predictions')
               for generated samples.
    -- mu2   : The sample mean over activations, precalculated on an
               representative data set.
    -- sigma1: The covariance matrix over activations for generated samples.
    -- sigma2: The covariance matrix over activations, precalculated on an
               representative data set.

    Returns:
    --   : The Frechet Distance.
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert (
        mu1.shape == mu2.shape
    ), "Training and test mean vectors have different lengths"
    assert (
        sigma1.shape == sigma2.shape
    ), "Training and test covariances have different dimensions"

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = (
            "fid calculation produces singular product; "
            "adding %s to diagonal of cov estimates"
        ) % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError("Imaginary component {}".format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean


def calculate_mu_sigma(features):
    # calculate "FDD" - Frechet Dust3r Distance:
    mu = np.mean(features, axis=0)
    sigma = np.cov(features, rowvar=False)
    return mu, sigma


# The bandwidth parameter for the Gaussian RBF kernel. See the paper for more
# details.
_SIGMA = 1
# The following is used to make the metric more human readable. See the paper
# for more details.
_SCALE = 10

def mmd_rbf(x, y):
    """Memory-efficient MMD implementation in JAX.

    This implements the minimum-variance/biased version of the estimator described
    in Eq.(5) of
    https://jmlr.csail.mit.edu/papers/volume13/gretton12a/gretton12a.pdf.
    As described in Lemma 6's proof in that paper, the unbiased estimate and the
    minimum-variance estimate for MMD are almost identical.

    Note that the first invocation of this function will be considerably slow due
    to JAX JIT compilation.

    Args:
      x: The first set of embeddings of shape (n, embedding_dim).
      y: The second set of embeddings of shape (n, embedding_dim).

    Returns:
      The MMD distance between x and y embedding sets.
    """
    x = torch.from_numpy(x)
    y = torch.from_numpy(y)

    x_sqnorms = torch.diag(torch.matmul(x, x.T))
    y_sqnorms = torch.diag(torch.matmul(y, y.T))

    gamma = 1 / (2 * _SIGMA**2)
    k_xx = torch.mean(
        torch.exp(-gamma * (-2 * torch.matmul(x, x.T) + torch.unsqueeze(x_sqnorms, 1) + torch.unsqueeze(x_sqnorms, 0)))
    )
    k_xy = torch.mean(
        torch.exp(-gamma * (-2 * torch.matmul(x, y.T) + torch.unsqueeze(x_sqnorms, 1) + torch.unsqueeze(y_sqnorms, 0)))
    )
    k_yy = torch.mean(
        torch.exp(-gamma * (-2 * torch.matmul(y, y.T) + torch.unsqueeze(y_sqnorms, 1) + torch.unsqueeze(y_sqnorms, 0)))
    )

    return _SCALE * (k_xx + k_yy - 2 * k_xy)


def mmd_poly(x, y, degree=3, coef=1):

    x = torch.from_numpy(x)
    y = torch.from_numpy(y)
    assert x.shape[1] == y.shape[1]

    gamma = 1 / x.shape[1]

    k_xx = (gamma * torch.matmul(x,x.T) + coef) ** degree
    A = (k_xx.sum() - torch.diag(k_xx).sum()) / (x.shape[0] * (x.shape[0]-1) )

    k_yy = (gamma * torch.matmul(y,y.T) + coef) ** degree
    C = (k_yy.sum() - torch.diag(k_yy).sum()) / (y.shape[0] * (y.shape[0]-1) )

    k_xy = (gamma * torch.matmul(x, y.T) + coef) ** degree 
    B = 2 * k_xy.sum() / (x.shape[0] * y.shape[0])

    return A - B + C


def distances(features1, features2):

    mu1, sigma1 = calculate_mu_sigma(features1)
    mu2, sigma2 = calculate_mu_sigma(features2)

    FD = calculate_frechet_distance(mu1, sigma1, mu2, sigma2)
    MMD_RBF = mmd_rbf(features1, features2).numpy().item()
    MMD_POLY = mmd_poly(features1, features2).numpy().item()

    return FD, MMD_RBF, MMD_POLY