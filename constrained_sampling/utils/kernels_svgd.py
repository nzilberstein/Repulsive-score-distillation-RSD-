import torch
import math
import numpy as np
import lpips
import pdb


class RBFSteinKernel():
    """
    A RBF kernel for use in the SVGD inference algorithm. The bandwidth of the kernel is chosen from the
    particles using a simple heuristic as in reference [1].

    :param float bandwidth_factor: Optional factor by which to scale the bandwidth, defaults to 1.0.
    :ivar float ~.bandwidth_factor: Property that controls the factor by which to scale the bandwidth
        at each iteration.

    References

    [1] "Stein Variational Gradient Descent: A General Purpose Bayesian Inference Algorithm,"
        Qiang Liu, Dilin Wang
    """

    def __init__(self, bandwidth_factor, power):
        """
        :param float bandwidth_factor: Optional factor by which to scale the bandwidth
        """
        self.bandwidth_factor = bandwidth_factor
        self.power = power

    def get_kernel(self, latent_pred_t, dino, model):
        """
        :X and Y are particle inputs with shape (batch_size, n_particles, num_channels, height, width)
        """
        latent_pred_t.requires_grad_(True)
        dino.requires_grad_(True)
        model.vae.decoder.requires_grad_(True)
        dino.train()
        model.vae.train()

        x_pred_z = model.decode_latents(latent_pred_t, stay_on_device=True)
        dino_out = dino(x_pred_z)

        latents_vec = dino_out.view(len(dino_out), -1)
        # N x N x d
        diff = latents_vec.unsqueeze(1) - latents_vec.unsqueeze(0)

        # remove the diag, make distance with shape N x N-1 x 1
        diff = diff[~torch.eye(diff.shape[0], dtype=bool)].view(diff.shape[0], -1, diff.shape[-1])

        # N x N x 1
        distance = torch.norm(diff, p=2, dim=-1, keepdim=True)
        num_images = latents_vec.shape[0]
        h_t = (distance.median(dim=1, keepdim=True)[0]) ** 2 / np.log(
            num_images - 1)
        weights = torch.exp(- (distance ** self.power / h_t))

        grad_phi = 2 * weights * diff / h_t
        grad_phi = grad_phi.sum(dim=1)

        eval_sum = torch.sum(dino_out * grad_phi.detach())
        deps_dx_backprop = torch.autograd.grad(eval_sum, latent_pred_t)[0]
        grad_phi = deps_dx_backprop.view_as(latents)
        K_svgd_z_mat_reg_sum = weights.sum(dim = 1)
        nabla_log = torch.div(grad_phi, K_svgd_z_mat_reg_sum.unsqueeze(-1).unsqueeze(-1))
        noise_pred = et - noise_t - gamma * (1-alpha_prod_t).sqrt() * nabla_log

        return K, grad_K
    
    def get_kernel_particles_LPIPS(self, x, y, dino, grad = True, ad = 1):
        """
        :X and Y are particle inputs with shape (batch_size, n_particles, num_channels, height, width)
        """
        # Reshape X and Y to have shape (batch_size, n_particles, num_channels*height*width)
        x = x.cuda().requires_grad_()

        dino_features = dino(x.view(x.size(0) * x.size(1), x.size(2), x.size(3), x.size(4)))


        # print(torch.norm(x_flat.unsqueeze(2) - y_flat.unsqueeze(1), p = 2, dim=-1))**2
        # self.loss_fn_alex(x.view(x.size(0) * x.size(1), x.size(2), x.size(3), x.size(4)), 
        #                                     y.view(y.size(0) * y.size(1), y.size(2), y.size(3), y.size(4)))   
        # dists = torch.zeros(x.size(0), x.size(1), y.size(1)).cuda()
        # grad_dist = torch.zeros_like(x).cuda()
        # for bs in range(x.size(0)):
        #     for i in range(x.size(1)):
        #         for j in range(y.size(1)):
        #             dists[bs, i, j] = self.loss_fn_alex(x[bs, i, :, :, :].unsqueeze(0), y[bs, j, :, :, :].unsqueeze(0))
        #             grad_dist[0,i,:,:,:] = torch.autograd.grad(dists[bs, i, j], x, create_graph=True)[0][bs, i, :, :, :]

        diff = dino_features.unsqueeze(1) - dino_features.unsqueeze(0)
        dists = torch.norm(diff, p=2, dim=-1, keepdim=True)
            # print(x_flat, y_flat)
            # dists = dists / x_flat.size(-1) # I think is better to normalize by the number of dimensions, otherwise the distances are too big
            # print(dists.shape)
        
        # print(dists)
        if self.bandwidth_factor < 0: # use median trick
            gamma = torch.median(dists)
            if gamma != 0:
                gamma = torch.sqrt(gamma / np.log(dists.size(1)))
                gamma = 1 / gamma**2
            else:
                gamma = 1
        else:
            gamma = self.bandwidth_factor
        
        gamma = gamma * ad
        # Compute the RBF kernel using the squared distances and gamma. 
        # The gradient is (-2/h * K(x_j, x) (x_j - x), so we need to broadcast on the first variable and sum the j terms).
        K = torch.exp(-gamma * dists)
        
        # TODO: CHECK SIGN (I think is ok because we are summing in the component of all negatives)
        # grad_K =  2 * gamma * torch.mul(K.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1), grad_dist.unsqueeze(-1))
        
        dK =  2 * gamma * K * diff 

        # pdb.set_trace()
        grad_K = torch.sum(dK, dim=1)
        del dK, dists, diff
        eval_sum = torch.sum(dino_features * grad_K.detach())
        
        grad_K = torch.autograd.grad(eval_sum, x, create_graph=True)[0]
        
        # TODO: Change this
        K = torch.squeeze(K, -1).unsqueeze(0)
        
        return K, grad_K
    
    def get_kernel(self, x, y, grad = True, ad = 1):
        """
        :X and Y are particle inputs with shape (batch_size, n_particles, num_channels, height, width)
        """
        # Reshape X and Y to have shape (batch_size, n_particles, num_channels*height*width)
        # x_flat = x.view(x.size(0), x.size(1), -1)
        # y_flat = y.view(y.size(0), y.size(1), -1)

        x_flat = x.view(x.size(0), -1)
        y_flat = y.view(y.size(0), -1)

        # print(torch.norm(x_flat.unsqueeze(2) - y_flat.unsqueeze(1), p = 2, dim=-1))**2

        # Compute the pairwise squared Euclidean distances between the samples
        with torch.cuda.amp.autocast():
            dists = torch.cdist(x_flat, y_flat, p=2)**2
            # print(dists)

        if self.bandwidth_factor < 0: # use median trick
            gamma = torch.median(dists)
            # print(dists, gamma)
            if gamma != 0:
                gamma = torch.sqrt(0.5 * gamma / np.log(dists.size(0) + 1))
                gamma = 1 / (2 * gamma**2)
            else:
                gamma = 1
        else:
            gamma = self.bandwidth_factor

        # print(gamma)
        gamma = gamma * ad
        # Compute the RBF kernel using the squared distances and gamma. 
        # The gradient is (-2/h * K(x_j, x) (x_j - x), so we need to broadcast on the first variable and sum the j terms).
        K = torch.exp(-gamma * dists)
        # TODO: CHECK SIGN
        dK = 2 * gamma * K.unsqueeze(-1) * (x_flat.unsqueeze(1) - y_flat.unsqueeze(0))
        grad_K = torch.sum(dK, dim=1)

        return K, grad_K
    

class IdentityKernel():
    """
    An identity kernel.
    """

    def __init__(self):
        """
        :param float bandwidth_factor: Optional factor by which to scale the bandwidth
        """


    def get_kernel_particles(self, x, y, grad = True, ad = 1):
        """
        :X and Y are particle inputs with shape (batch_size, n_particles, num_channels, height, width)
        """
        # Compute the identity matrix
        I = torch.eye(x.size(1)).cuda()
        I = I.reshape((1,x.size(1), x.size(1)))
        K = I.repeat(x.size(0), 1, 1)
        grad_K = torch.zeros_like(x)
        
        return K, grad_K
    
    def get_kernel_particles_LPIPS(self, x, y, grad = True, ad = 1, dino = None):
        """
        :X and Y are particle inputs with shape (batch_size, n_particles, num_channels, height, width)
        """
        # Compute the identity matrix
        I = torch.eye(x.size(1)).cuda()
        I = I.reshape((1,x.size(1), x.size(1)))
        K = I.repeat(x.size(0), 1, 1)
        grad_K = torch.zeros_like(x)
        
        return K, grad_K
    
    def get_kernel(self, x, y, grad = True, ad = 1):
        """
        :X and Y are particle inputs with shape (batch_size, n_particles, num_channels, height, width)
        """
        # Compute the identity matrix
        K = torch.eye(x.size(0)).cuda()
        grad_K = torch.zeros_like(x)
        
        return K, grad_K

def build_kernel_string_input(kernel_type, bandwidth_factor = -1, ad = 1):
    if kernel_type == 'rbf':
        return RBFSteinKernel(bandwidth_factor)
    elif kernel_type == 'identity':
        return IdentityKernel()
    else:
        raise ValueError(f'No kernel named {kernel_type}')

def build_kernel(cfg):
    if cfg.algo.kernel_type == 'rbf':
        return RBFSteinKernel(cfg.algo.bandwidth_factor)
    elif cfg.algo.kernel_type == 'identity':
        return IdentityKernel()
    else:
        raise ValueError(f'No kernel named {cfg.algo.name}')
    
def build_kernel_prior(cfg, bandwidth_factor = -1):
    if cfg.algo.kernel_type == 'rbf':
        return RBFSteinKernel(bandwidth_factor)
    elif cfg.algo.kernel_type == 'identity':
        return IdentityKernel()
    else:
        raise ValueError(f'No kernel named {cfg.algo.name}')