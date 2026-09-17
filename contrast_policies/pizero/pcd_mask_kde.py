import math
import torch


def gaussian_kernel_torch(x):
    return (1 / math.sqrt(2 * math.pi)) * torch.exp(-0.5 * x ** 2)

def scott_rule_torch(data):
    return 1.06 * torch.std(data, dim=-1) * data.shape[-1] ** (-1 / 5)

def kde_torch(x, data, bandwidth, kernel_func):
    # x -> (N, B) -> (N, B, 1)
    x = x.unsqueeze(-1)
    # data -> (N, B) -> (N, 1, B)
    data = data.unsqueeze(1)
    # (N, B, 1) - (N, 1, B) -> (N, B, B)
    bandwidth_ = bandwidth.unsqueeze(-1).unsqueeze(-1) if not isinstance(bandwidth, float) else bandwidth
    kernel_values = kernel_func((x - data) / bandwidth_)
    # (N, B, B) -> (N, B)
    bandwidth_ = bandwidth.unsqueeze(-1) if not isinstance(bandwidth, float) else bandwidth
    density_estimation = kernel_values.sum(dim=-1) / (data.shape[-1] * bandwidth_)
    return density_estimation

class PCDMaskKDE:
    def __init__(self,
                 alpha=0.2,
                 bandwidth_factor=1.0,
                 keep_threshold=0.5):
        self.alpha = alpha
        self.bandwidth_factor = bandwidth_factor
        self.keep_threshold = keep_threshold
    
    def __call__(self, data, contrast_data):
        return self.decode(data, contrast_data)
        
    def decode(self, data, contrast_data):
        B, T, D = data.shape
        N = T * D
        data = data.reshape(B, N).permute(1, 0)
        contrast_data = contrast_data.reshape(B, N).permute(1, 0)
        
        bandwidth = self.bandwidth_factor * scott_rule_torch(data)
        prob = kde_torch(data, data, bandwidth, gaussian_kernel_torch)
        
        contrast_bandwidth = self.bandwidth_factor * scott_rule_torch(contrast_data)
        contrast_prob = kde_torch(data, contrast_data, contrast_bandwidth, gaussian_kernel_torch)
        
        contrast_factor = prob / contrast_prob
        final_prob = prob * contrast_factor ** self.alpha
        
        final_prob[prob < self.keep_threshold * prob.max(dim=-1, keepdims=True).values] = 0.0
        final_prob = final_prob / final_prob.max(dim=-1, keepdims=True).values * prob.max(dim=-1, keepdims=True).values
        
        # [N, B] -> [N,] -> [T, D]
        sample = data[range(N), prob.argmax(dim=-1)].reshape(T, D)
        contrast_sample = data[range(N), final_prob.argmax(dim=-1)].reshape(T, D)

        # update x, y, z, roll, pitch, yaw
        sample[:, :6] = contrast_sample[:, :6]
        return sample.unsqueeze(0)
