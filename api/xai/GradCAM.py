import torch
import torch.nn.functional as F
import numpy as np

from scipy.interpolate import interp1d

class GradCAM():
    """
    This class applies Grad-CAM to the provided model for 1 input sample.
    Grad-CAM is applied here as described in Grad-CAM: Visual Explanations 
    from Deep Networks via Gradient-based Localization, by Ramprasaath R. 
    Selvaraju, Michael Cogswell, Abhishek Das, Ramakrishna Vedantam, Devi 
    Parikh, Dhruv Batra.
    
    This class is only for applying Grad-CAM to 1 dimensional convolutional
    layers.
    """
    
    def __init__(self):
        # Activations of the target layer and gradients w.r.t. to the
        # activations of the target layer that will be used to compute the
        # coarse map of feature maps importance:
        self.activations = None
        self.gradients   = None
        
        # List to store hooks that have been applied to a layer so that
        # they can be removed afterwards:
        self.hooks = []
        
        
    def forward_hook(self, target_layer, X, A):
        """Hook to store activations of the target layer.

        Args:
            target_layer (torch.nn.Conv1d): layer to which the hook will be attached.
            X (tensor): Input to the layer.
            A (tensor): Output of the layer.
        """
        self.activations = A.detach().squeeze()
        
        
    def backward_hook(self, target_layer, grad_in, grad_out):
        """Hook to store the gradients w.r.t. the activations
        of the target layer.

        Args:
            target_layer (torch.nn.Conv1d): layer to which the hook will be attached.
            grad_in (tuple): Gradients w.r.t. the input of the layer.
            grad_out (tuple): Gradients w.r.t. the output of the layer.
        """
        self.gradients = grad_out[0].detach().squeeze()
        
    
    def register_hooks(self, target_layer:torch.nn.Conv1d) -> None:
        """
        Register forward and backward hooks and save them in a list
        so that they can be removed when necessary.

        Args:
            target_layer (torch.nn.Module): Layer of the model to which
            the hooks will be applied.
        """
        
        if not isinstance(target_layer, torch.nn.Conv1d):
            raise Exception("This class only supports Grad-CAM applied to"
                            "1D convolutional layers.")
            
        self.hooks += [target_layer.register_forward_hook(self.forward_hook)]
        self.hooks += [target_layer.register_full_backward_hook(self.backward_hook)]
        
        
    
    def remove_hooks(self) -> None:
        """Remove all hooks applied by this class."""
        for hook in self.hooks:
            hook.remove()
            
    def compute(self, interp_samples:int=0, interp_type:str = 'quadratic') -> np.ndarray:
        """
        Using activations and gradients from the target layer, compute a coarse map
        of the feature maps importance.

        Args:
            interp_samples (int, optional): Number of samples to interpolate 
            the coarse map to. Usually, this should match the input of the model.
            Defaults to 0.
            interp_type (str, optional): Type of interpolation method. The options
            available are those for the 'kind' parameter in the the scipy.interp1d
            class. Defaults to 'quadratic'.

        Returns:
            np.ndarray: Coarse map of the feature maps importance.
        """
        
        # Compute the neuron importance weights
        alphas = torch.mean(self.gradients, dim=1) # n_channels x 1

        # Compute weighted feature map
        coarse_map = alphas[:, None]*self.activations # n_channels x n_samples
        coarse_map = F.relu(torch.sum(coarse_map, axis=0))    # n_samples x 1
        coarse_map = coarse_map.numpy()
        
        # Interpolate to match specified length
        if interp_samples > coarse_map.shape[0]:
            coarse_axis = np.linspace(0, interp_samples, coarse_map.shape[0])
            interp = interp1d(coarse_axis, coarse_map, kind=interp_type)
            coarse_map = interp(np.arange(interp_samples))

        return coarse_map
    