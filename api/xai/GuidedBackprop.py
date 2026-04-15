import torch
import torch.nn.functional as F

class GuidedBackprop():
    """
    This class applies Guided Backpropagation to the provided model 
    for 1 input sample. Guided Backpropagation is applied here as described
    in Striving for Simplicity: The All Convolutional Net, by Jost Tobias 
    Springenberg, Alexey Dosovitskiy, Thomas Brox, Martin Riedmiller.
    
    This class modifies the backpropagation behaviour of the ReLU layers
    of the input model.
    """
    def __init__(self, model):
        
        # Copy the model to avoid messing up with other uses
        self.model = model
        
        # Operation for applying Guided Backpropagation
        self.guided_backprop = lambda module, grad_in, grad_out: (F.relu(grad_in[0]),)
        
        # List of attached hooks
        self.hooks = []
        
        # Ensure ReLUs are inplace False and attach hooks to ReLUs
        self.change_inplace()
        self.register_hooks()

        
    def change_inplace(self, to:bool=False) -> None:
        """
        Change all the ReLU layers in the model to have
        inplace=False. This prevents mistakes when correcting
        the gradients when applying guided backpropagation.

        Args:
            to (bool, optional): Value of the inplace attribute.
            Defaults to False.
        """
        for _, module in self.model.named_modules():
            if isinstance(module, torch.nn.ReLU):
                setattr(module, 'inplace', to)


    def register_hooks(self):
        """
        Register forward and backward hooks and save them in a list
        so that they can be removed when necessary.
        """
        for _, module in self.model.named_modules():
            if isinstance(module, torch.nn.ReLU):
                self.hooks += [module.register_full_backward_hook(self.guided_backprop)]
                
        
    def remove_hooks(self) -> None:
        """Remove all hooks applied by this class."""
        for hook in self.hooks:
            hook.remove()
            
    def compute(self, X:tuple) -> torch.tensor:
        """
        Do forward and backward pass. Backward pass using the modified backward
        computations.

        Args:
            X (tuple): Input for the model. The first element should be the
            age and sex data. The second element should be the ECG data.

        Returns:
            torch.tensor: Guided gradients of the maximum logit output of the model
            w.r.t. to the input ECG.
        """
        
        # Activate gradient computation for the ECG data
        age_sex, ecg = X
        ecg = ecg.detach()
        ecg.requires_grad_()
        
        # Forward and backward pass:
        self.model.zero_grad()
        logits = self.model((age_sex, ecg))
        logits.max().backward()
        
        # Return absolute values of importances
        return ecg.grad.squeeze().abs()