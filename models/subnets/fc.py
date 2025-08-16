import torch
import torch.nn as nn
import torch.nn.functional as F

class FcEncoder(nn.Module):
    def __init__(self, input_dim, layers):
        ''' Fully Connect classifier
            fc+relu+bn+dropout， 最后分类128-4层是直接fc的
            Parameters:
            --------------------------
            input_dim: input feature dim
            layers: A list where each element can be an integer (hidden nodes), 
                    a string (activation function), "bn" (batch normalization), 
                    or a float (dropout rate).
        '''
        super().__init__()
        active_fns = {
            "relu": nn.ReLU,
            "leakyrelu": nn.LeakyReLU,
            'tanh': nn.Tanh,
            "sigmoid": nn.Sigmoid,
            "softmax": nn.Softmax,
        }
        
        self.all_layers = []
        current_dim = input_dim
        
        for layer in layers:
            if isinstance(layer, int):
                # Add a fully connected layer
                self.all_layers.append(nn.Linear(current_dim, layer))
                current_dim = layer
            elif isinstance(layer, str):
                if layer in active_fns:
                    # Add an activation function
                    self.all_layers.append(active_fns[layer]())
                elif layer == "bn":
                    # Add batch normalization
                    self.all_layers.append(nn.BatchNorm1d(current_dim))
            elif isinstance(layer, float):
                # Add dropout
                if 0 < layer < 1:
                    self.all_layers.append(nn.Dropout(layer))
        
        self.module = nn.Sequential(*self.all_layers)
    
    def forward(self, x):
        ## make layers to a whole module
        feat = self.module(x)
        return feat
    
if __name__ == "__main__":
    fc_encoder = FcEncoder(768, [768, "leakyrelu",768])
    # print(fc_encoder)