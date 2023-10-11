import torch
import numpy as np
from locally_connected import LocallyConnected1D
from typing import Callable, List, Optional, Sequence, Tuple, Union
import torch.nn as nn

class Print(nn.Module):
    """
    A class for debugging layers that are packaged with sequential
    """
    def __init__(self,name):
        self.name = name
        super(Print, self).__init__()

    def forward(self, x):
        print("debug:"+str(self.name))
        print(x.shape)
        return x


class MLP(torch.nn.Sequential):
    """This block implements the multi-layer perceptron (MLP) module.

    Args:
        in_channels (int): Number of channels of the input
        hidden_channels (List[int]): List of the hidden channel dimensions
        norm_layer (Callable[..., torch.nn.Module], optional): Norm layer that will be stacked on top of the linear layer. If ``None`` this layer won't be used. Default: ``None``
        activation_layer (Callable[..., torch.nn.Module], optional): Activation function which will be stacked on top of the normalization layer (if not None), otherwise on top of the linear layer. If ``None`` this layer won't be used. >
        inplace (bool, optional): Parameter for the activation layer, which can optionally do the operation in-place.
            Default is ``None``, which uses the respective default values of the ``activation_layer`` and Dropout layer.
        bias (bool): Whether to use bias in the linear layer. Default ``True``
        dropout (float): The probability for the dropout layer. Default: 0.0
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: List[int],
        norm_layer: Optional[Callable[..., torch.nn.Module]] = None,
        activation_layer: Optional[Callable[..., torch.nn.Module]] = torch.nn.ReLU,
        inplace: Optional[bool] = None,
        bias: bool = True,
        dropout: float = 0.0,
        keep_nr_of_paramters_equal = False
    ):
        # The addition of `norm_layer` is inspired from the implementation of TorchMultimodal:
        # https://github.com/facebookresearch/multimodal/blob/5dec8a/torchmultimodal/modules/layers/mlp.py
        params = {} if inplace is None else {"inplace": inplace}

        layers = []
        #layers.append(Print())
        in_dim = in_channels
        #input("in_dim:"+str(in_dim))
        #input("hidden_channels:"+str(hidden_channels))
        for id_dim, hidden_dim in enumerate(hidden_channels[:-1]):
            #TODO remove hardcoded values!
            assert int(hidden_dim/16) == (hidden_dim/16)
            assert int(in_dim/16) == (in_dim/16)
            #divide the layer in 16 groups with outputs/16  units in each that each is connectd to input/16 nr of units
            closest=99999999999999999999999999999999999999999999999

            best_y=1

            if keep_nr_of_paramters_equal:

                print("searching for a configuration that will result in about the same number of parameters")
                print("this strategy asumes that only one layer is sparse")
                for y in np.arange(1, 10, 0.1):
                    nr_of_vanilla_mlp_parameters = in_dim*hidden_dim+hidden_dim*hidden_channels[id_dim+1]
                    locally_connected_mlp_parameters = (in_dim/16)*int(y*(hidden_dim/16))*16+((int(y*(hidden_dim/16))*16)*hidden_channels[id_dim+1])
                    print("nr_of_vanilla_mlp_parameters    :"+str(nr_of_vanilla_mlp_parameters))
                    print("locally_connected_mlp_parameters:"+str(locally_connected_mlp_parameters))
                    print("dist: "+str(abs(nr_of_vanilla_mlp_parameters-locally_connected_mlp_parameters)))

                    if(abs(nr_of_vanilla_mlp_parameters-locally_connected_mlp_parameters))<closest:
                        best_y = y
                        closest = (abs(nr_of_vanilla_mlp_parameters-locally_connected_mlp_parameters))

                print("best_y:"+str(best_y))
                print("closest:"+str(closest))
                print("id_dim:"+str(id_dim))

                #layers.append(Print("before locally conected"))



            layers.append(LocallyConnected1D(in_shape=in_dim, out_channels=int(best_y*hidden_dim/16), receptive_field_shape=int(in_dim/16), stride=int(in_dim/16)))
            #layers.append(torch.nn.Linear(in_dim, hidden_dim, bias=bias))

            in_dim = (int(best_y*(hidden_dim/16))*16) #hidden_dim

            if norm_layer is not None:
                layers.append(norm_layer(hidden_dim))
            layers.append(activation_layer(**params))
            layers.append(torch.nn.Dropout(dropout, **params))

            #layers.append(Print())



        assert int(hidden_dim/16) == (hidden_dim/16)
        assert int(in_dim/16) == (in_dim/16)
        #replacing the fully conected 2nd layer with a locally conected layer
        #layers.append(LocallyConnected1D(in_shape=in_dim, out_channels=int(hidden_channels[-1]/16), receptive_field_shape=int(in_dim/16), stride=int(in_dim/16)))
        #input("input to last layer")
        #input(in_dim)
        #input(hidden_channels[-1])

        #outcommenting the original fully conected layer
        #layers.append(Print("before linear layer"))
        layers.append(torch.nn.Linear(in_dim, hidden_channels[-1], bias=bias))
        layers.append(torch.nn.Dropout(dropout, **params))

        super().__init__(*layers)
        #_log_api_usage_once(self)
