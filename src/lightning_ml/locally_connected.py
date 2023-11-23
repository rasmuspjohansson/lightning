"""
#keras version#

tf.keras.layers.LocallyConnected1D(
filters,
kernel_size,
strides=1,
padding='valid',
data_format=None,
activation=None,
use_bias=True,
kernel_initializer='glorot_uniform',
bias_initializer='zeros',
kernel_regularizer=None,
bias_regularizer=None,
activity_regularizer=None,
kernel_constraint=None,
bias_constraint=None,
implementation=1,
**kwargs
)
"""
"""
#pytorch conv1d#
torch.nn.Conv1d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', device=None, dtype=None)

"""
import torch
import numpy as np
import torch.nn as nn
class Locally_connected2D(nn.Module):

    def __init__(self, in_shape=(1, 28, 28), out_channels=16, receptive_field_shape=(5, 5), stride=1):
        super(Locally_connected2D, self).__init__()
        self.stride = stride
        self.in_channels = in_shape[0]
        self.out_channels = out_channels
        self.in_shape = in_shape
        self.receptive_field_shape = receptive_field_shape
        # input("in_shape:"+str(in_shape))
        # input("receptive_field_shape:"+str(receptive_field_shape))
        out_channels, self.receptive_fields_y , self.receptive_fields_x = self.__get_output_shape(in_shape, out_channels, receptive_field_shape, stride)
        self.out_shape = (out_channels, self.receptive_fields_y , self.receptive_fields_x)


        # create a 2 dimensional array with one nn.linear for each receptive field
        #input(str((self.in_channels * receptive_field_shape[0] * receptive_field_shape[1], out_channels)))
        self.hiddens = [[nn.Linear(self.in_channels * receptive_field_shape[0] * receptive_field_shape[1], out_channels)
                         for ix in range(self.receptive_fields_x)]
                        for i in range(self.receptive_fields_y)]
        # register all nn.linear's so pytorch knows about them TODO: check if we really need to flatten and if we actuyally could use numpy array instead of lists
        self.linears = nn.ModuleList(list(np.array(self.hiddens).flatten()))

    def __get_output_shape(self, in_shape, out_channels, receptive_field_shape, stride):
        """
        calculate the shape of the layer output
        only accepts combinations of receptive_field_shape and stride that fits with the in_shape
        @ return: (out_channels,n_receptive_fields_y,n_receptive_fields_x)
        """

        n_receptive_fields_y = (in_shape[-1] - receptive_field_shape[0]) / stride + 1
        assert int(n_receptive_fields_y) == n_receptive_fields_y, \
            str(int(n_receptive_fields_y)) + " is == " + str(
                n_receptive_fields_y) + ";invalid combination of input shape, receptive field shape and stride"
        n_receptive_fields_x = (in_shape[-2] - receptive_field_shape[1]) / stride + 1
        assert int(
            n_receptive_fields_x) == n_receptive_fields_x, "invalid combination of input shape, receptive field shape and stride"

        return ( out_channels, int(n_receptive_fields_y), int(n_receptive_fields_x))

    def forward(self, x):
        """
        Extract the receptivefieldsfrom the input
        send the receptive field through the linear layer connected to this receptive field
        """
        # input("locally connected input shape :"+str(x.shape))
        batch_size = x.shape[0]
        # calculate number of receptive fields with stride ==1

        # print("receptive fiels")
        # input(self.receptive_fields_x)
        # input(self.receptive_fields_y)

        output = []

        for i_receptive_field_y in range(self.receptive_fields_y):
            for i_receptive_field_x in range(self.receptive_fields_x):
                # get the receptive field
                receptive_field = x[:, :, (i_receptive_field_y * self.stride):(
                            i_receptive_field_y * self.stride + self.receptive_field_shape[0]),
                                  (i_receptive_field_x * self.stride):(
                                              i_receptive_field_x * self.stride + self.receptive_field_shape[1])]

                # send the receptive field through the linear layer connected to this receptive field. store the putput in the layer1_output list
                flattened_receptive_field = receptive_field.reshape([batch_size, -1])
                receptive_field_output = self.hiddens[i_receptive_field_y][i_receptive_field_x](flattened_receptive_field)

                output.append(receptive_field_output)
        result = torch.stack(output)  # torch.cat(layer1_output, dim=0, out=None)
        # input("after stack:"+str(result.shape))
        # input(layer_1_result.shape)

        # input(layer_1_result.shape)
        result = result.permute(1, 2, 0)
        # input(layer_1_result.shape)

        result = result.reshape(batch_size, self.out_channels, self.receptive_fields_y, self.receptive_fields_x)
        return result


class LocallyConnected1D(nn.Module):
    def __init__(self, in_shape=1000, out_channels=16, receptive_field_shape=100, stride=100):
        super(LocallyConnected1D, self).__init__()
        self.stride = stride
        self.in_channels = in_shape
        self.out_channels = out_channels
        self.in_shape = in_shape
        self.receptive_field_shape = receptive_field_shape
        #input("in_shape:"+str(in_shape))
        #input("receptive_field_shape:"+str(receptive_field_shape))

        #output channels and nr of values
        (self.number_of_output_channels,self.output_length )= self.__get_output_shape(in_shape, out_channels, receptive_field_shape, stride)



        # input("self.receptive_fields_y:"+str(self.receptive_fields_y))
        # input("self.receptive_fields_x "+str(self.receptive_fields_x ))

        # create a 1 dimensional array with one nn.linear for each receptive field
        print("creating hiddens with:"+str([ receptive_field_shape, self.number_of_output_channels]))
        self.hiddens = [nn.Linear(int(receptive_field_shape), int(self.number_of_output_channels))
                         for ix in range(self.output_length)]

        # register all nn.linear's so pytorch knows about them TODO: check if we really need to flatten and if we actuyally could use numpy array instead of lists
        self.linears = nn.ModuleList(list(np.array(self.hiddens).flatten()))
        #input("OBS permute operation  NEED TO BE DOUBLE CHECKED !!!! would be fatal if this was done wrong!!")

    def __get_output_shape(self, in_shape, out_channels, receptive_field_shape, stride):
        """
        calculate the shape of the layer output
        @ return: (out_channels,n_receptive_fields)
        """
        print("variables [in_shape, out_channels, receptive_field_shape, stride] are :"+str([in_shape, out_channels, receptive_field_shape, stride]))
        n_receptive_fields = (in_shape - receptive_field_shape) / stride + 1
        assert int(n_receptive_fields) == n_receptive_fields, \
            str(int(n_receptive_fields)) + " is == " + str(
                n_receptive_fields) + ";invalid combination of input shape, receptive field shape and stride"


        return (out_channels, int(n_receptive_fields))

    def forward(self, x):
        """
        Extract the receptive fields from the input
        send the receptive field through the linear layer connected to this receptive field

        TODO:not able to handle input with more dimensions than (batchsize,input_length)
            should also be able to handle input of shape  (batchsize,input_length,nr_of_channels)
        """
        #input("locally connected input shape :"+str(x.shape))
        #input("self.stride :"+str( self.stride))
        #input(" self.receptive_field_shape :"+str(  self.receptive_field_shape))

        input_shape = x.shape



        batch_size = x.shape[0]
        # calculate number of receptive fields with stride ==1

        # print("receptive fiels")
        # input(self.receptive_fields_x)
        # input(self.receptive_fields_y)

        output = []

        for i_receptive_field in range(self.output_length):
            # get the receptive field
            receptive_field = x[...,(i_receptive_field * self.stride):(
                        i_receptive_field * self.stride + self.receptive_field_shape)]
            #input("receptive_field.shape:"+str(receptive_field.shape))

            # send the receptive field through the linear layer connected to this receptive field. store the putput in the layer1_output list
            #tmp_input = receptive_field.reshape([batch_size, -1])
            #input("tmp_input.shape:"+str(tmp_input.shape))

            #tmp = self.hiddens[i_receptive_field](tmp_input)
            tmp = self.hiddens[i_receptive_field](receptive_field)

            output.append(tmp)
        result = torch.stack(output)  # torch.cat(layer1_output, dim=0, out=None)
        #input("after stack:"+str(result.shape))
        # input(layer_1_result.shape)

        # input(layer_1_result.shape)
        if len(result.shape) ==4:
            result = result.permute(1, 3,2, 0)
            #print("OBS THIS NEED TO BE DOUBLE CHECKED!!")
            result = result.reshape(batch_size,input_shape[1], self.out_channels* self.output_length)

        else:
            result = result.permute(1, 2, 0)
            result = result.reshape(batch_size, self.out_channels* self.output_length)
        return result
