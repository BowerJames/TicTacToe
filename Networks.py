import copy

import tensorflow.keras.initializers as init
from tensorflow.keras import Sequential, Input, Model
from tensorflow.keras.layers import Concatenate, Add,  LayerNormalization, Dense, Input, Softmax, ReLU, Conv2D, MaxPool2D, AveragePooling2D, BatchNormalization, Flatten, LeakyReLU, Dropout
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.losses import mean_squared_error
import tensorflow as tf
from tensorflow.keras import regularizers
from tensorflow.keras.models import clone_model


class ValueNetwork(Sequential):

    def __init__(self, input_size, h_layer, reg_coef=0.01):
        super(ValueNetwork, self).__init__()
        self.add(Dense(h_layer, activation='relu', kernel_regularizer=regularizers.l2(reg_coef), input_dim=input_size))
        self.add(Dense(1, kernel_regularizer=regularizers.l2(reg_coef)))


class PolicyNetwork(Sequential):
    def __init__(self, input_size, h_layer, output_dim):
        super(PolicyNetwork, self).__init__()
        self.add(Dense(h_layer, activation='relu',  input_dim=input_size))
        self.add(Dense(output_dim))

class ActorNetwork(Sequential):

    def __init__(self, input_size, h_layer, output_dim):
        super(ActorNetwork, self).__init__()
        self.add(Dense(h_layer, activation='relu', input_dim=input_size))
        self.add(Dense(output_dim))
        self.add(Softmax())

def value_network(inp_dim, h_layers, reg_coef=0.01):
    model = Sequential()
    for i in range(len(h_layers)):
        if i==0:
            model.add(Dense(h_layers[i], input_shape=(inp_dim,), kernel_regularizer=regularizers.l2(reg_coef)))
            model.add(ReLU())
        else:
            model.add(Dense(h_layers[i],kernel_regularizer=regularizers.l2(reg_coef)))
            model.add(ReLU())

    model.add(Dense(1, kernel_regularizer=regularizers.l2(reg_coef)))

    return model

def conv_value_network(inp_shape, filter_layers):
    model = Sequential()
    for i in range(len(filter_layers)):
        if i == 0:
            model.add(Conv2D(filter_layers[i], kernel_size=(3, 3), padding='same', use_bias=True))
            model.add(LeakyReLU())

        else:
            model.add(Conv2D(filter_layers[i], (3, 3), padding='same', use_bias=True))
            model.add(LeakyReLU())

    model.add(Flatten())
    model.add(Dense(32))
    model.add(LeakyReLU())
    model.add(Dense(1))


    return model

def state_to_action_net(inp_shape, filter_layers):
    model = Sequential(name='live')
    for i in range(len(filter_layers)):
        if i == 0:
            model.add(
                Conv2D(filter_layers[i], kernel_size=4, padding='same', use_bias=False, input_shape=inp_shape, dtype=tf.float64))
            model.add(ReLU())

        else:
            model.add(Conv2D(filter_layers[i], kernel_size=4, padding='same', use_bias=False, dtype=tf.float64))
            model.add(ReLU())
    model.add(Flatten())
    model.add(Dense(7, use_bias=True))


    return model

def state_to_action_net_zero(inp_shape, filter_layers):
    model = Sequential(name='target')
    for i in range(len(filter_layers)):
        if i == 0:
            model.add(
                Conv2D(filter_layers[i], kernel_size=4, padding='same', use_bias=False, input_shape=inp_shape))
            model.add(LayerNormalization())
            model.add(ReLU())

        else:
            model.add(Conv2D(filter_layers[i], kernel_size=4, padding='same', use_bias=False))
            model.add(LayerNormalization())
            model.add(ReLU())
    model.add(Flatten())
    model.add(Dropout(0.1))
    model.add(Dense(128, use_bias=True))
    model.add(Dropout(0.1))
    model.add(Dense(7, use_bias=True, kernel_initializer=init.zeros()))

    return model

def res_block(inputs, filters, conv_size):
    x = inputs
    for i in range(len(filters)):
        if i == len(filters) - 1:
            x = Conv2D(filters[i], conv_size, padding='same', use_bias=False)(x)
            x = BatchNormalization()(x)
        else:
            x = Conv2D(filters[i], conv_size, padding='same', use_bias=False)(x)
            x = BatchNormalization()(x)
            x = ReLU()(x)
    x = Add()([x, inputs])
    x = ReLU()(x)

    return x

def res_model(initial_filters, num_res, res_struc, dense):
    inputs = Input(shape=(6, 7, 3))

    for i in range(len(initial_filters)):
        if i == 0:
            x = Conv2D(initial_filters[i], 3, padding='same')(inputs)
            x = BatchNormalization()(x)
            x = ReLU()(x)
        else:
            x = Conv2D(initial_filters[i], 3, padding='same')(x)
            x = BatchNormalization()(x)
            x = ReLU()(x)
    for j in range(num_res):
        x = res_block(x, res_struc, 3)

    x = Conv2D(1, 1, padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Flatten()(x)
    for k in range(len(dense)):
        x = Dense(dense[k])(x)
        x = ReLU()(x)
    x = Dense(7)(x)
    return Model(inputs, x)

def res_model_zero(initial_filters, num_res, res_struc, dense):
    inputs = Input(shape=(6, 7, 3))

    for i in range(len(initial_filters)):
        if i == 0:
            x = Conv2D(initial_filters[i], 3, padding='same')(inputs)
            x = LayerNormalization()(x)
            x = ReLU()(x)
        else:
            x = Conv2D(initial_filters[i], 3, padding='same')(x)
            x = LayerNormalization()(x)
            x = ReLU()(x)
    for j in range(num_res):
        x = res_block(x, res_struc, 3)

    x = Conv2D(1, 1, padding='same')
    x = LayerNormalization()(x)
    x = Flatten()(x)
    for k in range(len(dense)):
        x = Dense(dense[k])(x)
        x = ReLU()(x)
    x = Dense(7, kernel_initializer=init.zeros())(x)
    return Model(inputs, x)


def feature_extraction(inputs, conv_filter_numbers, conv_filter_size):
    x = inputs
    for i in range(len(conv_filter_numbers)):
        x = Conv2D(conv_filter_numbers[i], conv_filter_size, padding='same')(x)
        x = ReLU()(x)

    return x

def action_block(inputs, hidden_layers, zero=False):
    x = inputs
    for i in range(len(hidden_layers)):
        x = Dense(hidden_layers[i])(x)
        x = Dropout(0.1)(x)
        x = ReLU()(x)

    if not zero:
        x = Dense(1)(x)
    else:
        x = Dense(1, kernel_initializer=init.zeros())(x)
    return x




def gpu_net(feature_extraction_filters, feature_extraction_kernel_size, res_block_structure, res_block_filter_size, num_res_blocks, action_net_layers, zero=False):
    state_input = Input(shape=(6, 7, 3))
    action_input = Input(shape=(7,))
    x = state_input
    x = feature_extraction(x, feature_extraction_filters, feature_extraction_kernel_size)
    for i in range(num_res_blocks):
        x = res_block(x, res_block_structure, res_block_filter_size)
    x = Conv2D(1, 1)(x)
    x = Flatten()(x)
    x = Concatenate()([x, action_input])
    x = action_block(x, action_net_layers, zero=zero)

    model = Model(inputs=[state_input, action_input], outputs=x)

    return model
