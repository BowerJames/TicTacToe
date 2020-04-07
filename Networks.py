from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Input


class ValueNetwork(Model):

    def __init__(self, h_layer):
        super().__init__()
        self.Input_Layer = Input((None,9))
        self.h_layer = Dense(h_layer, activation='sigmoid')
        self.output = Dense(1)

    def call(self, inputs):
        x = self.Input_Layer(inputs)
        x = self.h_layer(x)
