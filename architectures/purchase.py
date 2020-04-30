from torch.nn import Module, Linear
from torch.nn.functional import tanh

class Model(Module):
    def __init__(self, input_shape, nb_classes, *args, **kwargs):
        super(Model, self).__init__()
        self.fc1 = Linear(input_shape[0], 128)
        self.fc2 = Linear(128, nb_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = tanh(x)
        x = self.fc2(x)

        return x