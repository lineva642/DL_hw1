from torch import nn

class Network(nn.Module):
    def __init__(self, layers_list):
        #инициализируем родительский класс
        super(Network, self).__init__()
        
        layers = []
        for layer in layers_list:
            #добавляем заданный слой сети
            layers.append(nn.Linear(layer[0], layer[1]))
            #добавляем функцию активации
            layers.append(nn.Sigmoid())
        #нормируем выходы    
        layers.append(nn.Softmax())
        #соединяем слои в сеть
        self.net = nn.Sequential(*layers)
    
    #прямой проход по сети, где х - входной тензор    
    def forward(self, x):
        return self.net(x)        
