import torch
import torch.nn as nn

class SimpleNeuralNetwork(nn.Module):
    def __init__(self,num_classes = 10):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Sequential(
            nn.Linear(in_features= 3*32*32,out_features= 256),
            nn.ReLU()
        )
        self.fc2= nn.Sequential(
            nn.Linear(in_features=256, out_features=512),
            nn.ReLU()
        )
        self.fc3 = nn.Sequential(
            nn.Linear(in_features=512, out_features=1024),
            nn.ReLU()
        )
        self.fc4 = nn.Sequential(
            nn.Linear(in_features=1024, out_features=512),
            nn.ReLU()
        )
        self.fc5 = nn.Sequential(
            nn.Linear(in_features=512, out_features= num_classes),
            nn.ReLU()
        )
    def forward(self,x):
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        x = self.fc5(x)
        return x

if __name__ == '__main__':
    model = SimpleNeuralNetwork()
    input_data = torch.rand(8,3,32,32) # Supposing input of the CIFAR data
    #model.forward(input_data)
    if torch.cuda.is_available():
        model.cuda() # in_place function
        input_data = input_data.cuda()
    while True:
        result = model(input_data)
        print(result.shape)

# the result is the proportion of predicting the proportion of the image belongs to what classes