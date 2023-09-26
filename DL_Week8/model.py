import torch
import torch.nn as nn

class SimpleCNN(nn.Module):
      def __init__(self,num_classes=10):
          super().__init__()
          self.conv1 = self.make_block(in_channels=3,out_channels=8)
          self.conv2 = self.make_block(in_channels=8, out_channels=16)
          self.conv3 = self.make_block(in_channels=16, out_channels=32)
          self.conv4 = self.make_block(in_channels=32, out_channels=64)
          self.conv5 = self.make_block(in_channels=64, out_channels=128)
          #self.flatten = nn.Flatten()
          self.fc1 = nn.Sequential(
              nn.Linear(in_features=6272, out_features = 512),
              nn.Dropout(p=0.5),
              nn.LeakyReLU()
          )
          self.fc2 = nn.Sequential(
              nn.Linear(in_features=512, out_features=1024),
              nn.Dropout(p=0.5),
              nn.LeakyReLU()
          )
          self.fc3 = nn.Sequential(
              nn.Linear(in_features=1024, out_features=num_classes),
              nn.Dropout(p=0.5)
          )
      def make_block(self,in_channels,out_channels):
          return nn.Sequential(
              nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding="same"),
              nn.BatchNorm2d(num_features=out_channels),
              nn.LeakyReLU(),
              nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding="same"),
              nn.BatchNorm2d(num_features=out_channels),
              nn.LeakyReLU(),
              nn.MaxPool2d(kernel_size=2)
          )

      def forward(self,x):
          x = self.conv1(x)
          x = self.conv2(x)
          x = self.conv3(x)
          x = self.conv4(x)
          x = self.conv5(x)
          #x = self.flatten(x)
          #x = x.view(x.shape[0], x.shape[1]*x.shape[2]*x.shape[3]) # Reshape, flatten in tensor form
          x = x.view(x.shape[0],-1)
          x = self.fc1(x)
          x = self.fc2(x)
          x = self.fc3(x) # Shape [batch_size,num_classes]
          return x

if __name__ == '__main__':
    model = SimpleCNN()
    input_data = torch.rand(8,3,224,224) # Supposing input of the CIFAR data
    #model.forward(input_data)
    if torch.cuda.is_available():
        model.cuda() # in_place function
        input_data = input_data.cuda()
    while True:
        result = model(input_data)
        print(result.shape)    # Shape after 2DConv(8x8x222x222) as kernel_size=3 then size of 2 other dim will reduce by (kernel_size-1)
        # B X C X H X W
        break
# the result is the proportion of predicting the proportion of the image belongs to what classes