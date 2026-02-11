import torch
import torch.nn as nn

## HELPER FUNCTION
def print_model_map(model, x):
    print("Input:", x.shape)
    for i, layer in enumerate(model.conv_layers):
        x = layer(x)
        if layer.__class__.__name__ == "Conv2d":
            print(f"Conv layer {i:02d} ({layer.__class__.__name__}): {x.shape}")
        else:
            print(f"ELU layer {i:02d} ({layer.__class__.__name__}): {x.shape}")

    x = x.flatten(1)
    print("After flatten:", x.shape)

    for i, layer in enumerate(model.fc_layers):
        x = layer(x)
        if layer.__class__.__name__ == "Linear":
            print(f"FC layer {i:02d} ({layer.__class__.__name__}): {x.shape}")
        elif layer.__class__.__name__ == "ELU":
            print(f"ELU layer {i:02d} ({layer.__class__.__name__}): {x.shape}")
        else:
            print(f"Flatten layer {i:02d} ({layer.__class__.__name__}): {x.shape}")

    print("Output:", x.shape)
# Driving CNN model
class DrivingCNN(nn.Module):
    def __init__(self):
        super(DrivingCNN, self).__init__()
        # Copying NVIDIA's DAVE 2 architecture, 5 feature stack
        # Input shape (desired 3x66x200)
        # Special Exponential Linear Unit
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 24, 5, stride=2), nn.ELU(),
            nn.Conv2d(24, 36, 5, stride=2), nn.ELU(),
            nn.Conv2d(36, 48, 5, stride=2), nn.ELU(),
            nn.Conv2d(48, 64, 3), nn.ELU(),
            nn.Conv2d(64, 64, 3), nn.ELU()
        )
        # Flatten and Fully Connected Layers
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1152, 100), nn.ELU(),
            nn.Linear(100, 50), nn.ELU(),
            nn.Linear(50, 10), nn.ELU(),
            nn.Linear(10, 2) # Obtain the final vehicle control parameters, [steer, throttole]
        )
        
        with torch.no_grad():
            noise = torch.randn([88, 3, 66, 200])
            # out = self.conv_layers(noise)
            # out = self.fc_layers(out)
            print_model_map(self, noise)
            print("Desired output shape: ")#, out.shape)
            print("Number of trainable parameters: ", sum(p.numel() for p in self.parameters() if p.requires_grad))
        

    def forward(self, x):
        # Image Normalization to [-1, 1] range, so ELU trains faster
        x = x / 127.5 - 1.0
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x
