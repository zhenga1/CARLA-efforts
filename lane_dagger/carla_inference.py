import torch
import numpy as np
import carla
from carla_model import DrivingCNN

## CONFIGURATION
MODEL_PATH = "DrivingCNN_dagger.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

