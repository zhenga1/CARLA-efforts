import torch.optim as optim
import torch.nn as nn
import torch
from carla_model import DrivingCNN
from carla_dataset import *
from torch.utils.data import DataLoader
import h5py
from constants import *
from pathlib import Path
import wandb

# Hyperparameters
lr=1e-4
lr_decay = 0.6
lr_patience = 10
lr_threshold = 0.006
batch_size=64
architecture_name="NVIDIA_DAVE_2"


# 1. Initialize WandB at the start of the script
wandb.init(
    project="carla-lane-dagger",
    config={
        "learning_rate": lr,
        "batch_size": batch_size,
        "architecture": architecture_name,
        "dataset": "CORL2017"
    }
)

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
total_steps = 0
def train_model(model, scheduler, dataloader, criterion, optimizer, epochs=10):
    global total_steps
    # watch the model as training begins
    wandb.watch(model, log_freq=100) # Optional: logs gradients and topology
    for epoch in range(epochs):
        # Train model
        model.train()
        running_loss = 0.0
        if epoch > 0 and steps_accum > 0:
            average_loss = last_1000stepsloss / float(steps_accum)
            scheduler.step(average_loss) # reduce the learning rate
        last_1000stepsloss = 0
        steps_accum = 0
        for i, (images, targets) in enumerate(dataloader):
            images, targets = images.to(device), targets.to(device)
            #print("Image shape right before training: ", images.shape)
            concern_targets = targets[:, :2] # get only steer and throttle
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, concern_targets)

            # Backward pass
            optimizer.zero_grad() # clear gradients from previous step
            loss.backward() # calculate new gradients
            optimizer.step() # step through the optimizer to update model weights

            # Print loss
            running_loss += loss.item()
            last_1000stepsloss += loss.item()
            steps_accum += 1
            if total_steps % 100 == 99:  # print every 100 steps
                current_step = total_steps + 1
                current_loss = running_loss / 100
                current_lr = optimizer.param_groups[0]['lr']
                wandb.log({"epoch": epoch, "Loss": current_loss, "image": image_index, "learning_rate": current_lr}, step=current_step)
                print(f"Steps {current_step}, Image current {image_index}, Loss: {current_loss}")
                running_loss = 0.0

            if total_steps % 1000 == 999:
                ### UPDATE TRAINING
                average_loss = last_1000stepsloss / float(steps_accum)
                scheduler.step(average_loss) # reduce the learning rate
                last_1000stepsloss = 0 # reset
                steps_accum = 0 # reset

            total_steps += 1


def save_data_to_h5(filename, images, targets):
    with h5py.File(filename, 'w') as f:
        f.create_dataset("images_center", data=images)
        f.create_dataset("targets", data=targets)
    print("Data saved to", filename)

image_index = 0
if __name__=="__main__":
    ### Basic Parameters
    directory = Path(DIR_PATH_TRAIN)

    # Define model and everything else
    model = DrivingCNN().to(device)

    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    # approach where LR is reduced when loss plateaus
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=lr_decay, 
        patience=lr_patience, 
        threshold=lr_threshold, #only trigger when loss improves by 0.001, i.e. 0.1%
        verbose=True
    )
    dataset = CarlaH5GlobalDataset(directory)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    ##### 
    # ------ Leftover code from before ------
    ##### 

    # for file in directory.iterdir():
    #     if file.suffix == '.h5':
    #         train_model(model, dataset, dataloader, criterion, optimizer, file )
    #         image_index += 1
    #         #print("Finished training the file with index of: ", image_index)
    
    ##### 
    # ------ End of leftover code --------
    ##### 
    
    number_of_training_iterations = 100
    train_model(model, scheduler, dataloader, criterion, optimizer, number_of_training_iterations)
    torch.save(model.state_dict(), "DrivingCNN_dagger.pth")
    print("Training Complete. Model Saved.")