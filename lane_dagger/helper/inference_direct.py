import torch 
import numpy as np
import carla

##
import os
import sys
## Manual addition of the parent directory to PATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
##
from carla_model import DrivingCNN

## DEFINING SOME CONSTANT MODEL PATHS
MODEL_PATH = "../Trained_models/DrivingCNN_dagger.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
#####

# Image preprocessing 
def preprocess_image(rgb_image):
    """
    Matches Training preprocessing
    1. Converts CARLA raw image to NumPy
    2. Crops the image height from 88 to 66.
    3. Normalizes and permutes the image values to (1, 3, 66, 200) for PyTorch.
    """
    # Carla Raw Data is BGRA format; convert to RGB NumPy
    # (H, W, 4)
    img = np.reshape(np.array(rgb_image.raw_data), (rgb_image.height, rgb_image.width, 4))
    # now becomes (H, W, 3) after dropping alpha channel
    img = img[:, :, :3] # Drop the alpha channel

    ## Slicing. Here slice the width dim##
    img = img[11:-11, :, :] # crop image (in the HEIGHT axis)

    # Transform to a tensor and permute to CHW format, also normalize to [0, 1]
    img_tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
    print("Preprocessed image shape (C, H, W): ", img_tensor.shape) # should be (3, 66, 200)
    return img_tensor.unsqueeze(0).to(DEVICE) # Add batch dimension, shape becomes (1, 3, 66, 200) (so do per image at a time)

def attach_spectator(world, vehicle, bp_lib, camera=None):
    # will default to spectator camera if not provided
    if camera is None:
        camera = world.get_spectator()
    
    vehicle_transform = vehicle.get_transform()
    camera.set_transform(carla.Transform(vehicle_transform.location + carla.Location(x=1.5, z=20), carla.Rotation(pitch=-90)))
    print("Spectator attached successfully.    ")

def destroy_and_recreate(world, vehicle, camera, bp_lib):
    # Destroy the existing vehicle
    if vehicle is not None:
        vehicle.destroy()
        print("Vehicle destroyed.")

    if camera is not None:
        camera.destroy()
        print("Camera destroyed.")

    # Spawn a new vehicle
    bp = bp_lib.filter('vehicle.*model3*')[0]
    
    # get random spawn points and choose one randomly to spawn the vehicle
    spawn_points = world.get_map().get_spawn_points()
    new_vehicle = None
    while new_vehicle is None:
        spawn_point = spawn_points[np.random.randint(len(spawn_points))]
        #spawn_point = world.get_map().get_spawn_points()[0]
        new_vehicle = world.try_spawn_actor(bp, spawn_point)
    
    print("New vehicle spawned.")

    # 4. Attach Camera (Must match training sensor position!)
    cam_bp = bp_lib.find('sensor.camera.rgb')
    cam_bp.set_attribute('image_size_x', '200')
    cam_bp.set_attribute('image_size_y', '88')
    cam_bp.set_attribute('fov', '110')

    # Position matches CORL 2017 dataset sensor
    camera_transform = carla.Transform(carla.Location(x=2.0, z=1.4))
    new_camera =world.spawn_actor(cam_bp, camera_transform, attach_to=new_vehicle)
    print("New Camera created and attached to new vehicle successfully.")

    # attach spectator
    attach_spectator(world, new_vehicle, bp_lib)
    print("Spectator attached to new vehicle successfully.")
    return new_vehicle, new_camera

def run_replay():
    # 1. Load trained model
    model = DrivingCNN().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval() # Set to evaluation mode
    print("Model loaded successfully.")

    # 2. Connect to CARLA
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)
    world = client.get_world()
    blueprint_library = world.get_blueprint_library()

    # 3. Spawn Vehicle
    bp = blueprint_library.filter('vehicle.*model3*')[0]
    spawn_point = world.get_map().get_spawn_points()[0]
    vehicle = world.spawn_actor(bp, spawn_point)

    # 4. Attach Camera (Must match training sensor position!)
    cam_bp = blueprint_library.find('sensor.camera.rgb')
    cam_bp.set_attribute('image_size_x', '200')
    cam_bp.set_attribute('image_size_y', '88')
    cam_bp.set_attribute('fov', '110')

    # Position matches CORL 2017 dataset sensor
    camera_transform = carla.Transform(carla.Location(x=2.0, z=1.4))
    camera = world.spawn_actor(cam_bp, camera_transform, attach_to=vehicle)

    max_low_throttle_tolerance = 200
    low_throttle_count = 0
    low_throttle_threshold = 0.05

    max_low_position_change_tolerance = 200
    low_position_change_count = 0
    low_position_change_threshold = 0.001

    stop_condition = False
    prev_location = vehicle.get_transform().location

    def reinit_local_vars():
        nonlocal low_throttle_count, low_position_change_count, prev_location
        low_throttle_count = 0
        low_position_change_count = 0
        prev_location = vehicle.get_transform().location


    
    try:
        def on_image(image):
            nonlocal stop_condition, prev_location
            nonlocal vehicle, camera
            if stop_condition:
                # done in outside the callback
                #reinit_local_vars()
                #vehicle, camera = destroy_and_recreate(world, vehicle, camera, blueprint_library)
                # stop_condition = False
                return 
            # Preprocess and Predict
            input_tensor = preprocess_image(image)
            with torch.no_grad():
                prediction = model(input_tensor).cpu().numpy()[0]
            
            attach_spectator(world, vehicle, blueprint_library) # attach spectator to the vehicle to follow it around

            # prediction[0] = steer, prediction[1] = throttle
            steer = float(prediction[0])
            throttle = float(prediction[1])
            
            debug_test = f"Steer prediction: {steer:.3f}, Throttle prediction: {throttle:.3f}"
            debug_location = vehicle.get_transform().location + carla.Location(z=2.0) # 2.0 metres above the vehicle

            # Draw the relevant string
            world.debug.draw_string(
                debug_location, debug_test, draw_shadow=False, 
                color=carla.Color(r=255, g=0, b=0), # red color for visibility
                life_time=0.1
            )
            if throttle < low_throttle_threshold:
                nonlocal low_throttle_count
                low_throttle_count += 1
                if low_throttle_count >= max_low_throttle_tolerance:
                    stop_condition = True
            else:
                low_throttle_count = 0
            
            position_change = vehicle.get_transform().location
            position_change_distance = position_change.distance(prev_location)
            if position_change_distance < low_position_change_threshold:
                nonlocal low_position_change_count
                low_position_change_count += 1
                if low_position_change_count >= max_low_position_change_tolerance:
                    stop_condition = True
            else:
                low_position_change_count = 0

            prev_location = position_change

            # Apply to car
            control = carla.VehicleControl(
                throttle=max(0.0, throttle), 
                steer=steer, 
                brake=0.0
            )
            vehicle.apply_control(control)

        camera.listen(lambda image: on_image(image))
        
        # Keep the script running
        while True:
            world.wait_for_tick()

            if stop_condition:
                print("Resetting the camera")
                camera.stop() # stop listening to the camera feed to prevent multiple triggers
                print("Stop condition met. Reinitializing vehicle and camera...")
                # recreate the vehicle and camera
                vehicle, camera = destroy_and_recreate(world, vehicle, camera, blueprint_library)
                # reset the local variables
                reinit_local_vars()
                camera.listen(lambda image: on_image(image)) # start listening again
                # done
                stop_condition = False

    finally:
        print("Cleaning up...")
        camera.destroy()
        vehicle.destroy()

if __name__ == "__main__":
    run_replay()