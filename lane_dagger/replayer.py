import h5py
import time
import carla
from pathlib import Path

# 1. Connect to the CARLA server
host='127.0.0.1'
port=2000
client = carla.Client(host, port)
world = client.get_world()

def spawn_vehicle():
    # spawn vehicle
    bp = world.get_blueprint_library().filter('*model3*')[0]
    random_spawn_point = world.get_map().get_spawn_points()[0] # get the first spawn point
    spawn_point = world.get_map().get_spawn_points()[0] # get the first spawn point
    try: # spawn the vehicle
        vehicle = world.spawn_actor(bp, spawn_point)
    except Exception as e:
        print(e)
        spawn_point = world.get_map().get_spawn_points()[1] # get the second spawn point
        vehicle = world.spawn_actor(bp, spawn_point)
    return vehicle

def gather_globals(targets, i):
    row = targets[i]
    
    # Basic Controls
    steer          = row[0]
    throttle            = row[1]
    brake          = row[2]
    hand_brake     = row[3]  # Boolean (0.0 or 1.0)
    reverse_gear   = row[4]  # Boolean
    
    # Perturbations
    steer_noise    = row[5]
    gas_noise      = row[6]
    brake_noise    = row[7]
    
    # Localization
    pos_x          = row[8]
    pos_y          = row[9]
    speed          = row[10]
    
    # Collision Sensors (0.0 = no collision, >0.0 = collision intensity)
    coll_other     = row[11]
    coll_ped       = row[12]
    coll_car       = row[13]
    
    # Lane/Sidewalk Infractions
    opp_lane_inter = row[14]
    sidewalk_inter = row[15]
    
    # IMU / Physics
    accel_x        = row[16]
    accel_y        = row[17]
    accel_z        = row[18]
    platform_time  = row[19]
    game_time      = row[20]
    orient_x       = row[21]
    orient_y       = row[22]
    orient_z       = row[23]
    
    # High-Level Decision Data
    # 2: Follow, 3: Left, 4: Right, 5: Straight
    command        = int(row[24]) 
    noise_active   = row[25]
    camera_id      = row[26]
    camera_angle   = row[27]

    globals().update(locals())


def replay_h5(file_path):
    # 2. Open the H5 file
    with h5py.File(file_path, 'r') as f:
        print("Currently working on the file: ", file_path)
        # 3. Read the images and targets from the H5 file
        if not list(f.keys()):
            print("H5 file is empty. Exiting...")
            return
        # import pdb
        # pdb.set_trace()
        images = f['rgb'][:] # get the images (200, 88, 200, 3)
        targets = f['targets'] # get the targets (200, 28). There are 28 different labels

        vehicle = spawn_vehicle()
        # 4. Loop through the images and targets
        for i in range(len(images)):
            #print("Replaying image: ", i)
            # 5. Get the image and targets
            steer, throttle, brake = targets[i][0], targets[i][1], targets[i][2]
            gather_globals(targets, i)
            #print("Steer: ", steer, "Throttle: ", throttle, "Brake: ", brake)

            # loop through all the throttle and steer values
            try:
                sleep_time = 0.01
                # for i in range(len(steer)):
                control = carla.VehicleControl(
                    throttle=float(throttle), brake=float(brake), steer=float(steer))
                vehicle.apply_control(control)
                time.sleep(sleep_time) # sleep for 0.01 seconds

                spectator = world.get_spectator()
                transform = vehicle.get_transform()
                spectator.set_transform(carla.Transform(transform.location + carla.Location(z=20), carla.Rotation(pitch=-90)))
            except Exception as e:
                print(e)
        vehicle.destroy()
        print("Finished replaying the vehicles. Exiting...")
            # finally:
            #     vehicle.destroy()
            #     print("Finished replaying the vehicles. Exiting...")


            # # 6. Get the vehicle
            # vehicle = world.get_actors().filter('*model3*')[0]


            # # 7. Apply the target to the vehicle
            # vehicle.apply_control(carla.VehicleControl(throttle=throttle, brake=brake, steer=steer))
import os
from constants import *
if __name__ == '__main__':
    directory = Path(DIR_PATH_TRAIN)
    index = 0
    for file in directory.iterdir():
        if file.suffix == '.h5':
            replay_h5(file)
            print("Finished replaying the file with index of: ", index)
            index += 1
    #replay_h5(r"C:\Users\aaron\Downloads\CORL2017ImitationLearningData\AgentHuman\SeqTrain\data_06406.h5")
