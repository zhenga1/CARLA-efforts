# Copy of tutorial 3 from before

# All Imports
import numpy as np
import time
import carla # Simulation Library
import math 
import cv2


## CONNECT TO CARLA SIMULATOR
# Sim connection
client = carla.Client('localhost', 2000)


## Define the CARLA world
# We get the world instance and spawn the vehicle here
world = client.get_world()
# Get all the spawn points that are suitable
spawn_points = world.get_map().get_spawn_points() 

# Get Mini Car Blueprint
bp = world.get_blueprint_library().filter('*model3*')[0]

temp_vehicle = None
# Spawn the vehicle
for spawn_point in spawn_points:
    start_point = spawn_point
    temp_vehicle = world.try_spawn_actor(bp, start_point)
    if temp_vehicle is not None:
        break
vehicle = temp_vehicle

### RGB Camera. Mount the offset of the camera w.r.t to the vehicle
CAMERA_POS_X = -5
CAMERA_POS_Z = 3

camera_bp = world.get_blueprint_library().filter('sensor.camera.rgb')[0]
camera_bp.set_attribute('image_size_x', '640')
camera_bp.set_attribute('image_size_y', '480')

camera_init_trans = carla.Transform(carla.Location(x=CAMERA_POS_X, z=CAMERA_POS_Z))
# spawing this camera actor attached to the vehicle
camera = world.spawn_actor(camera_bp, camera_init_trans, attach_to=vehicle)

def camera_callback(image, data_dict):
    data_dict['image'] = np.reshape(np.copy(image.raw_data),(image.height,image.width,4))

image_w = camera_bp.get_attribute('image_size_x').as_int()
image_h = camera_bp.get_attribute('image_size_y').as_int()

camera_data = {'image': np.zeros((image_h,image_w,4))}
# this actually opens a live stream from the camera
camera.listen(lambda image: camera_callback(image,camera_data))


#### This is the part where we get the car to move forward
### LOGIC FOR MOVING THE CAR
'''
This is new Bit for tutorial 4
Create control functions, so we can push the car along the route
'''
# define the speed constants
PREFERRED_SPEED = 30 
SPEED_THRESHOLD = 2 # We want cur speed to be this amount within Preferred speed 

#adding params to display text to image
font = cv2.FONT_HERSHEY_SIMPLEX
# org - defining lines to display telemetry values on the screen
org = (30, 30) # this line will be used to show current speed
org2 = (30, 50) # this line will be used for future steering angle
org3 = (30, 70) # and another line for future telemetry outputs
org4 = (30, 90) # and another line for future telemetry outputs
org3 = (30, 110) # and another line for future telemetry outputs
fontScale = 0.5
# white color
color = (255, 255, 255)
# Line thickness of 2 px
thickness = 1

def maintain_speed(s):
    '''
    This is a function to maintain the speed of the car
    '''
    if s > PREFERRED_SPEED:
        return 0
    elif s < PREFERRED_SPEED - SPEED_THRESHOLD:
        return 0.8 # hit gas
    else:
        return 0.4 # hit a little gas
    

### SHOW CAMERA FEED
cv2.namedWindow('RGB Camera', cv2.WINDOW_AUTOSIZE)
cv2.imshow('RGB Camera', camera_data['image'])


## MAIN LOOP
done = False
stuck_steps = 0
stuck_threshold = 0.1
stuck_count_max = 50
while True:
    # Carla Tick
    world.tick() # get next timestep

    # deal with quit event with cv2
    if cv2.waitKey(1) == ord('q'):
        done = True
        break

    # Get the next image from the camera
    image = camera_data['image']

    steering_angle = 0
    # Get the current speed of the car
    v = vehicle.get_velocity()
    speed = round(3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2))
    if speed < stuck_threshold:
        stuck_steps += 1
    if stuck_steps > stuck_count_max:
        done = True
        break
    image = cv2.putText(image, 'Speed: '+str(int(speed))+' kmh', org2, 
                        font, fontScale, color, thickness, cv2.LINE_AA)
    estimated_throttle = maintain_speed(speed)
    # Now apply accelerator
    vehicle.apply_control(carla.VehicleControl(throttle=estimated_throttle,
                                               steer=steering_angle))
    cv2.imshow('RGB Camera', image)

# Clean up stuff
cv2.destroyAllWindows()
camera.stop()
for actor in world.get_actors().filter('*vehicle*'):
    actor.destroy()
for sensor in world.get_actors().filter('*sensor*'):
    sensor.destroy()



    


