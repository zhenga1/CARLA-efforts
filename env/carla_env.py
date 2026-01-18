import carla
import numpy as np
import random
import time

class CarlaLaneEnv:
    def __init__(self, max_steps=500, client_setup_timeout=10.0, camera_setup_timeout=5.0, add_min_speed_penalty=True,
                  speed_reward_multiplier=0.45, slow_speed_threshold=0.1, max_stuck_step_count=50):
        # define class properties
        self.client_setup_timeout = client_setup_timeout
        self.camera_setup_timeout = camera_setup_timeout
        self.add_min_speed_penalty = add_min_speed_penalty
        self.speed_reward_multiplier = speed_reward_multiplier
        self.slow_speed_threshold = slow_speed_threshold
        self.max_stuck_step_count = max_stuck_step_count
        
        # IP address of the CARLA Client, and the TCP port to communicate with
        # Two ports are needed, always the given port, given port + 1
        self.client = carla.Client('localhost', 2000)
        self.client.set_timeout(self.client_setup_timeout) # nonblocking option on the client
        self.world = self.client.get_world() # retrieve the current world

        self.bp_lib = self.world.get_blueprint_library() # retrieve the blueprint library

        # max # of steps per episode
        self.max_steps = max_steps
        self.current_step = 0
        
        # keep track of how many steps the vehicle has been stuck for
        self.stuck_steps = 0 

        # variables to keep track of actors
        self.vehicle = None
        self.camera = None
        self.image = None
    
    def reset(self):
        # destroy existing actors to start fresh 
        if self.vehicle:
            self.vehicle.destroy()
            self.vehicle = None
        if self.camera:
            self.camera.destroy()
            self.camera = None
        
        self.world.tick()  # advance the simulation to ensure clean state
        
        # resetting the step counter and the stuck counter
        self.current_step = 0
        self.stuck_steps = 0

        # making sure the synchronous mode is enabled
        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.05
        self.world.apply_settings(settings)
        # Call tick to start the simulation
        self.world.tick()

        # spawn the vehicle at a random spawn point with ROBUST error protection
        spawn_points = self.world.get_map().get_spawn_points()
        bp = self.bp_lib.filter("vehicle.*model3*")[0]  # Tesla Model 3
        random.shuffle(spawn_points)
        temp_vehicle = None
        for spawn in spawn_points[:50]:
            temp_vehicle = self.world.try_spawn_actor(bp, spawn)
            if temp_vehicle is not None:
                break
        if temp_vehicle is None:
            raise RuntimeError("Failed to spawn vehicle after multiple attempts.")
        
        self.vehicle = temp_vehicle
        print("Vehicle spawned:", self.vehicle.id)

        self._attach_camera()  # get the camera
        self.world.tick()  # tick to get the first image

        # try to get the spectator location and transform it to look at vehicle
        spectator = self.world.get_spectator()
        spectator.set_transform(
            carla.Transform(spawn.location + carla.Location(x=-6,z=3),
            carla.Rotation(pitch=-15))
        )

        time_waited = 0.0
        while self.image is None:
            self.world.tick()  # wait until the first image is received
            print("Waiting for camera image..., waited {:.1f}s".format(time_waited))
            time.sleep(0.1)
            time_waited += 0.1
            if time_waited >= self.camera_setup_timeout:
                raise TimeoutError("Timeout while waiting for camera image.")
        return self.image

    def _follow_vehicle(self, first_person=False):
        spectator = self.world.get_spectator()
        transform = self.vehicle.get_transform()

        # Vehicle orientation vector (world frame)
        forward = transform.get_forward_vector()
        right = transform.get_right_vector()

        # offsets desired
        distance_back = 6.0
        height = 3.0
        lateral_offset = 0.0  # keep 0 unless you want cinematic angle

        # Compute camera location and rotation in world coordinates
        cam_location = (
            transform.location
            - forward * distance_back
            + right * lateral_offset
            + carla.Location(z=height)
        )

        cam_rotation = carla.Rotation(
            pitch=-15,
            yaw=transform.rotation.yaw,
            roll=0.0
        )

        spectator.set_transform(
            carla.Transform(cam_location, cam_rotation)
        )

    def _attach_camera(self):
        cam_bp = self.bp_lib.find('sensor.camera.rgb')
        cam_bp.set_attribute("image_size_x", "160")
        cam_bp.set_attribute("image_size_y", "120")
        cam_bp.set_attribute("fov", "90")

        self.camera = self.world.spawn_actor(
            cam_bp,
            # This is the sensor camera's relative transform to the vehicle
            carla.Transform(
                carla.Location(x=1.5, z=2.4),
                carla.Rotation(pitch=-10)
            ),
            attach_to=self.vehicle
        )
        self.camera.listen(self._on_image) # register callback for image data
        print("Camera attached")
    
    def _on_image(self, image):
        img_array = np.frombuffer(image.raw_data, dtype=np.uint8)
        self.image = img_array.reshape((image.height, image.width, 4))[:, :, :3] # RGB format

    def _lane_deviation(self):
        """
        Returns:
            deviation (float): lateral distance from lane center (meters)
            lane_width (float): width of current lane
            on_lane (bool): whether vehicle is still on a driving lane
        """
        transform = self.vehicle.get_transform()
        location = transform.location

        # project some vehicle location to the nearest lane waypoint
        waypoint = self.world.get_map().get_waypoint(
            location, project_to_road=True, lane_type=carla.LaneType.Driving
        )

        if waypoint is None:
            return None, None, False
        
        lane_center = waypoint.transform.location
        lane_width = waypoint.lane_width

        deviation = location.distance(lane_center)

        return deviation, lane_width, deviation < (lane_width / 2)
    
    def _lane_reward(self):
        """
        Docstring for _lane_reward
        
        :param self: CarlaLaneEnv instance
        :return: reward, episode end or not
        """
        deviation, lane_width, on_lane = self._lane_deviation()
        if not on_lane:
            return -10.0, True  # heavy penalty for leaving the lane
        
        # Normalize deviation: 0 = center, 1 = edge of lane
        norm_dev = deviation / (lane_width / 2)

        # Reward shaping
        reward = 1.0 - norm_dev  # max reward at center, min at edge
        reward = max(reward, -1.0) # ensure reward is not too negative

        done = norm_dev > 1.2 # off-lane threshold
        print(f"Lane reward={reward:.2f}, deviation={deviation:.2f}")

        return reward, done
    
    def _speed_reward(self):
        velocity = self.vehicle.get_velocity()
        speed = np.linalg.norm([velocity.x, velocity.y])
        if speed < self.slow_speed_threshold:
            self.stuck_steps += 1
        else:
            self.stuck_steps = 0
        speed_bonus = self.speed_reward_multiplier * speed
        if self.add_min_speed_penalty and speed < self.slow_speed_threshold:
            speed_bonus -= 1.0  # penalty for being nearly stationary
        return speed_bonus
        

    def step(self, steer, throttle, first_person=False):
        self.vehicle.apply_control(
            carla.VehicleControl(
                throttle=float(throttle),
                steer=float(steer)
            )
        )
        self.world.tick()  # advance the simulation
        self._follow_vehicle(first_person)

        lane_reward, hasfail = self._lane_reward()

        # calculate speed reward, increment stuck steps
        speed_reward = self._speed_reward()

        # increment step counters
        self.current_step += 1

        steps_done = self.current_step >= self.max_steps

        done = hasfail or steps_done

        # calculate reward
        reward = lane_reward + speed_reward # speed
        reward -= 0.3   # constant time penalty so staying still = BAD

        # deal with case of being stuck for too long
        if self.stuck_steps > self.max_stuck_step_count:   # ~2.5 seconds
            reward -= 5.0
            done = True


        return self.image, reward, done
    
    def close(self):
        if self.camera:
            self.camera.destroy()
        if self.vehicle:
            self.vehicle.destroy()
