import carla
import numpy as np
import random
import time

class CarlaLaneEnv:
    def __init__(self, max_steps=500):
        # IP address of the CARLA Client, and the TCP port to communicate with
        # Two ports are needed, always the given port, given port + 1
        self.client = carla.Client('localhost', 2000)
        self.client.set_timeout(10.0) # nonblocking option on the client
        self.world = self.client.get_world() # retrieve the current world

        self.bp_lib = self.world.get_blueprint_library() # retrieve the blueprint library

        # max # of steps per episode
        self.max_steps = max_steps
        self.current_step = 0

        # variables to keep track of actors
        self.vehicle = None
        self.camera = None
        self.image = None
    
    def reset(self):
        if self.vehicle:
            self.vehicle.destroy()
        
        # resetting the step counter
        self.current_step = 0

        # making sure the synchronous mode is enabled
        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.05
        self.world.apply_settings(settings)
        # Call tick to start the simulation
        self.world.tick()

        spawn = random.choice(self.world.get_map().get_spawn_points())
        bp = self.bp_lib.filter("vehicle.*model3*")[0]
        self.vehicle = self.world.spawn_actor(bp, spawn)
        print("Vehicle spawned:", self.vehicle.id)

        self._attach_camera()  # get the camera
        self.world.tick()  # tick to get the first image

        # try to get the spectator location and transform it to look at vehicle
        spectator = self.world.get_spectator()
        spectator.set_transform(
            carla.Transform(spawn.location + carla.Location(x=-6,z=3),
            carla.Rotation(pitch=-15))
        )

        return self.image

    def _follow_vehicle(self, first_person=False):
        spectator = self.world.get_spectator()
        transform = self.vehicle.get_transform()

        x_bias = -6 if first_person else -10
        spectator.set_transform(
            carla.Transform(
                transform.location + carla.Location(x=x_bias, z=3),
                carla.Rotation(
                    pitch=-15,
                    yaw=transform.rotation.yaw
                )
            )
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
        reward = max(reward, -1.0) # ensure non-negative

        done = norm_dev > 1.2 # off-lane threshold
        print(f"Lane reward={reward:.2f}, deviation={deviation:.2f}")

        return reward, done

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

        velocity = self.vehicle.get_velocity()
        speed = np.linalg.norm([velocity.x, velocity.y])
        speed_bonus = 0.05 * speed

        self.current_step += 1
        steps_done = self.current_step >= self.max_steps

        done = hasfail or steps_done

        reward = lane_reward + speed_bonus # speed
        #done = False

        return self.image, reward, done
    
    def close(self):
        if self.camera:
            self.camera.destroy()
        if self.vehicle:
            self.vehicle.destroy()
