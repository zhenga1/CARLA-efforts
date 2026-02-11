import carla

host = "127.0.0.1"
port = 2000
client = carla.Client(host, port)
world = client.get_world()

# Get every actor spawned in this world
actors = world.get_actors()

# # Destroy all actors (CAN"T Do this because this destroys the spectators)
# for actor in actors:
#     actor.destroy()
# Filter for things that can be destroyed (vehicles, sensors, walkers)

for actor in actors:
    if 'vehicle' in actor.type_id or 'sensor' in actor.type_id or 'walker' in actor.type_id:
        actor.destroy()

print("Scene cleaned up!")

