import glob
import os
import sys

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import style

try:
    sys.path.insert(
        0,
        "C:/Apps/sourceDocker/ad-dev-stack/pkg/CarlaPy/WindowsNoEditor/PythonAPI/carla",
    )
    sys.path.append(
        "C:/Apps/sourceDocker/ad-dev-stack/pkg/CarlaPy/WindowsNoEditor/PythonAPI/carla/agents"
    )
    sys.path.append(
        glob.glob(
            "../carla/dist/carla-*%d.%d-%s.egg"
            % (
                sys.version_info.major,
                sys.version_info.minor,
                "win-amd64" if os.name == "nt" else "linux-x86_64",
            )
        )[0]
    )
except IndexError:
    pass

import time
import random
import math
import traceback
import carla
from agents.navigation import controller
from agents.navigation.global_route_planner import GlobalRoutePlanner

client = carla.Client("localhost", 2000)
client.set_timeout(10.0)
print("client connected")


def tic():
    global startTime_for_tictoc
    startTime_for_tictoc = time.time()


def toc():
    if "startTime_for_tictoc" in globals():
        print(
            "Elapsed time is " + str(time.time() - startTime_for_tictoc) + " seconds."
        )
    else:
        print("Toc: start time not set")


def onWorldTick(self, timestamp):
    """Gets informations from the world at every tick"""
    self._server_clock.tick()
    self.server_fps = self._server_clock.get_fps()
    self.frame = timestamp.frame_count
    self.simulation_time = timestamp.elapsed_seconds


def destroyActorsInBatches(actors, batchSize):
    for i in range(0, len(actors), batchSize):
        batch = carla.command.Batch(
            [carla.command.DestroyActor(x) for x in actors[i : i + batchSize]]
        )
        client.apply_batch(batch)


class CarlaSim:
    def spawnVehicle(
        self,
        spawnPoint=carla.Transform(
            carla.Location(x=-5.5, y=-80.0, z=0.25),
            carla.Rotation(pitch=0.0, yaw=90.0, roll=0.000000),
        ),
        # If no spawn point is provided it spawns vehicle in x=27.607,y=3.68402,z=0.02
    ):
        world = client.get_world()
        blueprintLibrary = world.get_blueprint_library()
        bp = blueprintLibrary.filter("vehicle.*")[7]
        vehicle = world.spawn_actor(bp, spawnPoint)
        return vehicle

    def euclideanDistance(self, egoLocation, targetWp, threshold=2.5):
        distance = (
            (egoLocation.x - targetWp.transform.location.x) ** 2
            + (egoLocation.y - targetWp.transform.location.y) ** 2
        ) ** 0.5
        return distance <= threshold

    def getLaneCentroids(self):
        world = client.get_world()
        _map = world.get_map()

        # Get a list of all the waypoints in the map
        waypoints = _map.generate_waypoints(distance=2.0)

        # For each waypoint, get the center of the corresponding lane
        lane_centroids = [
            [wp.transform.location.x, wp.transform.location.y]
            for wp in waypoints
            if wp.lane_type == carla.LaneType.Driving
        ]

        return lane_centroids

    def runCarla(self, callback):
        actorList = []
        world = client.get_world()
        world = client.load_world("Town03")

        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.1
        world.apply_settings(settings)

        # Set the weather to cloudy
        weather = carla.WeatherParameters(
            cloudiness=30.0, precipitation=0.0, sun_altitude_angle=45.0
        )
        world.set_weather(weather)

        # Retrieve the spectator object
        spectator = world.get_spectator()
        actorList.append(spectator)

        # Add ego vehicle
        egoVehicle = self.spawnVehicle()

        # Get the vehicle location transform
        egoLocation = egoVehicle.get_location()

        # Create a spectator camera
        spectatorTransform = carla.Transform(
            egoLocation + carla.Location(z=50), carla.Rotation(pitch=-90)
        )
        spectator = world.get_spectator()
        spectator.set_transform(spectatorTransform)

        lane_centroids = self.getLaneCentroids()

        _map = world.get_map()
        samplingResolution = 2
        grp = GlobalRoutePlanner(_map, samplingResolution)
        # grp.setup()
        spawnPoints = world.get_map().get_spawn_points()
        a = carla.Location(spawnPoints[0].location)
        b = carla.Location(spawnPoints[100].location)
        route = grp.trace_route(a, b)
        wpTransform = [wpList[0].transform for wpList in route]
        waypoints = [waypoint.location for waypoint in wpTransform]

        waypoints = []
        for i in range(len(route)):
            waypoints.append(route[i][0])
            world.debug.draw_point(
                route[i][0].transform.location,
                color=carla.Color(r=255, g=0, b=0),
                size=0.2,
                life_time=120.0,
            )

        wpLocation = []
        for i in range(len(waypoints)):
            wpLocation.append(waypoints[i].transform.location)

        # Define the PID controller parameters
        args_lateral = {"K_P": 1.0, "K_D": 0.0, "K_I": 0.07, "dt": 0.1}
        args_longitudinal = {"K_P": 1.0, "K_D": 0.0, "K_I": 0.01, "dt": 0.1}

        # Initialize the PID controller
        pidControl = controller.VehiclePIDController(
            egoVehicle, args_lateral, args_longitudinal
        )

        i = 0
        targetWp = waypoints[0]
        refSpeed = 5

        # Main loop to update the spectator's transform with the vehicle's transform
        while True:
            # Get the vehicle transform
            egoTransform = egoVehicle.get_transform()

            # Move the spectator behind the vehicle
            fixSpectatorPose = carla.Location(x=-4, z=2.5)
            spectatorTransform = carla.Transform(
                egoTransform.transform(fixSpectatorPose), egoTransform.rotation
            )
            spectator.set_transform(spectatorTransform)

            egoLocation = egoVehicle.get_location()
            control = pidControl.run_step(refSpeed, targetWp)

            # Apply control values to ego vehicle
            # egoVehicle.apply_control(carla.VehicleControl(0.0, 0.0))
            egoVehicle.apply_control(control)

            if i == (len(waypoints) - 1):
                print("last waypoint reached")
                break

            if self.euclideanDistance(egoLocation, targetWp):
                control = pidControl.run_step(refSpeed, targetWp)
                egoVehicle.apply_control(control)
                i = i + 1
                targetWp = waypoints[i]

            egoVehicleInformation = {
                "position": egoTransform.location,
                "orientation": egoTransform.rotation,
                "velocity": egoVehicle.get_velocity(),
                "angular": egoVehicle.get_angular_velocity(),
                "control": egoVehicle.get_control(),
                "attributes": egoVehicle.get_physics_control(),
            }

            physicsControl = egoVehicle.get_physics_control()
            world.tick()

        settings = world.get_settings()
        settings.synchronous_mode = False
        settings.no_rendering_mode = False
        settings.fixed_delta_seconds = None
        world.apply_settings(settings)

        for actor in [x for x in actorList if x]:
            client.apply_batch([carla.command.DestroyActor(actor)])
        egoVehicle.destroy()
        print("\nAll actors destroyed")
        print("End of sim")


if __name__ == "__main__":
    try:
        CarlaSim().runCarla(0)
    except KeyboardInterrupt:
        pass
    finally:
        print("\ndone.")
