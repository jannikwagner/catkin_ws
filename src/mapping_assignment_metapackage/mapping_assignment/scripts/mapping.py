#!/usr/bin/env python3

"""
    # {Jannik Wagner}
    # {19971213-1433}
    # {wagne@kth.se}
"""

# Python standard library
from math import cos, sin, atan2, fabs

# Numpy
import numpy as np

# "Local version" of ROS messages
from local.geometry_msgs import Pose, PoseStamped, Quaternion
from local.sensor_msgs import LaserScan
from local.map_msgs import OccupancyGridUpdate

from grid_map import GridMap


class Mapping:
    def __init__(self, unknown_space, free_space, c_space, occupied_space,
                 radius, optional=None):
        self.unknown_space = unknown_space
        self.free_space = free_space
        self.c_space = c_space
        self.occupied_space = occupied_space
        self.allowed_values_in_map = {"self.unknown_space": self.unknown_space,
                                      "self.free_space": self.free_space,
                                      "self.c_space": self.c_space,
                                      "self.occupied_space": self.occupied_space}
        self.radius = radius
        self.__optional = optional

    def get_yaw(self, q):
        """Returns the Euler yaw from a quaternion.
        :type q: Quaternion
        """
        return atan2(2 * (q.w * q.z + q.x * q.y),
                     1 - 2 * (q.y * q.y + q.z * q.z))

    def raytrace(self, start, end):
        """Returns all cells in the grid map that has been traversed
        from start to end, including start and excluding end.
        start = (x, y) grid map index
        end = (x, y) grid map index
        """
        (start_x, start_y) = start
        (end_x, end_y) = end
        x = start_x
        y = start_y
        (dx, dy) = (fabs(end_x - start_x), fabs(end_y - start_y))
        n = dx + dy
        x_inc = 1
        if end_x <= start_x:
            x_inc = -1
        y_inc = 1
        if end_y <= start_y:
            y_inc = -1
        error = dx - dy
        dx *= 2
        dy *= 2

        traversed = []
        for i in range(0, int(n)):
            traversed.append((int(x), int(y)))

            if error > 0:
                x += x_inc
                error -= dy
            else:
                if error == 0:
                    traversed.append((int(x + x_inc), int(y)))
                y += y_inc
                error += dx

        return traversed

    def add_to_map(self, grid_map, x, y, value):
        """Adds value to index (x, y) in grid_map if index is in bounds.
        Returns weather (x, y) is inside grid_map or not.
        """
        if value not in self.allowed_values_in_map.values():
            raise Exception("{0} is not an allowed value to be added to the map. "
                            .format(value) + "Allowed values are: {0}. "
                            .format(self.allowed_values_in_map.keys()) +
                            "Which can be found in the '__init__' function.")

        if self.is_in_bounds(grid_map, x, y):
            grid_map[x, y] = value
            return True
        return False

    def is_in_bounds(self, grid_map, x, y):
        """Returns weather (x, y) is inside grid_map or not."""
        if x >= 0 and x < grid_map.get_width():
            if y >= 0 and y < grid_map.get_height():
                return True
        return False

    def update_map(self, grid_map, pose, scan):
        """Updates the grid_map with the data from the laser scan and the pose.

        For E: 
            Update the grid_map with self.occupied_space.

            Return the updated grid_map.

            You should use:
                self.occupied_space  # For occupied space

                You can use the function add_to_map to be sure that you add
                values correctly to the map.

                You can use the function is_in_bounds to check if a coordinate
                is inside the map.

        For C:
            Update the grid_map with self.occupied_space and self.free_space. Use
            the raytracing function found in this file to calculate free space.

            You should also fill in the update (OccupancyGridUpdate()) found at
            the bottom of this function. It should contain only the rectangle area
            of the grid_map which has been updated.

            Return both the updated grid_map and the update.

            You should use:
                self.occupied_space  # For occupied space
                self.free_space      # For free space

                To calculate the free space you should use the raytracing function
                found in this file.

                You can use the function add_to_map to be sure that you add
                values correctly to the map.

                You can use the function is_in_bounds to check if a coordinate
                is inside the map.

        :type grid_map: GridMap
        :type pose: PoseStamped
        :type scan: LaserScan
        """

        # Current yaw of the robot
        robot_yaw = self.get_yaw(pose.pose.orientation)
        robot_position = np.array([pose.pose.position.x, pose.pose.position.y])
        # The origin of the map [m, m, rad]. This is the real-world pose of the
        # cell (0,0) in the map.
        origin = grid_map.get_origin()
        origin_position = np.array([origin.position.x, origin.position.y])
        # The map resolution [m/cell]
        resolution = grid_map.get_resolution()

        """
        Fill in your solution here
        """

        # E

        points = []

        for i, ray_range in enumerate(scan.ranges):
            if ray_range <= scan.range_min or ray_range >= scan.range_max:
                continue
            ray_angle = scan.angle_min + i*scan.angle_increment
            angle = ray_angle + robot_yaw
            # pos_laser = np.array([scan_range, 0])
            pos = np.array((cos(angle), sin(angle)))*ray_range

            # pos = rotate(pos, robot_yaw)
            pos = pos + robot_position
            pos = pos - origin_position
            pos = pos / resolution
            pos = pos.astype(int)
            x, y = pos

            # C
            points.append((x, y, self.occupied_space))
            pos_robot = robot_position - origin_position
            pos_robot = pos_robot / resolution
            pos_robot = pos_robot.astype(int)
            ray_points = self.raytrace(pos_robot, pos)
            for point in ray_points:
                x_p, y_p = point
                x_p, y_p = int(x_p), int(y_p)
                self.add_to_map(grid_map, x_p, y_p, self.free_space)
                points.append((x_p, y_p, self.free_space))

            # E
            self.add_to_map(grid_map, x, y, self.occupied_space)

        """
        For C only!
        Fill in the update correctly below.
        """
        # self.inflate_map(grid_map)

        x_min = min(p[0] for p in points)
        x_max = max(p[0] for p in points)
        y_min = min(p[1] for p in points)
        y_max = max(p[1] for p in points)

        update_width = x_max - x_min + 1
        update_height = y_max - y_min + 1

        # x_min = max(0, x_min - self.radius)
        # y_min = max(0, y_min - self.radius)
        # x_max = min(grid_map.get_width()-1, x_max + self.radius)
        # y_min = min(grid_map.get_height()-1, y_max + self.radius)

        # Only get the part that has been updated
        update = OccupancyGridUpdate()
        # The minimum x index in 'grid_map' that has been updated
        UPDATE = True
        if UPDATE:
            update.x = x_min
            # The minimum y index in 'grid_map' that has been updated
            update.y = y_min
            # Maximum x index - minimum x index + 1
            update.width = update_width
            # Maximum y index - minimum y index + 1
            update.height = update_height
            # The map data inside the rectangle, in row-major order.
            data = grid_map[x_min:x_max +
                            1, y_min:y_max+1].astype(int)
            # print(data)
            # print(data.min())
            # print(data.max())
            # print(set(data.flatten()))
            # # data = [list(row) for row in data]
            # # print(data)
            update.data = list(data.flatten())
        # update.data = []

        # Return the updated map together with only the
        # part of the map that has been updated
        return grid_map, update

    def inflate_map(self, grid_map):
        """For C only!
        Inflate the map with self.c_space assuming the robot
        has a radius of self.radius.

        Returns the inflated grid_map.

        Inflating the grid_map means that for each self.occupied_space
        you calculate and fill in self.c_space. Make sure to not overwrite
        something that you do not want to.


        You should use:
            self.c_space  # For C space (inflated space).
            self.radius   # To know how much to inflate.

            You can use the function add_to_map to be sure that you add
            values correctly to the map.

            You can use the function is_in_bounds to check if a coordinate
            is inside the map.

        :type grid_map: GridMap
        """

        """
        Fill in your solution here
        """

        pixel_radius = self.radius
        offsets = []
        for dx in range(-pixel_radius, pixel_radius+1):
            for dy in range(-pixel_radius, pixel_radius+1):
                if dx**2 + dy**2 <= pixel_radius**2:
                    offsets.append((dx, dy))

        for x in range(grid_map.get_width()):
            for y in range(grid_map.get_height()):
                if grid_map[x, y] == self.occupied_space:
                    for dx, dy in offsets:
                        x_p = x + dx
                        y_p = y + dy
                        # print(x_p, y_p)
                        if self.is_in_bounds(grid_map, x_p, y_p) and grid_map[x_p, y_p] != self.occupied_space:
                            self.add_to_map(grid_map, x_p, y_p, self.c_space)

                            # Return the inflated map
        return grid_map


def rotate(v, alpha):
    M = np.array([
        [cos(alpha), -sin(alpha)],
        [sin(alpha), cos(alpha)]
    ])
    return np.dot(M, v)
