#!/usr/bin/env python3

# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

# Author: Ryan De Iaco
# Additional Comments: Carlos Wang
# Date: October 29, 2018

import numpy as np
from math import sin, cos, pi, sqrt

class VelocityPlanner:
    def __init__(self, time_gap, a_max, slow_speed, stop_line_buffer):
        self._time_gap         = time_gap
        self._a_max            = a_max
        self._slow_speed       = slow_speed
        self._stop_line_buffer = stop_line_buffer
        self._prev_trajectory  = [[0.0, 0.0, 0.0]]

    # Computes an open loop speed estimate based on the previously planned
    # trajectory, and the timestep since the last planning cycle.
    # Input: timestep is in seconds
    def get_open_loop_speed(self, timestep):
        if len(self._prev_trajectory) == 1:
            return self._prev_trajectory[0][2] 
        
        # If simulation time step is zero, give the start of the trajectory as the
        # open loop estimate.
        if timestep < 1e-4:
            return self._prev_trajectory[0][2]

        for i in range(len(self._prev_trajectory)-1):
            distance_step = np.linalg.norm(np.subtract(self._prev_trajectory[i+1][0:2], 
                                                       self._prev_trajectory[i][0:2]))
            velocity = self._prev_trajectory[i][2]
            time_delta = distance_step / velocity
           
            # If time_delta exceeds the remaining time in our simulation timestep, 
            # interpolate between the velocity of the current step and the velocity
            # of the next step to estimate the open loop velocity.
            if time_delta > timestep:
                v1 = self._prev_trajectory[i][2]
                v2 = self._prev_trajectory[i+1][2]
                v_delta = v2 - v1
                interpolation_ratio = timestep / time_delta
                return v1 + interpolation_ratio * v_delta

            # Otherwise, keep checking.
            else:
                timestep -= time_delta

        # Simulation time step exceeded the length of the path, which means we have likely
        # stopped. Return the end velocity of the trajectory.
        return self._prev_trajectory[-1][2]

    def compute_velocity_profile(self, path, desired_speed, ego_state, closed_loop_speed):
        profile = []
        start_speed = ego_state[3]
        profile = self.nominal_profile(path, start_speed, desired_speed)
        if len(profile) > 1:
            interpolated_state = [(profile[1][0] - profile[0][0]) * 0.1 + profile[0][0], 
                                  (profile[1][1] - profile[0][1]) * 0.1 + profile[0][1], 
                                  (profile[1][2] - profile[0][2]) * 0.1 + profile[0][2]]
            del profile[0]
            profile.insert(0, interpolated_state)

        self._prev_trajectory = profile

        return profile

    
    def nominal_profile(self, path, start_speed, desired_speed):

        profile = []
        # Compute distance travelled from start speed to desired speed using
        # a constant acceleration.
        if desired_speed < start_speed:
            accel_distance = calc_distance(start_speed, desired_speed, -self._a_max)
        else:
            accel_distance = calc_distance(start_speed, desired_speed, self._a_max)

        # Here we will compute the end of the ramp for our velocity profile.
        # At the end of the ramp, we will maintain our final speed.
        ramp_end_index = 0
        distance = 0.0
        while (ramp_end_index < len(path[0])-1) and (distance < accel_distance):
            distance += np.linalg.norm([path[0][ramp_end_index+1] - path[0][ramp_end_index], 
                                        path[1][ramp_end_index+1] - path[1][ramp_end_index]])
            ramp_end_index += 1

        # Here we will actually compute the velocities along the ramp.
        vi = start_speed
        for i in range(ramp_end_index):
            dist = np.linalg.norm([path[0][i+1] - path[0][i], 
                                   path[1][i+1] - path[1][i]])
            if desired_speed < start_speed:
                vf = calc_final_speed(vi, -self._a_max, dist)
                # clamp speed to desired speed
                if vf < desired_speed:
                    vf = desired_speed
            else:
                vf = calc_final_speed(vi, self._a_max, dist)
                # clamp speed to desired speed
                if vf > desired_speed:
                    vf = desired_speed

            profile.append([path[0][i], path[1][i], vi])
            vi = vf

        # If the ramp is over, then for the rest of the profile we should
        # track the desired speed.
        for i in range(ramp_end_index+1, len(path[0])):
            profile.append([path[0][i], path[1][i], desired_speed])

        return profile

def calc_distance(v_i, v_f, a):
    """Computes the distance given an initial and final speed, with a constant
    acceleration.
    
    args:
        v_i: initial speed (m/s)
        v_f: final speed (m/s)
        a: acceleration (m/s^2)
    returns:
        d: the final distance (m)
    """
    pass

    # TODO: INSERT YOUR CODE BETWEEN THE DASHED LINES
    # ------------------------------------------------------------------
    d = ( np.square(v_f) - np.square(v_i) ) / 2 * a
    return d
    # ------------------------------------------------------------------

######################################################
######################################################
# MODULE 7: COMPUTE FINAL SPEED WITH CONSTANT ACCELERATION
#   Read over the function comments to familiarize yourself with the
#   arguments and necessary variables to return. Then follow the TODOs
#   (top-down) and use the surrounding comments as a guide.
######################################################
######################################################
# Using v_f = sqrt(v_i^2 + 2ad), compute the final speed for a given
# acceleration across a given distance, with initial speed v_i.
# Make sure to check the discriminant of the radical. If it is negative,
# return zero as the final speed.
def calc_final_speed(v_i, a, d):
    """Computes the final speed given an initial speed, distance travelled, 
    and a constant acceleration.
    
    args:
        v_i: initial speed (m/s)
        a: acceleration (m/s^2)
        d: distance to be travelled (m)
    returns:
        v_f: the final speed (m/s)
    """
    pass

    # TODO: INSERT YOUR CODE BETWEEN THE DASHED LINES
    # ------------------------------------------------------------------
    v_f = np.sqrt(abs(np.square(v_i) + 2 * a * d ))
    return v_f
    # ------------------------------------------------------------------

