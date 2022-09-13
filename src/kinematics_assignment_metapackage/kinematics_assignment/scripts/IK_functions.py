#! /usr/bin/env python3

"""
    # {Jannik Wagner}
    # {wagne@kth.se}
"""

import math

def scara_IK(point):
    x = point[0]
    y = point[1]
    z = point[2]

    l0 = 0.07
    l1 = 0.3
    l2 = 0.35

    q3 = z

    x_ = x - l0

    c2 = (x_**2 + y**2 - l1**2 - l2**2) / (2 * l1 * l2)
    s2 = math.sqrt(1-c2**2) # add sign

    q2 = math.atan2(s2, c2)

    s1 = ((l1 + l2 * c2) * y - l2 * s2 * x_) / (x_**2 + y**2)
    c1 = ((l1 + l2 * c2) * x_ - l2 * s2 * y) / (x_**2 + y**2)

    q1 = math.atan2(s1, c1)

    q = [q1, q2, q3]

    """
    Fill in your IK solution here and return the three joint values in q
    """

    return q

def kuka_IK(point, R, joint_positions):
    x = point[0]
    y = point[1]
    z = point[2]
    q = joint_positions #it must contain 7 elements

    """
    Fill in your IK solution here and return the seven joint values in q
    """

    return q
