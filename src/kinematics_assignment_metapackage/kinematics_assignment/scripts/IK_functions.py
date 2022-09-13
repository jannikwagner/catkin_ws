#! /usr/bin/env python3

"""
    # {Jannik Wagner}
    # {wagne@kth.se}
"""

import math
import random

scara_l = 0.07, 0.3, 0.35


def scara_IK(point):

    q = geometric(point)

    """
    Fill in your IK solution here and return the three joint values in q
    """

    return q


def analytic(point):
    l0, l1, l2 = scara_l
    x, y, z = point

    q3 = z

    x_W = x - l0

    c2 = (x_W**2 + y**2 - l1**2 - l2**2) / (2 * l1 * l2)
    s2 = math.sqrt(1-c2**2)  # add sign

    q2 = math.atan2(s2, c2)

    s1 = ((l1 + l2 * c2) * y - l2 * s2 * x_W) / (x_W**2 + y**2)
    c1 = ((l1 + l2 * c2) * x_W - l2 * s2 * y) / (x_W**2 + y**2)

    q1 = math.atan2(s1, c1)

    q = [q1, q2, q3]
    return q


def geometric(point):
    l0, l1, l2 = scara_l
    x, y, z = point

    q3 = z

    x_W = x - l0

    c2 = (x_W**2 + y**2 - l1**2 - l2**2) / (2 * l1 * l2)
    q2 = math.acos(c2)

    alpha = math.atan2(y, x_W)
    beta = math.acos((x_W**2 + y**2 + l1**2 - l2**2) /
                     (2 * l1 * math.sqrt(x_W**2 + y**2)))

    q1 = alpha + beta * (1 if q2 < 0 else -1)

    q = [q1, q2, q3]
    return q


def kuka_IK(point, R, joint_positions):
    x = point[0]
    y = point[1]
    z = point[2]
    q = joint_positions  # it must contain 7 elements

    """
    Fill in your IK solution here and return the seven joint values in q
    """

    return q


def scara_FK(q):
    q1, q2, q3 = q
    l0, l1, l2 = scara_l

    x = l0 + math.cos(q1) * l1 + math.cos(q1+q2) * l2
    y = math.sin(q1) * l1 + math.sin(q1+q2) * l2
    z = q3

    return (x, y, z)


def test():

    l0, l1, l2 = scara_l

    point = (l0+l1+l2-0.1, 0, 0)
    print(scara_IK(point))

    for i in range(100):

        q = random.random()*2*math.pi, random.random()*2*math.pi, random.random()
        print("q:", q)
        point = scara_FK(q)
        print("x:", point)
        q_ana = analytic(point)
        q_geo = geometric(point)
        print("q_ana:", q_ana)
        print("q_geo:", q_geo)


if __name__ == "__main__":
    test()
