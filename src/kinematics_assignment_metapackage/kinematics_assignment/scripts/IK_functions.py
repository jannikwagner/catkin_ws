#! /usr/bin/env python3

"""
    # {Jannik Wagner}
    # {wagne@kth.se}
"""

import math
import random
import numpy as np

SCARA_L = 0.07, 0.3, 0.35

M = 0.39
L = 0.4
kuka_values = [0.322, L, M, 0.078]  # ?

KUKA_DH = np.array([
    [math.pi / 2, 0, 0],
    [- math.pi / 2, 0, 0],
    [- math.pi / 2, L, 0],
    [math.pi / 2, 0, 0],
    [math.pi / 2, M, 0],
    [- math.pi / 2, 0, 0],
    [0, 0, 0],
])


def scara_IK(point):

    q = geometric_scara_IK(point)

    """
    Fill in your IK solution here and return the three joint values in q
    """

    return q


def analytic_scara_IK(point):
    l0, l1, l2 = SCARA_L
    x, y, z = point

    q3 = z

    x_W = x - l0

    c2 = (x_W**2 + y**2 - l1**2 - l2**2) / (2 * l1 * l2)
    s2 = math.sqrt(1-c2**2)  # add sign

    q2 = math.atan2(s2, c2)

    s1 = ((l1 + l2 * c2) * y - l2 * s2 * x_W) / (x_W**2 + y**2)
    c1 = ((l1 + l2 * c2) * x_W - l2 * s2 * y) / (x_W**2 + y**2)

    q1 = math.atan2(s1, c1)

    q = np.array((q1, q2, q3))
    return q


def geometric_scara_IK(point):
    l0, l1, l2 = SCARA_L
    x, y, z = point

    q3 = z

    x_W = x - l0

    c2 = (x_W**2 + y**2 - l1**2 - l2**2) / (2 * l1 * l2)
    q2 = math.acos(c2)

    alpha = math.atan2(y, x_W)
    beta = math.acos((x_W**2 + y**2 + l1**2 - l2**2) /
                     (2 * l1 * math.sqrt(x_W**2 + y**2)))

    q1 = alpha + beta * (1 if q2 < 0 else -1)

    q = np.array((q1, q2, q3))
    return q


def scara_FK(q):
    q1, q2, q3 = q
    l0, l1, l2 = SCARA_L

    x = l0 + math.cos(q1) * l1 + math.cos(q1+q2) * l2
    y = math.sin(q1) * l1 + math.sin(q1+q2) * l2
    z = q3

    return np.array((x, y, z))


def kuka_IK(point, R, joint_positions):
    euler_desired = get_euler_rotation(R)
    X_desired = np.array((*point, *euler_desired))

    threshold = 0.1
    q_current = np.array(joint_positions)
    X_current = kuka_FK(q_current)
    while np.sum((X_current - X_desired)**2) > threshold:
        X_diff = X_current - X_desired
        J = get_jacobian(q_current)
        J_inv = invert(J)
        q_diff = J_inv @ X_diff
        q_current -= q_diff
        X_current = kuka_FK(q_current)

    """
    Fill in your IK solution here and return the seven joint values in q
    """

    return q_current


def kuka_FK(q):
    T = np.eye(4)
    euler = np.zeros(3)
    for (a_i, alpha_i, d_i), q_i in zip(KUKA_DH, q):
        im1_T_i = get_transform(a_i, alpha_i, d_i, q_i)
        T = T @ im1_T_i
        euler = update_euler(euler, alpha_i, q_i)

    r = T @ np.array([0, 0, 0, 1])
    return np.array((*r, *euler))


def get_transform(a, alpha, d, q):
    raise NotImplementedError()


def update_euler(euler, alpha_i, q_i):
    raise NotImplementedError()


def get_euler_rotation(R):
    raise NotImplementedError()


def get_jacobian(q):
    raise NotImplementedError()


def invert(J):
    raise NotImplementedError()


def test_scara():
    VERBOSE = False
    l0, l1, l2 = SCARA_L

    point = (l0+l1+l2-0.1, 0, 0)
    print(scara_IK(point))

    gamma_0 = 0*math.pi
    d_gamma = 2*math.pi

    for _ in range(100):

        q = gamma_0+random.random()*d_gamma, gamma_0+random.random() * \
            d_gamma, random.random()
        point = scara_FK(q)

        q_ana = analytic_scara_IK(point)
        q_geo = geometric_scara_IK(point)

        x_ana = scara_FK(q_ana)
        x_geo = scara_FK(q_geo)

        if VERBOSE or np.sum((point - x_geo)**2) > 10**-10:
            print("q:", q)
            print("x:", point)
            print("q_ana:", q_ana)
            print("q_geo:", q_geo)
            print("x_ana:", x_ana)
            print("x_geo:", x_geo)
            print(np.sum((point - x_geo)**2))
            print(_)


def test_kuka():
    pass


if __name__ == "__main__":
    # test_scara()
    test_kuka()
