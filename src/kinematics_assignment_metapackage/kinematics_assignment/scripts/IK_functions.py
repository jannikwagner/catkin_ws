#! /usr/bin/env python3

"""
    # {Jannik Wagner}
    # {wagne@kth.se}
"""

import math
import random
import numpy as np

SCARA_L = 0.07, 0.3, 0.35

D4 = 0.078
M = 0.39
L = 0.4
D1 = 0.311
kuka_values = [D1, L, M, D4]  # ?

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
    T = 200
    threshold = 1e-6

    x_desired = np.array(point)
    R_desired = np.array(R)
    q_current = np.array(joint_positions, dtype=np.float64)

    for t in range(T):
        x_current, R_current, Ts = kuka_FK(q_current)
        dx = x_current - x_desired
        # dR = R_current - R_desired
        dR = d_rot3(R_desired, R_current)
        dX = np.concatenate((dx, dR))

        J = get_jacobian(x_current, Ts)
        J_inv = invert(J)
        dq = J_inv @ dX
        q_current -= dq

        if np.sum(dX**2) < threshold:
            break

    return q_current


def d_rot3(R1, R2):
    return 0.5 * (np.cross(R1[:, 0], R2[:, 0]) + np.cross(R1[:, 1], R2[:, 1]) + np.cross(R1[:, 2], R2[:, 2]))


def kuka_FK(q):
    T = np.eye(4)
    Ts = [T]
    for (alpha_i, d_i, a_i), q_i in zip(KUKA_DH, q):
        im1_T_i = get_transform(alpha_i, d_i, a_i, q_i)
        T = T @ im1_T_i
        Ts.append(T)

    R = T[:3, :3]
    r = T @ np.array([0, 0, D1, 1])
    r = r[:3] / r[3]
    # what about D4?
    return r, R, Ts


def get_transform(alpha, d, a, theta):
    Trans_z_d = get_translation_z(d)
    R_z_q = get_rotation_z(theta)
    Trans_x_a = get_translation_x(a)
    R_x_alpha = get_rotation_x(alpha)

    T = Trans_z_d @ R_z_q @ Trans_x_a @ R_x_alpha

    return T


def get_translation_z(d):
    return np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, d],
        [0, 0, 0, 1]
    ]).astype(float)


def get_rotation_z(theta):
    return np.array([
        [math.cos(theta), - math.sin(theta), 0, 0],
        [math.sin(theta), math.cos(theta), 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ]).astype(float)


def get_translation_x(a):
    return np.array([
        [1, 0, 0, a],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ]).astype(float)


def get_rotation_x(alpha):
    return np.array([
        [1, 0, 0, 0],
        [0, math.cos(alpha), -math.sin(alpha), 0],
        [0, math.sin(alpha), math.cos(alpha), 0],
        [0, 0, 0, 1]
    ]).astype(float)


def get_jacobian(X, Ts):
    J = []

    pe = X
    p0 = np.array([0, 0, D4, 1])
    z0 = np.array([0, 0, 1])

    for t in range(len(Ts)-1):
        T = Ts[t]
        R = T[:3, :3]
        z = R @ z0
        p = T @ p0
        p = p[:3] / p[3]
        jp = np.cross(z, pe - p)
        jo = z
        J.append(np.concatenate((jp, jo)))

    J = np.array(J).T
    return J


def invert(J):
    return np.linalg.pinv(J)


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
    VERBOSE = False
    q0 = (0, 0, 0, 0, 0, 0, 0)

    point = (D1+L+M+D4-0.1, 0, 0)
    R = np.eye(3)
    print(kuka_IK(point, R, q0))

    gamma_0 = 0*math.pi
    d_gamma = 2*math.pi

    for _ in range(100):

        q = gamma_0+np.random.rand(7)*d_gamma
        point, R, Ts = kuka_FK(q)

        q_ana = kuka_IK(point, R, q0)

        x_ana, R_ana, Ts_ana = kuka_FK(q_ana)

        if VERBOSE or np.sum((point - x_ana)**2) > 10**-6:
            print(_)
            print("q:", q)
            print("x:", point)
            print("q_ana:", q_ana)
            print("x_ana:", x_ana)
            print(np.sum((point - x_ana)**2))


if __name__ == "__main__":
    # test_scara()
    test_kuka()
