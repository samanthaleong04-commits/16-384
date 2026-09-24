import numpy as np

# Joint origins of the xArm7, from UFACTORY's model, in mm.
# Shoulder (joint 1) to elbow (joint 4 axis): 293 out, 52.5 across.
# Elbow to wrist (joint 7 axis): 418.5 back the other way, 77.5 across.
LINK1_MM = np.array([293.0, 52.5])
LINK2_MM = np.array([-418.5, 77.5])

LOCKED_INDICES = np.array([1, 2, 4, 5])
LOCKED_ANGLES_DEG = np.array([90.0, 90.0, -90.0, 90.0])
RECORD_RATE = 100.0


def robot_info():
    robot_info = {}
    # length of the links [m], from the joint origins above
    robot_info['link_lengths'] = np.array([
        np.hypot(*LINK1_MM) / 1000.0,  # 0.2977 m
        np.hypot(*LINK2_MM) / 1000.0,  # 0.4256 m
    ])
    return robot_info


def q2rr(q):
    """(theta1, theta2) of the planar RR, from a 7-joint configuration."""
    A1 = np.arctan2(LINK1_MM[1], LINK1_MM[0])  # 10.16 deg
    A2 = np.arctan2(LINK2_MM[1], LINK2_MM[0]) - A1  # 159.35 deg
    return q[0] + A1, A2 - q[3]


def qd2rr(qd):
    """(theta1_dot, theta2_dot) of the planar RR, from 7 joint velocities."""
    return qd[0], -qd[3]


def rr2q(theta1, theta2):
    """The 7-joint configuration at the planar RR's (theta1, theta2)."""
    A1 = np.arctan2(LINK1_MM[1], LINK1_MM[0])  # 10.16 deg
    A2 = np.arctan2(LINK2_MM[1], LINK2_MM[0]) - A1  # 159.35 deg
    q = np.zeros(7)
    q[LOCKED_INDICES] = np.radians(LOCKED_ANGLES_DEG)
    q[0], q[3] = theta1 - A1, A2 - theta2
    return q
