import numpy as np

from robot_info import robot_info


def forward_kinematics_RR(theta1, theta2):
    """
    Returns the forward kinematics for an RR robot given the joint angle positions in radians.
    """
    l1, l2 = robot_info()['link_lengths']

    # ============ BEGIN STUDENT SECTION ==============
    # To-Do 1: Compute the homogeneous transformation matrices H_2_0 and H_3_0

    s1 = np.sin(theta1)
    c1 = np.cos(theta1)
    c12 = np.cos(theta1 + theta2)
    s12 = np.sin(theta1 + theta2)

    H_2_0 = np.array([
        [c12, -s12, l1*c1],
        [s12,  c12, l1*s1],
        [0,    0,   1]
    ])
    H_3_0 = np.array([
        [c12, -s12, l1*c1 + l2*c12],
        [s12,  c12, l1*s1 + l2*s12],
        [0,    0,   1]
    ])  
    # ============ END STUDENT SECTION ==============

    return {
        'H_2_0': H_2_0,  # the elbow, in the base frame
        'H_3_0': H_3_0,  # the end effector, in the base frame
    }


def jacobian_RR(theta1, theta2):
    """
    Returns the end-effector Jacobian of an RR robot given the joint angle positions
    in radians.
    """
    l1, l2 = robot_info()['link_lengths']

    # ============ BEGIN STUDENT SECTION ==============
    # To-Do 2: Compute the Jacobian matrix J
    c1 = np.cos(theta1)
    s1 = np.sin(theta1)
    c12 = np.cos(theta1 + theta2)
    s12 = np.sin(theta1 + theta2)

    J = np.array([
        [-l1*s1 - l2*s12,   -l2*s12],
        [ l1*c1 + l2*c12,    l2*c12]
    ])
    # ============ END STUDENT SECTION ==============

    return J


def load_trajectory(path):
    """
    Loads a trajectory recorded by `record.py` and returns its RR joint angles.
    """
    data = np.loadtxt(path, delimiter=',')
    theta1 = data[:, 0]
    theta2 = data[:, 1]
    return theta1, theta2


def run_trajectory(arm, path):
    """
    Plays the recording at `path` on the arm. `arm` gives you two methods:
        arm.set_position(theta1, theta2)       moves the arm to one pose and waits
                                               until it has arrived there
        arm.servo_to_position(theta1, theta2)  streams one sample to the arm and waits
                                               until it is time for the next one
    Streaming has to start from where the arm already is, so move it to the first
    sample before servoing through the rest.
    """
    # 2 Length-N arrays
    theta1, theta2 = load_trajectory(path)
    arm.set_position(theta1[0], theta2[0])
    for th1, th2 in zip(theta1[1:], theta2[1:]):
        arm.servo_to_position(th1, th2)
