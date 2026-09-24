import numpy as np

def forward_kinematics_RRR(theta1, theta2, theta3, l1, l2, l3):
    """
    Returns the forward kinematics for an RRR robot given the joint angle positions in radians.
    """
    # NOTE: The convention is that links in 2D lie along the x-axis of the starting frame, 
    # and the joint angles are measured counter-clockwise from the x-axis of the previous link.
    
    # To-Do 1: Compute the homogeneous transformation matrices H_1_0, H_2_1, H_3_2, H_4_3, H_5_4, H_6_5
    H_1_0 = np.array([
    [np.cos(theta1), -np.sin(theta1), 0],
    [np.sin(theta1), np.cos(theta1), 0],
    [0, 0, 1]
    ])
    H_2_1 = np.array([
    [1, 0, l1],
    [0, 1, 0],
    [0, 0, 1]
    ])
    H_3_2 = np.array([
    [np.cos(theta2), -np.sin(theta2), 0],
    [np.sin(theta2), np.cos(theta2), 0],
    [0, 0, 1]
    ])
    H_4_3 = np.array([
    [1, 0, l2],
    [0, 1, 0],
    [0, 0, 1]
    ])
    H_5_4 = np.array([
    [np.cos(theta3), -np.sin(theta3), 0],
    [np.sin(theta3), np.cos(theta3), 0],
    [0, 0, 1]
    ])
    H_6_5 = np.array([
    [1, 0, l3],
    [0, 1, 0],
    [0, 0, 1]
    ])
    
    # To-Do 2: Compute the homogeneous transformation matrix H_6_0
    H_6_0 = H_1_0 @ H_2_1 @ H_3_2 @ H_4_3 @ H_5_4 @ H_6_5 # TODO
    
    return {
        'H_1_0': H_1_0,
        'H_2_1': H_2_1,
        'H_3_2': H_3_2,
        'H_4_3': H_4_3,
        'H_5_4': H_5_4,
        'H_6_5': H_6_5,
        'H_6_0': H_6_0
    }