import sys
sys.path.append('..')

import numpy as np
from common.robot_info import robot_info

def jacobian_coms_RR(theta):
    """
    jacobian_coms_RR

    Returns a vector of Jacobian matrices, corresponding
    to points at the center of mass of each link, given the
    joint angle positions [rad]. The Jacobians computed here
    are relative to R^2 only (i.e., no theta term).

    The function returns N 2xN matrices, where N is the
    number of links (and also the number of joints).

    Each matrix describes the differential relationship between
    a vector of joint angle velocities and the (x,y) motion of the
    corresponding point.

    Hints
    - 'theta' is a vector. Individual angles can be selected
       using indices, e.g., theta1 = theta[0]
    """

    # Get information about the robot:
    robot = robot_info()
    # Extract length of the links
    l1, l2 = robot['link_lengths']

    # Ensure theta is a 1D array
    theta = np.squeeze(theta)

    # --------------- BEGIN STUDENT SECTION ----------------------------------
    # Define the Jacobians for the frames below.  Feel free to define helper
    # variables.
    theta1 = theta[0]
    theta2 = theta[1]

    s1 = np.sin(theta1)
    c1 = np.cos(theta1)

    s12 = np.sin(theta1 + theta2)
    c12 = np.cos(theta1 + theta2)

    J_COM_1 = np.array([
        [-l1/2 * s1, 0],
        [ l1/2 * c1, 0]
    ])

    J_COM_2 = np.array([
        [-l1*s1 - l2/2 * s12, -l2/2 * s12],
        [ l1*c1 + l2/2 * c12,  l2/2 * c12]
    ])

    # --------------- END STUDENT SECTION ------------------------------------

    # Pack into a more readable format. DO NOT CHANGE!
    jacobians = {
        'J_COM_1': J_COM_1,
        'J_COM_2': J_COM_2
    }
    return jacobians
