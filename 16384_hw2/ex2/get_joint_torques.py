import sys
sys.path.append("..")

import numpy as np
from ex1.jacobian_link_ends_RR import jacobian_link_ends_RR

def get_joint_torques(theta, desired_force):
    """
    Calculates the joint torques required to result in a desired force
    vector (in world coordinates).

    Args:
    theta (np.array): Joint angles
    desired_force (np.array): Desired force vector (2x1)

    Returns:
    np.array: Joint torques
    """
    theta = np.squeeze(theta)
    desired_force = np.squeeze(desired_force)

    # --------------- BEGIN STUDENT SECTION ----------------------------------
    # Use the Jacobian to find the joint torques necessary for the end
    # effector to exert the given force (given the joint configuration theta).
    # Assume 'desired_force' is a 2x1 (column) vector.
    # Hint: You may find the jacobian_link_ends_RR function useful.

    # Your code here

    jacobians = jacobian_link_ends_RR(theta)
    J = jacobians['J_END_2']

    torques = J.T @ desired_force

    torque1 = torques[0]
    torque2 = torques[1]

    # --------------- END STUDENT SECTION ------------------------------------

    # Pack into a more readable format.
    return np.array([torque1, torque2])
