import sys
sys.path.append("..")

import numpy as np
from common.robot_info import robot_info
from ex1.jacobian_link_ends_RR import jacobian_link_ends_RR
from ex1.jacobian_coms_RR import jacobian_coms_RR

def get_grav_comp_torques(theta, gravity):
    """
    Calculates the joint torques required to cancel out effects due to
    gravity.

    Args:
    theta (np.array): Joint angles
    gravity (np.array): Gravity vector (2x1)

    Returns:
    np.array: Joint torques for gravity compensation
    """
    # Get information about the robot:
    robot = robot_info()

    # Extract mass of the links, joint, and end effector [kg]
    m_link_1 = robot['link_masses'][0]
    m_link_2 = robot['link_masses'][1]
    m_joint_1 = robot['joint_masses'][0]
    m_joint_2 = robot['joint_masses'][1]
    m_end_effector = robot['end_effector_mass']
    theta = np.squeeze(theta)
    gravity = np.squeeze(gravity)

    # --------------- BEGIN STUDENT SECTION ----------------------------------
    # Use the Jacobian to calculate the joint torques to compensate for the
    # weights of the joints, links, and end effector (assuming the acceleration
    # due to gravity is given by 'gravity', and it is a 2x1 (column) vector).
    # Hint: You may find the jacobian_link_ends_RR and jacobian_coms_RR functions useful.

    # Your code here
    jacobian_ends = jacobian_link_ends_RR(theta)
    jacobian_coms = jacobian_coms_RR(theta)

    J_END_1 = jacobian_ends['J_END_1']
    J_END_2 = jacobian_ends['J_END_2']

    J_COM_1 = jacobian_coms['J_COM_1']
    J_COM_2 = jacobian_coms['J_COM_2']

    force_link_1 = m_link_1 * gravity
    force_link_2 = m_link_2 * gravity
    force_joint_2 = m_joint_2 * gravity
    force_end_effector = m_end_effector * gravity

    torque_link_1 = J_COM_1.T @ force_link_1
    torque_link_2 = J_COM_2.T @ force_link_2
    torque_joint_2 = J_END_1.T @ force_joint_2
    torque_end_effector = J_END_2.T @ force_end_effector

    torque_gravity = (
        torque_link_1
        + torque_link_2
        + torque_joint_2
        + torque_end_effector
    )

    torque1 = -torque_gravity[0]
    torque2 = -torque_gravity[1]
    # --------------- END STUDENT SECTION ------------------------------------

    # Pack into a more readable format.
    return np.array([torque1, torque2])
