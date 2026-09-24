import numpy as np
import matplotlib.pyplot as plt
import pickle
import sys
sys.path.append("..")

from common.robot_info import load_data
from get_joint_torques import get_joint_torques
from get_grav_comp_torques import get_grav_comp_torques

def sample_torques():
    data = load_data()
    theta = data['theta']

    # Calculate joint torques for (1) applying a desired force and (2) gravity compensation
    n = len(theta)
    tau_desired_force = np.zeros((n, 2))
    tau_grav_comp = np.zeros((n, 2))

    # End effector should exert this force [N]
    desired_force = np.array([-1, 1])

    for i in range(n):
        # Get the joint angle for this timestep
        th = theta[i, :]

        # Get the torques needed to apply an end effector force
        tau = get_joint_torques(th, desired_force)
        tau_desired_force[i] = tau

        # Get the torques needed to compensate for gravity in the negative y direction
        tau = get_grav_comp_torques(th, np.array([0, -9.8]))
        tau_grav_comp[i] = tau

    # Plot actual data
    # Torques for desired force
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
    ax1.plot(tau_desired_force[:, 0], 'k-', linewidth=1)
    ax1.set_title('Joint 1 torque for end effector to exert force of [-1, 1] N')
    ax1.set_xlabel('timestep')
    ax1.set_ylabel('Joint torque [Nm]')
    ax2.plot(tau_desired_force[:, 1], 'k-', linewidth=1)
    ax2.set_title('Joint 2 torque for end effector to exert force of [-1, 1] N')
    ax2.set_xlabel('timestep')
    ax2.set_ylabel('Joint torque [Nm]')
    plt.tight_layout()
    plt.savefig('torques_ee.png')
    plt.close()

    # Torques for gravity compensation
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
    ax1.plot(tau_grav_comp[:, 0], 'k-', linewidth=1)
    ax1.set_title('Joint 1 torque for gravity compensation')
    ax1.set_xlabel('timestep')
    ax1.set_ylabel('Joint torque [Nm]')
    ax2.plot(tau_grav_comp[:, 1], 'k-', linewidth=1)
    ax2.set_title('Joint 2 torque for gravity compensation')
    ax2.set_xlabel('timestep')
    ax2.set_ylabel('Joint torque [Nm]')
    plt.tight_layout()
    plt.savefig('torques_grav_comp.png')
    plt.close()

    # Save computed torques as pickle files
    with open('tau_desired_force.pkl', 'wb') as f:
        pickle.dump(tau_desired_force, f)

    with open('tau_grav_comp.pkl', 'wb') as f:
        pickle.dump(tau_grav_comp, f)

if __name__ == "__main__":
    sample_torques()
