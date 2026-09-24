import numpy as np
import matplotlib.pyplot as plt
import pickle
import sys
sys.path.append('..')

from common.robot_info import robot_info, load_data
from forward_kinematics_RR import forward_kinematics_RR
from jacobian_link_ends_RR import jacobian_link_ends_RR
from jacobian_coms_RR import jacobian_coms_RR

def sample_path():
    # Load a subset of the sample data so that the plot is less cluttered.
    data = load_data()
    theta = data['theta'][2400:3200, :]
    theta_dot = data['theta_dot'][2400:3200, :]

    n = len(theta)
    x = np.zeros((n, 4))
    y = np.zeros((n, 4))
    x_dot = np.zeros((n, 4))
    y_dot = np.zeros((n, 4))

    robot = robot_info()
    l1, l2 = robot['link_lengths']

    # --------------- BEGIN STUDENT SECTION ----------------------------------
    # Fill in x/y positions and velocities for each point, using the forward
    # kinematics and Jacobian functions you have written.
    for i in range(n):
        th = theta[i, :]
        th_dot = theta_dot[i, :]
        frames = forward_kinematics_RR(th)
        jac_coms = jacobian_coms_RR(th)
        jac_ends = jacobian_link_ends_RR(th)
        H_2_0 = frames['H_2_0']

        # Fill in center of mass of link 1 position/velocity:
        x[i, 0] = H_2_0[0, 2] / 2
        y[i, 0] = H_2_0[1, 2] / 2
        velocity_com_1 = jac_coms['J_COM_1'] @ th_dot
        x_dot[i, 0] = velocity_com_1[0]
        y_dot[i, 0] = velocity_com_1[1]

        # Fill in distal end of link 1 position/velocity:
        x[i, 1] = H_2_0[0, 2]
        y[i, 1] = H_2_0[1, 2]

        velocity_end_1 = jac_ends['J_END_1'] @ th_dot
        x_dot[i, 1] = velocity_end_1[0]
        y_dot[i, 1] = velocity_end_1[1]

        # Fill in center of mass of link 2 position/velocity:
        H_3_0 = frames['H_3_0']

        x[i, 2] = (H_2_0[0, 2] + H_3_0[0, 2]) / 2
        y[i, 2] = (H_2_0[1, 2] + H_3_0[1, 2]) / 2

        velocity_com_2 = jac_coms['J_COM_2'] @ th_dot
        x_dot[i, 2] = velocity_com_2[0]
        y_dot[i, 2] = velocity_com_2[1]

        # Fill in distal end of link 2 position/velocity:
        x[i, 3] = H_3_0[0, 2]
        y[i, 3] = H_3_0[1, 2]

        velocity_end_2 = jac_ends['J_END_2'] @ th_dot
        x_dot[i, 3] = velocity_end_2[0]
        y_dot[i, 3] = velocity_end_2[1]
    # --------------- END STUDENT SECTION ------------------------------------

    # Save the results to a pickle file
    results = {
        'COM_Link_1': {
            'x': x[:, 0],
            'y': y[:, 0],
            'x_dot': x_dot[:, 0],
            'y_dot': y_dot[:, 0]
        },
        'End_Link_1': {
            'x': x[:, 1],
            'y': y[:, 1],
            'x_dot': x_dot[:, 1],
            'y_dot': y_dot[:, 1]
        },
        'COM_Link_2': {
            'x': x[:, 2],
            'y': y[:, 2],
            'x_dot': x_dot[:, 2],
            'y_dot': y_dot[:, 2]
        },
        'End_Link_2': {
            'x': x[:, 3],
            'y': y[:, 3],
            'x_dot': x_dot[:, 3],
            'y_dot': y_dot[:, 3]
        }
    }
    with open('results.pkl', 'wb') as f:
        pickle.dump(results, f)

    subsample_resolution = 25
    x_sub = x[::subsample_resolution, :]
    y_sub = y[::subsample_resolution, :]
    x_dot_sub = x_dot[::subsample_resolution, :]
    y_dot_sub = y_dot[::subsample_resolution, :]

    plt.figure(figsize=(10, 10))
    colors = ['#FFB3B3', '#FF0000', '#808080', '#000000']
    labels = ['COM Link 1', 'End Link 1', 'COM Link 2', 'End Link 2']
    
    for i in range(4):
        plt.plot(x[:, i], y[:, i], color=colors[i], label=labels[i], alpha=0.5)
        
        scale = 0.1
        plt.quiver(x_sub[:, i], y_sub[:, i],
                   x_dot_sub[:, i], y_dot_sub[:, i],
                   scale=scale, color=colors[i], width=0.003)

    plt.title('Plot of link positions and velocities over a sample run.')
    plt.xlabel('x [m]')
    plt.ylabel('y [m]')
    plt.axis('equal')
    plt.legend()
    
    margin = 0.1
    plt.xlim([np.min(x) - margin, np.max(x) + margin])
    plt.ylim([np.min(y) - margin, np.max(y) + margin])
    
    plt.grid(True)
    plt.savefig('path.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    sample_path()
