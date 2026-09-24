import numpy as np
import matplotlib.pyplot as plt

'''
EXAMPLE METHODS
'''

# Example usage
# joints = [0.5, 0.3] (rad, rad)
# linkLengths = [0.310, 0.1] (m, m)
# workspace_analysis_RR(thetas, linkLengths)

def endEffector_RR(joints, linkLengths):
    s1 = np.sin(joints[0])
    c1 = np.cos(joints[0])
    s12 = np.sin(joints[0] + joints[1])
    c12 = np.cos(joints[0] + joints[1])

    l1 = linkLengths[0]
    l2 = linkLengths[1]

    x = l1 * c1 + l2 * c12
    y = l1 * s1 + l2 * s12
    return x, y

def workspace_analysis_RR(
        nSamples = 50, minJoint=[0, 0], 
        maxJoint=[2 *np.pi, 2*np.pi], 
        linkLengths=[0.3, 0.1]):

    theta_1 = np.linspace(minJoint[0], maxJoint[0], nSamples)
    theta_2 = np.linspace(minJoint[1], maxJoint[1], nSamples)
    xs = np.zeros(nSamples**2)
    ys = np.zeros(nSamples**2)

    for i in range(nSamples):
        for j in range(nSamples):
            joints = [theta_1[i], theta_2[j]]
            index = i * nSamples + j
            xs[index], ys[index] = endEffector_RR(joints, linkLengths)

    return xs, ys


