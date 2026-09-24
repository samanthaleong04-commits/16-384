import numpy as np


# Example usage
# joints = [3.14, 2.5, 1.5]
# linkLengths = [0.375, 0.310, 0.1]
def endEffector_RRR(joints, linkLengths):
    # TODO
    s1 = np.sin(joints[0])
    c1 = np.cos(joints[0])
    s12 = np.sin(joints[0] + joints[1])
    c12 = np.cos(joints[0] + joints[1])
    s123 = np.sin(joints[0] + joints[1] + joints[2])
    c123 = np.cos(joints[0] + joints[1] + joints[2])

    l1 = linkLengths[0]
    l2 = linkLengths[1]    
    l3 = linkLengths[2] 

    x = l1 * c1 + l2 * c12 + l3 * c123
    y = l1 * s1 + l2 * s12 + l3 * s123

    return x, y


def workspace_analysis_RRR(
        nSamples = 50, minJoint=[0, 0, 0], 
        maxJoint=[2*np.pi, 2*np.pi, 2*np.pi], 
        linkLengths=[0.3, 0.2, 0.1]):
    
    theta_1 = np.linspace(minJoint[0], maxJoint[0], nSamples)
    theta_2 = np.linspace(minJoint[1], maxJoint[1], nSamples)
    theta_3 = np.linspace(minJoint[2], maxJoint[2], nSamples)
    xs = np.zeros(nSamples**3)
    ys = np.zeros(nSamples**3)
    for i in range(nSamples):
        for j in range(nSamples):
            for k in range(nSamples):

                joints = [theta_1[i], theta_2[j], theta_3[k]]
                index = i * nSamples**2 + j * nSamples + k
                xs[index], ys[index] = endEffector_RRR(joints, linkLengths)
    return xs, ys

