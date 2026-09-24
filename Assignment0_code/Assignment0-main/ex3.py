import numpy as np

def main():
    # Define vectors u and v in 3D space
    u = np.array([1, 2, 3])  # Populate with appropriate values
    v = np.array([4, 5, 6])  # Populate with appropriate values

    # TODO: Compute and print the dot product of u and v
    dot_product = np.dot(u, v)

    # TODO: Compute and print the cross product of u and v
    cross_product = np.cross(u, v)

    # TODO: Analyze the geometric relationship based on dot and cross product
    angle_between_vectors = np.arccos(dot_product / (np.linalg.norm(u) * np.linalg.norm(v)))
    
    print("Dot product of u and v:", dot_product)
    print("Cross product of u and v:", cross_product)
    print("Angle between vectors in radians:", angle_between_vectors)

if __name__ == "__main__":
    main()