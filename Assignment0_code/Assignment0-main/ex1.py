import numpy as np

def dot_product(A, B):
    # TODO: Compute the dot product of matrices A and B

    result = np.dot(A, B)
    return result

def matrix_transpose(A):
    # TODO: Compute the transpose of matrix A
    result = A.T
    return result

def matrix_determinant(A):
    # TODO: Compute the determinant of matrix A
    result = np.linalg.det(A)
    return result

def matrix_inverse(A):
    # TODO: Compute the inverse of matrix A
    result = np.linalg.inv(A)
    return result

def matrix_rank(B):
    # TODO: Calculate the rank of matrix B
    result = np.linalg.matrix_rank(B)
    return result

def vector_norm(B):
    # TODO: Calculate the vector norm of B
    result = np.linalg.norm(B)
    return result

def multiply_matrix_vector(A, vector):
    # TODO: Multiply matrix A by a vector
    result = A @ vector
    return result

def main():
    # TODO: Define matrices A and B with appropriate values
    A = np.array([[1, 2],
                  [3, 4]])  
    B = np.array([[5, 6, 7],
                  [8, 9, 10]])  
    
    # Perform operations
    A_dot_B = dot_product(A, B)
    A_transpose = matrix_transpose(A)
    A_det = matrix_determinant(A)
    A_inv = matrix_inverse(A)
    B_rank = matrix_rank(B)
    B_norm = vector_norm(B)
    
    # TODO: Define a vector with appropriate values
    vector = np.array([8, 8])
  # Populate with appropriate values
    
    A_vector = multiply_matrix_vector(A, vector)
    vector_norm_result = vector_norm(A_vector)

    # Print results
    print("Dot product of A and B:", A_dot_B)
    print("Transpose of A:", A_transpose)
    print("Determinant of A:", A_det)
    print("Inverse of A:", A_inv)
    print("Rank of B:", B_rank)
    print("Norm of B:", B_norm)
    print("A multiplied by vector:", A_vector)
    print("Norm of A*vector:", vector_norm_result)

if __name__ == "__main__":
    main()