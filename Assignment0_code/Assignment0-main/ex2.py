import sympy as sp

def main():
    # TODO: Define symbolic variables
    x, y = sp.symbols('x y')  # Replace with the correct code to define symbolic variables

    # TODO: Define the function f(x, y)
    f = x**2 * sp.sin(y) + y**3 * sp.cos(x)  # Replace with the correct function definition

    # TODO: Compute partial derivatives with respect to x and y
    f_x = sp.diff(f, x)
    f_y = sp.diff(f, y)

    # TODO: Evaluate the derivatives at specific points (x, y) = (π, π/2)
    eval_f_x = f_x.subs({x: sp.pi, y: sp.pi/2})
    eval_f_y = f_y.subs({x: sp.pi, y: sp.pi/2})

    # TODO: Compute the directional derivative of f along the vector [1, 1] at the point (1, 2)
    direction = sp.Matrix([1, 1]) / sp.Matrix([1, 1]).norm()  # Define the direction vector
    grad_f = sp.Matrix([sp.diff(f, x), sp.diff(f, y)]).subs({x: 1, y: 2})  # Define the gradient and substitute specific points
    dir_derivative = grad_f.dot(direction)  # Compute the directional derivative

    # Print results
    print("Partial derivative f_x:", f_x)
    print("Partial derivative f_y:", f_y)
    print("Evaluated f_x at (π, π/2):", eval_f_x)
    print("Evaluated f_y at (π, π/2):", eval_f_y)
    print("Directional derivative at (1, 2):", dir_derivative)

if __name__ == '__main__':
    main()