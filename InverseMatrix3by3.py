def calculate_determinant_3x3(matrix):
    """Calculate the determinant of a 3x3 matrix."""
    a, b, c = matrix[0]
    d, e, f = matrix[1]
    g, h, i = matrix[2]
    
    det = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)
    return det


def calculate_cofactor_matrix(matrix):
    """Calculate the cofactor matrix (matrix of minors with sign adjustments)."""
    cofactor = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
    
    for i in range(3):
        for j in range(3):
            # Get the 2x2 minor by excluding row i and column j
            minor = []
            for row in range(3):
                if row != i:
                    minor_row = []
                    for col in range(3):
                        if col != j:
                            minor_row.append(matrix[row][col])
                    minor.append(minor_row)
            
            # Calculate determinant of 2x2 minor
            minor_det = minor[0][0] * minor[1][1] - minor[0][1] * minor[1][0]
            
            # Apply checkerboard sign pattern: (-1)^(i+j)
            sign = (-1) ** (i + j)
            cofactor[i][j] = sign * minor_det
    
    return cofactor


def transpose_matrix(matrix):
    """Transpose a 3x3 matrix."""
    transposed = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
    for i in range(3):
        for j in range(3):
            transposed[j][i] = matrix[i][j]
    return transposed


def multiply_matrix_by_scalar(matrix, scalar):
    """Multiply every element of a matrix by a scalar."""
    result = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
    for i in range(3):
        for j in range(3):
            result[i][j] = matrix[i][j] * scalar
    return result


def calculate_inverse_3x3(matrix):
    """
    Calculate the inverse of a 3x3 matrix.
    
    Formula: A^(-1) = (1/det(A)) * adj(A)
    where adj(A) = transpose of cofactor matrix
    """
    # Step 1: Calculate determinant
    det = calculate_determinant_3x3(matrix)
    
    # Check if matrix is invertible
    if det == 0:
        return None, "Matrix is singular (determinant = 0), inverse does not exist"
    
    # Step 2: Calculate cofactor matrix
    cofactor = calculate_cofactor_matrix(matrix)
    
    # Step 3: Transpose cofactor matrix to get adjugate matrix
    adjugate = transpose_matrix(cofactor)
    
    # Step 4: Multiply adjugate by (1/determinant)
    inverse = multiply_matrix_by_scalar(adjugate, 1/det)
    
    return inverse, None


def print_matrix(matrix, title="Matrix"):
    """Print a matrix in a readable format."""
    print(f"\n{title}:")
    for row in matrix:
        print("  [", end="")
        for j, val in enumerate(row):
            if j < len(row) - 1:
                print(f"{val:8.4f}", end="  ")
            else:
                print(f"{val:8.4f}", end="")
        print("]")


def verify_inverse(original, inverse):
    """Verify that A * A^(-1) = I (identity matrix)."""
    result = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
    
    # Matrix multiplication
    for i in range(3):
        for j in range(3):
            for k in range(3):
                result[i][j] += original[i][k] * inverse[k][j]
    
    return result


# Example Usage
if __name__ == "__main__":
    # Example 1: Regular invertible matrix
    print("=" * 50)
    print("EXAMPLE 1: Invertible Matrix")
    print("=" * 50)
    
    matrix1 = [
        [2, 3, 1],
        [4, 1, 5],
        [3, 2, 1]
    ]
    
    print_matrix(matrix1, "Original Matrix A")
    
    # Calculate determinant
    det = calculate_determinant_3x3(matrix1)
    print(f"\nDeterminant: {det}")
    
    # Calculate inverse
    inverse1, error = calculate_inverse_3x3(matrix1)
    
    if error:
        print(f"\nError: {error}")
    else:
        print_matrix(inverse1, "Inverse Matrix A^(-1)")
        
        # Verify the inverse
        identity = verify_inverse(matrix1, inverse1)
        print_matrix(identity, "Verification: A * A^(-1) (should be Identity)")
    
    # Example 2: Another invertible matrix
    print("\n" + "=" * 50)
    print("EXAMPLE 2: Another Invertible Matrix")
    print("=" * 50)
    
    matrix2 = [
        [1, 0, 2],
        [2, 1, 0],
        [0, 1, 1]
    ]
    
    print_matrix(matrix2, "Original Matrix B")
    
    det2 = calculate_determinant_3x3(matrix2)
    print(f"\nDeterminant: {det2}")
    
    inverse2, error2 = calculate_inverse_3x3(matrix2)
    
    if error2:
        print(f"\nError: {error2}")
    else:
        print_matrix(inverse2, "Inverse Matrix B^(-1)")
        identity2 = verify_inverse(matrix2, inverse2)
        print_matrix(identity2, "Verification: B * B^(-1)")
    
    # Example 3: Singular matrix (non-invertible)
    print("\n" + "=" * 50)
    print("EXAMPLE 3: Singular Matrix (Non-invertible)")
    print("=" * 50)
    
    matrix3 = [
        [1, 2, 3],
        [2, 4, 6],
        [1, 1, 1]
    ]
    
    print_matrix(matrix3, "Original Matrix C")
    
    det3 = calculate_determinant_3x3(matrix3)
    print(f"\nDeterminant: {det3}")
    
    inverse3, error3 = calculate_inverse_3x3(matrix3)
    
    if error3:
        print(f"\n{error3}")
    else:
        print_matrix(inverse3, "Inverse Matrix C^(-1)")