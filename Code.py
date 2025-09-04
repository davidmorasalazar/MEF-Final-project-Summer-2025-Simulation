import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
class Matrix:

    def generate(self, n, m, x=0):
        self.mat = [[0] * m for i in range(n)]
        self.n = n
        self.m = m

        if n == m:
            for i in range(n):
                self.mat[i][i] = x

    def __init__(self, A, B=1, x=0):

        if isinstance(A, int):
            self.generate(A, B, x)
            return

        for l in A:
            if not isinstance(l, list) or len(l) != len(A[0]):
                raise Exception("Table rows must be consistent")

            for x in l:
                try:
                    float(x)
                except ValueError:
                    raise Exception("Table elements must be convertible to float")

        self.mat = [[float(A[i][j]) for j in range(len(A[i]))] for i in range(len(A))]
        self.n = len(A)
        self.m = len(A[0]) 

    def to_list(self):
        return [l[:] for l in self.mat]

    def __getitem__(self, key):
        return self.mat[key]

    def __setitem__(self, key, value):
        self.mat[key] = value

    def print(self):
        for l in self.mat: 
            print(l)
        print()
    def __eq__(self, other):
        if (self.n, self.m) != (other.n, other.m):
            return False

        for i in range(self.n):
            for j in range(self.m):
                if abs(self[i][j] - other[i][j]) > 1e-10:
                    return False

        return True

    def __ne__(self, other):
        return not self.mat == other.mat

    def __add__(self, other):
        if self.n != other.n or self.m != other.m:
            raise Exception("Incorrect dimensions.")
        result = Matrix(self.n, self.m)

        for i in range(self.n):
            for j in range(self.m):
                result[i][j] = self[i][j] + other[i][j]

        return result


    def __sub__(self, other):
        if self.n != other.n or self.m != other.m:
            raise Exception("Incorrect dimensions.")
        result = Matrix(self.n, self.m)

        for i in range(self.n):
            for j in range(self.m):
                result[i][j] = self[i][j] - other[i][j]

        return result


    def __mul__(self, other):
        if self.m != other.n:
            raise Exception("Incorrect dimensions.")

        result = Matrix(self.n, other.m)

        for i in range(self.n):
            for j in range(other.m):
                for k in range(self.m):
                    result[i][j] += self[i][k] * other[k][j]

        return result

    def transpose(self):
        result = Matrix(self.m, self.n)

        for i in range(self.n):
            for j in range(self.m):
                result[j][i] = self[i][j]

        return result

    def symmetry(self):
        return self.mat == self.transpose().mat

    def row_swap(self, i, j):
        self[i], self[j] = self[j], self[i]

    def row_multiply(self, i, x):
        if x == 0:
            raise Exception("Cannot multiply by 0")
        self[i] = [x * e for e in self[i]]

    def row_add(self, i, j, x):
        self[j] = [e_j + x * e_i for e_i, e_j in zip(self[i], self[j])]

    def _REF(self):
        ref_matrix = self
        rowA, colA = self.n, self.m

        row, col = 0, 0  # i, j
        all_zeroes = True
        pivot_row = 0

        while row < rowA and col < colA:
            # Step 2: Find the pivot element
            for k in range(row, rowA):
                for l in range(col, colA):
                    if ref_matrix[k][l] != 0:
                        all_zeroes = False
                        # Step 3: Update row and column indices
                        col = l
                        pivot_row = k
                        break
                if not all_zeroes:
                    break
            if all_zeroes:
                break

            # Step 4: Swap rows to bring the pivot element to the current row
            ref_matrix.row_swap(row, pivot_row)

            # Step 6: Perform row operations to make all elements below the pivot element zero
            for k in range(rowA):
                if k <= row:
                    continue
                ref_pivot_row_value = ref_matrix[k][col] / ref_matrix[row][col]
                for i in range(colA):
                    ref_result = ref_matrix[k][i] - ref_pivot_row_value * ref_matrix[pivot_row][i]
                    ref_matrix[k][i] = ref_result

            # Step 7: Move to the next row and column
            row += 1
            col += 1

        return ref_matrix

    def REF(self):
        res = Matrix(self.to_list())
        res._REF()
        return res
    
    def rref_and_inverse(self):
        rowA, colA = self.n, self.m

        rref_matrix = self
        inverse_matrix = Matrix(self.n, self.m, 1)
        row, col = 0, 0  # i, j
        all_zeroes = True

        pivot_row = 0

        while row < rowA and col < colA:
            # Step 2: Find the pivot element
            for k in range(row, rowA):
                for l in range(col, colA):
                    if A[k][l] != 0:
                        all_zeroes = False
                        # Step 3: Update row and column indices
                        col = l
                        pivot_row = k
                        break
                if not all_zeroes:
                    break
            if all_zeroes:
                break

            # Step 4: Swap rows to bring the pivot element to the current row
            rref_matrix[row], rref_matrix[pivot_row] = rref_matrix[pivot_row], rref_matrix[row]
            inverse_matrix[row], inverse_matrix[pivot_row] = inverse_matrix[pivot_row], inverse_matrix[row]

            # Step 5: Normalize the pivot row
            pivot_value = rref_matrix[row][col]
            for i in range(rowA):
                rref_matrix[row][i] /= pivot_value
                inverse_matrix[row][i] /= pivot_value

            # Step 6: Perform row operations to make all elements except the pivot zero
            for k in range(rowA):
                if k == row:
                    continue
                rref_pivot_row_value = rref_matrix[k][col]
                for i in range(colA):
                    rref_matrix[k][i] -= rref_pivot_row_value * rref_matrix[row][i]
                    inverse_matrix[k][i] -= rref_pivot_row_value * inverse_matrix[row][i]

            # Step 7: Move to the next row and column
            row += 1
            col += 1

        return rref_matrix, inverse_matrix
    
    def RREF(self):
        if self.n != self.m:
            raise Exception("Incorrect dimensions")
        res = Matrix(self.to_list())
        result = res.rref_and_inverse()
        return result[0]
    
    def inverse(self):
        if self.n != self.m:
            raise Exception("Incorrect dimensions")
        res = Matrix(self.to_list())
        result = res.rref_and_inverse()
        return result[1]

A = [[2,1,7],
     [4,1,5], 
     [-6,-2,9]]

A = Matrix(A)

B = [[5,2,3],
     [-4,-1,6], 
     [7,5,2]]

B = Matrix(B)


A.transpose().print()
print("Are matrices equal?", A==B)
print("1. row, 2. column of matrix A:", A[0][1])

(A+B).print()
(A-B).print()
(A*B).print()
B.row_swap(0, 1)
B.print()
B.row_multiply(1, 2)
B.print()

print('Test REF, _REF')

A.REF().print() # creates REF from a copy of matrix A, A is unchanged
A.print()
A._REF().print() # A prevede do REF
A.print()


print("Test RREF, inverse")

B.RREF().print() #Reduce matrices to REF (Row Echelon Form) and RREF (Reduced Row Echelon Form)  creates RREF from a copy of matrix B, B is unchanged
B.print()
B.inverse().print() # return an inverse from a copy of matrix B, B is unchanged
B.print()

points = B
##Code lines to use pandas and matplotlib
# Split into x, y, z coordinates
x = [p[0] for p in points]
y = [p[1] for p in points]
z = [p[2] for p in points]

# Create 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot the points
ax.scatter(x, y, z, c='r', marker='o', s=100)

# Add labels
ax.set_xlabel("X axis")
ax.set_ylabel("Y axis")
ax.set_zlabel("Z axis")
ax.set_title("3D Points from Matrix")

# Show plot
plt.show()