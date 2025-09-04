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


print()
print('Test REF, _REF')

A.REF().print() # creates REF from a copy of matrix A, A is unchanged
A.print()
A._REF() # A prevede do REF
A.print()

print()
print("Test RREF, inverse")

B.RREF().print() # creates RREF from a copy of matrix B, B is unchanged
B.print()
B.inverse().print() # return an inverse from a copy of matrix B, B is unchanged
B.print()