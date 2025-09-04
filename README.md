Matrix Library – User Manual

This is the user manual for the semester project “Matrix Library”, created in Python.
Matrices are treated here as 2D arrays.
________________________________________

Functions:

• def generate(self, n, m, x=0)

Generates an n × m matrix; x is placed on the main diagonal (default: 0)

• def to_list(self)

Creates a copy of the matrix as a list

• def __getitem__(self, key)

Returns the element A[i][j], e.g., print(A[0][2])

• def print(self)

Prints the rows of the matrix line by line, e.g., A.print()

• def __eq__(self, other)

Equality check, e.g., print(A == B)

• def __ne__(self, other)

Inequality check, e.g., print(A != B)

• def __add__(self, other)

Matrix addition, e.g., (A + B).print()

• def __sub__(self, other)

Matrix subtraction, e.g., (A - B).print()

• def __mul__(self, other)

Matrix multiplication, e.g., (A * B).print()

• def transpose(self)

Transposes the matrix

• def symmetry(self)

Checks for symmetry

• def row_swap(self, i, j)

Swaps rows i and j

• def row_multiply(self, i, x)

Multiplies row i by a non-zero number x

• def row_add(self, i, j, x)

Adds x times row i to row j

• def _REF(self)

Converts the matrix to REF (Row Echelon Form)

• def REF(self)

Returns a copy of the matrix in REF form

• def rref_and_inverse(self)

Returns the RREF (Reduced Row Echelon Form) and the inverse matrix

• def RREF(self)

Returns a copy of the matrix in RREF form

• def inverse(self)

Returns the inverse of a copy of the matrix

Finally, an example of plotting matrices is shown.
