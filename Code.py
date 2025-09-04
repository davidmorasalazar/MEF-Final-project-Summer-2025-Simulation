cimport pandas as pd
import matplotlib as plt
import numpy as np
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