# -*- coding: utf-8 -*-
class Solution:
    """
    Get all distinct N-Queen solutions
    @param n: The number of queens
    @return: All distinct solutions
    """
    def solveNQueens(self, n):
        self.ret = []
        self.limit = n
        self._solveNQueens(0, [])

    def _solveNQueens(self, row, queens):
        if row == self.limit:
            self.printBoard(queens)
        else:
            for col in range(self.limit):
                if self.check(queens, row, col):
                    queens.append([row, col])
                    self._solveNQueens(row + 1, queens)
                    queens.pop()

    def check(self, queens, row, col):
        for queen in queens:
            if (col == queen[1]) or (abs(row - queen[0]) == abs(col - queen[1])):
                return False
        return True

    def printBoard(self, queens):
        self.ret.append([])
        for i in range(self.limit):
            row = []
            for j in range(self.limit):
                if j == queens[i][j]:
                    row.append('Q')
                else:
                    row.append('.')
                print(row)
            ret[-1].append(row)
            
solution = Solution()
solution.solveNQueens(3)
print(solution.limit)
print(solution.ret)
