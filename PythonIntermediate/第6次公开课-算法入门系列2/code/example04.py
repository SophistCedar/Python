"""
骑士巡逻
"""
import sys

SIZE = 8

def display(board):
    """显示棋盘"""
    for row in board:
        for col in row:
            print(f'{col}'.rjust(2, '0'), end=' ')
        print()


def patrol(board, i=0, j=0, step=1):
    """巡逻"""
    if 0 <= i < SIZE and 0 <= j < SIZE and board[i][j] == 0: # 收敛条件
        board[i][j] = step
        if step == SIZE * SIZE: # 走完条件
            display(board)
            sys.exit(0)
        patrol(board, i + 1, j + 2, step + 1) # 递归公式
        patrol(board, i + 2, j + 1, step + 1)
        patrol(board, i + 2, j - 1, step + 1)
        patrol(board, i + 1, j - 2, step + 1)
        patrol(board, i - 1, j - 2, step + 1)
        patrol(board, i - 2, j - 1, step + 1)
        patrol(board, i - 2, j + 1, step + 1)
        patrol(board, i - 1, j + 2, step + 1)
        board[i][j] = 0 # 回溯设置


def main():
    """主函数"""
    board = [[0] * SIZE for _ in range(SIZE)]
    patrol(board)


if __name__ == '__main__':
    main()
