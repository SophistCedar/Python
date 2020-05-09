"""
迷宫寻路
"""
import random
import sys

WALL = -1
ROAD = 0

ROWS = 10
COLS = 10


def find_way(maze, i=0, j=0, step=1):
    """走迷宫"""
    if 0 <= i < ROWS and 0 <= j < COLS and maze[i][j] == ROAD: # 收敛条件
        maze[i][j] = step # 设置走过的路为第几步数
        if i == ROWS - 1 and j == COLS - 1: # 走出去迷宫的判断
            print('=' * 20) # 分隔线
            display(maze)
            sys.exit(0) # 程序终止
        find_way(maze, i + 1, j, step + 1) # 递归公式
        find_way(maze, i, j + 1, step + 1)
        find_way(maze, i - 1, j, step + 1)
        find_way(maze, i, j - 1, step + 1)
        maze[i][j] = ROAD 
        # 下右上左都没有走通的时候，设置之前走过的那个路为ROAD，此为回溯设置，如果走通则会继续执行程序，执行不到这个语句


def reset(maze):
    """重置迷宫"""
    for i in range(ROWS):
        for j in range(COLS):
            num = random.randint(1, 10)
            maze[i][j] = WALL if num > 7 else ROAD # 70%的概率是WALL
    maze[0][0] = maze[ROWS - 1][COLS - 1] = ROAD # 左上角和右下角是路


def display(maze):
    """显示迷宫"""
    for row in maze:
        for col in row:
            if col == WALL:
                print('■', end=' ')
            elif col == ROAD:
                print('□', end=' ')
            else:
                print(f'{col}'.ljust(2), end='')
        print()


def main():
    """主函数"""
    maze = [[0] * COLS for _ in range(ROWS)] # 列表生成式创建列表
    reset(maze)
    display(maze)
    find_way(maze)
    print('没有出路!!!')


if __name__ == '__main__':
    main()
