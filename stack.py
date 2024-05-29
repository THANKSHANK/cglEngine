"""
stack.py, a simple stack class for 3x3 matrices
"""
__author__ = "Zihan Wang"
import glm
class Stack:
    def __init__(self):
        self.items = [glm.mat4(1)]  # Initialize with identity matrix

    def isEmpty(self):
        return len(self.items) == 0

    def push(self, M):
        current_top = self.items[-1]
        new_top = current_top * M
        self.items.append(new_top)

    def pop(self):
        if self.isEmpty():
            raise IndexError("Pop from an empty stack")
        return self.items.pop()

    def top(self):
        if self.isEmpty():
            raise IndexError("Top from an empty stack")
        return self.items[-1]

    def __len__(self):
        return len(self.items)

    def __str__(self):
        return str(self.items)
