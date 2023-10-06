import random
import time

def insertion_sort(arr):
    for i in range(1, len(arr)):
        key = arr[i]
        j = i - 1
        while j >= 0 and key < arr[j]:
            arr[j + 1] = arr[j]
            j -= 1
        arr[j + 1] = key

def quick_sort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quick_sort(left) + middle + quick_sort(right)

def shell_sort(arr):
    n = len(arr)
    gap = n // 2
    while gap > 0:
        for i in range(gap, n):
            temp = arr[i]
            j = i
            while j >= gap and arr[j - gap] > temp:
                arr[j] = arr[j - gap]
                j -= gap
            arr[j] = temp
        gap //= 2

def selection_sort(arr):
    n = len(arr)
    for i in range(n):
        min_index = i
        for j in range(i+1, n):
            if arr[j] < arr[min_index]:
                min_index = j
        arr[i], arr[min_index] = arr[min_index], arr[i]

def merge_sort(arr):
    if len(arr) > 1:
        mid = len(arr) // 2
        left_half = arr[:mid]
        right_half = arr[mid:]
        merge_sort(left_half)
        merge_sort(right_half)
        i = j = k = 0
        while i < len(left_half) and j < len(right_half):
            if left_half[i] < right_half[j]:
                arr[k] = left_half[i]
                i += 1
            else:
                arr[k] = right_half[j]
                j += 1
            k += 1
        while i < len(left_half):
            arr[k] = left_half[i]
            i += 1
            k += 1
        while j < len(right_half):
            arr[k] = right_half[j]
            j += 1
            k += 1

if __name__ == "__main__":
    length = 5
    numslst = []
    for i in range(0, 4):
        nums = []
        for j in range(0, length):
            x = random.randint(0,10)
            nums.append(x)
        numslst.append(nums)
        nums = []
        length *= 10;
    for i in range(0, 4):
        start_time = time.time()
        insertion_sort(numslst[i])
        end_time = time.time()
        execution_time = end_time - start_time
        print(f'数列长度为{5*pow(10,i)}时，插入排序耗时：{execution_time}')

        start_time = time.time()
        quick_sort(numslst[i])
        end_time = time.time()
        execution_time = end_time - start_time
        print(f'数列长度为{5*pow(10,i)}时，快速排序耗时：{execution_time}')

        start_time = time.time()
        shell_sort(numslst[i])
        end_time = time.time()
        execution_time = end_time - start_time
        print(f'数列长度为{5*pow(10,i)}时，希尔排序耗时：{execution_time}')

        start_time = time.time()
        selection_sort(numslst[i])
        end_time = time.time()
        execution_time = end_time - start_time
        print(f'数列长度为{5*pow(10,i)}时，选择排序耗时：{execution_time}')

        start_time = time.time()
        merge_sort(numslst[i])
        end_time = time.time()
        execution_time = end_time - start_time
        print(f'数列长度为{5*pow(10,i)}时，归并排序耗时：{execution_time}')