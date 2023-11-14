import numpy as np

if __name__ == "__main__":
    mtx = np.array([[2, 1], [4, 5]])
    feature = np.linalg.eig(mtx)
    print("特征值为:", feature[0])
    print("特征向量矩阵为:\n", feature[1])

    max_iterations = 100
    tolerance = 1e-6
    vec = np.array([0, 1], dtype='float64')
    for _ in range(max_iterations):
        vec_new = np.dot(mtx, vec)
        vec_new /= np.linalg.norm(vec_new)
        eigenvalue = np.dot(vec, np.dot(mtx, vec))
        if np.linalg.norm(vec - vec_new) < tolerance:
            print("估计的最大特征值：", eigenvalue)
            print("对应的特征向量：", vec)
            break
        vec = vec_new
