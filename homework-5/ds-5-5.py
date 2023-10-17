import numpy as np

if __name__ == "__main__":
    mtx = np.array([[1, -1, 4],
                    [2, 1, 3],
                    [1, 3, -1]])

    cov_mtx = np.cov(mtx)
    print("协方差矩阵为：\n", cov_mtx)

    feature = np.linalg.eig(cov_mtx)
    print("特征值为:", feature[0])
    print("特征向量矩阵为:\n", feature[1])


    def power_iteration(matrix, dimensions):
        n = matrix.shape[0]
        eigenvectors = []

        for _ in range(dimensions):
            vec = np.random.rand(n)
            for _ in range(100):
                vec = np.dot(matrix, vec)
                vec /= np.linalg.norm(vec)
            eigenvalue = np.dot(vec, np.dot(matrix, vec))
            eigenvectors.append((eigenvalue, vec))
            matrix -= eigenvalue * np.outer(vec, vec)

        return eigenvectors

    dimension = 3
    eigenpairs = power_iteration(cov_mtx, dimension)

    for eigenvalue, eigenvector in eigenpairs:
        print("Eigenvalue:", eigenvalue)
        print("Eigenvector:", eigenvector)
