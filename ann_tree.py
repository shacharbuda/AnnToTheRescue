from typing import List
import numpy as np

N = 5  # Number of dimensions for the vectors
K = 10  # Number of nearest neighbors to find
# LEAF_SIZE is the maximum number of vectors in a leaf node
# we limit it to 10 to avoid missing too many neighbors.
LEAF_SIZE = min(10, K)

class Vector:
    def __init__(self, coordinates: List[float]):
        self.coordinates = coordinates

    def __repr__(self):
        return f"Vector({self.coordinates})"

class Hyperplane:
    def __init__(self, data: List[Vector]):
        # create two random points in the data
        indices = list(range(len(data)))
        i1, i2 = np.random.choice(indices, 2, replace=False)
        p1, p2 = data[i1], data[i2]
        v1, v2 = np.array(p1.coordinates), np.array(p2.coordinates)
        # create a direction vector from the two points
        self.direction_vector = v2 - v1
        # calculate the midpoint for the hyperplane
        self.mid_vector_for_hyperplane = (v1 + v2) / 2

    def is_below(self, vector: Vector) -> bool:
        # The hyperplane divides the space into two half-spaces
        v = np.array(vector.coordinates)
        return np.dot(v - self.mid_vector_for_hyperplane, self.direction_vector) < 0

class AnnNode:
    def __init__(self,
                hyperplane: Hyperplane = None,
                leftNode = None,
                rightNode = None,
                leaf_vectors: List[Vector] = None):
        self.hyperplane = hyperplane
        self.leftNode = leftNode
        self.rightNode = rightNode
        self.leaf_vectors = leaf_vectors

    def is_leaf(self) -> bool:
        return self.leaf_vectors is not None


class AnnTree:
    def __init__(self):
        self.data = []
        self.root = None

    def fit(self, data: List[Vector]):
        self.data = data
        self.root = self._build_node(data)

    def add(self, vector: Vector):
        # Add a new vector to the tree and rebuild it
        self.data.append(vector)
        self.root = self._build_node(self.data)

    def find_k_nearest_neighbors(self, query_vec: Vector, k: int = K) -> List[Vector]:
        def descend(node: AnnNode) -> List[Vector]:
            if node.is_leaf():
                return node.leaf_vectors

            if node.hyperplane.is_below(query_vec):
                # Query vector is below the hyperplane, go left
                nearest = descend(node.leftNode)
                other_side = node.rightNode
            else:
                # Query vector is above the hyperplane, go right
                nearest = descend(node.rightNode)
                other_side = node.leftNode

            # Check if we need to explore the other side
            if len(nearest) < k:
                nearest.extend(descend(other_side))

            # Sort the nearest vectors by distance to the query vector
            return sorted(nearest, key=lambda v: np.linalg.norm(np.array(v.coordinates) - np.array(query_vec.coordinates)))[:k]

        return descend(self.root)

    def _build_node(self, data: List[Vector]) -> AnnNode:
        if not data:
            return None

        # Stop condition for leaf node
        if len(data) <= LEAF_SIZE:
            return AnnNode(leaf_vectors=data)

        random_dividing_hyperplane = Hyperplane(data)

        left_vectors = []
        right_vectors = []

        for vector in data:
            if random_dividing_hyperplane.is_below(vector):
                left_vectors.append(vector)
            else:
                right_vectors.append(vector)

        left_node = self._build_node(left_vectors)
        right_node = self._build_node(right_vectors)

        return AnnNode(random_dividing_hyperplane, left_node, right_node)

if __name__ == "__main__":
    # Example usage using plot and colors
    import matplotlib.pyplot as plt

    N = 2  # For visualization, we will use 2 dimensions

    # Generate random data
    np.random.seed(49)  # For reproducibility
    data = np.random.rand(100, N) * 100  # 100 random points in N-dimensional space
    data = [Vector(point) for point in data]
    # Create and fit the AnnTree
    ann_tree = AnnTree()
    ann_tree.fit(data)
    # Define a query point
    query_point = Vector(np.random.rand(N) * 100)
    # Find K nearest neighbors
    neighbors = ann_tree.find_k_nearest_neighbors(query_point, K)
    # Convert neighbors to indices for plotting
    neighbors = [data.index(neighbor) for neighbor in neighbors]
    # Convert query point to numpy array for plotting
    query_point = np.array(query_point.coordinates)

    # Plot the data points
    plt.figure(figsize=(10, 8))
    for i, point in enumerate(data):
        if i in neighbors:
            plt.scatter(point.coordinates[0], point.coordinates[1], color='red', label='Neighbor' if i == neighbors[0] else "")
        else:
            plt.scatter(point.coordinates[0], point.coordinates[1], color='blue', label='Data Point' if i == 0 else "")
    # Plot the query point
    plt.scatter(query_point[0], query_point[1], color='green', marker='x', s=100, label='Query Point')
    plt.title('AnnTree Nearest Neighbors')
    plt.legend()
    plt.grid()
    plt.show()

