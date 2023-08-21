# k-Nearest Neighbours Study Project

k-Nearest Neighbours (kNN) is a classic algorithm in supervised learning, using proximity to classify or predict the grouping of a data point. The algorithm's intuition is relatively simple: to find k nearest data points according to the queried data point (where k is a model hyperparameter). From here, we could easily point out a critical challenge for this approach, which is the complexity of calculating distances, storing, and comparing them. For the sake of study, I work on this project with some objectives in mind:

1. Optimising the time complexity as much as I can, with access to a sufficiently large memory available (4GB RAM) and
2. Structuring the project in an object-oriented manner, embracing modularisation for usability.

Some approaches utilise approximation methods to trade off a little bit of accuracy with time efficiency [1-3]. To be clear, my project is not about those. In this study, I mainly focus on the optimisation problem by addressing the exact kNN algorithm.

## References
1. Alexandr Andoni and Piotr Indyk. 2006. Near-Optimal Hashing Algorithms for Approximate Nearest Neighbor in High Dimensions. In Proceedings of the 47th Annual IEEE Symposium on Foundations of Computer Science (FOCS '06). IEEE Computer Society, USA, 459–468. https://doi.org/10.1109/FOCS.2006.49
2. Alexandr Andoni and Ilya Razenshteyn. 2015. Optimal Data-Dependent Hashing for Approximate Near Neighbors. In Proceedings of the forty-seventh annual ACM symposium on Theory of Computing (STOC '15). Association for Computing Machinery, New York, NY, USA, 793–801. https://doi.org/10.1145/2746539.2746553
3. Mingmou Liu, Xiaoyin Pan, and Yitong Yin. 2018. Randomized Approximate Nearest Neighbor Search with Limited Adaptivity. ACM Trans. Parallel Comput. 5, 1, Article 3 (March 2018), 26 pages. https://doi.org/10.1145/3209884
