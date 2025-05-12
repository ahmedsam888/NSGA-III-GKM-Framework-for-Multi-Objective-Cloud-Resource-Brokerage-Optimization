import numpy as np
import matplotlib.pyplot as plt
import random
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.core.problem import Problem
from pymoo.optimize import minimize
from pymoo.operators.crossover.sbx import SimulatedBinaryCrossover
from pymoo.operators.mutation.pm import PolynomialMutation
from pymoo.termination import get_termination
from sklearn.cluster import KMeans

# 🔹 Cloud brokerage parameters
execution_time = np.array([0.2525, 0.1541, 0.19])
hourly_price = np.array([0.1, 0.125, 0.143])
energy_consumption = np.array([190, 220, 240])

# 🔹 Number of clients and providers
N = 10  
M = 3    

# 🔹 Definition of random parameters
np.random.seed(42)
L_ij = np.random.uniform(0.05, 0.5, size=(N, M))
T_j = execution_time
P_i = np.random.uniform(0.2, 0.5, N)
C_j = hourly_price
A_j = np.array([5, 8, 12])
R_i = np.random.randint(1, 3, size=N)

# 🔹 Definition of the problem
class CloudBrokerProblem(Problem):
    def __init__(self):
        super().__init__(
            n_var=N * M,
            n_obj=3,
            n_constr=N + M,  
            xl=0, xu=1
        )

    def _evaluate(self, X, out, *args, **kwargs):
        X = X.reshape(-1, N, M)

        # 🔹 Objectives
        RT = np.sum(X * (L_ij + T_j), axis=(1, 2))
        E = np.sum(X * energy_consumption, axis=(1, 2))
        P = np.sum(X * (P_i[:, None] - C_j[None, :]), axis=(1, 2))

        # 🔹 Constraints
        g1 = np.sum(X, axis=2) - R_i[None, :]
        g2 = A_j[None, :] - np.sum(X, axis=1)

        g1 = np.maximum(g1, -1)  # Relaxation
        g2 = np.maximum(g2, -1)  # Relaxation

        out["F"] = np.column_stack([RT, E, -P])
        out["G"] = np.hstack([g1.flatten(), g2.flatten()]).reshape(X.shape[0], -1)

# 🔹 Genetic K-Means++ Algorithm (GKM++)
def genetic_kmeans_plus_plus(pop_size_gkm, n_gen_gkm, initial_k, a, b):
    population = np.random.rand(pop_size_gkm, N * M)
    best_centers = None
    best_fit = float('-inf')

    for _ in range(n_gen_gkm):
        # 1️⃣ Adaptive selection of the number of clusters
        n_clusters = min(random.randint(2, pop_size_gkm), len(population))

        # 2️⃣ Initialization of centers with K-Means++
        kmeans = KMeans(n_clusters=n_clusters, init='k-means++', n_init=10, random_state=42)
        kmeans.fit(population)
        centers = kmeans.cluster_centers_

        # 3️⃣ Calculation of quality metrics
        E = np.sum([np.linalg.norm(population[i] - centers[kmeans.labels_[i]])**2 for i in range(len(population))])
        if len(centers) > 1:
            G_b = np.sum([
                np.linalg.norm(centers[i] - centers[j])**2
                for i in range(len(centers))
                for j in range(i+1, len(centers))
            ]) / (2 * len(centers) * (len(centers) - 1))
        else:
            G_b = 0  # If only one center, separation is zero

        # 4️⃣ Fitness function
        fitD = G_b / (b + a * E)

        # 5️⃣ Selection of the best clustering
        if fitD > best_fit:
            best_fit = fitD
            best_centers = centers

    return best_centers

# 🔹 Algorithm parameters
pop_size = 100  
n_generations = 250
pop_size_gkm = 5  
n_gen_gkm = 10  
initial_k = 2  
a, b = 2, 1.2  

# 🔹 Execution of the hybrid NSGA-III + GKM++ algorithm
def hybrid_nsga3_gkm():
    ref_dirs = get_reference_directions("das-dennis", 3, n_partitions=12)

    # 🔹 Initialization with GKM++
    initial_population = genetic_kmeans_plus_plus(pop_size_gkm, n_gen_gkm, initial_k, a, b)

    # 🔹 Configuration of the NSGA-III algorithm
    algorithm = NSGA3(
        pop_size=pop_size,
        ref_dirs=ref_dirs,
        sampling=initial_population,  
        crossover=SimulatedBinaryCrossover(prob=0.9, eta=15),
        mutation=PolynomialMutation(prob=0.1, eta=15)
    )

    termination = get_termination("n_gen", n_generations)
    
    res = minimize(
        CloudBrokerProblem(),
        algorithm,
        termination,
        verbose=True
    )
    
    if res.F is not None:
        print("\n✅ Optimal solutions found:")
        print(res.X.reshape(-1, N, M))  
        print("\n📊 Objective values (RT, E, P):")
        print(res.F)

        plt.figure(figsize=(8,6))
        plt.scatter(res.F[:, 0], res.F[:, 1], c=-res.F[:, 2], cmap='viridis')
        plt.xlabel("Response Time (RT)")
        plt.ylabel("Energy Consumption (E)")
        plt.colorbar(label="Broker Profit (P)")
        plt.title("Pareto Front: Time vs Energy vs Profit")
        plt.show()
    else:
        print("\n❌ No valid solution found. Try adjusting constraints or parameters.")

# 🔹 Execution
hybrid_nsga3_gkm()