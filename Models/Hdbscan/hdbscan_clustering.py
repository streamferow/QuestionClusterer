import hdbscan, random, warnings, matplotlib.pyplot as plt
from tqdm import tqdm
from functools import partial
from Data.Embedded import embedded_data
from sklearn.metrics import silhouette_score
from deap import base, creator, tools, algorithms

reduced = embedded_data.load_tensor("/Users/ivan/Desktop/final_reduced_embeddings.pt")
warnings.filterwarnings("ignore", message=".*force_all_finite.*")


def cluster(reduced, min_cluster_size, min_samples):
    clustering = hdbscan.HDBSCAN(min_samples=min_samples,
                                 min_cluster_size=min_cluster_size,
                                 metric='cosine',
                                 algorithm='generic')
    return clustering.fit_predict(reduced)


def plot_clusters(reduced_embeddings, labels):
    plt.figure(figsize=(10,10))
    plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], c=labels, cmap='viridis', s=5)
    plt.legend()
    plt.grid(True)
    plt.show()


def evaluate(reduced_embeddings, individual, target_sizes=(400, 200, 50)):
    min_cluster_size, min_samples = individual
    labels = cluster(reduced, min_cluster_size=min_cluster_size, min_samples=min_samples)

    valid_mask = labels != -1
    valid_labels, valid_points = labels[valid_mask], reduced[valid_mask]

    unique_labels = set(valid_labels)
    n_clusters = len(unique_labels)

    if n_clusters != 3:
        penalty = abs(3 - n_clusters)
        return (0.01 / penalty,)
    elif n_clusters < 2:
        return (0.0,)

    cluster_sizes = [sum(valid_labels == label) for label in unique_labels]
    cluster_sizes.sort()
    target_sizes = sorted(target_sizes)
    size_penalty = sum(abs(cs - ts) / ts for cs, ts in zip(cluster_sizes, target_sizes))

    score = silhouette_score(valid_points, valid_labels, metric='cosine')
    fitness = score - size_penalty * 0.4

    return (fitness,)


def start_genetic_algorithm(ngen):
    for gen in tqdm(range(ngen), desc="эволюция"):
        offspring = algorithms.varAnd(population, toolbox, cxpb=0.5, mutpb=0.1)
        fits = list(map(toolbox.evaluate, offspring))

        for fit, ind in zip(fits, offspring):
            ind.fitness.values = fit

        population = toolbox.select(offspring, k=len(population))

    top_ind = tools.selBest(population, k=1)[0]
    print(f"лучшие параметры: min_cluster_size={top_ind[0]}, min_samples={top_ind[1]}")

creator.create("FitnessMax", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("min_cluster_size", random.randint, 5, 100)
toolbox.register("min_samples", random.randint, 1, 100)

toolbox.register("individual", tools.initCycle, creator.Individual,
                 (toolbox.min_cluster_size, toolbox.min_samples), n=1)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("evaluate", evaluate)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutUniformInt, low=[5, 1], up=[100, 100], indpb=0.2)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("evaluate", partial(evaluate, reduced))

population = toolbox.population(n=30)
ngen = 50
# start_genetic_algorithm(ngen)


# labels = cluster(reduced, min_cluster_size=49, min_samples=9)
# plot_clusters(reduced, labels)
