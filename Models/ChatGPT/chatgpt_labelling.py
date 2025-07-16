from sklearn.metrics.pairwise import cosine_distances
from Models.Hdbscan import hdbscan_clustering
from Data.Embedded import embedded_data
from Data.Processed import processed_data
from Data.Raw import raw_data
import matplotlib.pyplot as plt
from openai import OpenAI
import numpy as np


embeddings = embedded_data.load_tensor("/Users/ivan/Desktop/final_embeddings.pt")
reduced = embedded_data.load_tensor("/Users/ivan/Desktop/final_reduced_embeddings.pt")
labels = hdbscan_clustering.cluster(reduced, min_cluster_size=49, min_samples=9)
texts = raw_data.load_text_from_json("/Users/ivan/PycharmProjects/PostsFilter/Data/Raw/posts.json")
df = processed_data.texts_to_df(texts)


client = OpenAI(
    api_key="sk-vQEH0SfRn_FcEgrg8IgUZQ",
    base_url="https://api.ai-mediator.ru/v1"
)


def get_top_questions(embeddings, labels, questions_df, top_n):
    clusters = np.unique(labels)
    top_questions_in_each_cluster = {}

    for cluster_id in clusters:
        cluster_indices = np.where(labels == cluster_id)[0]
        cluster_embeddings = embeddings[cluster_indices]
        cluster_questions = questions_df.iloc[cluster_indices]["text"].tolist()

        centroid = cluster_embeddings.mean(axis=0).reshape(1, -1)
        distances = cosine_distances(cluster_embeddings, centroid).flatten()

        top_indices = distances.argsort()[:top_n]
        top_questions = [cluster_questions[i] for i in top_indices]
        top_questions_in_each_cluster[cluster_id] = top_questions

    return top_questions_in_each_cluster


def label_clusters_batch(top_questions, client, model_name="gpt-4o"):
    prompt = (
        "You are a highly intelligent assistant trained to analyze and categorize user questions from an online community.\n\n"
        "You are provided with 25 representative questions from several clusters. Each cluster contains related questions.\n"
        "Your task is to determine the main topic that best describes each cluster.\n\n"
        "Instructions:\n"
        "- For each cluster, specify one topic with 1-3 words.\n"
        "- Topics must be specific and meaningful (e.g., \"спорт\", \"студенческое жильё\", \"здоровье\").\n"
        "- If the questions mention sports, include \"спорт\".\n"
        "- Do NOT USE vague or generic words such as: \"учебные вопросы\", \"обсуждение\", \"университет\", \"МИФИ\", \"академические\", \"процесс\", \"трудности\", \"дисциплины\".\n"
        "- Each cluster should have a semantically unique topic — avoid repeating the same topic meaning.\n"
        "- Return ONLY the list of topics in the following format:\n"
        "  Кластер <ID>: <topic in Russian>\n"
        "- Do NOT add any explanations or extra text.\n\n"
        "Here are the clusters:\n\n"
    )

    for cluster_id, questions in top_questions.items():
        prompt += f"Кластер {cluster_id}:\n"
        prompt += "\n".join(f"{i + 1}. {q}" for i, q in enumerate(questions))
        prompt += "\n\n"

    prompt += "Ответ:"

    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": "You are an intelligent assistant and an expert in text topic classification."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.4,
        max_tokens=1000,
    )

    output = response.choices[0].message.content.strip()
    cluster_labels = {}
    for line in output.splitlines():
        if ":" in line:
            cluster_id_str, label = line.split(":", 1)
            try:
                cluster_id = int(cluster_id_str.strip().replace("Кластер", "").strip())
                cluster_labels[cluster_id] = label.strip()
            except ValueError:
                continue

    return cluster_labels


def plot_clusters_with_labels(reduced_embeddings, labels, cluster_labels_dict):
    plt.figure(figsize=(12, 8))
    unique_labels = np.unique(labels)

    for cluster_id in unique_labels:
        cluster_points = reduced_embeddings[labels == cluster_id]
        label = cluster_labels_dict.get(cluster_id, f"кластер {cluster_id}")
        plt.scatter(
            cluster_points[:, 0],
            cluster_points[:, 1],
            label=label,
            alpha=0.6,
            s=20
        )

    plt.title("кластеры и их метки")
    plt.xlabel("компонента 1")
    plt.ylabel("компонента 2")
    plt.legend(loc='best', fontsize='small', frameon=True)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# top = get_top_questions(embeddings, labels, df, top_n=25)
# print(top)
# labels = label_clusters_batch(top, client)
# for cluster, label in labels.items():
#     print(f"{cluster}: {label}")


clusters={
    -1: "Сложность семестров",
    0: "Студенческая жизнь, спорт",
    1: "Обучение",
    2: "Экзамены",
}



# plot_clusters_with_labels(reduced, labels, clusters)
