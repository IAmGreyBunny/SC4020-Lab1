import joblib

# import torch.nn as nn
# from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd

# import pandas as pd
# from pandas.core.common import flatten

# from extract_features import *
from common import *

from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image


EMBEDDINGS_PATH = Path(
    "output/zappos_embeddings_N50025.pkl"
)

QUERY_INDEX = 0
TOP_K = 10
#select the first image in the saved sample to be used as the query image for the search
QUERY_INDEX = 0
#specifies how many similar images to retrieve
TOP_K = 20


def get_subcategory(path):
    """Read the subcategory from the Zappos directory structure."""
    return Path(path).relative_to(IMAGE_ROOT).parts[1]


def calculate_metrics(query_index, top_indices, labels):
    """Calculate precision and recall for one query and one label type."""
    query_label = labels[query_index]
    relevant_retrieved = sum(labels[index] == query_label for index in top_indices)
    relevant_available = sum(label == query_label for label in labels) - 1

    precision = relevant_retrieved / len(top_indices)
    recall = relevant_retrieved / relevant_available if relevant_available else 0.0

    return precision, recall, relevant_retrieved, relevant_available


def save_metrics(row):
    """Save the result without duplicating the same N, K and query setting."""
    output_path = Path("output/query_metrics.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    new_result = pd.DataFrame([row])
    if output_path.exists():
        results = pd.read_csv(output_path)
        same_run = (
            (results["dataset_size"] == row["dataset_size"])
            & (results["k"] == row["k"])
            & (results["query_index"] == row["query_index"])
        )
        results = results.loc[~same_run]
        new_result = pd.concat([results, new_result], ignore_index=True)

    new_result.to_csv(output_path, index=False)
    print("Metrics saved to:", output_path)

# each image has a list of 512 numbers, so embedding matrix has shape (1000, 512) for 1000 images
def find_similar_images(query_index, embeddings, top_k):
    # embedding contains the representations of all 1000 images as matrix
    query_embedding = embeddings[query_index]

    # Embeddings were already normalized, so their dot product
    # is equivalent to cosine similarity.
    similarity_scores = embeddings @ query_embedding

    # exclude the query image itself from the results by setting its similarity score to negative infinity
    similarity_scores[query_index] = -np.inf

    #sort the scores in descending order and retrieve the indices of the top_k most similar images
    top_indices = np.argsort(similarity_scores)[::-1][:top_k]

    return top_indices, similarity_scores[top_indices]

def display_results(
    query_index,
    top_indices,
    similarity_scores,
    paths,
    categories,
):
    #[Query] [Rank 1] [Rank 2] [Rank 3] [Rank 4] [Rank 5]
    #[Rank 6] [Rank 7] [Rank 8] [Rank 9] [Rank 10] [Unused]
    total_images = len(top_indices) + 1
    columns = 5
    rows = int(np.ceil(total_images / columns))

    fig, axes = plt.subplots(
        rows,
        columns,
        figsize=(15, rows * 3.5),
    )

    axes = np.atleast_1d(axes).ravel()

    query_image = Image.open(paths[query_index]).convert("RGB")

    axes[0].imshow(query_image)
    axes[0].set_title(
        f"Query\n{categories[query_index]}"
    )
    axes[0].axis("off")
    # connect the result index to its similarity score
    for position, (image_index, score) in enumerate(
        zip(top_indices, similarity_scores),
        start=1,
    ):
        image = Image.open(paths[image_index]).convert("RGB")

        axes[position].imshow(image)
        axes[position].set_title(
            f"Rank {position}\n"
            f"{categories[image_index]}\n"
            f"Score: {score:.3f}"
        )
        axes[position].axis("off")

    # Hide any unused subplot.
    for axis in axes[len(top_indices) + 1:]:
        axis.axis("off")

    plt.tight_layout()

    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(RESULT_PATH, dpi=150)
    plt.show()

    print("Result saved to:", RESULT_PATH)

def main():
    data = joblib.load(EMBEDDINGS_PATH)

    embeddings = data["embeddings"]
    paths = data["paths"]
    categories = data["categories"]
    subcategories = [get_subcategory(path) for path in paths]

    print("Loaded embeddings:", embeddings.shape)
    print("Query image:", paths[QUERY_INDEX])
    print("Query category:", categories[QUERY_INDEX])
    print("Query subcategory:", subcategories[QUERY_INDEX])

    #perform the search, calling similarity search function and receive the top image positions and scores
    top_indices, scores = find_similar_images(
        QUERY_INDEX,
        embeddings,
        TOP_K,
    )

    print("\nTop results:")

    for rank, (index, score) in enumerate(
        zip(top_indices, scores),
        start=1,
    ):
        print(
            f"{rank}. {categories[index]} "
            f"| similarity={score:.4f} "
            f"| {paths[index]}"
        )

    category_precision, category_recall, category_hits, category_total = calculate_metrics(
        QUERY_INDEX, top_indices, categories
    )
    subcategory_precision, subcategory_recall, subcategory_hits, subcategory_total = calculate_metrics(
        QUERY_INDEX, top_indices, subcategories
    )

    print(f"\nCategory Precision@{TOP_K}: {category_precision:.2%} ({category_hits}/{TOP_K})")
    print(f"Category Recall@{TOP_K}: {category_recall:.2%} ({category_hits}/{category_total})")
    print(f"Subcategory Precision@{TOP_K}: {subcategory_precision:.2%} ({subcategory_hits}/{TOP_K})")
    print(f"Subcategory Recall@{TOP_K}: {subcategory_recall:.2%} ({subcategory_hits}/{subcategory_total})")

    save_metrics({
        "dataset_size": len(paths),
        "k": TOP_K,
        "query_index": QUERY_INDEX,
        "query_category": categories[QUERY_INDEX],
        "query_subcategory": subcategories[QUERY_INDEX],
        "category_precision": category_precision,
        "category_recall": category_recall,
        "subcategory_precision": subcategory_precision,
        "subcategory_recall": subcategory_recall,
    })

    global RESULT_PATH
    RESULT_PATH = Path(
        f"output/search_N{len(paths)}_K{TOP_K}_Q{QUERY_INDEX}.png"
    )

    display_results(
        QUERY_INDEX,
        top_indices,
        scores,
        paths,
        categories,
    )


if __name__ == "__main__":
    main()


# device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


# def recommend(filename, model, embeddings, n=10):
#     embedding = get_embedding(model, filename)
#     print(embedding)
#     similarity_scores = cosine_similarity(embedding.unsqueeze(0), embeddings)
#     similarity_scores = list(flatten(similarity_scores))
#     similarity_scores_df = pd.DataFrame(similarity_scores, columns=['Score'])
#     similarity_scores_df = similarity_scores_df.sort_values(by=['Score'], ascending=False)

#     print(similarity_scores_df['Score'][:10])

#     topN = similarity_scores_df[:n].index
#     topN = list(flatten(topN))
#     images = list(flatten([df[df.index==i]['image'] for i in topN]))
    
#     return images


# if __name__ == '__main__':
#     df = prepare_data()
#     embeddings = joblib.load('output/embeddings.pkl')
#     model = get_model(device)

#     recommendations = recommend('output/56913.jpg', model, embeddings)
#     show_recommendations('output/56913.jpg', recommendations, 'output/recommendations.png')
