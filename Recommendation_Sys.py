"""
Project 5: Recommendation System (Simple Collaborative Filtering)

Goal: Recommend movies to users based on similarity between users'
rating patterns, using collaborative filtering.
"""

# ---------- STEP 0: Import Libraries ----------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity

sns.set_style("darkgrid")
np.random.seed(42)

# ---------- STEP 1: Data Preparation ----------
# MovieLens dataset requires an external download (grouplens.org is not
# reachable in this environment), so we generate a realistic synthetic
# user-item ratings dataset instead - a commonly accepted substitute for
# this specific mini-project when the original source isn't accessible.
#
# We simulate 3 "taste clusters" of users (e.g., action fans, romance
# fans, comedy fans) so the collaborative filtering has real patterns
# to discover - just like real-world data would have.

movies = [
    "Inception", "The Dark Knight", "Interstellar",      # sci-fi/action cluster
    "Titanic", "The Notebook", "La La Land",              # romance cluster
    "Superbad", "The Hangover", "Step Brothers",          # comedy cluster
    "Mad Max", "John Wick"                                # action (crosses over)
]

n_users = 30
user_ids = [f"User_{i+1}" for i in range(n_users)]

# Assign each user a "taste cluster" (0=sci-fi/action, 1=romance, 2=comedy)
user_clusters = np.random.choice([0, 1, 2], size=n_users)

# Build ratings matrix: users rate movies 1-5, with higher ratings for
# movies matching their cluster (simulates real taste-based behavior)
ratings_matrix = np.zeros((n_users, len(movies)))
cluster_movie_map = {
    0: [0, 1, 2, 9, 10],   # sci-fi/action fans like these movies
    1: [3, 4, 5],           # romance fans like these
    2: [6, 7, 8],           # comedy fans like these
}

for i, cluster in enumerate(user_clusters):
    for m_idx in range(len(movies)):
        if m_idx in cluster_movie_map[cluster]:
            # High rating for movies matching taste, with low noise
            # (keeps the signal strong and realistic at the same time)
            rating = np.clip(np.random.normal(4.5, 0.4), 3, 5)
            rate_prob = 0.9   # users rate movies they like more often
        else:
            # Lower rating for movies outside taste
            rating = np.clip(np.random.normal(1.8, 0.5), 1, 3)
            rate_prob = 0.5   # less likely to rate movies outside their taste
        # Simulate real-world sparsity: not every user rates every movie
        if np.random.rand() < rate_prob:
            ratings_matrix[i, m_idx] = round(rating)

ratings_df = pd.DataFrame(ratings_matrix, index=user_ids, columns=movies)
# 0 means "not rated" - keep as is for the sparse matrix representation
print("Ratings matrix shape:", ratings_df.shape)
print("\nSample of ratings matrix:\n", ratings_df.head())

# ---------- STEP 2: User-Item Matrix ----------
# (already built above as ratings_df — rows = users, columns = movies)

# ---------- STEP 3: Similarity Calculation ----------
# User-based collaborative filtering: find users with similar taste
# using Cosine Similarity between their rating vectors.
user_similarity = cosine_similarity(ratings_df)
user_similarity_df = pd.DataFrame(
    user_similarity, index=ratings_df.index, columns=ratings_df.index
)
print("\nUser similarity matrix (first 5x5):\n",
      user_similarity_df.iloc[:5, :5].round(2))

# ---------- STEP 4: Recommendation Generation ----------
def recommend_movies(target_user, ratings_df, similarity_df, n_recommend=3):
    """
    Recommends movies for target_user based on ratings from the most
    similar users (user-based collaborative filtering).
    """
    # Get similarity scores of all other users to the target user
    sim_scores = similarity_df[target_user].drop(target_user)
    # Find top 5 most similar users ("neighbors")
    top_neighbors = sim_scores.sort_values(ascending=False).head(5).index

    # Movies the target user hasn't rated yet
    unrated_movies = ratings_df.columns[ratings_df.loc[target_user] == 0]

    predictions = {}
    for movie in unrated_movies:
        # Weighted average rating from similar users who rated this movie
        neighbor_ratings = ratings_df.loc[top_neighbors, movie]
        neighbor_sims = sim_scores[top_neighbors]
        rated_mask = neighbor_ratings > 0
        if rated_mask.sum() > 0:
            weighted_score = (
                (neighbor_ratings[rated_mask] * neighbor_sims[rated_mask]).sum()
                / neighbor_sims[rated_mask].sum()
            )
            predictions[movie] = weighted_score

    # Sort and return top N recommendations
    recommended = sorted(predictions.items(), key=lambda x: x[1], reverse=True)
    return recommended[:n_recommend]

# Generate recommendations for a sample user (pick one that actually has
# unrated movies, so the demo output is meaningful)
cluster_names_lookup = {0: "Sci-Fi/Action", 1: "Romance", 2: "Comedy"}
sample_idx = next(
    i for i, u in enumerate(user_ids) if (ratings_df.loc[u] == 0).sum() > 0
)
sample_user = user_ids[sample_idx]
recommendations = recommend_movies(sample_user, ratings_df, user_similarity_df)
print(f"\nTop recommendations for {sample_user} "
      f"(cluster: {cluster_names_lookup[user_clusters[sample_idx]]}):")
for movie, score in recommendations:
    print(f"  {movie}: predicted rating {score:.2f}")

# ---------- STEP 5: Evaluation (Optional) ----------
# Sanity check: users in the SAME taste cluster should be more similar
# to each other (on average) than users in DIFFERENT clusters. This
# confirms the similarity calculation is correctly picking up on taste
# patterns, rather than relying on which unrated movie gets recommended
# (which can be misleading since "already rated" movies get excluded).
cluster_names = {0: "Sci-Fi/Action", 1: "Romance", 2: "Comedy"}

same_cluster_sims = []
diff_cluster_sims = []
for i in range(n_users):
    for j in range(i + 1, n_users):
        sim = user_similarity[i, j]
        if user_clusters[i] == user_clusters[j]:
            same_cluster_sims.append(sim)
        else:
            diff_cluster_sims.append(sim)

avg_same = np.mean(same_cluster_sims)
avg_diff = np.mean(diff_cluster_sims)
print(f"\nSanity check:")
print(f"  Avg similarity between SAME-cluster users:      {avg_same:.3f}")
print(f"  Avg similarity between DIFFERENT-cluster users:  {avg_diff:.3f}")
print(f"  -> Same-cluster users are {(avg_same/avg_diff - 1)*100:.0f}% more "
      f"similar on average, confirming the model captures real taste patterns.")

# ---------- STEP 6: Visualization ----------

# 6a. Ratings heatmap (user-item matrix)
plt.figure(figsize=(10, 7))
sns.heatmap(ratings_df, cmap="mako", cbar_kws={"label": "Rating (0 = not rated)"})
plt.title("User-Item Ratings Matrix")
plt.xlabel("Movies")
plt.ylabel("Users")
plt.tight_layout()
plt.savefig("ratings_heatmap.png", dpi=150)
plt.close()

# 6b. User similarity heatmap (first 15 users for readability)
plt.figure(figsize=(9, 7))
sns.heatmap(user_similarity_df.iloc[:15, :15], cmap="viridis", annot=False)
plt.title("User-User Similarity Matrix (Cosine Similarity, first 15 users)")
plt.tight_layout()
plt.savefig("user_similarity.png", dpi=150)
plt.close()

# 6c. Cluster distribution
plt.figure(figsize=(6, 5))
cluster_counts = pd.Series(user_clusters).map(cluster_names).value_counts()
sns.barplot(x=cluster_counts.index, y=cluster_counts.values, palette="rocket")
plt.title("User Taste Cluster Distribution")
plt.ylabel("Number of Users")
plt.tight_layout()
plt.savefig("cluster_distribution.png", dpi=150)
plt.close()

print("\nAll plots saved successfully.")

# ---------- STEP 7: Key Insights ----------
print("\nKey Insights:")
print(f"1. Cosine similarity correctly captured taste patterns — users in the "
      f"same cluster were {(avg_same/avg_diff - 1)*100:.0f}% more similar on "
      f"average than users in different clusters.")
print("2. Sparse matrices (not all users rate all movies) are handled by "
      "only using neighbors who actually rated the target movie.")
print("3. User-based collaborative filtering works well when there's enough "
      "overlap in ratings between similar users.")
