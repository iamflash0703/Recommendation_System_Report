# 🎥 Movie Recommendation System — Collaborative Filtering

Built a user-based collaborative filtering recommendation system that suggests movies based on similarity between users' rating patterns.

## 📌 Overview
This project covers the full recommendation system workflow — building a user-item ratings matrix, calculating user similarity, generating recommendations, and validating that the model correctly captures taste patterns.

## 🛠️ Tools & Libraries
- Python
- Pandas, NumPy
- Scikit-learn (cosine similarity)
- Matplotlib, Seaborn

## 🔍 Steps Followed
1. **Data Preparation** — Generated a realistic synthetic ratings dataset (30 users, 11 movies) with 3 taste clusters: Sci-Fi/Action, Romance, Comedy.
2. **User-Item Matrix** — Built a sparse ratings matrix (not every user rates every movie, like real-world data).
3. **Similarity Calculation** — Used Cosine Similarity to measure taste similarity between every pair of users.
4. **Recommendation Generation** — For a target user, found the 5 most similar users and predicted ratings for unrated movies using a weighted average.
5. **Evaluation** — Verified that same-cluster users showed significantly higher similarity than cross-cluster users, confirming the model works.

## 📊 Results
- **Same-cluster user similarity:** 0.796 (avg)
- **Different-cluster user similarity:** 0.412 (avg)
- **→ 93% higher similarity within taste clusters** — confirms the model correctly captures user preferences

## 📁 Files
- `recommendation_system.py` — full commented Python script
- `*.png` — ratings heatmap, user similarity matrix, cluster distribution
