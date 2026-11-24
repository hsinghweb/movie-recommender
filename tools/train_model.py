"""
Train a logistic regression model for movie recommendations.

Model story:
  For any pair of movies (A, B), we model the probability that a user
  who liked A (rated >= 4) will also like B as:
    P(likes B | likes A) = sigma(w1*genre_overlap + w2*avg_rating_B + w3*popularity_B + bias)
  where sigma is the logistic function. We learn the weights from MovieLens data.

Features:
  1. Genre overlap: Jaccard similarity of genre sets between movie A and B
  2. Average rating: Mean rating of candidate movie B
  3. Popularity: log(number of ratings) of candidate movie B

We pre-compute the top 5 recommendations for the most popular movies and
save them as JSON for the React app.
"""

import json
import math
import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, log_loss

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'ml-latest-small')
APP_DIR = os.path.join(os.path.dirname(__file__), '..', 'app')

# ── 1. Load data ──────────────────────────────────────────────────────────
print("Loading data...")
ratings = pd.read_csv(os.path.join(DATA_DIR, 'ratings.csv'))
movies = pd.read_csv(os.path.join(DATA_DIR, 'movies.csv'))

print(f"  {len(ratings)} ratings, {len(movies)} movies, {ratings['userId'].nunique()} users")

# ── 2. Compute movie stats ───────────────────────────────────────────────
print("Computing movie stats...")
movie_stats = ratings.groupby('movieId').agg(
    avg_rating=('rating', 'mean'),
    num_ratings=('rating', 'count')
).reset_index()

# Parse genres into sets
movies['genre_set'] = movies['genres'].apply(
    lambda g: set(g.split('|')) if g != '(no genres listed)' else set()
)

movies = movies.merge(movie_stats, on='movieId', how='inner')

# Keep only movies with at least 20 ratings (enough data to be meaningful)
MIN_RATINGS = 20
popular = movies[movies['num_ratings'] >= MIN_RATINGS].copy()
popular_ids = set(popular['movieId'].tolist())
print(f"  {len(popular)} movies with >= {MIN_RATINGS} ratings")

# ── 3. Build the "liked" sets ────────────────────────────────────────────
# A user "likes" a movie if they rated it >= 4
print("Building liked-by sets...")
liked = ratings[(ratings['rating'] >= 4) & (ratings['movieId'].isin(popular_ids))]
user_liked = liked.groupby('userId')['movieId'].apply(set).to_dict()

# ── 4. Build training data (pairs of movies) ─────────────────────────────
print("Building training pairs...")

# For efficiency, sample pairs from users who liked multiple movies
all_pairs = []
movie_lookup = popular.set_index('movieId')

# Genre Jaccard similarity helper
def genre_jaccard(g1, g2):
    if not g1 or not g2:
        return 0.0
    return len(g1 & g2) / len(g1 | g2)

pair_count = 0
positive_count = 0

# For each user, their liked movies form positive pairs.
# We also sample negative pairs (liked A, did NOT like B).
rated_by_user = ratings[ratings['movieId'].isin(popular_ids)].groupby('userId')

np.random.seed(42)
popular_list = list(popular_ids)

for uid, group in rated_by_user:
    user_likes = set(group[group['rating'] >= 4]['movieId'])
    user_dislikes = set(group[group['rating'] < 4]['movieId'])

    if len(user_likes) < 2:
        continue

    likes_list = list(user_likes)

    # Positive pairs: both liked (sample up to 5 pairs per user)
    if len(likes_list) > 1:
        n_pos = min(5, len(likes_list) * (len(likes_list) - 1) // 2)
        for _ in range(n_pos):
            i, j = np.random.choice(len(likes_list), 2, replace=False)
            a, b = likes_list[i], likes_list[j]
            if a in movie_lookup.index and b in movie_lookup.index:
                ma, mb = movie_lookup.loc[a], movie_lookup.loc[b]
                all_pairs.append({
                    'genre_overlap': genre_jaccard(ma['genre_set'], mb['genre_set']),
                    'avg_rating': mb['avg_rating'],
                    'log_popularity': math.log(mb['num_ratings']),
                    'label': 1
                })
                positive_count += 1
                pair_count += 1

    # Negative pairs: liked A but not B (sample same number)
    if user_dislikes:
        dislikes_list = list(user_dislikes)
        n_neg = min(n_pos, len(dislikes_list))
        for _ in range(n_neg):
            a = likes_list[np.random.randint(len(likes_list))]
            b = dislikes_list[np.random.randint(len(dislikes_list))]
            if a in movie_lookup.index and b in movie_lookup.index:
                ma, mb = movie_lookup.loc[a], movie_lookup.loc[b]
                all_pairs.append({
                    'genre_overlap': genre_jaccard(ma['genre_set'], mb['genre_set']),
                    'avg_rating': mb['avg_rating'],
                    'log_popularity': math.log(mb['num_ratings']),
                    'label': 0
                })
                pair_count += 1

print(f"  {pair_count} pairs ({positive_count} positive)")

# ── 5. Train / test split ────────────────────────────────────────────────
print("Training logistic regression...")
df_pairs = pd.DataFrame(all_pairs)
X = df_pairs[['genre_overlap', 'avg_rating', 'log_popularity']].values
y = df_pairs['label'].values

# Lock away 20% as the test set (opened only once, at the end)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"  Train: {len(X_train)}, Locked test: {len(X_test)}")

# ── 6. Fit logistic regression ───────────────────────────────────────────
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train, y_train)

# Training metrics
train_pred = model.predict(X_train)
train_proba = model.predict_proba(X_train)[:, 1]
print(f"  Train accuracy: {accuracy_score(y_train, train_pred):.4f}")

# ── 7. Open the locked test set (once!) ──────────────────────────────────
print("\n=== Opening locked test set ===")
test_pred = model.predict(X_test)
test_proba = model.predict_proba(X_test)[:, 1]
test_acc = accuracy_score(y_test, test_pred)
test_loss = log_loss(y_test, test_proba)
print(f"  Test accuracy:  {test_acc:.4f}")
print(f"  Test log loss:  {test_loss:.4f}")

# ── 8. Calibration analysis ─────────────────────────────────────────────
print("\nCalibration analysis:")
n_bins = 10
bins = np.linspace(0, 1, n_bins + 1)
calibration_data = []
total_gap = 0
total_count = 0

for i in range(n_bins):
    mask = (test_proba >= bins[i]) & (test_proba < bins[i + 1])
    if mask.sum() > 0:
        predicted = test_proba[mask].mean()
        actual = y_test[mask].mean()
        count = int(mask.sum())
        gap = abs(predicted - actual)
        total_gap += gap * count
        total_count += count
        calibration_data.append({
            'bin': f"{bins[i]:.1f}-{bins[i+1]:.1f}",
            'predicted': round(predicted, 4),
            'actual': round(actual, 4),
            'count': count,
            'gap': round(gap, 4)
        })
        print(f"  Bin {bins[i]:.1f}-{bins[i+1]:.1f}: predicted={predicted:.3f}, actual={actual:.3f}, gap={gap:.3f} (n={count})")

avg_gap = total_gap / total_count if total_count > 0 else 0
print(f"\n  Average gap: {avg_gap:.4f}")

# ── 9. Model weights ────────────────────────────────────────────────────
feature_names = ['genre_overlap', 'avg_rating', 'log_popularity']
weights = model.coef_[0].tolist()
bias = model.intercept_[0]
print(f"\nModel weights:")
for name, w in zip(feature_names, weights):
    print(f"  {name}: {w:.4f}")
print(f"  bias: {bias:.4f}")

# ── 10. Pre-compute recommendations for each popular movie ──────────────
print("\nPre-computing recommendations...")

# Keep top 300 most popular movies for the app (to keep JSON small)
TOP_N_MOVIES = 300
top_movies = popular.nlargest(TOP_N_MOVIES, 'num_ratings').copy()
top_ids = set(top_movies['movieId'].tolist())

recommendations = {}
top_lookup = top_movies.set_index('movieId')

for mid_a in top_ids:
    ma = top_lookup.loc[mid_a]
    scores = []

    for mid_b in top_ids:
        if mid_b == mid_a:
            continue
        mb = top_lookup.loc[mid_b]

        features = np.array([[
            genre_jaccard(ma['genre_set'], mb['genre_set']),
            mb['avg_rating'],
            math.log(mb['num_ratings'])
        ]])
        prob = model.predict_proba(features)[0][1]
        scores.append((mid_b, prob))

    scores.sort(key=lambda x: x[1], reverse=True)
    top5 = scores[:5]
    recommendations[int(mid_a)] = [
        {
            'movieId': int(m),
            'probability': round(p, 4)
        }
        for m, p in top5
    ]

# ── 11. Build the app data JSON ─────────────────────────────────────────
print("Building app data...")

# Movie info for the app
movie_info = {}
for _, row in top_movies.iterrows():
    mid = int(row['movieId'])
    movie_info[mid] = {
        'title': row['title'],
        'genres': row['genres'],
        'avgRating': round(row['avg_rating'], 2),
        'numRatings': int(row['num_ratings'])
    }

# Extract year from title if present (e.g. "Toy Story (1995)")
for mid, info in movie_info.items():
    title = info['title']
    if title.endswith(')') and '(' in title:
        try:
            year_str = title[title.rfind('(') + 1:title.rfind(')')]
            year = int(year_str)
            info['year'] = year
            info['title'] = title[:title.rfind('(')].strip()
        except ValueError:
            pass

app_data = {
    'movies': movie_info,
    'recommendations': recommendations,
    'model': {
        'weights': {name: round(w, 4) for name, w in zip(feature_names, weights)},
        'bias': round(bias, 4),
        'features': feature_names,
        'trainSize': len(X_train),
        'testSize': len(X_test),
        'testAccuracy': round(test_acc, 4),
        'testLogLoss': round(test_loss, 4),
        'avgGap': round(avg_gap, 4),
        'calibration': calibration_data
    },
    'guess': {
        'statement': "I'm 60% sure the model will get at least 65% of predictions right on the test set",
        'threshold': 0.65,
        'actual': round(test_acc, 4),
        'correct': test_acc >= 0.65
    }
}

output_path = os.path.join(APP_DIR, 'moviedata.json')
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(app_data, f, separators=(',', ':'))

file_size = os.path.getsize(output_path)
print(f"\nSaved to {output_path}")
print(f"File size: {file_size / 1024:.1f} KB")
print(f"\nMovies in app: {len(movie_info)}")
print(f"Test accuracy: {test_acc:.4f} (guess threshold: 0.65, correct: {test_acc >= 0.65})")
print("\nDone!")
