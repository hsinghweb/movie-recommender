"""
Train a logistic regression model for movie recommendations.

Model story:
  For any pair of movies (A, B), we model the probability that a user
  who liked A (rated >= 4) will also like B as:
    P(likes B | likes A) = sigma(w1*genre_overlap + w2*avg_rating
        + w3*log_popularity + w4*year_proximity + w5*same_primary_genre
        + w6*log_co_rating + w7*rating_var_sim + bias)
  where sigma is the logistic function. We learn the weights from MovieLens data.

Features (7 total):
  1. Genre overlap: Jaccard similarity of genre sets between movie A and B
  2. Average rating: Mean rating of candidate movie B
  3. Log popularity: log(number of ratings) of candidate movie B
  4. Year proximity: 1 / (1 + |yearA - yearB| / 10)  — close eras score higher
  5. Same primary genre: 1 if the first-listed genre matches, 0 otherwise
  6. Log co-rating: log(1 + number of users who rated both A and B)
  7. Rating variance similarity: 1 / (1 + |stdA - stdB|)

Trained on ml-25m (25 million ratings, 162K users, 62K movies).
Pre-computes top 10 recommendations for the 1000 most popular movies.
"""

import json
import math
import os
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, log_loss

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'ml-25m')
APP_DIR = os.path.join(os.path.dirname(__file__), '..', 'app')

# ── 1. Load data ──────────────────────────────────────────────────────────
print("Loading data (ml-25m — this may take a moment)...")
ratings = pd.read_csv(os.path.join(DATA_DIR, 'ratings.csv'))
movies = pd.read_csv(os.path.join(DATA_DIR, 'movies.csv'))

print(f"  {len(ratings):,} ratings, {len(movies):,} movies, {ratings['userId'].nunique():,} users")

# ── 2. Compute movie stats ───────────────────────────────────────────────
print("Computing movie stats...")
movie_stats = ratings.groupby('movieId').agg(
    avg_rating=('rating', 'mean'),
    num_ratings=('rating', 'count'),
    rating_std=('rating', 'std')
).reset_index()

# Fill NaN std (movies with 1 rating) with 0
movie_stats['rating_std'] = movie_stats['rating_std'].fillna(0)

# Parse genres into sets and extract primary genre
movies['genre_set'] = movies['genres'].apply(
    lambda g: set(g.split('|')) if g != '(no genres listed)' else set()
)
movies['primary_genre'] = movies['genres'].apply(
    lambda g: g.split('|')[0] if g != '(no genres listed)' else ''
)

# Parse year from title
def parse_year(title):
    if title.endswith(')') and '(' in title:
        try:
            return int(title[title.rfind('(') + 1:title.rfind(')')])
        except ValueError:
            return None
    return None

movies['year'] = movies['title'].apply(parse_year)

movies = movies.merge(movie_stats, on='movieId', how='inner')

# Keep only movies with at least 50 ratings (more selective for larger dataset)
MIN_RATINGS = 50
popular = movies[movies['num_ratings'] >= MIN_RATINGS].copy()
popular_ids = set(popular['movieId'].tolist())
print(f"  {len(popular):,} movies with >= {MIN_RATINGS} ratings")

# ── 3. Select the top 1000 movies for the app ────────────────────────────
TOP_N_MOVIES = 1000
top_movies = popular.nlargest(TOP_N_MOVIES, 'num_ratings').copy()
top_ids = set(top_movies['movieId'].tolist())
top_lookup = top_movies.set_index('movieId')
print(f"  Top {TOP_N_MOVIES} movies selected for the app")

# ── 4. Build co-rating matrix for top movies ─────────────────────────────
print("Building co-rating matrix (sparse)...")
# Filter ratings to top movies only
top_ratings = ratings[ratings['movieId'].isin(top_ids)].copy()

# Create mappings for sparse matrix
movie_to_idx = {mid: i for i, mid in enumerate(sorted(top_ids))}
idx_to_movie = {i: mid for mid, i in movie_to_idx.items()}

# Build sparse user-movie matrix (binary: did user rate this movie?)
user_ids = top_ratings['userId'].values
movie_idxs = top_ratings['movieId'].map(movie_to_idx).values
data = np.ones(len(user_ids), dtype=np.float32)

n_users_total = top_ratings['userId'].max() + 1
n_movies_top = len(top_ids)

user_movie = sparse.csr_matrix(
    (data, (user_ids, movie_idxs)),
    shape=(n_users_total, n_movies_top)
)

# Co-rating matrix: M^T @ M gives (n_movies × n_movies) co-rating counts
co_rating_matrix = (user_movie.T @ user_movie).toarray()
print(f"  Co-rating matrix: {co_rating_matrix.shape}")

# ── 5. Build "liked" sets ────────────────────────────────────────────────
print("Building liked-by sets...")
liked = ratings[(ratings['rating'] >= 4) & (ratings['movieId'].isin(popular_ids))]
user_liked = liked.groupby('userId')['movieId'].apply(set).to_dict()

# ── 6. Feature helpers ───────────────────────────────────────────────────
def genre_jaccard(g1, g2):
    if not g1 or not g2:
        return 0.0
    return len(g1 & g2) / len(g1 | g2)

def year_proximity(y1, y2):
    if y1 is None or y2 is None:
        return 0.5  # neutral when year unknown
    return 1.0 / (1.0 + abs(y1 - y2) / 10.0)

def same_primary(pg1, pg2):
    if not pg1 or not pg2:
        return 0.0
    return 1.0 if pg1 == pg2 else 0.0

def rating_var_sim(std1, std2):
    return 1.0 / (1.0 + abs(std1 - std2))

def get_co_rating(mid_a, mid_b):
    """Get co-rating count for two movies (only for top movies)."""
    if mid_a in movie_to_idx and mid_b in movie_to_idx:
        return co_rating_matrix[movie_to_idx[mid_a], movie_to_idx[mid_b]]
    return 0

def safe_float(val, default=0.0):
    """Convert to float, returning default if NaN or None."""
    if val is None:
        return default
    try:
        f = float(val)
        return default if math.isnan(f) else f
    except (ValueError, TypeError):
        return default

def compute_features(ma, mb, mid_a, mid_b):
    """Compute all 7 features for a movie pair."""
    year_a = ma.get('year')
    year_b = mb.get('year')
    if year_a is not None and not isinstance(year_a, (int, float)):
        year_a = None
    elif year_a is not None and math.isnan(float(year_a)):
        year_a = None
    if year_b is not None and not isinstance(year_b, (int, float)):
        year_b = None
    elif year_b is not None and math.isnan(float(year_b)):
        year_b = None

    return [
        genre_jaccard(ma['genre_set'], mb['genre_set']),
        safe_float(mb['avg_rating']),
        math.log(max(1, safe_float(mb['num_ratings'], 1))),
        year_proximity(year_a, year_b),
        same_primary(ma.get('primary_genre', ''), mb.get('primary_genre', '')),
        math.log(1 + get_co_rating(mid_a, mid_b)),
        rating_var_sim(safe_float(ma['rating_std']), safe_float(mb['rating_std'])),
    ]

# ── 7. Build training data (pairs of movies) ─────────────────────────────
print("Building training pairs (sampling from users)...")

movie_lookup = popular.set_index('movieId')
all_pairs = []
positive_count = 0
pair_count = 0

# Sample users for training (use a subset for efficiency with 162K users)
rated_by_user = ratings[ratings['movieId'].isin(popular_ids)].groupby('userId')
np.random.seed(42)

# Sample up to 30000 users for training efficiency
user_groups = list(rated_by_user)
if len(user_groups) > 30000:
    sampled_indices = np.random.choice(len(user_groups), 30000, replace=False)
    user_groups = [user_groups[i] for i in sampled_indices]
    print(f"  Sampled 30,000 users from {len(rated_by_user):,}")
else:
    print(f"  Using all {len(user_groups):,} users")

for uid, group in user_groups:
    user_likes = set(group[group['rating'] >= 4]['movieId'])
    user_dislikes = set(group[group['rating'] < 4]['movieId'])

    if len(user_likes) < 2:
        continue

    likes_list = list(user_likes)

    # Positive pairs: both liked (sample up to 5 pairs per user)
    n_pos = min(5, len(likes_list) * (len(likes_list) - 1) // 2)
    for _ in range(n_pos):
        i, j = np.random.choice(len(likes_list), 2, replace=False)
        a, b = likes_list[i], likes_list[j]
        if a in movie_lookup.index and b in movie_lookup.index:
            ma, mb = movie_lookup.loc[a], movie_lookup.loc[b]
            features = compute_features(ma, mb, a, b)
            all_pairs.append(features + [1])
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
                features = compute_features(ma, mb, a, b)
                all_pairs.append(features + [0])
                pair_count += 1

print(f"  {pair_count:,} pairs ({positive_count:,} positive)")

# ── 8. Train / test split ────────────────────────────────────────────────
print("Training logistic regression...")
feature_names = [
    'genre_overlap', 'avg_rating', 'log_popularity',
    'year_proximity', 'same_primary_genre', 'log_co_rating',
    'rating_var_similarity'
]

df_pairs = pd.DataFrame(all_pairs, columns=feature_names + ['label'])
X = df_pairs[feature_names].values
y = df_pairs['label'].values

# Lock away 20% as the test set (opened only once, at the end)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"  Train: {len(X_train):,}, Locked test: {len(X_test):,}")

# ── 9. Fit logistic regression ───────────────────────────────────────────
model = LogisticRegression(max_iter=2000, random_state=42)
model.fit(X_train, y_train)

# Training metrics
train_pred = model.predict(X_train)
train_proba = model.predict_proba(X_train)[:, 1]
print(f"  Train accuracy: {accuracy_score(y_train, train_pred):.4f}")

# ── 10. Open the locked test set (once!) ─────────────────────────────────
print("\n=== Opening locked test set ===")
test_pred = model.predict(X_test)
test_proba = model.predict_proba(X_test)[:, 1]
test_acc = accuracy_score(y_test, test_pred)
test_loss = log_loss(y_test, test_proba)
print(f"  Test accuracy:  {test_acc:.4f}")
print(f"  Test log loss:  {test_loss:.4f}")

# ── 11. Calibration analysis ─────────────────────────────────────────────
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

# ── 12. Model weights ────────────────────────────────────────────────────
weights = model.coef_[0].tolist()
bias = model.intercept_[0]
print(f"\nModel weights:")
for name, w in zip(feature_names, weights):
    print(f"  {name}: {w:.4f}")
print(f"  bias: {bias:.4f}")

# ── 13. Pre-compute recommendations for top 1000 movies ──────────────────
print(f"\nPre-computing recommendations for {TOP_N_MOVIES} movies...")

recommendations = {}

# Progress tracking
total = len(top_ids)
done = 0

for mid_a in top_ids:
    ma = top_lookup.loc[mid_a]
    scores = []

    for mid_b in top_ids:
        if mid_b == mid_a:
            continue
        mb = top_lookup.loc[mid_b]

        features = np.array([compute_features(ma, mb, mid_a, mid_b)])
        prob = model.predict_proba(features)[0][1]
        scores.append((mid_b, prob))

    scores.sort(key=lambda x: x[1], reverse=True)
    top_recs = scores[:10]
    recommendations[int(mid_a)] = [
        {
            'movieId': int(m),
            'probability': round(p, 4)
        }
        for m, p in top_recs
    ]

    done += 1
    if done % 100 == 0:
        print(f"  {done}/{total} movies done...")

# ── 14. Build the app data JSON ─────────────────────────────────────────
print("Building app data...")

movie_info = {}
for _, row in top_movies.iterrows():
    mid = int(row['movieId'])
    title = row['title']
    year = row.get('year')

    # Extract year from title if present
    display_title = title
    if year is not None and not pd.isna(year):
        year = int(year)
        if title.endswith(')') and '(' in title:
            display_title = title[:title.rfind('(')].strip()
    else:
        year = None

    movie_info[mid] = {
        'title': display_title,
        'genres': row['genres'],
        'avgRating': round(row['avg_rating'], 2),
        'numRatings': int(row['num_ratings']),
    }
    if year is not None:
        movie_info[mid]['year'] = year

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
    },
    'dataset': 'ml-25m'
}

output_path = os.path.join(APP_DIR, 'moviedata.json')
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(app_data, f, separators=(',', ':'))

file_size = os.path.getsize(output_path)
print(f"\nSaved to {output_path}")
print(f"File size: {file_size / 1024:.1f} KB")
print(f"\nMovies in app: {len(movie_info)}")
print(f"Recommendations per movie: 10")
print(f"Test accuracy: {test_acc:.4f} (guess threshold: 0.65, correct: {test_acc >= 0.65})")
print("\nDone!")
