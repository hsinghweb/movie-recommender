# Movie Recommender — Probability for AI Final Project

## 1. The Question

**If a user liked a movie, which other movies will they rate 4 stars or higher?**

This is a question that matters to anyone who builds or uses a recommendation system — from Netflix and YouTube to small streaming services. At its core, recommendation is a probability problem: given what we know about a person's taste, how confident can we be that they'll enjoy something new?

We set out to answer this using **logistic regression**, one of the foundational tools of probabilistic modeling, trained on real movie rating data.

---

## 2. The Guess (Before Looking at the Data)

Before training the model or examining any results, we wrote down our prediction:

> **"I'm 60% sure the model will get at least 65% of predictions right on the test set."**

This guess reflects the belief that a simple three-feature model can do better than random chance (50%) but might struggle since movie preference is complex and personal.

---

## 3. The Data

### Source
- **Dataset:** MovieLens Latest Small
- **Provider:** GroupLens Research, University of Minnesota
- **URL:** https://grouplens.org/datasets/movielens/latest/
- **License:** Permitted for research and education use
- **Citation:** F. Maxwell Harper and Joseph A. Koncel-Kedziorski. *The MovieLens Datasets: History and Context.* ACM Transactions on Interactive Intelligent Systems (TiiS), 5(4), Article 19 (December 2015).

### What's in it
| Stat | Value |
|------|-------|
| Total ratings | 100,836 |
| Unique users | 610 |
| Unique movies | 9,742 |
| Rating scale | 0.5 to 5.0 (half-star increments) |
| Movies with ≥ 20 ratings | 1,297 |

We filtered to movies with at least 20 ratings to ensure we had enough data for meaningful statistics. For the final app, we kept the **300 most popular movies** to keep the JSON file small (93 KB).

---

## 4. The Model

### 4.1 The Probability Story

For any pair of movies (A, B), we model the probability that a user who liked movie A (rated it ≥ 4 stars) will also like movie B as:

```
P(likes B | likes A) = σ(w₁ · genre_overlap + w₂ · avg_rating_B + w₃ · log_popularity_B + bias)
```

where **σ** is the **logistic (sigmoid) function**:

```
σ(x) = 1 / (1 + e^(-x))
```

This function takes any real number and squeezes it into a probability between 0 and 1. It's the core of logistic regression, which the course taught as "σ of a weighted sum."

### 4.2 The Features (Three Inputs)

We chose three features that capture different aspects of movie similarity:

| # | Feature | What it measures | How it's computed |
|---|---------|-----------------|-------------------|
| 1 | **Genre overlap** | How similar two movies are in genre | Jaccard similarity: \|genres_A ∩ genres_B\| / \|genres_A ∪ genres_B\| (ranges from 0 to 1) |
| 2 | **Average rating** | How well-liked the candidate movie is | Mean of all ratings for movie B (ranges from ~1 to 5) |
| 3 | **Log popularity** | How widely known the candidate movie is | log(number of ratings for movie B) |

**Why these three?**
- **Genre overlap** captures content similarity — if you liked a sci-fi movie, you might like another sci-fi movie.
- **Average rating** captures quality — highly rated movies are liked by many people regardless of genre.
- **Log popularity** captures exposure — popular movies have more ratings, which could mean they appeal broadly (we use log to compress the scale since some movies have thousands of ratings).

### 4.3 The Parameters (Weights)

The model has **four parameters** (three weights + one bias) that we learn from data:

| Parameter | Learned Value | Interpretation |
|-----------|--------------|----------------|
| w₁ (genre_overlap) | **+0.4048** | More genre overlap → slightly higher probability of liking |
| w₂ (avg_rating) | **+1.7937** | Higher average rating → much higher probability of liking |
| w₃ (log_popularity) | **−0.0138** | Popularity barely matters (near zero) |
| bias | **−6.5172** | Baseline (shifted so probabilities start low and require positive evidence) |

**Key insight:** The average rating weight (1.79) is **4.4× larger** than genre overlap (0.40), meaning the model relies primarily on "is this a good movie?" rather than "is this a similar movie?" Popularity has essentially no effect.

---

## 5. Training Process

### 5.1 Building Training Pairs

From the MovieLens data, we constructed pairs of movies for each user:

- **Positive pair (label = 1):** User liked both movie A and movie B (both rated ≥ 4 stars)
- **Negative pair (label = 0):** User liked movie A but did NOT like movie B (A rated ≥ 4, B rated < 4)

For each user who liked at least 2 movies, we sampled up to 5 positive pairs and an equal number of negative pairs. This produced:

| Split | Pairs | Purpose |
|-------|-------|---------|
| **Training set** | 4,756 | Used to learn the weights |
| **Locked test set** | 1,189 | Locked away, opened only once at the end |
| **Total** | 5,945 (3,024 positive, 2,921 negative) | |

The split was **80/20**, stratified by label (preserving the positive/negative ratio), with random seed 42 for reproducibility.

### 5.2 Training the Model

We used **scikit-learn's LogisticRegression** with:
- Maximum 1,000 iterations (converged well before that)
- Random seed 42 for reproducibility
- Default L2 regularization (prevents overfitting)

The training process finds the weights that maximize the likelihood of the observed labels — equivalently, it minimizes the log loss (cross-entropy) on the training data.

**Training accuracy: 64.95%**

---

## 6. Results

### 6.1 Locked Test Set (Opened Once)

| Metric | Value |
|--------|-------|
| **Test accuracy** | **68.04%** |
| **Test log loss** | **0.6094** |
| **Guess threshold** | 65% |
| **Guess correct?** | ✅ **Yes** (68.04% > 65%) |

The model correctly predicts whether a user will like a movie **68% of the time** — above our pre-registered guess of 65%. This is better than random chance (50%) but far from perfect, which makes sense given we're using only three simple features.

### 6.2 Calibration Analysis

Calibration measures whether the model's predicted probabilities match reality. If the model says "70% chance you'll like this," do 70% of users actually like it?

| Predicted Bin | Predicted % | Actual % | Gap | Count |
|---------------|-------------|----------|-----|-------|
| 0.0 – 0.1 | 7.6% | 0.0% | 0.076 | 11 |
| 0.1 – 0.2 | 16.0% | 15.5% | 0.005 | 58 |
| 0.2 – 0.3 | 25.4% | 25.2% | 0.002 | 135 |
| 0.3 – 0.4 | 35.9% | 34.8% | 0.011 | 112 |
| 0.4 – 0.5 | 44.4% | 38.2% | 0.062 | 220 |
| 0.5 – 0.6 | 55.8% | 58.0% | 0.022 | 207 |
| 0.6 – 0.7 | 64.7% | 68.0% | 0.033 | 269 |
| 0.7 – 0.8 | 73.6% | 76.0% | 0.024 | 167 |
| 0.8 – 0.9 | 81.6% | 90.0% | 0.084 | 10 |

**Average gap: 0.029**

An average gap of **0.029** means the model's probabilities are off by only about **3 percentage points** on average. This is **well calibrated** — when the model says 60%, roughly 58–62% of users actually liked the movie.

The model is very slightly **underconfident** in the higher probability ranges (predicts 73.6% but actual is 76.0%), meaning it's a bit more reliable than it thinks.

### 6.3 Score (Log₁₀ Likelihood)

The score is the log₁₀ likelihood of the test data under the model:
- **Log loss (natural log):** 0.6094 per pair
- **Equivalent log₁₀:** −0.2647 per pair
- **Total score (1,189 test pairs):** −314.7

For comparison, a coin-flip model (always predicting 50%) would score log₁₀(0.5) = −0.301 per pair, giving a total of −357.9. Our model scores **43 points higher**, confirming it has learned useful patterns.

---

## 7. What Worked and What Didn't

### ✅ What Worked
1. **The model beats random chance** — 68% accuracy vs 50% baseline
2. **Calibration is excellent** — average gap of 0.029 means the probabilities are trustworthy
3. **Average rating is a strong signal** — highly rated movies tend to be liked by everyone
4. **The pre-computed approach works** — 93 KB JSON file serves 300 movies instantly with no backend

### ❌ What Didn't Work
1. **Popularity doesn't help** — the weight is essentially zero (−0.014), meaning how many people rated a movie tells us nothing about whether you'll like it
2. **Genre similarity is secondary** — it helps (weight 0.40) but is dwarfed by average rating (weight 1.79), so the model tends to recommend generally popular, highly-rated movies to everyone
3. **Recommendations lack personalization** — since we only use movie-level features (not user-specific features), the recommendations are similar for everyone. A user who loves horror films gets recommended the same top-rated dramas as everyone else
4. **Limited to popular movies** — we only pre-computed recommendations for 300 movies to keep the file small. The full dataset has 9,742 movies

### 💡 What Could Be Better
- **User-level features:** Tracking individual genre preferences would add personalization
- **Collaborative filtering:** Using "users who liked A also liked B" patterns directly
- **More features:** Year gap, director overlap, cast overlap, movie keywords
- **A small neural network:** Could capture non-linear patterns that logistic regression misses

---

## 8. The App

The project is delivered as a **React web app** (TypeScript) that runs entirely in the browser with no network calls. All data is pre-computed and embedded as a JSON file.

### Features
- **Search & Recommend:** Type a movie name, pick it from a dropdown, and see the top 5 movies the model predicts you'll like, with probabilities
- **Model Details:** Interactive chart of the learned weights, the fitted formula, and training/test statistics
- **Results:** The pre-registered guess (correct!), a calibration chart comparing predicted vs actual probabilities, and a discussion of what worked and what didn't

### Technical Stack
- **React 18** with TypeScript
- **Recharts** for data visualization (bar charts)
- **react-icons/fa** for icons
- Pre-computed data in JSON (93 KB)
- Total code: 27 KB

### Files
| File | Size | Description |
|------|------|-------------|
| `app/App.tsx` | 27 KB | The complete React app |
| `app/moviedata.json` | 93 KB | Pre-computed movie data and recommendations |
| `tools/train_model.py` | ~6 KB | Python script that trains the model and generates the JSON |

---

## 9. How to Reproduce

### Prerequisites
- Node.js (v18+)
- Python 3.10+ with numpy, pandas, scikit-learn

### Steps
```bash
# 1. Install dependencies
npm install

# 2. Download and extract MovieLens data to data/ml-latest-small/

# 3. Train the model and generate app data
python tools/train_model.py

# 4. Verify the app passes all checks
node tools/check.mjs

# 5. Preview the app
npm run dev
```

---

## 10. Course Concepts Used

| Concept | How we used it |
|---------|---------------|
| **Logistic regression** | The core model: σ of a weighted sum of features |
| **Parameters** | Four learned parameters: three weights + bias |
| **Score (log₁₀ likelihood)** | Evaluated model quality vs a baseline |
| **Locked test set** | 20% of data locked away, opened once at the end |
| **Calibration & average gap** | Checked if predicted probabilities match reality (gap = 0.029) |
| **Bernoulli distribution** | Each "likes/doesn't like" outcome is a Bernoulli trial |
| **Pre-registered guess** | "I'm 60% sure the model will get ≥ 65% accuracy" — correct! |

---

## 11. Data Source

**MovieLens Latest Small Dataset**
- GroupLens Research, University of Minnesota
- https://grouplens.org/datasets/movielens/latest/
- Licensed for research and education
- F. Maxwell Harper and Joseph A. Koncel-Kedziorski. *The MovieLens Datasets.* ACM TiiS, 5(4), 2015.
