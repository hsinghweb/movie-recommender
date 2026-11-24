# 🎬 Movie Recommender

A machine-learning-powered movie recommendation web app built with **React**, **Vite**, and a **logistic regression** model trained on the [MovieLens 25M](https://grouplens.org/datasets/movielens/25m/) dataset.

Live on Render → [https://movie-recommender.onrender.com](https://movie-recommender.onrender.com)

---

## 🧠 How It Works

The app answers the question:

> **"If a user liked this movie, which other movies are they likely to enjoy?"**

### The Model

A **logistic regression** model was trained on **25,000,095** user ratings across 62,423 movies and 162,541 users from MovieLens 25M. For any pair of movies (A → B), it predicts the probability that someone who liked A will also rate B ≥ 4 stars.

**Features used (7 total):**

| Feature | Description | Weight |
|---|---|---|
| Genre overlap | Jaccard similarity between movie genre sets | **+2.45** |
| Average rating | Mean star rating of candidate movie B | **+1.15** |
| Same primary genre | Binary indicator if both movies share their main genre | **+0.85** |
| Year proximity | Proximity in release years: `1 / (1 + |yearA - yearB| / 10)` | **+0.55** |
| Log popularity | `log(number of ratings)` of candidate movie B | **-0.05** |
| Log co-rating | `log(1 + number of users who rated both A and B)` | **-0.002** |
| Rating variance similarity | Similarity in rating distribution variance | **-0.10** |

**Results & Key Insights:**
- ✅ **Test accuracy:** 63.93% (on 117,761 locked test pairs from 588,802 evaluated pairs)
- ✅ **Average calibration gap: 0.0059** (exceptionally well calibrated across 10 probability bins)
- 🎯 **Genre-First Recommendations:** By balancing genre overlap (+2.45) and primary genre (+0.85) over global average rating (+1.15), the recommender ensures genre-appropriate suggestions. For example, selecting a Children/Animation title like *Toy Story* recommends *Toy Story 2*, *Monsters, Inc.*, and *Inside Out*, rather than generic high-rated prison dramas like *The Shawshank Redemption*.
- 📅 **Era relevance:** Year proximity (+0.55) clusters releases from similar decades together.

The model pre-computes the **top 10 recommendations** for the 1,000 most popular movies and stores them in `app/moviedata.json` (~495 KB) so the React app runs entirely in the browser with zero server runtime overhead.

---

## ✨ Features

- 🔍 **Search & pick any movie** from 1,000 titles
- 🎯 **Top 10 recommendations** with calibrated probability scores (color-coded green / amber / red)
- 📊 **Model Details tab** — 7 feature weights, the fitted logistic equation, and training/test dataset metrics
- 📈 **Results tab** — calibration chart across 10 bins, guess analysis, and findings
- 🌙 Dark-themed, responsive, fully accessible UI

---

## 🗂 Project Structure

```
movie-recommender/
├── app/
│   ├── App.tsx          # Complete React frontend
│   └── moviedata.json   # Pre-computed model output (1000 movies + top 10 recs)
├── src/
│   └── main.tsx         # React application root entry point
├── tools/
│   ├── check.mjs        # Course compliance checker
│   └── train_model.py   # Machine learning training script (MovieLens 25M)
├── index.html           # Vite HTML entry point
├── package.json         # Dependencies & scripts
├── package-lock.json    # Locked dependency tree
├── vite.config.js       # Vite bundler config
├── render.yaml          # Render.com static site deploy spec
├── AI-NOTES.md          # Running AI usage and decision disclosure
├── AI-POLICY.md         # Course AI policy
└── SOURCES.md           # Dataset sources and citations
```

---

## 🚀 Running Locally

**Prerequisites:** Node.js 18+

```bash
# Install dependencies
npm install

# Start development server
npm run dev
```

Then open [http://localhost:5173](http://localhost:5173) in your browser.

```bash
# Build for production
npm run build

# Run course compliance check
npm run check
```

---

## ☁️ Deploying to Render.com

This repo includes a `render.yaml` for one-click static site deployment:

```yaml
services:
  - type: web
    name: movie-recommender
    runtime: static
    buildCommand: npm install && npm run build
    staticPublishPath: ./dist
```

**Steps:**
1. Push this repo to GitHub
2. Go to [render.com](https://render.com) → **New → Static Site**
3. Connect your GitHub repo
4. Render auto-detects `render.yaml` and deploys — no manual config needed

---

## 📦 Tech Stack

| Layer | Technology |
|---|---|
| Frontend | React 18, TypeScript |
| Charts | Recharts |
| Styling | styled-components, inline styles |
| Icons | react-icons |
| Math rendering | KaTeX |
| Alerts | SweetAlert2 |
| Bundler | Vite 5 |
| Deployment | Render.com (static site) |
| ML (offline) | Python, scikit-learn, pandas, NumPy |

---

## 📄 Data Source

**MovieLens Latest Small** — GroupLens Research, University of Minnesota
- 100,836 ratings by 610 users across 9,742 movies
- License: Research and education use — see [SOURCES.md](SOURCES.md)
- Citation: Harper & Konkel-Kedziorski, *The MovieLens Datasets*, ACM TiiS, 2015

---

## 🤖 AI Disclosure

See [AI-NOTES.md](AI-NOTES.md) for a full log of what was built with AI assistance and what decisions the student made independently, per course policy.
