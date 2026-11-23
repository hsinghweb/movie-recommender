# 🎬 Movie Recommender

A machine-learning-powered movie recommendation web app built with **React**, **Vite**, and a **logistic regression** model trained on the [MovieLens Latest Small](https://grouplens.org/datasets/movielens/latest/) dataset.

Live on Render → *(add your Render URL here once deployed)*

---

## 🧠 How It Works

The app answers the question:

> **"If a user liked this movie, which other movies are they likely to enjoy?"**

### The Model

A **logistic regression** model was trained on 100,836 real user ratings from MovieLens. For any pair of movies (A → B), it predicts the probability that someone who liked A will also rate B ≥ 4 stars.

**Features used:**

| Feature | Description |
|---|---|
| Genre overlap | Jaccard similarity between the two movies' genre sets |
| Average rating | Mean star rating of the candidate movie B |
| Log popularity | `log(number of ratings)` of candidate movie B |

**Results:**
- ✅ **Test accuracy: 68.04%** (above the 65% pre-registered guess threshold)
- ✅ **Average calibration gap: 0.029** (well-calibrated probabilities)
- Strongest predictor: average rating (weight **+1.79**)
- Genre overlap helps (weight **+0.40**) but isn't dominant
- Popularity barely matters (weight **−0.01**)

The model pre-computes the **top 5 recommendations** for the 300 most popular movies and stores them in `app/moviedata.json` so the React app works entirely in the browser — no server needed.

---

## ✨ Features

- 🔍 **Search & pick any movie** from 300 popular titles
- 🎯 **Top 5 recommendations** with probability scores (colour-coded green / amber / red)
- 📊 **Model Details tab** — feature weights bar chart, the logistic regression formula, training/test stats
- 📈 **Results tab** — calibration chart, guess verification, what worked / didn't
- 🌙 Dark-themed, fully responsive UI

---

## 🗂 Project Structure

```
movie-recommender/
├── app/
│   ├── App.tsx          # Entire React frontend
│   └── moviedata.json   # Pre-computed model output (movies + recommendations)
├── tools/
│   └── check.mjs        # Course compliance checker
├── index.html           # Vite entry point
├── package.json         # Dependencies & build scripts
├── package-lock.json    # Locked dependency versions
├── vite.config.js       # Vite configuration
├── render.yaml          # Render.com deployment config
├── AI-NOTES.md          # AI usage disclosure
├── AI-POLICY.md         # AI policy
└── SOURCES.md           # Dataset citations & licenses
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
