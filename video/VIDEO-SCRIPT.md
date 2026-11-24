# Your 60-Second Video Script

## Overview
- **Total time:** 60 seconds (~140 words)
- **Format:** Screen recording of your app + voiceover (camera can stay off)
- **Tool:** Use OBS, Windows Game Bar (Win+G), or your phone to record

---

## Script (Read This Aloud)

### 🎬 Part 1: Why It's Interesting (0–15 sec)
**[SHOW: Your thumbnail, then switch to the app's main screen]**

> "Have you ever finished a movie and wished Netflix could read your mind
> for what to watch next? I built a movie recommender that uses probability
> to predict which movies you'll love — based on just one movie you liked."

*(~35 words)*

---

### 🧠 Part 2: The Probability Highlight (15–40 sec)
**[SHOW: The "Model Details" tab — scroll through the weights chart and formula]**

> "I used logistic regression from a hundred thousand real MovieLens ratings.
> The model takes three features — genre overlap, average rating, and popularity —
> and outputs the probability you'll rate a movie four stars or higher.
> The strongest signal? Simply how highly rated the movie is.
> Genre similarity helps, but great movies get liked by everyone."

*(~55 words)*

---

### 📊 Part 3: Results (40–60 sec)
**[SHOW: The "Results" tab — your guess box, then the calibration chart]**

> "Before looking at the data, I guessed the model would get at least
> sixty-five percent right. It scored sixty-eight percent on a locked test set
> I never trained on. The calibration is strong too — an average gap of
> just zero-point-zero-three. It works, but it mostly recommends popular
> highly-rated films. A richer model could do better."

*(~55 words)*

---

## What to Show on Screen (Visual Plan)

| Time | What's on screen | What you're saying |
|------|-----------------|-------------------|
| 0–5s | Your **thumbnail** | "Have you ever finished a movie..." |
| 5–15s | App **Recommend tab** — search "Toy Story", show results | "...predict which movies you'll love" |
| 15–25s | Switch to **Model Details tab** — show weights chart | "I used logistic regression..." |
| 25–35s | Scroll to the **formula** section | "...genre overlap, average rating, popularity..." |
| 35–40s | Scroll to **stats cards** (68% accuracy, 0.029 gap) | "...great movies get liked by everyone" |
| 40–50s | Switch to **Results tab** — show the green **guess box** | "Before looking at the data, I guessed..." |
| 50–60s | Scroll to **calibration chart** | "...average gap of just 0.03. It works..." |

## Recording Tips

1. **Practice twice** before recording — aim for natural, not memorized
2. **Open your app** at http://localhost:5173/ before you start recording
3. **Pre-load** a search (type "Toy Story") so the recommendations are ready
4. **Scroll slowly** through each tab so viewers can see the charts
5. **Speak clearly** at a relaxed pace — 60 seconds is enough
6. Camera off is fine — just the app + your voice works great
7. **Add captions** if you can (YouTube auto-captions, or CapCut free)
8. Export as **MP4**, under 250 MB
