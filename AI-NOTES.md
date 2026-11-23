# AI Notes

## What the student decided
- **Topic:** Movie recommendation using logistic regression (probability model)
- **Question:** "If a user liked a movie, can we predict which other movies they will rate ≥ 4 stars?"
- **Guess before looking:** "I'm 60% sure the model will get at least 65% of predictions right on the test set"
- **Features chosen:** Genre overlap (Jaccard), average rating, popularity (log of number of ratings)
- **Approach:** Logistic regression (from the options: conditional probability, logistic regression, co-rating probability)

## What the AI did
1. Set up the project structure from the PAI final project starter bundle.
2. Downloaded the MovieLens Latest Small dataset (100K ratings, 9742 movies, 610 users).
3. Wrote a Python script (`tools/train_model.py`) to:
   - Build training pairs: for each user, pairs of (liked movie A, liked/disliked movie B)
   - Computed three features: genre Jaccard similarity, average rating of B, log(number of ratings of B)
   - Split data 80/20 into train/locked test sets (stratified, random seed 42)
   - Trained a logistic regression model using scikit-learn
   - Evaluated on the locked test set (opened once, at the end)
   - Computed calibration bins and average gap
   - Pre-computed top 5 recommendations for the 300 most popular movies
   - Saved all results as `app/moviedata.json` (93 KB)
4. Built the React app (`app/App.tsx`) with:
   - A movie search/picker with dropdown
   - Top 5 recommendations display with probabilities
   - Model details tab (weights chart, formula, stats)
   - Results tab (guess check, calibration chart, what worked/didn't)
   - Data source citation in the footer
5. Ran `node tools/check.mjs` — passed.

## Results
- **Test accuracy:** 68.04% (above the 65% guess threshold — guess was correct!)
- **Average calibration gap:** 0.029 (well calibrated)
- **Strongest feature:** avg_rating (weight 1.79)
- **Genre overlap weight:** 0.40 (helpful but not dominant)
- **Popularity weight:** −0.01 (barely matters)
