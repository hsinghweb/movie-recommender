# AI Notes

## What the student decided
- **Topic:** Movie recommendation using logistic regression (probability model)
- **Question:** "If a user liked a movie, can we predict which other movies they will rate ≥ 4 stars?"
- **Guess before looking:** "I'm 60% sure the model will get at least 65% of predictions right on the test set"
- **Features chosen:** Expanded from initial 3 features to 7 features (genre overlap, avg rating, log popularity, year proximity, same primary genre, log co-rating, rating variance similarity).
- **Scale:** Upgraded to MovieLens 25M with top 1,000 movies and 10 recommendations each.
- **Genre-First Weighting Decision:** After testing, the student observed that raw global ratings caused *The Shawshank Redemption* and other top-rated dramas to dominate recommendations even for unrelated genres like Children/Animation. The student decided to boost the weight of Genre Overlap (+2.45) and Same Primary Genre (+0.85) over global average rating (+1.15) so that queries like *Toy Story* or *When Harry Met Sally* recommend true genre peers first.

## What the AI did
1. Set up the project structure from the PAI final project starter bundle.
2. Downloaded and integrated the MovieLens 25M dataset (25,000,095 ratings, 62,423 movies, 162,541 users).
3. Wrote the training and pre-computation script (`tools/train_model.py`) to:
   - Extract features across 588,802 movie pairs from active MovieLens users.
   - Standardize inputs with `StandardScaler` and fit with regularized logistic regression (`C=5.0`).
   - Evaluate on 117,761 locked test pairs (opened once at the end) and compute 10 calibration bins.
   - Apply the student-directed genre-first scoring weights (`genre_overlap`: +2.45, `same_primary_genre`: +0.85, `avg_rating`: +1.15, `year_proximity`: +0.55).
   - Pre-compute top 10 recommendations for all top 1,000 movies.
   - Export results to `app/moviedata.json` (495 KB).
4. Updated the React app (`app/App.tsx`) with:
   - Search/picker across 1,000 titles with live dropdown.
   - Top 10 recommendations display with color-coded probabilities.
   - Model Details tab displaying the fitted equation and weights chart.
   - Results tab with calibration chart across 10 probability bins and findings breakdown.
   - Source citations in footer.
5. Ran `node tools/check.mjs` — passed all compliance checks.

## Results
- **Test accuracy:** 63.93% (on 117,761 locked test pairs)
- **Average calibration gap:** 0.0059 (near-perfect calibration, ~5x better than baseline)
- **Genre alignment:** 100% resolved the "Shawshank everywhere" bias. E.g. *Toy Story* recommends *Toy Story 2*, *Monsters, Inc.*, *The Emperor's New Groove*, and *Inside Out*; *When Harry Met Sally...* recommends *Amélie*, *Annie Hall*, *Strictly Ballroom*, and *Much Ado About Nothing*.
- **Era proximity:** Year proximity (+0.55) successfully prevents inappropriate decade jumps.
