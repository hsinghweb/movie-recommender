import { useMemo, useState, useCallback } from 'react';
import {
  BarChart, Bar, CartesianGrid, Line, LineChart,
  ResponsiveContainer, Tooltip, XAxis, YAxis, Cell,
  ScatterChart, Scatter, ZAxis, Legend
} from 'recharts';
import { FaFilm, FaStar, FaUsers, FaSearch, FaChartBar, FaBrain, FaCheckCircle, FaTimesCircle } from 'react-icons/fa';
import data from './moviedata.json';

/* ── Types ────────────────────────────────────────────────────────────── */
interface MovieInfo {
  title: string;
  genres: string;
  avgRating: number;
  numRatings: number;
  year?: number;
}

interface Recommendation {
  movieId: number;
  probability: number;
}

interface CalibrationBin {
  bin: string;
  predicted: number;
  actual: number;
  count: number;
  gap: number;
}

const movies: Record<string, MovieInfo> = data.movies;
const recommendations: Record<string, Recommendation[]> = data.recommendations;
const model = data.model;
const guess = data.guess;

const movieList = Object.entries(movies)
  .map(([id, info]) => ({ id, ...info }))
  .sort((a, b) => b.numRatings - a.numRatings);

/* ── Helpers ──────────────────────────────────────────────────────────── */
function starDisplay(rating: number): string {
  const full = Math.floor(rating);
  const half = rating - full >= 0.5 ? '½' : '';
  return '★'.repeat(full) + half + '☆'.repeat(5 - full - (half ? 1 : 0));
}

function probBar(prob: number): string {
  const pct = Math.round(prob * 100);
  return `${pct}%`;
}

function getColor(prob: number): string {
  if (prob >= 0.7) return '#2e7d32';
  if (prob >= 0.5) return '#f57f17';
  return '#c62828';
}

/* ── Styles ───────────────────────────────────────────────────────────── */
const COLORS = {
  bg: '#0f1117',
  card: '#1a1d27',
  cardHover: '#242838',
  accent: '#6c63ff',
  accentLight: '#8b83ff',
  text: '#e8e6f0',
  textMuted: '#9a97ad',
  gold: '#ffd700',
  green: '#4caf50',
  red: '#ef5350',
  border: '#2d3148',
};

const styles = {
  page: {
    fontFamily: "'Segoe UI', system-ui, -apple-system, sans-serif",
    fontSize: 17,
    color: COLORS.text,
    backgroundColor: COLORS.bg,
    minHeight: '100vh',
    padding: '0 16px 48px',
  } as React.CSSProperties,
  container: {
    maxWidth: 900,
    margin: '0 auto',
  } as React.CSSProperties,
  hero: {
    textAlign: 'center' as const,
    padding: '40px 0 24px',
  } as React.CSSProperties,
  h1: {
    fontSize: 32,
    fontWeight: 700,
    color: '#fff',
    margin: '0 0 8px',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    gap: 12,
  } as React.CSSProperties,
  subtitle: {
    fontSize: 18,
    color: COLORS.textMuted,
    margin: 0,
  } as React.CSSProperties,
  section: {
    backgroundColor: COLORS.card,
    borderRadius: 12,
    padding: 24,
    marginBottom: 20,
    border: `1px solid ${COLORS.border}`,
  } as React.CSSProperties,
  sectionTitle: {
    fontSize: 22,
    fontWeight: 600,
    color: '#fff',
    margin: '0 0 16px',
    display: 'flex',
    alignItems: 'center',
    gap: 10,
  } as React.CSSProperties,
  searchContainer: {
    position: 'relative' as const,
    marginBottom: 16,
  } as React.CSSProperties,
  searchInput: {
    width: '100%',
    padding: '14px 16px 14px 44px',
    fontSize: 17,
    backgroundColor: COLORS.bg,
    color: COLORS.text,
    border: `2px solid ${COLORS.border}`,
    borderRadius: 10,
    outline: 'none',
    boxSizing: 'border-box' as const,
  } as React.CSSProperties,
  searchIcon: {
    position: 'absolute' as const,
    left: 14,
    top: '50%',
    transform: 'translateY(-50%)',
    color: COLORS.textMuted,
  } as React.CSSProperties,
  dropdown: {
    position: 'absolute' as const,
    top: '100%',
    left: 0,
    right: 0,
    backgroundColor: COLORS.card,
    border: `1px solid ${COLORS.border}`,
    borderRadius: 10,
    maxHeight: 300,
    overflowY: 'auto' as const,
    zIndex: 100,
    marginTop: 4,
  } as React.CSSProperties,
  dropdownItem: {
    padding: '12px 16px',
    cursor: 'pointer',
    borderBottom: `1px solid ${COLORS.border}`,
    fontSize: 16,
  } as React.CSSProperties,
  recCard: {
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
    padding: '14px 16px',
    backgroundColor: COLORS.bg,
    borderRadius: 10,
    marginBottom: 10,
    border: `1px solid ${COLORS.border}`,
  } as React.CSSProperties,
  probBadge: (prob: number) => ({
    display: 'inline-block',
    padding: '4px 12px',
    borderRadius: 20,
    fontSize: 15,
    fontWeight: 600,
    color: '#fff',
    backgroundColor: getColor(prob),
    minWidth: 48,
    textAlign: 'center' as const,
  }) as React.CSSProperties,
  genrePill: {
    display: 'inline-block',
    padding: '3px 10px',
    borderRadius: 14,
    fontSize: 13,
    backgroundColor: COLORS.border,
    color: COLORS.textMuted,
    marginRight: 6,
    marginBottom: 4,
  } as React.CSSProperties,
  statsGrid: {
    display: 'grid',
    gridTemplateColumns: 'repeat(auto-fit, minmax(180px, 1fr))',
    gap: 12,
  } as React.CSSProperties,
  statCard: {
    backgroundColor: COLORS.bg,
    borderRadius: 10,
    padding: 16,
    textAlign: 'center' as const,
    border: `1px solid ${COLORS.border}`,
  } as React.CSSProperties,
  statValue: {
    fontSize: 28,
    fontWeight: 700,
    color: COLORS.accent,
    margin: 0,
  } as React.CSSProperties,
  statLabel: {
    fontSize: 14,
    color: COLORS.textMuted,
    margin: '4px 0 0',
  } as React.CSSProperties,
  guessBox: (correct: boolean) => ({
    padding: 16,
    borderRadius: 10,
    backgroundColor: correct ? '#1b3a1b' : '#3a1b1b',
    border: `1px solid ${correct ? COLORS.green : COLORS.red}`,
    display: 'flex',
    alignItems: 'flex-start',
    gap: 12,
  }) as React.CSSProperties,
  weightBar: {
    display: 'flex',
    alignItems: 'center',
    gap: 12,
    marginBottom: 10,
  } as React.CSSProperties,
  weightLabel: {
    minWidth: 140,
    fontSize: 15,
    color: COLORS.textMuted,
  } as React.CSSProperties,
  weightValue: {
    fontSize: 15,
    fontWeight: 600,
    minWidth: 60,
    textAlign: 'right' as const,
  } as React.CSSProperties,
  formula: {
    padding: 16,
    backgroundColor: COLORS.bg,
    borderRadius: 10,
    fontFamily: "'Courier New', monospace",
    fontSize: 15,
    overflowX: 'auto' as const,
    border: `1px solid ${COLORS.border}`,
    lineHeight: 1.6,
  } as React.CSSProperties,
  tabs: {
    display: 'flex',
    gap: 0,
    marginBottom: 20,
    borderRadius: 10,
    overflow: 'hidden',
    border: `1px solid ${COLORS.border}`,
  } as React.CSSProperties,
  tab: (active: boolean) => ({
    flex: 1,
    padding: '12px 16px',
    fontSize: 16,
    fontWeight: active ? 600 : 400,
    color: active ? '#fff' : COLORS.textMuted,
    backgroundColor: active ? COLORS.accent : COLORS.card,
    border: 'none',
    cursor: 'pointer',
    textAlign: 'center' as const,
  }) as React.CSSProperties,
  footer: {
    textAlign: 'center' as const,
    fontSize: 14,
    color: COLORS.textMuted,
    padding: '24px 0',
    borderTop: `1px solid ${COLORS.border}`,
    marginTop: 24,
  } as React.CSSProperties,
};

/* ── Main App ─────────────────────────────────────────────────────────── */
export default function App() {
  const [query, setQuery] = useState('');
  const [selectedMovie, setSelectedMovie] = useState<string | null>(null);
  const [showDropdown, setShowDropdown] = useState(false);
  const [activeTab, setActiveTab] = useState<'recommend' | 'model' | 'calibration'>('recommend');

  const filteredMovies = useMemo(() => {
    if (!query.trim()) return movieList.slice(0, 20);
    const q = query.toLowerCase();
    return movieList
      .filter(m => m.title.toLowerCase().includes(q) || m.genres.toLowerCase().includes(q))
      .slice(0, 20);
  }, [query]);

  const recs = useMemo(() => {
    if (!selectedMovie || !recommendations[selectedMovie]) return [];
    return recommendations[selectedMovie];
  }, [selectedMovie]);

  const selectedInfo = selectedMovie ? movies[selectedMovie] : null;

  const handleSelect = useCallback((id: string) => {
    setSelectedMovie(id);
    setQuery(movies[id].title + (movies[id].year ? ` (${movies[id].year})` : ''));
    setShowDropdown(false);
  }, []);

  const calibrationChart = useMemo(() => {
    return (model.calibration as CalibrationBin[]).map(bin => ({
      ...bin,
      predicted: +(bin.predicted * 100).toFixed(1),
      actual: +(bin.actual * 100).toFixed(1),
    }));
  }, []);

  const weightData = useMemo(() => {
    const w = model.weights as Record<string, number>;
    return [
      { name: 'Average Rating', weight: w.avg_rating, absWeight: Math.abs(w.avg_rating) },
      { name: 'Genre Overlap', weight: w.genre_overlap, absWeight: Math.abs(w.genre_overlap) },
      { name: 'Popularity (log)', weight: w.log_popularity, absWeight: Math.abs(w.log_popularity) },
    ];
  }, []);

  return (
    <div style={styles.page}>
      <div style={styles.container}>
        {/* Hero */}
        <header style={styles.hero}>
          <h1 style={styles.h1}>
            <FaFilm aria-hidden="true" style={{ color: COLORS.accent }} />
            Movie Recommender
          </h1>
          <p style={styles.subtitle}>
            A logistic regression model that predicts which movies you will like
          </p>
        </header>

        {/* Model Story */}
        <section style={styles.section} aria-label="Model story">
          <h2 style={styles.sectionTitle}>
            <FaBrain aria-hidden="true" style={{ color: COLORS.accent }} />
            The Probability Story
          </h2>
          <p style={{ margin: '0 0 12px', lineHeight: 1.6 }}>
            <strong>Question:</strong> If you liked a movie, which other movies will you rate 4 stars or higher?
          </p>
          <p style={{ margin: '0 0 12px', lineHeight: 1.6 }}>
            <strong>Model:</strong> For any pair of movies (A, B), the probability that a fan of A
            also likes B is modeled using <strong>logistic regression</strong>:
          </p>
          <div style={styles.formula} role="math" aria-label="P of likes B given likes A equals sigma of w1 times genre overlap plus w2 times average rating of B plus w3 times log popularity of B plus bias">
            P(likes B | likes A) = σ( w₁·genre_overlap + w₂·avg_rating + w₃·log_popularity + bias )
          </div>
          <p style={{ margin: '12px 0 0', lineHeight: 1.6, color: COLORS.textMuted }}>
            σ is the logistic function: σ(x) = 1 / (1 + e⁻ˣ). We learned the weights from {model.trainSize.toLocaleString()} training
            pairs built from MovieLens ratings, then tested on {model.testSize.toLocaleString()} locked pairs.
          </p>
        </section>

        {/* Tabs */}
        <div style={styles.tabs} role="tablist" aria-label="Main sections">
          <button
            role="tab"
            aria-selected={activeTab === 'recommend'}
            style={styles.tab(activeTab === 'recommend')}
            onClick={() => setActiveTab('recommend')}
          >
            <FaSearch aria-hidden="true" /> Recommend
          </button>
          <button
            role="tab"
            aria-selected={activeTab === 'model'}
            style={styles.tab(activeTab === 'model')}
            onClick={() => setActiveTab('model')}
          >
            <FaChartBar aria-hidden="true" /> Model Details
          </button>
          <button
            role="tab"
            aria-selected={activeTab === 'calibration'}
            style={styles.tab(activeTab === 'calibration')}
            onClick={() => setActiveTab('calibration')}
          >
            <FaStar aria-hidden="true" /> Results
          </button>
        </div>

        {/* ── Tab: Recommend ──────────────────────────────────────────── */}
        {activeTab === 'recommend' && (
          <section style={styles.section} aria-label="Movie recommendations">
            <h2 style={styles.sectionTitle}>
              <FaSearch aria-hidden="true" style={{ color: COLORS.accent }} />
              Find Similar Movies
            </h2>
            <p style={{ margin: '0 0 16px', color: COLORS.textMuted }}>
              Type a movie name, pick it from the list, and see the top 5 movies our model thinks you will also like.
            </p>

            {/* Search */}
            <div style={styles.searchContainer}>
              <FaSearch aria-hidden="true" style={styles.searchIcon} />
              <label htmlFor="movie-search" style={{ position: 'absolute', left: -9999 }}>
                Search for a movie
              </label>
              <input
                id="movie-search"
                type="text"
                placeholder="Search movies... e.g. Toy Story, Pulp Fiction"
                value={query}
                onChange={e => { setQuery(e.target.value); setShowDropdown(true); setSelectedMovie(null); }}
                onFocus={() => setShowDropdown(true)}
                onBlur={() => setTimeout(() => setShowDropdown(false), 200)}
                style={styles.searchInput}
                autoComplete="off"
              />
              {showDropdown && filteredMovies.length > 0 && (
                <div style={styles.dropdown} role="listbox" aria-label="Movie suggestions">
                  {filteredMovies.map(m => (
                    <div
                      key={m.id}
                      role="option"
                      aria-selected={selectedMovie === m.id}
                      style={{
                        ...styles.dropdownItem,
                        backgroundColor: selectedMovie === m.id ? COLORS.cardHover : 'transparent',
                      }}
                      onMouseDown={() => handleSelect(m.id)}
                    >
                      <strong>{m.title}</strong>
                      {m.year ? ` (${m.year})` : ''}
                      <span style={{ color: COLORS.textMuted, marginLeft: 8, fontSize: 14 }}>
                        {starDisplay(m.avgRating)} · {m.numRatings} ratings
                      </span>
                    </div>
                  ))}
                </div>
              )}
            </div>

            {/* Selected movie info */}
            {selectedInfo && (
              <div style={{ ...styles.recCard, backgroundColor: COLORS.accent + '18', borderColor: COLORS.accent, marginBottom: 20 }}>
                <div>
                  <div style={{ fontSize: 20, fontWeight: 600, color: '#fff' }}>
                    {selectedInfo.title} {selectedInfo.year ? `(${selectedInfo.year})` : ''}
                  </div>
                  <div style={{ marginTop: 6 }}>
                    {selectedInfo.genres.split('|').map(g => (
                      <span key={g} style={styles.genrePill}>{g}</span>
                    ))}
                  </div>
                  <div style={{ marginTop: 6, color: COLORS.gold, fontSize: 15 }}>
                    {starDisplay(selectedInfo.avgRating)} {selectedInfo.avgRating.toFixed(1)} · <FaUsers aria-hidden="true" style={{ fontSize: 13 }} /> {selectedInfo.numRatings} ratings
                  </div>
                </div>
              </div>
            )}

            {/* Recommendations */}
            {recs.length > 0 && (
              <div>
                <h3 style={{ fontSize: 18, color: '#fff', margin: '0 0 12px' }}>
                  Top 5 Recommendations
                </h3>
                {recs.map((rec, i) => {
                  const info = movies[String(rec.movieId)];
                  if (!info) return null;
                  return (
                    <div key={rec.movieId} style={styles.recCard}>
                      <div style={{ flex: 1 }}>
                        <div style={{ fontSize: 17, fontWeight: 500 }}>
                          <span style={{ color: COLORS.accent, marginRight: 8 }}>#{i + 1}</span>
                          {info.title} {info.year ? `(${info.year})` : ''}
                        </div>
                        <div style={{ marginTop: 4 }}>
                          {info.genres.split('|').map(g => (
                            <span key={g} style={styles.genrePill}>{g}</span>
                          ))}
                        </div>
                        <div style={{ marginTop: 4, fontSize: 14, color: COLORS.gold }}>
                          {starDisplay(info.avgRating)} {info.avgRating.toFixed(1)}
                        </div>
                      </div>
                      <div style={{ textAlign: 'right' as const, marginLeft: 12 }}>
                        <div style={styles.probBadge(rec.probability)}>
                          {probBar(rec.probability)}
                        </div>
                        <div style={{ fontSize: 12, color: COLORS.textMuted, marginTop: 4 }}>
                          probability
                        </div>
                      </div>
                    </div>
                  );
                })}
              </div>
            )}

            {selectedMovie && recs.length === 0 && (
              <p style={{ color: COLORS.textMuted, textAlign: 'center', padding: 24 }}>
                No recommendations found for this movie.
              </p>
            )}

            {!selectedMovie && (
              <p style={{ color: COLORS.textMuted, textAlign: 'center', padding: 24 }}>
                Pick a movie above to see recommendations.
              </p>
            )}
          </section>
        )}

        {/* ── Tab: Model Details ──────────────────────────────────────── */}
        {activeTab === 'model' && (
          <section style={styles.section} aria-label="Model details">
            <h2 style={styles.sectionTitle}>
              <FaChartBar aria-hidden="true" style={{ color: COLORS.accent }} />
              Model Parameters
            </h2>

            {/* Stats grid */}
            <div style={styles.statsGrid}>
              <div style={styles.statCard}>
                <p style={styles.statValue}>{model.trainSize.toLocaleString()}</p>
                <p style={styles.statLabel}>Training pairs</p>
              </div>
              <div style={styles.statCard}>
                <p style={styles.statValue}>{model.testSize.toLocaleString()}</p>
                <p style={styles.statLabel}>Locked test pairs</p>
              </div>
              <div style={styles.statCard}>
                <p style={styles.statValue}>{(model.testAccuracy * 100).toFixed(1)}%</p>
                <p style={styles.statLabel}>Test accuracy</p>
              </div>
              <div style={styles.statCard}>
                <p style={{ ...styles.statValue, color: COLORS.green }}>
                  {model.avgGap.toFixed(3)}
                </p>
                <p style={styles.statLabel}>Average gap</p>
              </div>
            </div>

            {/* Weights */}
            <h3 style={{ fontSize: 18, color: '#fff', margin: '20px 0 12px' }}>
              Learned Weights
            </h3>
            <p style={{ color: COLORS.textMuted, margin: '0 0 12px', fontSize: 15 }}>
              Bigger weight = stronger influence. Average rating matters most, genre overlap helps, and popularity barely matters.
            </p>
            <div role="img" aria-label="Bar chart of model weights" style={{ width: '100%', height: 220 }}>
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={weightData} layout="vertical" margin={{ top: 8, right: 32, bottom: 8, left: 120 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke={COLORS.border} />
                  <XAxis type="number" stroke={COLORS.textMuted} tick={{ fill: COLORS.textMuted }} />
                  <YAxis type="category" dataKey="name" stroke={COLORS.textMuted} tick={{ fill: COLORS.text }} width={110} />
                  <Tooltip
                    contentStyle={{ backgroundColor: COLORS.card, border: `1px solid ${COLORS.border}`, color: COLORS.text }}
                  />
                  <Bar dataKey="weight" radius={[0, 6, 6, 0]}>
                    {weightData.map((entry, index) => (
                      <Cell key={index} fill={entry.weight >= 0 ? COLORS.accent : COLORS.red} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>

            {/* Formula with actual numbers */}
            <h3 style={{ fontSize: 18, color: '#fff', margin: '20px 0 12px' }}>
              Fitted Model
            </h3>
            <div style={styles.formula} role="math" aria-label="The fitted logistic regression equation with learned weights">
              P(likes B | likes A) = σ({(model.weights as Record<string, number>).avg_rating} · avg_rating
              {' '}+ {(model.weights as Record<string, number>).genre_overlap} · genre_overlap
              {' '}{(model.weights as Record<string, number>).log_popularity} · log_pop
              {' '}{model.bias > 0 ? '+ ' : '- '}{Math.abs(model.bias)})
            </div>
          </section>
        )}

        {/* ── Tab: Results & Calibration ──────────────────────────────── */}
        {activeTab === 'calibration' && (
          <section style={styles.section} aria-label="Results and calibration">
            <h2 style={styles.sectionTitle}>
              <FaStar aria-hidden="true" style={{ color: COLORS.gold }} />
              Results
            </h2>

            {/* Guess */}
            <h3 style={{ fontSize: 18, color: '#fff', margin: '0 0 12px' }}>
              My Guess (Before Looking)
            </h3>
            <div style={styles.guessBox(guess.correct)}>
              {guess.correct
                ? <FaCheckCircle aria-hidden="true" style={{ color: COLORS.green, fontSize: 22, flexShrink: 0, marginTop: 2 }} />
                : <FaTimesCircle aria-hidden="true" style={{ color: COLORS.red, fontSize: 22, flexShrink: 0, marginTop: 2 }} />
              }
              <div>
                <p style={{ margin: 0, fontWeight: 600 }}>
                  &ldquo;{guess.statement}&rdquo;
                </p>
                <p style={{ margin: '8px 0 0', color: COLORS.textMuted }}>
                  Actual test accuracy: <strong style={{ color: '#fff' }}>{(guess.actual * 100).toFixed(1)}%</strong>
                  {' '}(threshold: {guess.threshold * 100}%) —{' '}
                  <span style={{ color: guess.correct ? COLORS.green : COLORS.red, fontWeight: 600 }}>
                    {guess.correct ? 'Correct!' : 'Incorrect'}
                  </span>
                </p>
              </div>
            </div>

            {/* Calibration chart */}
            <h3 style={{ fontSize: 18, color: '#fff', margin: '24px 0 12px' }}>
              Calibration: Predicted vs. Actual
            </h3>
            <p style={{ color: COLORS.textMuted, margin: '0 0 12px', fontSize: 15 }}>
              A well-calibrated model has bars close together. When we predict 60%, about 60% of users
              really liked the movie. Average gap: <strong style={{ color: COLORS.green }}>{model.avgGap.toFixed(3)}</strong> — the model is well calibrated.
            </p>
            <div role="img" aria-label="Calibration chart showing predicted versus actual percentages" style={{ width: '100%', height: 320 }}>
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={calibrationChart} margin={{ top: 8, right: 16, bottom: 40, left: 16 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke={COLORS.border} />
                  <XAxis
                    dataKey="bin"
                    stroke={COLORS.textMuted}
                    tick={{ fill: COLORS.textMuted, fontSize: 13 }}
                    label={{ value: 'Predicted probability bin', position: 'insideBottom', offset: -28, fill: COLORS.textMuted }}
                  />
                  <YAxis
                    stroke={COLORS.textMuted}
                    tick={{ fill: COLORS.textMuted }}
                    label={{ value: 'Percentage (%)', angle: -90, position: 'insideLeft', fill: COLORS.textMuted }}
                  />
                  <Tooltip
                    contentStyle={{ backgroundColor: COLORS.card, border: `1px solid ${COLORS.border}`, color: COLORS.text }}
                  />
                  <Legend verticalAlign="top" />
                  <Bar dataKey="predicted" name="Predicted %" fill={COLORS.accent} radius={[4, 4, 0, 0]} />
                  <Bar dataKey="actual" name="Actual %" fill={COLORS.gold} radius={[4, 4, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>

            {/* What worked, what didn't */}
            <h3 style={{ fontSize: 18, color: '#fff', margin: '24px 0 12px' }}>
              What Worked and What Didn&apos;t
            </h3>
            <div style={{ lineHeight: 1.7 }}>
              <p style={{ margin: '0 0 8px' }}>
                <strong style={{ color: COLORS.green }}>✓ What worked:</strong> The model achieves 68% accuracy, beating the 65% threshold.
                It is well calibrated (average gap 0.029). Average rating is the strongest signal — highly rated movies tend to be liked together.
              </p>
              <p style={{ margin: '0 0 8px' }}>
                <strong style={{ color: COLORS.red }}>✗ What didn&apos;t work:</strong> Popularity (number of ratings) barely helps — its weight is near zero.
                The model recommends mostly high-rated movies regardless of genre match, because avg_rating dominates.
                A richer model could use user-specific features or more movie metadata.
              </p>
              <p style={{ margin: 0 }}>
                <strong style={{ color: COLORS.accent }}>→ Takeaway:</strong> Even a simple three-feature logistic regression captures useful patterns,
                but the strongest signal is just &ldquo;good movies tend to be liked by everyone.&rdquo; Genre similarity adds nuance but doesn&apos;t dominate.
              </p>
            </div>
          </section>
        )}

        {/* Footer */}
        <footer style={styles.footer}>
          <p style={{ margin: 0 }}>
            Data: <a href="https://grouplens.org/datasets/movielens/" style={{ color: COLORS.accentLight }}>
              MovieLens Latest Small
            </a> (GroupLens, University of Minnesota).
            {' '}F. Maxwell Harper and Joseph A. Koncel-Kedziorski. The MovieLens Datasets.
            {' '}Licensed for research and education.
          </p>
          <p style={{ margin: '8px 0 0' }}>
            {Object.keys(movies).length} movies · {model.trainSize + model.testSize} rating pairs · Logistic regression
          </p>
        </footer>
      </div>
    </div>
  );
}
