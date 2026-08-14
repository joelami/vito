// ── Vito — Alpine.js SPA ─────────────────────────────────────────────────
const API = ''; // same origin

async function apiGet(path) {
  const r = await fetch(API + path);
  if (!r.ok) throw new Error(`API ${r.status}: ${path}`);
  return r.json();
}

async function apiSend(path, method, body) {
  const r = await fetch(API + path, {
    method,
    headers: { 'Content-Type': 'application/json' },
    body: body !== undefined ? JSON.stringify(body) : undefined,
  });
  if (!r.ok) {
    let detail = `API ${r.status}`;
    try { const j = await r.json(); if (j.detail) detail = j.detail; } catch (e) {}
    throw new Error(detail);
  }
  if (r.status === 204) return null;
  return r.json();
}

// ── Formatting helpers ───────────────────────────────────────────────────
function fmt(v, dec = 2) {
  if (v === null || v === undefined || Number.isNaN(v)) return '—';
  return Number(v).toFixed(dec);
}

function fmtInt(v) {
  if (v === null || v === undefined || Number.isNaN(v)) return '—';
  return Number(v).toLocaleString('en-US');
}

// Raw dates come in two shapes: ESPN's full UTC timestamp ("2026-08-12T01:45Z")
// and manual-entry's plain date ("2026-08-12"). Both rendered as-is were a raw
// ISO string sitting in the UI — this converts to the viewer's local time zone
// (a "Z" timestamp is meaningless to read literally; the game is at a real local
// time) in a short, human date format instead.
function fmtDate(raw) {
  if (!raw) return '—';
  // Three shapes seen across this app: ESPN's full UTC timestamp
  // ("2026-08-12T01:45Z"), manual-entry's plain date ("2026-08-12"), and
  // SQLite's own datetime('now') format ("2026-08-11 14:30:16" — space
  // instead of "T", no timezone suffix, but genuinely UTC by SQLite's own
  // convention). Normalize the SQLite shape into the same ISO-with-Z form
  // the first case already handles, rather than a third parsing branch.
  const sqlDatetime = /^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}$/;
  if (sqlDatetime.test(raw)) raw = raw.replace(' ', 'T') + 'Z';
  const hasTime = raw.includes('T');
  const d = hasTime ? new Date(raw) : new Date(raw + 'T00:00:00');
  if (Number.isNaN(d.getTime())) return raw;
  const opts = hasTime
    ? { weekday: 'short', month: 'short', day: 'numeric', hour: 'numeric', minute: '2-digit' }
    : { weekday: 'short', month: 'short', day: 'numeric', year: 'numeric' };
  return d.toLocaleString('en-US', opts);
}

// Market-aware line display. Spreads carry their own sign (a positive line
// IS the underdog's real number, so "+3.5" is correct as-is); totals don't —
// the line itself is just a threshold, so tacking a "+" onto "Under 9.0"
// reads like a contradiction ("plus" and "under" clashing). Under gets an
// explicit "-" instead, purely as a visual convention to tell the two sides
// apart at a glance; Over is shown unsigned.
function lineDisplay(o) {
  if (o.line === null || o.line === undefined) return '';
  if (o.market === 'total') {
    return ' (' + (o.side === 'under' ? '-' : '') + o.line + ')';
  }
  return ' (' + (o.line > 0 ? '+' : '') + o.line + ')';
}

const MARKET_LABELS = { moneyline: 'Moneyline', spread: 'Spread', total: 'Total' };
function marketLabel(m) { return MARKET_LABELS[m] || m; }

function sideLabel(o) {
  const map = { home: 'Home', away: 'Away', over: 'Over', under: 'Under' };
  return map[o.side] || o.side;
}


// ── Chart.js helpers (draw/destroy pattern, dataviz-skill palette) ──────
// Diverging polarity pair (positive/negative), single accent for line charts —
// values pulled from CSS custom properties so the chart matches the theme.
let activeCharts = {};

function cssVar(name) {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

function destroyChart(id) {
  if (activeCharts[id]) { activeCharts[id].destroy(); delete activeCharts[id]; }
}

function destroyAllCharts() {
  Object.keys(activeCharts).forEach(destroyChart);
}

// Diverging bar chart: one bar per category, colored by sign of value.
// Values shown ± SE via the accompanying data table (see interaction.md:
// tooltips enhance, they never gate — the SE is also directly readable in
// the table under each chart).
function drawDivergingBar(canvasId, labels, values, opts = {}) {
  destroyChart(canvasId);
  const ctx = document.getElementById(canvasId);
  if (!ctx) return;
  const pos = cssVar('--chart-pos') || '#2f7d4f';
  const neg = cssVar('--chart-neg') || '#a13a2e';
  const grid = cssVar('--chart-grid') || '#e2d5b8';
  const axis = cssVar('--chart-axis') || '#8a7a5f';
  activeCharts[canvasId] = new Chart(ctx, {
    type: 'bar',
    data: {
      labels,
      datasets: [{
        data: values,
        backgroundColor: values.map(v => (v >= 0 ? pos : neg)),
        borderRadius: 4,
        borderSkipped: false,
        maxBarThickness: 40,
      }],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      animation: { duration: 250 },
      scales: {
        x: { grid: { display: false }, ticks: { color: axis, font: { size: 11 } } },
        y: {
          grid: { color: grid },
          // round before appending '%' — Chart.js auto-generated tick values can carry
          // floating-point noise (e.g. -0.2000000000000002) on small/negative ranges
          ticks: { color: axis, font: { size: 11 }, callback: (v) => (Math.round(v * 100) / 100) + '%' },
          title: { display: !!opts.yLabel, text: opts.yLabel || '', color: axis, font: { size: 11 } },
        },
      },
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: {
            label: (ctx) => {
              const extra = (opts.extra && opts.extra[ctx.dataIndex]) || '';
              return `ROI ${ctx.parsed.y.toFixed(2)}%${extra}`;
            },
          },
        },
      },
    },
  });
}

// Single-series line (bankroll curve), with a muted dashed baseline at the
// starting bankroll so over/under performance reads at a glance without any
// "beating the market" framing.

// CFB is ratings-only (no live ESPN sync — see main.py's startup comment on
// why) so it will always show zero picks on Suggestions/Parlay; both of
// those pages already skip any league with zero picks, so including it
// here is safe and only actually surfaces on the Ratings tab.
const LEAGUE_ORDER = ['NFL', 'CFB', 'MLB', 'NBA', 'NHL'];

// ── Main App ──────────────────────────────────────────────────────────────
function app() {
  return {
    route: { page: 'suggestions' },
    toasts: [],

    suggestions: {
      loading: true,
      data: null,
      mlbProbables: {},   // espn_event_id -> {home, away}
    },

    liveRecord: {
      loading: true,
      data: null,
    },

    parlay: {
      loading: true,
      data: null,
      selected: {},       // legKey -> pick object
      combining: false,
      combineResult: null,
      combineError: '',
    },

    ratings: {
      loading: true,
      list: [],
      sort: 'rank',
      order: 'asc',
      league: 'NFL',
    },

    livetrack: {
      loading: true,
      bySport: {},   // sport -> /api/forward-test response
    },

    parlayTrack: {
      loading: true,
      data: null,   // /api/forward-test/parlays response
    },

    // ── Init / routing ──────────────────────────────────────────────────
    async init() {
      window.addEventListener('hashchange', () => this.parseRoute());
      this.parseRoute();
    },

    parseRoute() {
      const hash = window.location.hash.replace('#', '').replace(/^\//, '') || 'suggestions';
      const page = hash.split('/')[0] || 'suggestions';
      const valid = ['suggestions', 'parlay', 'ratings', 'livetrack'];
      this.route = { page: valid.includes(page) ? page : 'suggestions' };
      this.onRouteChange(this.route.page);
    },

    navigate(page) {
      window.location.hash = `#/${page}`;
    },

    onRouteChange(page) {
      destroyAllCharts();
      if (page === 'suggestions') this.loadSuggestions();
      else if (page === 'parlay') this.loadSuggestions();
      else if (page === 'ratings') this.loadRatings();
      else if (page === 'livetrack') this.loadForwardTrack();
    },

    toast(message, type = 'success') {
      const id = Date.now() + Math.random();
      this.toasts.push({ id, message, type });
      setTimeout(() => { this.toasts = this.toasts.filter(t => t.id !== id); }, 3500);
    },

    // ── Shared helpers ───────────────────────────────────────────────────
    fmt, fmtInt, fmtDate, marketLabel, sideLabel, lineDisplay,

    sortedOpps(g) {
      return [...(g.opportunities || [])].sort((a, b) => b.edge_pct - a.edge_pct);
    },

    bestOpp(g) {
      const opps = g.opportunities || [];
      if (!opps.length) return null;
      return opps.reduce((best, o) => (best === null || o.edge_pct > best.edge_pct) ? o : best, null);
    },

    bestEdge(g) {
      const b = this.bestOpp(g);
      return b ? b.edge_pct : null;
    },

    // ── Suggestions of the Day (main page) ──────────────────────────────
    // Every currently-live, already-qualifying edge across every league in
    // one place — the model already decided what qualifies (harness.py's
    // snapshot_new_picks, at the honest opening price); this page only reads
    // and groups it. Shared by both the Suggestions and Parlay tabs so they
    // never show a different picture of "what's live right now."
    async loadSuggestions() {
      const alreadyLoaded = this.suggestions.data !== null;
      this.suggestions.loading = !alreadyLoaded;
      this.parlay.loading = !alreadyLoaded;
      try {
        const data = await apiGet('/api/suggestions/daily');
        this.suggestions.data = data;
        this.parlay.data = data;
        if ((data.sports.MLB && data.sports.MLB.count) > 0) this.loadMlbProbables();
      } catch (e) {
        this.toast('Failed to load suggestions: ' + e.message, 'error');
      } finally {
        this.suggestions.loading = false;
        this.parlay.loading = false;
      }
    },

    // "Vito's record since going live" — the real, cross-league forward-test
    // track record (see /api/forward-test/summary's docstring). Lives on
    // Live Track Record, loaded alongside that page's per-league detail.
    async loadLiveRecord() {
      this.liveRecord.loading = true;
      try {
        this.liveRecord.data = await apiGet('/api/forward-test/summary');
      } catch (e) { /* silent — bonus summary, not required */ }
      finally { this.liveRecord.loading = false; }
    },

    // MLB-only context flag: who's announced to start and their season
    // ERA/W-L — see sports/mlb/probables.py. Not a model input, purely
    // informational, so a failure here shouldn't block the suggestions
    // page itself (no toast — this is a nice-to-have, not core data).
    async loadMlbProbables() {
      try {
        const data = await apiGet('/api/mlb/probable-pitchers');
        this.suggestions.mlbProbables = data.probables || {};
      } catch (e) { /* silent — probables are a bonus, not required */ }
    },

    probablesFor(p) {
      return this.suggestions.mlbProbables[p.espn_event_id] || null;
    },

    probableLabel(side) {
      if (!side) return 'not yet announced';
      return side.name + (side.era ? ' (' + side.era + ' ERA)' : '');
    },

    get leagueOrder() { return LEAGUE_ORDER; },

    leaguePicks(sport) {
      const d = this.suggestions.data;
      return (d && d.sports && d.sports[sport] && d.sports[sport].picks) || [];
    },

    get totalSuggestionCount() {
      const d = this.suggestions.data;
      if (!d) return 0;
      return LEAGUE_ORDER.reduce((sum, sp) => sum + ((d.sports[sp] && d.sports[sp].count) || 0), 0);
    },

    get leaguesLiveToday() {
      const d = this.suggestions.data;
      if (!d) return 0;
      return LEAGUE_ORDER.filter(sp => (d.sports[sp] && d.sports[sp].count) > 0).length;
    },

    pickMatchupLabel(p) {
      return `${p.away_team} @ ${p.home_team}`;
    },

    pickSideLine(p) {
      return marketLabel(p.market) + ' ' + sideLabel(p) + lineDisplay(p);
    },

    // ── Parlay (suggested + manual builder) ─────────────────────────────
    legKey(p) {
      return `${p.sport}:${p.espn_event_id}:${p.market}:${p.side}:${p.line ?? ''}`;
    },

    isLegSelected(p) {
      return !!this.parlay.selected[this.legKey(p)];
    },

    toggleLeg(p) {
      const k = this.legKey(p);
      if (this.parlay.selected[k]) delete this.parlay.selected[k];
      else this.parlay.selected[k] = p;
      this.parlay.combineResult = null;
      this.parlay.combineError = '';
    },

    get selectedLegCount() {
      return Object.keys(this.parlay.selected).length;
    },

    get selectedGameCount() {
      const games = new Set(Object.values(this.parlay.selected).map(p => `${p.sport}:${p.espn_event_id}`));
      return games.size;
    },

    clearSelectedLegs() {
      this.parlay.selected = {};
      this.parlay.combineResult = null;
      this.parlay.combineError = '';
    },

    async combineSelectedLegs() {
      const legs = Object.values(this.parlay.selected).map(p => ({
        game_key: `${p.sport}:${p.espn_event_id}`, market: p.market, side: p.side, line: p.line,
        model_prob: p.model_prob, market_odds: p.market_odds,
        market_fair_prob: p.market_fair_prob, confidence: p.confidence,
      }));
      if (legs.length < 1) {
        this.parlay.combineError = 'Select at least one pick below.';
        return;
      }
      this.parlay.combining = true;
      this.parlay.combineError = '';
      try {
        this.parlay.combineResult = await apiSend('/api/parlay/combine', 'POST', { legs, kelly_frac: 0.25 });
      } catch (e) {
        this.parlay.combineError = e.message;
      } finally {
        this.parlay.combining = false;
      }
    },

    // ── Ratings ───────────────────────────────────────────────────────────
    async loadRatings() {
      this.ratings.loading = true;
      try {
        this.ratings.list = await apiGet('/api/ratings?sport=' + this.ratings.league);
      } catch (e) {
        this.toast('Failed to load ratings: ' + e.message, 'error');
      } finally { this.ratings.loading = false; }
    },

    switchRatingsLeague(sp) {
      if (this.ratings.league === sp) return;
      this.ratings.league = sp;
      this.ratings.sort = 'rank';
      this.ratings.order = 'asc';
      this.loadRatings();
    },

    sortRatingsBy(col) {
      if (this.ratings.sort === col) {
        this.ratings.order = this.ratings.order === 'asc' ? 'desc' : 'asc';
      } else {
        this.ratings.sort = col;
        this.ratings.order = col === 'team' ? 'asc' : 'asc';
      }
    },

    get sortedRatings() {
      const { sort, order } = this.ratings;
      const dir = order === 'asc' ? 1 : -1;
      return [...this.ratings.list].sort((a, b) => {
        if (typeof a[sort] === 'string') return a[sort].localeCompare(b[sort]) * dir;
        return (a[sort] - b[sort]) * dir;
      });
    },

    // ── Live Track Record (forward test) ────────────────────────────────
    // The harness's real, ongoing track record: every qualifying edge found
    // in a real (not historical) game, logged at the honest opening price
    // and settled against the real result. This is where "how is Vito
    // doing" lives — the cross-league summary (loadLiveRecord, above) plus
    // a per-league breakdown, one section per sport, each league's own
    // model kept fully separate the same way it is everywhere else in
    // this app. CFB excluded — it's never live-synced (ratings-only, see
    // core/dispatch.py's LIVE_SPORTS), so it would only ever show empty.
    liveSports: ['NFL', 'MLB', 'NBA', 'NHL'],

    async loadForwardTrack() {
      this.livetrack.loading = true;
      this.loadLiveRecord();
      this.loadParlayTrack();
      try {
        const results = await Promise.all(
          this.liveSports.map(sp => apiGet('/api/forward-test?sport=' + sp).then(data => [sp, data]))
        );
        this.livetrack.bySport = Object.fromEntries(results);
      } catch (e) {
        this.toast('Failed to load live track record: ' + e.message, 'error');
      } finally { this.livetrack.loading = false; }
    },

    sortedForwardPicks(sp) {
      const picks = (this.livetrack.bySport[sp] && this.livetrack.bySport[sp].picks) || [];
      return [...picks].sort((a, b) => (b.date || '').localeCompare(a.date || ''));
    },

    // How Vito's actual suggested parlays (2/3/4/5-leg combos surfaced on
    // Suggestions/Parlay) have done for real — the parlay counterpart to
    // the per-league picks above. Cross-league by nature (a parlay's legs
    // can span sports), so it's one flat log, not bucketed under liveSports.
    async loadParlayTrack() {
      this.parlayTrack.loading = true;
      try {
        this.parlayTrack.data = await apiGet('/api/forward-test/parlays');
      } catch (e) { /* silent — bonus detail, not required for the page to work */ }
      finally { this.parlayTrack.loading = false; }
    },

    sortedForwardParlays() {
      const parlays = (this.parlayTrack.data && this.parlayTrack.data.parlays) || [];
      return [...parlays].sort((a, b) => (b.snapshotted_at || '').localeCompare(a.snapshotted_at || ''));
    },

    parlayLegSummary(p) {
      return (p.legs || [])
        .map(l => `${l.sport} ${l.away_team} @ ${l.home_team} — ${marketLabel(l.market)} ${sideLabel(l)}${lineDisplay(l)}`)
        .join('  ·  ');
    },

    forwardResultLabel(p) {
      if (!p.settled) return 'Pending';
      const map = { win: 'Win', loss: 'Loss', push: 'Push' };
      return map[p.result] || 'Pending';
    },

    forwardResultClass(p) {
      if (!p.settled) return 'pending';
      return ['win', 'loss', 'push'].includes(p.result) ? p.result : 'pending';
    },

  };
}

