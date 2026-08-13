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
function drawBankrollLine(canvasId, labels, values, startValue) {
  destroyChart(canvasId);
  const ctx = document.getElementById(canvasId);
  if (!ctx) return;
  const accent = cssVar('--chart-line') || '#a15c2e';
  const grid = cssVar('--chart-grid') || '#e2d5b8';
  const axis = cssVar('--chart-axis') || '#8a7a5f';
  const baseline = cssVar('--chart-baseline') || '#cdb98c';

  const datasets = [{
    label: 'Bankroll',
    data: values,
    borderColor: accent,
    backgroundColor: accent + '1f',
    fill: true,
    tension: 0.25,
    borderWidth: 2,
    pointRadius: 0,
    pointHoverRadius: 4,
    pointHoverBackgroundColor: accent,
    pointHoverBorderColor: cssVar('--surface') || '#faf6ec',
    pointHoverBorderWidth: 2,
  }];

  if (startValue !== undefined && startValue !== null) {
    datasets.push({
      label: 'Starting bankroll',
      data: labels.map(() => startValue),
      borderColor: baseline,
      borderDash: [4, 4],
      borderWidth: 1,
      pointRadius: 0,
      fill: false,
    });
  }

  activeCharts[canvasId] = new Chart(ctx, {
    type: 'line',
    data: { labels, datasets },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      animation: { duration: 250 },
      interaction: { mode: 'index', intersect: false },
      scales: {
        x: { grid: { display: false }, ticks: { color: axis, font: { size: 10 }, maxTicksLimit: 8 } },
        y: { grid: { color: grid }, ticks: { color: axis, font: { size: 11 } } },
      },
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: {
            label: (ctx) => `${ctx.dataset.label}: ${ctx.parsed.y.toFixed(2)}`,
          },
        },
      },
    },
  });
}

// CFB is ratings-only (no live ESPN sync — see main.py's startup comment on
// why) so it will always show zero picks on Suggestions/Parlay; both of
// those pages already skip any league with zero picks, so including it
// here is safe and only actually surfaces on the Ratings tab.
const LEAGUE_ORDER = ['NFL', 'CFB', 'MLB', 'NBA', 'NHL'];

// ── Main App ──────────────────────────────────────────────────────────────
function app() {
  return {
    route: { page: 'suggestions' },
    teams: [],
    toasts: [],

    suggestions: {
      loading: true,
      data: null,
      mlbProbables: {},   // espn_event_id -> {home, away}
    },

    qbStatuses: {},        // team display name -> starter_out() dict

    parlay: {
      loading: true,
      data: null,
      selected: {},       // legKey -> pick object
      combining: false,
      combineResult: null,
      combineError: '',
    },

    dash: {
      loading: true,
      games: [],
      showForm: false,
      saving: false,
      formError: '',
      form: emptyManualGameForm(),
      synced: {
        loading: true,
        games: [],
      },
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
      data: null,
    },

    betlog: {
      loading: true,
      bets: [],
      filterStatus: '',
      showForm: false,
      saving: false,
      formError: '',
      form: emptyBetForm(),
      bankroll: null,
      bankrollLoading: true,
    },

    // ── Init / routing ──────────────────────────────────────────────────
    async init() {
      window.addEventListener('hashchange', () => this.parseRoute());
      this.parseRoute();
      try {
        const list = await apiGet('/api/ratings');
        this.teams = (list || []).map(r => r.team);
      } catch (e) { /* ratings tab load will surface the error */ }
    },

    parseRoute() {
      const hash = window.location.hash.replace('#', '').replace(/^\//, '') || 'suggestions';
      const page = hash.split('/')[0] || 'suggestions';
      const valid = ['suggestions', 'parlay', 'dashboard', 'ratings', 'livetrack', 'betlog'];
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
      else if (page === 'dashboard') { this.loadDashboard(); this.loadSyncedGames(); }
      else if (page === 'ratings') this.loadRatings();
      else if (page === 'livetrack') this.loadForwardTrack();
      else if (page === 'betlog') { this.loadBets(); this.loadBankroll(); }
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

    // ── Dashboard ─────────────────────────────────────────────────────────
    async loadDashboard() {
      this.dash.loading = true;
      try {
        const games = await apiGet('/api/games/manual');
        (games || []).forEach(g => { g._best = this.bestOpp(g); g._source = 'manual'; });
        this.dash.games = games || [];
      } catch (e) {
        this.toast('Failed to load games: ' + e.message, 'error');
      } finally { this.dash.loading = false; }
    },

    get sortedDashGames() {
      return this.sortGamesByEdge(this.dash.games);
    },

    // Real games synced from ESPN (schedule + live DraftKings odds), scored
    // through the model — this is the primary Dashboard source; manual entry
    // is the secondary/fallback path for props or games ESPN doesn't have.
    async loadSyncedGames() {
      this.dash.synced.loading = true;
      try {
        const games = await apiGet('/api/games/upcoming?sport=NFL');
        (games || []).forEach(g => { g._best = this.bestOpp(g); g._source = 'synced'; });
        this.dash.synced.games = games || [];
      } catch (e) {
        this.toast('Failed to load synced games: ' + e.message, 'error');
      } finally { this.dash.synced.loading = false; }
      this.loadQbStatuses();
    },

    // NFL-only context flag: starting-QB availability (Out/Doubtful/Questionable),
    // see sports/nfl/injuries.py. Purely informational — silent on failure,
    // same reasoning as MLB probables above.
    async loadQbStatuses() {
      try {
        const data = await apiGet('/api/injuries/qb-status');
        this.qbStatuses = data.statuses || {};
      } catch (e) { /* silent — bonus context, not required */ }
    },

    qbFlagFor(teamName) {
      const s = this.qbStatuses[teamName];
      if (!s) return null;
      if (s.likely_out) return { label: 'QB ' + s.status_abbr + ': ' + s.name, cls: 'out' };
      if (s.questionable) return { label: 'QB Q: ' + s.name, cls: 'questionable' };
      return null;
    },

    get sortedSyncedGames() {
      return this.sortGamesByEdge(this.dash.synced.games);
    },

    sortGamesByEdge(games) {
      return [...games].sort((a, b) => {
        const ea = a._best ? a._best.edge_pct : -Infinity;
        const eb = b._best ? b._best.edge_pct : -Infinity;
        return eb - ea;
      });
    },

    async submitManualGame() {
      const f = this.dash.form;
      if (!f.date || !f.home_team || !f.away_team) {
        this.dash.formError = 'Date, home team, and away team are required.';
        return;
      }
      if (f.home_team === f.away_team) {
        this.dash.formError = 'Home and away team must be different.';
        return;
      }
      this.dash.formError = '';
      this.dash.saving = true;
      try {
        await apiSend('/api/games/manual', 'POST', f);
        this.toast('Game added');
        this.dash.form = emptyManualGameForm();
        this.dash.showForm = false;
        await this.loadDashboard();
      } catch (e) {
        this.dash.formError = e.message;
      } finally { this.dash.saving = false; }
    },

    async deleteManualGame(id) {
      if (!confirm('Remove this game?')) return;
      try {
        await apiSend(`/api/games/manual/${id}`, 'DELETE');
        this.toast('Game removed');
        this.dash.games = this.dash.games.filter(g => g.id !== id);
      } catch (e) {
        this.toast('Failed to remove game: ' + e.message, 'error');
      }
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
    // and settled against the real result. Distinct from Backtest (replays
    // history) and Bet Log (only bets the user actually placed). Reuses the
    // Backtest tab's stat-card / honesty-banner / table patterns as-is —
    // no new chart or palette needed here.
    async loadForwardTrack() {
      this.livetrack.loading = true;
      try {
        this.livetrack.data = await apiGet('/api/forward-test?sport=NFL');
      } catch (e) {
        this.toast('Failed to load live track record: ' + e.message, 'error');
      } finally { this.livetrack.loading = false; }
    },

    get sortedForwardPicks() {
      const picks = (this.livetrack.data && this.livetrack.data.picks) || [];
      return [...picks].sort((a, b) => (b.date || '').localeCompare(a.date || ''));
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

    // ── Bet Log ───────────────────────────────────────────────────────────
    async loadBets() {
      this.betlog.loading = true;
      try {
        const qs = this.betlog.filterStatus ? `?status=${encodeURIComponent(this.betlog.filterStatus)}` : '';
        this.betlog.bets = await apiGet('/api/bets' + qs);
      } catch (e) {
        this.toast('Failed to load bets: ' + e.message, 'error');
      } finally { this.betlog.loading = false; }
    },

    async loadBankroll() {
      this.betlog.bankrollLoading = true;
      try {
        const data = await apiGet('/api/bankroll');
        this.betlog.bankroll = data;
        await this.$nextTick();
        if (data.curve && data.curve.length) {
          drawBankrollLine(
            'chartBetlogBankroll',
            data.curve.map(p => p.date),
            data.curve.map(p => Number(p.bankroll.toFixed(2))),
            data.starting_bankroll
          );
        }
      } catch (e) {
        this.toast('Failed to load bankroll: ' + e.message, 'error');
      } finally { this.betlog.bankrollLoading = false; }
    },

    async submitBet() {
      const f = this.betlog.form;
      if (!f.game_label || !f.market || !f.side || !f.odds_taken || !f.stake || !f.placed_at) {
        this.betlog.formError = 'Game, market, side, odds, stake, and date are required.';
        return;
      }
      this.betlog.formError = '';
      this.betlog.saving = true;
      try {
        await apiSend('/api/bets', 'POST', f);
        this.toast('Bet logged');
        this.betlog.form = emptyBetForm();
        this.betlog.showForm = false;
        await this.loadBets();
        await this.loadBankroll();
      } catch (e) {
        this.betlog.formError = e.message;
      } finally { this.betlog.saving = false; }
    },

    async updateBetResult(bet, result) {
      try {
        await apiSend(`/api/bets/${bet.id}`, 'PUT', { result });
        this.toast('Bet updated');
        await this.loadBets();
        await this.loadBankroll();
      } catch (e) {
        this.toast('Failed to update bet: ' + e.message, 'error');
      }
    },

    async updateBetClosing(bet, closingOddsStr) {
      const closing_odds = closingOddsStr === '' ? null : Number(closingOddsStr);
      try {
        await apiSend(`/api/bets/${bet.id}`, 'PUT', { closing_odds });
        this.toast('Closing odds saved');
        await this.loadBets();
      } catch (e) {
        this.toast('Failed to save closing odds: ' + e.message, 'error');
      }
    },

    async deleteBet(id) {
      if (!confirm('Delete this bet?')) return;
      try {
        await apiSend(`/api/bets/${id}`, 'DELETE');
        this.toast('Bet deleted');
        this.betlog.bets = this.betlog.bets.filter(b => b.id !== id);
        await this.loadBankroll();
      } catch (e) {
        this.toast('Failed to delete bet: ' + e.message, 'error');
      }
    },

  };
}

function emptyManualGameForm() {
  return {
    date: new Date().toISOString().slice(0, 10),
    home_team: '', away_team: '',
    home_odds: null, away_odds: null,
    home_line: null, home_line_odds: null, away_line_odds: null,
    total_line: null, over_odds: null, under_odds: null,
  };
}

function emptyBetForm() {
  return {
    sport: 'NFL', game_label: '', market: 'moneyline', side: 'home',
    line: null, odds_taken: null, stake: null,
    placed_at: new Date().toISOString().slice(0, 10), notes: '',
  };
}
