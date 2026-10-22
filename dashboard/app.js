/* app.js -- replay dashboard (plan 9.2), React 18 UMD + htm, no build step.
 *
 * Everything shown is PRECOMPUTED: results/traces/*.json replay the exact
 * simulations that produced full_comparison.csv; charts are the four PNGs
 * from results/charts/. No RL inference runs in the browser (plan rule).
 * Play mechanics: the trace is a list of state-change events; one tick per
 * event, cadence set by the speed slider.
 */
const R = window.React, D = window.ReactDOM;
const html = window.htm.bind(R.createElement);

function useData() {
  const [data, setData] = R.useState(null);
  const [error, setError] = R.useState(null);
  R.useEffect(() => {
    fetch("data.json")
      .then((r) => { if (!r.ok) throw new Error(r.status); return r.json(); })
      .then(setData)
      .catch((e) => setError(`Cannot load dashboard/data.json (${e}). ` +
        `Serve the repo root: python -m http.server 8000`));
  }, []);
  return [data, error];
}

function fetchTrace(path, onOk, onErr) {
  fetch(path)
    .then((r) => { if (!r.ok) throw new Error(r.status); return r.json(); })
    .then(onOk)
    .catch((e) => onErr(`Cannot load trace ${path} (${e})`));
}

/* WHY one tick = one trace event: events are state changes in the event-
 * driven simulator; replaying them in order reproduces the real episode. */
function ReplayArea({ scenario, policy, path }) {
  const [trace, setTrace] = R.useState(null);
  const [error, setError] = R.useState(null);
  const [playing, setPlaying] = R.useState(false);
  const [speed, setSpeed] = R.useState(25);       // events per second
  const [idx, setIdx] = R.useState(0);
  // MUST be before the early returns below (rules of hooks: same count on
  // every render of a mounted component, including the first no-trace one).
  const endJob = R.useRef(null);

  R.useEffect(() => {                              // reset on selection change
    setTrace(null); setError(null); setPlaying(false); setIdx(0);
    if (path) fetchTrace(path, setTrace, setError);
  }, [path]);

  R.useEffect(() => {                              // playback loop
    if (!playing || !trace) return undefined;
    const ms = Math.max(1000 / speed, 5);
    const h = setInterval(() => {                  // advance one event per tick
      setIdx((i) => {
        if (i + 1 >= trace.length) { setPlaying(false); return i; }
        return i + 1;
      });
    }, ms);
    return () => clearInterval(h);
  }, [playing, speed, trace]);

  if (error) return html`<p class="err">${error}</p>`;
  if (!trace) return html`<p>Loading trace...</p>`;

  const ev = trace[Math.min(idx, trace.length - 1)];
  const jobsDone = ev.jobs_done;
  const last = trace[trace.length - 1];
  if (!endJob.current || endJob.current.n !== last.jobs_done) {
    endJob.current = { n: last.jobs_done, when: performance.now() };
  }
  // WHY a brief flash: makes the completion counter respond visibly.
  const flash = performance.now() - endJob.current.when < 150;

  return html`<div>
    <div class="row" style=${{ marginBottom: 8 }}>
      <button onClick=${() => {
        if (idx + 1 >= trace.length) setIdx(0);
        setPlaying(!playing);
      }}>${playing ? "Pause" : "Play"}</button>
      <button onClick=${() => { setIdx(0); setPlaying(false); }}
              disabled=${playing}>Restart</button>
      <label>Speed
        <input type="range" min="5" max="200" value=${speed}
               onChange=${(e) => setSpeed(+e.target.value)}
               style=${{ width: 140 }} />
      </label>
      <span>${speed} events/s</span>
    </div>

    <div>
      <label>Time slider (event ${idx + 1} / ${trace.length})</label>
      <input type="range" min="0" max=${trace.length - 1} value=${idx}
             onChange=${(e) => { setPlaying(false); setIdx(+e.target.value); }} />
    </div>

    <div class="row" style=${{ margin: "10px 0" }}>
      <div class="stat">t = ${ev.time.toFixed(1)}<small>simulated s</small></div>
      <div class="stat">${ev.queue_len}<small>queuing jobs</small></div>
      <div class="stat" style=${flash ? { color: "#2ca02c" } : null}>
        ${jobsDone}<small>jobs completed</small></div>
    </div>

    <div class="bars">
      ${ev.server_cpu_used.map((u, k) => html`<${R.Fragment} key=${k}>
        <span style=${{ fontSize: ".8rem" }}>Server ${k}</span>
        <div class="barbg"><div
          class=${"bar" + (u > 0.85 ? " warn" : "")}
          style=${{ width: `${(u * 100).toFixed(1)}%` }} /></div>
        <span style=${{ fontSize: ".8rem", align: "right" }}>
          ${(u * 100).toFixed(0)}%</span>
      <//>`) }
    </div>
    <p style=${{ fontSize: ".8rem", color: "#5a6b7b" }}>
      Policy: ${policy} &middot; Scenario: ${scenario} &middot; red bar = CPU
      &gt; 85% &middot; bars show CPU; memory is the second resource in the
      saved trace.</p>
  </div>`;
}

function SummaryTable({ table }) {
  return html`<div style=${{ maxHeight: 420, overflow: "auto" }}>
    <table><thead><tr>${table.columns.map((c) => html`<th key=${c}>${c}</th>`)}</tr></thead>
      <tbody>${table.rows.map((r, i) => html`<tr key=${i}>${r.map((v, j) =>
        html`<td key=${j}>${typeof v === "number" ? v.toLocaleString() : v}</td>`)}</tr>`)}</tbody></table>
  </div>`;
}

function App() {
  const [data, error] = useData();
  const [scenario, setScenario] = R.useState(null);
  const [policy, setPolicy] = R.useState(null);

  R.useEffect(() => {
    if (data && !scenario) { setScenario("heavy"); setPolicy("Round Robin"); }
  }, [data]);

  if (error) return html`<div class="card"><p class="err">${error}</p></div>`;
  if (!data || !scenario) return html`<p>Loading dashboard data...</p>`;

  const policies = Object.keys(data.traces[scenario]);
  const path = policy && data.traces[scenario][policy];

  return html`<div>
    <div class="card">
      <div class="row">
        <label>Scenario: <select value=${scenario}
          onChange=${(e) => { setScenario(e.target.value); }}>
          ${data.scenarios.map((s) => html`<option key=${s}>${s}</option>`)}
        </select></label>
        <label>Policy: <select value=${policy}
          onChange=${(e) => setPolicy(e.target.value)}>
          ${policies.map((p) => html`<option key=${p}>${p}</option>`)}
        </select></label>
        <span style=${{ fontSize: ".8rem", color: "#5a6b7b" }}>
          RL traces use each algorithm's seed-0 run (see results/analysis.md,
          ppo_seed2 excluded).</span>
      </div>
    </div>

    <div class="card"><h2>Cluster replay (10 servers, CPU bars)</h2>
      <${ReplayArea} scenario=${scenario} policy=${policy} path=${path} />
    </div>

    <div class="card"><h2>Comparison (results/summary_table.csv)</h2>
      <${SummaryTable} table=${data.summary} />
    </div>

    <div class="card charts"><h2>Report charts (results/charts/)</h2>
      ${data.charts.map((c) => html`<img key=${c} src=${c} alt=${c} />`)}
    </div>
  </div>`;
}

D.createRoot(document.getElementById("root")).render(html`<${App} />`);
