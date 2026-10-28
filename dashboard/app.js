/* app.js -- replay dashboard (plan 9.2), React 18 UMD + htm, no build step.
 *
 * Everything shown is PRECOMPUTED: results/traces/*.json replay the exact
 * simulations that produced full_comparison.csv; charts are the PNGs from
 * results/charts/. No RL inference runs in the browser (plan rule).
 * Play mechanics: the trace is a list of state-change events; one tick per
 * event, cadence set by the speed slider.
 *
 * Two tabs (addendum 5.3):
 *   Replay & comparison -- scenario/policy replay + summary table + charts
 *   Soft computing      -- FIS membership plots + default/GA-tuned rule tables
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

/* Generic JSON loader for the decision-inspector files. */
function useJson(path) {
  const [data, setData] = R.useState(null);
  const [error, setError] = R.useState(null);
  R.useEffect(() => {
    setData(null); setError(null);
    if (!path) return;
    fetch(path)
      .then((r) => { if (!r.ok) throw new Error(r.status); return r.json(); })
      .then(setData)
      .catch((e) => setError(`Cannot load ${path} (${e})`));
  }, [path]);
  return [data, error];
}

/* Step through ONE recorded decision and show the fuzzy arithmetic that
 * produced it: crisp inputs -> memberships -> fired rules -> weighted score ->
 * chosen server. Data comes from softcomputing/record_decisions.py, i.e. from
 * the same seeded episodes that produced the result tables. */
function DecisionInspector({ soft }) {
  const [policy, setPolicy] = R.useState("FIS");
  const [scenario, setScenario] = R.useState("heavy");
  const [idx, setIdx] = R.useState(0);
  const [playing, setPlaying] = R.useState(false);
  const [speed, setSpeed] = R.useState(2);        // decisions per second
  const decisions = (soft && soft.decisions) || {};
  const path = decisions[policy] && decisions[policy][scenario];
  const [rec, error] = useJson(path);

  R.useEffect(() => { setIdx(0); setPlaying(false); }, [path]);
  R.useEffect(() => {
    if (!playing || !rec) return undefined;
    const h = setInterval(() => {                  // one decision per tick
      setIdx((i) => {
        if (i + 1 >= rec.decisions.length) { setPlaying(false); return i; }
        return i + 1;
      });
    }, Math.max(1000 / speed, 150));
    return () => clearInterval(h);
  }, [playing, speed, rec]);

  if (!soft || !soft.decisions) {
    return html`<div class="card"><h2>Fuzzy decision inspector</h2>
      <p class="muted">No decision recording found. Run
        <code>python -m softcomputing.record_decisions</code> and regenerate
        <code>dashboard/data.json</code>.</p></div>`;
  }
  if (!path) {
    return html`<div class="card"><h2>Fuzzy decision inspector</h2>
      <p class="err">No recorded decisions for ${policy} / ${scenario}.</p>
      <p class="muted">Run <code>python -m softcomputing.record_decisions</code>.</p>
    </div>`;
  }
  if (error) return html`<div class="card"><p class="err">${error}</p></div>`;
  if (!rec) return html`<div class="card"><p>Loading decisions...</p></div>`;

  const d = rec.decisions[Math.min(idx, rec.decisions.length - 1)];
  const cands = d.candidates || [];
  const maxScore = Math.max(0.001, ...cands.map((c) => c.score));
  const waiting = d.server === null;
  const top = (d.rules && d.rules[0]) || null;

  return html`<div class="card">
    <h2>Fuzzy decision inspector (step through one decision)</h2>
    <div class="row" style=${{ marginBottom: 8 }}>
      <label>Policy: <select value=${policy}
        onChange=${(e) => setPolicy(e.target.value)}>
        ${["FIS", "GA-FIS"].map((p) => html`<option key=${p}>${p}</option>`)}
      </select></label>
      <label>Scenario: <select value=${scenario}
        onChange=${(e) => setScenario(e.target.value)}>
        ${["light", "heavy", "bursty"].map((s) => html`<option key=${s}>${s}</option>`)}
      </select></label>
      <button onClick=${() => {
        if (idx + 1 >= rec.decisions.length) setIdx(0);
        setPlaying(!playing);
      }}>${playing ? "Pause" : "Play"}</button>
      <label>Speed
        <input type="range" min="1" max="12" value=${speed}
               onChange=${(e) => setSpeed(+e.target.value)}
               style=${{ width: 110 }} />
      </label>
      <span class="muted">${speed} decisions/s</span>
    </div>

    <label>Decision ${idx + 1} of ${rec.decisions.length} sampled
      (${rec.total_decisions} total)</label>
    <input type="range" min="0" max=${rec.decisions.length - 1} value=${idx}
           onChange=${(e) => { setPlaying(false); setIdx(+e.target.value); }} />

    <div class="row" style=${{ margin: "10px 0" }}>
      <div class="stat">t = ${d.time.toFixed(1)}<small>simulated s</small></div>
      <div class="stat">${d.queue_len}<small>queuing jobs</small></div>
      <div class="stat">${d.job[0].toFixed(2)} / ${d.job[1].toFixed(2)}
        <small>job cpu / mem</small></div>
      <div class="stat">${waiting ? "WAIT" : "server " + d.server}
        <small>chosen action</small></div>
      <div class="stat">${waiting ? "—" : d.score.toFixed(3)}
        <small>fuzzy score</small></div>
    </div>

    ${waiting
      ? html`<p class="muted">No candidate server can fit this job, so the
          policy emits WAIT (action ${d.action}).</p>`
      : html`<div>
          <p class="muted">${top ? html`<b>Why:</b> ${top.leftover} /
            ${top.load} / ${top.pressure} fires at ${top.strength.toFixed(3)}
            (z=${top.z.toFixed(2)}); the weighted average over
            ${d.rules.length} active rules is
            ${d.num.toFixed(3)} / ${d.den.toFixed(3)} =
            <b>${d.score.toFixed(3)}</b>.` : null}
            Ties go to the lowest server index.</p>

          <div class="grid2">
            <div>
              <h3>Membership degrees (chosen server)</h3>
              ${["leftover", "load", "pressure"].map((name) => html`<div key=${name}>
                <p class="muted" style=${{ margin: "6px 0 2px" }}>
                  <b>${name}</b> = ${d.crisp[name].toFixed(3)}</p>
                <div class="bars">
                  ${Object.entries(d.memberships[name]).map(([set, v]) => html`
                    <${R.Fragment} key=${set}>
                      <span style=${{ fontSize: ".8rem" }}>${set}</span>
                      <div class="barbg"><div class="bar mem"
                        style=${{ width: `${(v * 100).toFixed(1)}%` }} /></div>
                      <span style=${{ fontSize: ".8rem", textAlign: "right" }}>
                        ${v.toFixed(2)}</span>
                    <//>`)}
                </div>
              </div>`)}
            </div>
            <div>
              <h3>Candidate servers (score, higher = better)</h3>
              <div class="bars">
                ${cands.map((c) => html`<${R.Fragment} key=${c.server}>
                  <span style=${{ fontSize: ".8rem" }}>
                    s${c.server}${c.server === d.server ? " *" : ""}</span>
                  <div class="barbg"><div
                    class=${"bar" + (c.server === d.server ? "" : " alt")}
                    style=${{ width: `${(c.score / maxScore * 100).toFixed(1)}%` }} /></div>
                  <span style=${{ fontSize: ".8rem", textAlign: "right" }}>
                    ${c.score.toFixed(2)}</span>
                <//>`)}
              </div>
              <p class="muted">* chosen. Crisp inputs per candidate:
                leftover = free capacity after placing the job, load = current
                utilization, pressure = queue length (same for all servers).</p>
            </div>
          </div>

          <h3>Fired rules (strength = min of the three memberships)</h3>
          <table class="rules">
            <thead><tr><th>Leftover</th><th>Load</th><th>Pressure</th>
              <th>z</th><th>Strength</th><th>z x strength</th></tr></thead>
            <tbody>${d.rules.map((r) => html`<tr key=${r.i}>
              <td>${r.leftover}</td><td>${r.load}</td><td>${r.pressure}</td>
              <td>${r.z.toFixed(2)}</td><td>${r.strength.toFixed(3)}</td>
              <td>${r.contribution.toFixed(3)}</td></tr>`)}</tbody>
          </table>
        </div>`}

    <p class="muted">Recorded from the same seeded episode that produced the
      results table: avg wait ${Number(rec.metrics.avg_wait).toFixed(1)} s,
      ${rec.metrics.invalid_actions} invalid actions, completion
      ${(rec.metrics.completion_rate * 100).toFixed(0)}%.
      ${rec.ga_seed !== undefined ? `GA seed ${rec.ga_seed}.` : ""}</p>
  </div>`;
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

  if (!path) return html`<p class="muted">No saved trace for this
    policy/scenario.</p>`;
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
      <div class="stat" style=${flash ? { color: "#1b9e77" } : null}>
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
    <p class="muted">Policy: ${policy} · Scenario: ${scenario}
      · red bar = CPU > 85% · bars show CPU; memory is the
      second resource in the saved trace.</p>
  </div>`;
}

function SummaryTable({ table }) {
  return html`<div style=${{ maxHeight: 420, overflow: "auto" }}>
    <table><thead><tr>${table.columns.map((c) => html`<th key=${c}>${c}</th>`)}</tr></thead>
      <tbody>${table.rows.map((r, i) => html`<tr key=${i}>${r.map((v, j) =>
        html`<td key=${j}>${typeof v === "number" ? v.toLocaleString() : v}</td>`)}</tr>`)}</tbody></table>
  </div>`;
}

/* Group the policy dropdown by family (heuristic / soft computing / RL). */
function PolicySelect({ value, onChange, policies, families }) {
  const groups = {};
  policies.forEach((p) => {
    const f = families[p] || "Other";
    if (!groups[f]) groups[f] = [];
    groups[f].push(p);
  });
  return html`<select value=${value} onChange=${(e) => onChange(e.target.value)}>
    ${Object.keys(groups).map((f) => html`<optgroup key=${f} label=${f}>
      ${groups[f].map((p) => html`<option key=${p}>${p}</option>`)}
    </optgroup>`)}
  </select>`;
}

function RuleTable({ rows, cuts, title }) {
  if (!rows) return null;
  const cutText = cuts ? Object.entries(cuts)
    .map(([k, v]) => `${k}=(${v[0].toFixed(2)}, ${v[1].toFixed(2)})`)
    .join("  ·  ") : null;
  return html`<div>
    <h3>${title}</h3>
    ${cutText ? html`<p class="muted">Cut points: ${cutText}</p>` : null}
    <table class="rules">
      <thead><tr><th>Leftover</th><th>Load</th><th>Pressure</th>
        <th>Suitability z</th></tr></thead>
      <tbody>${rows.map((r, i) => html`<tr key=${i}>
        <td>${r[0]}</td><td>${r[1]}</td><td>${r[2]}</td>
        <td>${Number(r[3]).toFixed(2)}</td></tr>`)}</tbody>
    </table>
  </div>`;
}

function SmallTable({ rows, columns }) {
  if (!rows || !rows.length) return html`<p class="muted">No rows.</p>`;
  return html`<table>
    <thead><tr>${columns.map((c) => html`<th key=${c}>${c}</th>`)}</tr></thead>
    <tbody>${rows.map((r, i) => html`<tr key=${i}>${columns.map((c) => {
      const v = r[c];
      return html`<td key=${c}>${typeof v === "number"
        ? (Number.isInteger(v) ? v.toLocaleString() : v.toFixed(2))
        : String(v)}</td>`;
    })}</tr>`)}</tbody></table>`;
}

function SoftComputingTab({ soft }) {
  if (!soft) return html`<div class="card"><p class="err">
    dashboard/data.json has no softcomputing block.</p></div>`;
  return html`<div>
    <div class="card">
      <h2>Soft computing: FIS and GA-tuned FIS</h2>
      <p>Two non-learning policies share the same <code>policy(obs, env)</code>
        interface and the same evaluation harness as the heuristics and RL
        agents. The <b>Fuzzy Inference Scheduler</b> scores every server that
        can fit the head job from three linguistic inputs (leftover slack,
        server load, queue pressure) and picks the best. The
        <b>GA-tuned FIS</b> keeps the same 18 rules but lets a genetic
        algorithm evolve the membership breakpoints and rule consequents on
        <code>data/scenarios/train_pool.csv</code> only.</p>
      ${soft.missing ? html`<p class="err">${soft.missing}</p>` : null}
      ${soft.selected_seed !== null ? html`<p class="muted">
        Selected GA seed: <b>${soft.selected_seed}</b> ·
        training fitness ${Number(soft.train_fitness).toFixed(3)}
        (default FIS ${Number(soft.default_fitness).toFixed(3)}). Selection uses
        training fitness only; the three scenarios stay held out.</p>` : null}
    </div>

    <${DecisionInspector} soft=${soft} />

    ${soft.memberships.length ? html`<div class="card">
      <h2>Membership functions</h2>
      ${soft.memberships.map((src) => html`<figure key=${src}>
        <img src=${src} alt=${src} />
        <figcaption>${src.split("/").pop()}</figcaption></figure>`)}
    </div>` : null}

    ${soft.convergence ? html`<div class="card">
      <h2>GA convergence</h2>
      <img src=${soft.convergence} alt="GA convergence" />
    </div>` : null}

    <div class="card">
      <h2>Rule base: default vs GA-tuned</h2>
      <div class="grid2">
        <${RuleTable} rows=${soft.default_rules} cuts=${soft.default_cuts}
          title="Default FIS (hand-designed)" />
        <${RuleTable} rows=${soft.tuned_rules} cuts=${soft.tuned_cuts}
          title="GA-FIS (evolved)" />
      </div>
      <p class="muted">Rule strength = min of the three memberships; the
        score is the strength-weighted average of the 18 consequents
        (zero-order Sugeno).</p>
    </div>

    ${soft.seed_summary.length ? html`<div class="card">
      <h2>GA seed robustness (training fitness)</h2>
      <${SmallTable} rows=${soft.seed_summary}
        columns=${["seed", "fitness", "default_fitness", "selected"]} />
      <p class="muted">One row per independent GA run; the selected seed is
        the one with the best training fitness.</p>
    </div>` : null}

    ${soft.robustness.length ? html`<div class="card">
      <h2>GA seed robustness (held-out scenarios)</h2>
      <${SmallTable} rows=${soft.robustness}
        columns=${["ga_seed", "train_fitness", "selected", "scenario",
                   "avg_wait", "avg_cpu_util", "invalid_actions"]} />
      <p class="muted">Supplementary check only: every seed's chromosome on
        light/heavy/bursty. The main comparison uses the selected seed.</p>
    </div>` : null}
  </div>`;
}

function App() {
  const [data, error] = useData();
  // WHY the hash: bookmarks/deep links; also lets a headless check load the
  // Soft computing tab directly (dashboard/#soft).
  const [tab, setTab] = R.useState(
    () => (window.location.hash === "#soft" ? "soft" : "replay"));
  const [scenario, setScenario] = R.useState(null);
  const [policy, setPolicy] = R.useState(null);

  const goTab = (next) => {
    setTab(next);
    window.location.hash = next === "soft" ? "soft" : "";
  };

  R.useEffect(() => {
    if (data && !scenario) { setScenario("heavy"); setPolicy("Round Robin"); }
  }, [data]);

  if (error) return html`<div class="card"><p class="err">${error}</p></div>`;
  if (!data || !scenario) return html`<p>Loading dashboard data...</p>`;

  const policies = Object.keys(data.traces[scenario]);
  const path = policy && data.traces[scenario][policy];

  return html`<div>
    <div class="tabs">
      <button class=${"tab" + (tab === "replay" ? " active" : "")}
        onClick=${() => goTab("replay")}>Replay & comparison</button>
      <button class=${"tab" + (tab === "soft" ? " active" : "")}
        onClick=${() => goTab("soft")}>Soft computing</button>
    </div>

    ${tab === "soft"
      ? html`<${SoftComputingTab} soft=${data.softcomputing} />`
      : html`<div>
        <div class="card">
          <div class="row">
            <label>Scenario: <select value=${scenario}
              onChange=${(e) => { setScenario(e.target.value); }}>
              ${data.scenarios.map((s) => html`<option key=${s}>${s}</option>`)}
            </select></label>
            <label>Policy: <${PolicySelect} value=${policy}
              onChange=${setPolicy} policies=${policies}
              families=${data.families || {}} /></label>
            <span class="muted">Deterministic policies have one run; RL
              traces use each algorithm's seed-0 run (see
              results/analysis.md, ppo_seed2 excluded).</span>
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
      </div>`}
  </div>`;
}

D.createRoot(document.getElementById("root")).render(html`<${App} />`);
