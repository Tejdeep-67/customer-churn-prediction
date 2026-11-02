// Churn what-if simulator.
// Same maths as the sklearn pipeline in src/train.py:
//   numeric:  (x - mean) / scale * coef
//   category: coef of the selected value (one-hot)
//   p = 1 / (1 + e^-z)
// All weights come from model.json, which src/train.py writes.

let MODEL = null;
const $ = (id) => document.getElementById(id);
const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
const ADDONS = ["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"];
const ADDON_NAMES = {
  OnlineSecurity: "online security", OnlineBackup: "online backup", DeviceProtection: "device protection",
  TechSupport: "tech support", StreamingTV: "streaming TV", StreamingMovies: "streaming movies",
};
const sigmoid = (z) => 1 / (1 + Math.exp(-z));

const PRESETS = {
  new: { tenure: 2, MonthlyCharges: 95, Contract: "Month-to-month", PaymentMethod: "Electronic check", PaperlessBilling: true,
    InternetService: "Fiber optic", addons: ["StreamingTV"], PhoneService: true, MultipleLines: false,
    gender: "Female", SeniorCitizen: false, Partner: false, Dependents: false },
  loyal: { tenure: 60, MonthlyCharges: 65, Contract: "Two year", PaymentMethod: "Credit card (automatic)", PaperlessBilling: false,
    InternetService: "DSL", addons: ["OnlineSecurity", "TechSupport", "OnlineBackup"], PhoneService: true, MultipleLines: true,
    gender: "Male", SeniorCitizen: false, Partner: true, Dependents: true },
  mid: { tenure: 29, MonthlyCharges: 70, Contract: "One year", PaymentMethod: "Bank transfer (automatic)", PaperlessBilling: true,
    InternetService: "Fiber optic", addons: ["OnlineBackup"], PhoneService: true, MultipleLines: false,
    gender: "Male", SeniorCitizen: false, Partner: true, Dependents: false },
};

// ---------- reading the form ----------
function segValue(name) {
  return document.querySelector(`.seg[data-name="${name}"] [aria-pressed="true"]`).dataset.v;
}

function readForm() {
  const on = (id) => $(id).checked;
  const internet = segValue("InternetService");
  const c = {
    tenure: Number($("tenure").value),
    MonthlyCharges: Number($("MonthlyCharges").value),
    gender: segValue("gender"),
    SeniorCitizen: on("SeniorCitizen") ? "1" : "0",
    Partner: on("Partner") ? "Yes" : "No",
    Dependents: on("Dependents") ? "Yes" : "No",
    PhoneService: on("PhoneService") ? "Yes" : "No",
    MultipleLines: !on("PhoneService") ? "No phone service" : on("MultipleLines") ? "Yes" : "No",
    InternetService: internet,
    Contract: segValue("Contract"),
    PaperlessBilling: on("PaperlessBilling") ? "Yes" : "No",
    PaymentMethod: $("PaymentMethod").value,
  };
  for (const a of ADDONS) c[a] = internet === "No" ? "No internet service" : on(a) ? "Yes" : "No";
  // the form has no total charges field, so estimate it as tenure x monthly bill
  c.TotalCharges = c.tenure * c.MonthlyCharges;
  return c;
}

// ---------- the model ----------
function label(col, val) {
  switch (col) {
    case "tenure": return "Tenure";
    case "charges": return "Monthly + total charges";
    case "Contract": return val === "Month-to-month" ? "Month-to-month contract" : val + " contract";
    case "InternetService": return val === "No" ? "No internet" : val === "DSL" ? "DSL internet" : "Fiber optic internet";
    case "PaymentMethod": return "Pays by " + val.replace(" (automatic)", " (auto)").toLowerCase();
    case "PaperlessBilling": return val === "Yes" ? "Paperless billing" : "Paper bills";
    case "SeniorCitizen": return val === "1" ? "Senior citizen" : "Not a senior citizen";
    case "Partner": return val === "Yes" ? "Has a partner" : "No partner";
    case "Dependents": return val === "Yes" ? "Has dependents" : "No dependents";
    case "PhoneService": return val === "Yes" ? "Has a phone line" : "No phone line";
    case "MultipleLines": return val === "Yes" ? "Multiple lines" : val === "No" ? "Single line" : "No phone line";
    case "gender": return val;
    default:
      if (ADDONS.includes(col)) return val === "Yes" ? "Has " + ADDON_NAMES[col] : val === "No" ? "No " + ADDON_NAMES[col] : "No internet add-ons";
      return col + ": " + val;
  }
}

function predict(c) {
  const M = MODEL;
  let z = M.intercept;
  let base = M.intercept;
  const parts = [];
  // tenure and charges: standardised, so an average customer adds ~0
  const num = (col) => ((c[col] - M.numeric[col].mean) / M.numeric[col].scale) * M.numeric[col].coef;
  const t = num("tenure");
  const ch = num("MonthlyCharges") + num("TotalCharges"); // these two overlap, so they are shown as one bar
  z += t + ch;
  parts.push({ col: "tenure", label: `Tenure (${c.tenure} mo)`, value: t });
  parts.push({ col: "charges", label: label("charges"), value: ch });
  for (const [col, cats] of Object.entries(M.categorical)) {
    const w = cats[String(c[col])] ?? 0;
    const avg = M.average_contribution[col];
    z += w;
    base += avg;
    parts.push({ col, label: label(col, String(c[col])), value: w - avg });
  }
  return { z, prob: sigmoid(z), base: sigmoid(base), parts };
}

function riskLevel(p) {
  if (p >= 0.7) return "High";
  if (p >= 0.4) return "Medium";
  return "Low";
}

function percentile(p) {
  // share of the test customers with a lower score
  const q = MODEL.test_percentiles;
  let i = 0;
  while (i < q.length && q[i] < p) i++;
  return Math.max(0, Math.min(100, i - 1));
}

// ---------- drawing ----------
function countTo(el, to, fmt) {
  const from = el._v ?? 0;
  el._v = to;
  if (reduceMotion) { el.textContent = fmt(to); return; }
  cancelAnimationFrame(el._raf);
  const t0 = performance.now();
  const step = (t) => {
    const k = Math.min(1, (t - t0) / 650);
    el.textContent = fmt(from + (to - from) * (1 - Math.pow(1 - k, 3)));
    if (k < 1) el._raf = requestAnimationFrame(step);
  };
  el._raf = requestAnimationFrame(step);
}

const GLOW = { Low: "rgba(198,244,50,.32)", Medium: "rgba(255,181,71,.34)", High: "rgba(255,77,141,.38)" };
let lastRisk = null;

function render() {
  if (!MODEL) return;
  syncUi();
  const c = readForm();
  const r = predict(c);
  const pct = r.prob * 100;
  const risk = riskLevel(r.prob);

  countTo($("prob"), pct, (x) => Math.round(x) + "%");
  $("g-fill").style.strokeDashoffset = (282.7 * (1 - r.prob)).toFixed(1);
  $("needle").style.transform = `rotate(${-90 + 180 * r.prob}deg)`;
  const badge = $("risk");
  badge.textContent = risk + " risk";
  badge.className = "badge " + risk;
  if (lastRisk && lastRisk !== risk && !reduceMotion) { void badge.offsetWidth; badge.classList.add("pop"); }
  lastRisk = risk;
  $("result").style.setProperty("--glow", GLOW[risk]);
  const pc = percentile(r.prob);
  $("pct").textContent = `riskier than ${pc}% of customers`;
  $("pbar-dot").style.left = pc + "%";
  $("mini-prob").textContent = Math.round(pct) + "%";
  $("mini-risk").textContent = risk + " risk";
  $("mini-risk").style.color = risk === "High" ? "var(--rose)" : risk === "Medium" ? "var(--amber)" : "var(--lime)";

  drawDrivers(r);
  drawWhatIf(c, r);
}

function drawDrivers(r) {
  $("baseline").textContent = `vs a typical customer (${Math.round(r.base * 100)}%)`;
  const top = r.parts.filter((p) => Math.abs(p.value) > 0.03).sort((a, b) => Math.abs(b.value) - Math.abs(a.value)).slice(0, 7);
  const max = Math.max(1.2, ...top.map((p) => Math.abs(p.value)));
  const ul = $("drivers");
  // reuse rows by column so the bars animate between states instead of re-appearing
  const existing = new Map([...ul.children].map((li) => [li.dataset.col, li]));
  const frag = [];
  top.forEach((p, i) => {
    let li = existing.get(p.col);
    if (!li) {
      li = document.createElement("li");
      li.dataset.col = p.col;
      li.innerHTML = '<span class="name"></span><span class="track"><i class="bar"></i></span><span class="val"></span>';
      li.style.animationDelay = i * 35 + "ms";
    }
    existing.delete(p.col);
    li.querySelector(".name").textContent = p.label;
    li.querySelector(".name").title = p.label;
    const bar = li.querySelector(".bar");
    bar.className = "bar " + (p.value > 0 ? "up" : "down");
    bar.style.width = ((Math.abs(p.value) / max) * 50).toFixed(1) + "%";
    li.querySelector(".val").textContent = (p.value > 0 ? "+" : "\u2212") + Math.abs(p.value).toFixed(2);
    frag.push(li);
  });
  existing.forEach((li) => li.remove());
  frag.forEach((li) => ul.appendChild(li));
  if (!top.length) ul.innerHTML = '<li><span class="name">Looks like a typical customer</span></li>';
}

// try a few changes a retention team could actually offer and keep the ones that help
function drawWhatIf(c, r) {
  const ideas = [];
  const tryIt = (text, change, apply) => {
    const p = predict({ ...c, ...change }).prob;
    const d = (p - r.prob) * 100;
    if (d <= -1) ideas.push({ text, d, apply });
  };
  if (c.Contract === "Month-to-month") {
    tryIt("Move to a 1-year contract", { Contract: "One year" }, () => setSeg("Contract", "One year"));
    tryIt("Move to a 2-year contract", { Contract: "Two year" }, () => setSeg("Contract", "Two year"));
  } else if (c.Contract === "One year") {
    tryIt("Move to a 2-year contract", { Contract: "Two year" }, () => setSeg("Contract", "Two year"));
  }
  if (c.InternetService !== "No") {
    for (const a of ["OnlineSecurity", "TechSupport"]) {
      if (c[a] === "No") tryIt(`Add free ${ADDON_NAMES[a]}`, { [a]: "Yes" }, () => { $(a).checked = true; });
    }
  }
  if (c.PaymentMethod === "Electronic check" || c.PaymentMethod === "Mailed check") {
    tryIt("Switch to automatic card payment", { PaymentMethod: "Credit card (automatic)" }, () => { $("PaymentMethod").value = "Credit card (automatic)"; });
  }
  ideas.sort((a, b) => a.d - b.d);
  const box = $("whatif");
  box.innerHTML = "";
  if (!ideas.length) {
    box.innerHTML = '<p class="none">No easy wins left: this customer already has the low-risk options. Bill kept the same in every suggestion.</p>';
    return;
  }
  ideas.slice(0, 4).forEach((it, i) => {
    const b = document.createElement("button");
    b.type = "button";
    b.style.animationDelay = i * 50 + "ms";
    b.innerHTML = `<span>${it.text}</span><span class="delta">${it.d.toFixed(0)} pts</span>`;
    b.addEventListener("click", () => { it.apply(); render(); });
    box.appendChild(b);
  });
}

// ---------- controls ----------
function setSeg(name, v) {
  document.querySelectorAll(`.seg[data-name="${name}"] button`).forEach((b) => b.setAttribute("aria-pressed", String(b.dataset.v === v)));
}

function moveThumb(seg) {
  const b = seg.querySelector('[aria-pressed="true"]');
  const th = seg.querySelector(".thumb");
  th.style.width = b.offsetWidth + "px";
  th.style.transform = `translateX(${b.offsetLeft - 4}px)`;
  th.style.left = "4px";
}

function syncUi() {
  const tenure = +$("tenure").value, bill = +$("MonthlyCharges").value;
  $("tenure-val").textContent = tenure === 1 ? "1 month" : tenure + " months";
  $("MonthlyCharges-val").textContent = "$" + bill;
  for (const id of ["tenure", "MonthlyCharges"]) {
    const el = $(id);
    el.style.setProperty("--fill", ((el.value - el.min) / (el.max - el.min)) * 100 + "%");
  }
  $("addons").classList.toggle("disabled", segValue("InternetService") === "No");
  $("MultipleLines").disabled = !$("PhoneService").checked;
  document.querySelectorAll(".seg").forEach(moveThumb);
}

function applyPreset(name) {
  const p = PRESETS[name];
  $("tenure").value = p.tenure;
  $("MonthlyCharges").value = p.MonthlyCharges;
  $("PaymentMethod").value = p.PaymentMethod;
  setSeg("Contract", p.Contract);
  setSeg("InternetService", p.InternetService);
  setSeg("gender", p.gender);
  for (const a of ADDONS) $(a).checked = p.addons.includes(a);
  for (const k of ["PaperlessBilling", "PhoneService", "MultipleLines", "SeniorCitizen", "Partner", "Dependents"]) $(k).checked = p[k];
  render();
}

function bind() {
  document.querySelectorAll(".seg").forEach((seg) => {
    seg.insertAdjacentHTML("afterbegin", '<span class="thumb" aria-hidden="true"></span>');
    seg.addEventListener("click", (e) => {
      const b = e.target.closest("button");
      if (!b) return;
      setSeg(seg.dataset.name, b.dataset.v);
      render();
    });
  });
  $("form").addEventListener("input", render);
  $("form").addEventListener("change", render);
  document.querySelectorAll("[data-preset]").forEach((b) => b.addEventListener("click", () => applyPreset(b.dataset.preset)));
  window.addEventListener("resize", () => document.querySelectorAll(".seg").forEach(moveThumb));

  // little floating result bar on phones when the gauge is scrolled away
  new IntersectionObserver(([en]) => $("mini").classList.toggle("show", !en.isIntersecting && en.boundingClientRect.top < 0))
    .observe($("result"));

  // nav highlight
  const links = document.querySelectorAll(".nav-links a[href^='#']");
  new IntersectionObserver((entries) => entries.forEach((en) => {
    if (en.isIntersecting) links.forEach((a) => a.classList.toggle("is-active", a.getAttribute("href") === "#" + en.target.id));
  }), { rootMargin: "-40% 0px -55% 0px" }).observe($("model"));
  new IntersectionObserver((entries) => entries.forEach((en) => {
    if (en.isIntersecting) links.forEach((a) => a.classList.toggle("is-active", a.getAttribute("href") === "#sim"));
  }), { rootMargin: "-40% 0px -55% 0px" }).observe($("sim"));

  // model section reveal + counters
  const io = new IntersectionObserver((entries) => entries.forEach((en) => {
    if (!en.isIntersecting) return;
    en.target.classList.add("in");
    en.target.querySelectorAll("[data-to]").forEach((el) => {
      const to = +el.dataset.to, dec = +el.dataset.dec, suf = el.dataset.suf || "";
      el._v = 0;
      countTo(el, to, (x) => x.toFixed(dec) + suf);
    });
    io.unobserve(en.target);
  }), { threshold: 0.25 });
  document.querySelectorAll(".metrics, .how").forEach((el) => { el.classList.add("reveal"); io.observe(el); });
}

fetch("model.json")
  .then((r) => r.json())
  .then((m) => {
    MODEL = m;
    bind();
    render();
  })
  .catch(() => {
    $("prob").textContent = "!";
    $("pct").textContent = "could not load model.json (run a local server in docs/)";
  });
