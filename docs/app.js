// Same maths as the sklearn pipeline:
//   numeric:  (x - mean) / scale * coef
//   category: coef of the selected value (one-hot)
//   p = 1 / (1 + e^-z)
// The numbers come from model.json, which src/train.py writes.

let MODEL = null;

const ADDONS = ["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"];

// readable names for the reasons list
const NICE = {
  tenure: "Short time with the company",
  MonthlyCharges: "Monthly charges",
  TotalCharges: "Total charges",
  "Contract=Month-to-month": "Month-to-month contract",
  "InternetService=Fiber optic": "Fiber optic internet",
  "PaymentMethod=Electronic check": "Pays by electronic check",
  "OnlineSecurity=No": "No online security",
  "TechSupport=No": "No tech support",
  "OnlineBackup=No": "No online backup",
  "DeviceProtection=No": "No device protection",
  "PaperlessBilling=Yes": "Paperless billing",
  "SeniorCitizen=1": "Senior citizen",
  "StreamingTV=Yes": "Streaming TV",
  "StreamingMovies=Yes": "Streaming movies",
  "MultipleLines=Yes": "Multiple lines",
  "Partner=No": "No partner",
  "Dependents=No": "No dependents",
};

function readForm() {
  const v = id => document.getElementById(id).value;
  const internet = v("InternetService");
  const c = {
    tenure: Number(v("tenure")),
    MonthlyCharges: Number(v("MonthlyCharges")),
    gender: v("gender"),
    SeniorCitizen: v("SeniorCitizen"),
    Partner: v("Partner"),
    Dependents: v("Dependents"),
    PhoneService: v("PhoneService"),
    MultipleLines: v("PhoneService") === "No" ? "No phone service" : v("MultipleLines"),
    InternetService: internet,
    Contract: v("Contract"),
    PaperlessBilling: v("PaperlessBilling"),
    PaymentMethod: v("PaymentMethod"),
  };
  for (const a of ADDONS) {
    c[a] = internet === "No" ? "No internet service" : (document.getElementById(a).checked ? "Yes" : "No");
  }
  // no total charges field in the form, estimate it from tenure * monthly
  c.TotalCharges = c.tenure * c.MonthlyCharges;
  return c;
}

function predict(c) {
  let z = MODEL.intercept;
  const parts = [];

  for (const [col, p] of Object.entries(MODEL.numeric)) {
    const contrib = ((c[col] - p.mean) / p.scale) * p.coef;
    z += contrib;
    parts.push({ name: col, value: contrib });
  }
  for (const [col, cats] of Object.entries(MODEL.categorical)) {
    const key = String(c[col]);
    if (key in cats) {
      z += cats[key];
      parts.push({ name: col + "=" + key, value: cats[key] });
    }
  }
  const prob = 1 / (1 + Math.exp(-z));
  return { prob, parts };
}

function riskLevel(p) {
  if (p >= 0.7) return "High";
  if (p >= 0.4) return "Medium";
  return "Low";
}

function update() {
  if (!MODEL) return;
  document.getElementById("tenure-val").textContent = document.getElementById("tenure").value;
  document.getElementById("MonthlyCharges-val").textContent = document.getElementById("MonthlyCharges").value;
  document.getElementById("addons").classList.toggle("disabled", document.getElementById("InternetService").value === "No");

  const c = readForm();
  const { prob, parts } = predict(c);
  const pct = Math.round(prob * 100);

  document.getElementById("prob").textContent = pct + "%";
  document.querySelector(".gauge").style.setProperty("--p", pct + "%");
  const r = riskLevel(prob);
  const riskEl = document.getElementById("risk");
  riskEl.textContent = r + " risk";
  riskEl.className = "risk " + r;

  // top 3 things pushing the score up
  const reasons = parts
    // charges are left out of the reasons: they overlap with tenure and the services,
    // so on their own they give confusing explanations
    .filter(p => p.value > 0.05 && p.name !== "MonthlyCharges" && p.name !== "TotalCharges")
    .sort((a, b) => b.value - a.value)
    .slice(0, 3)
    .map(p => NICE[p.name] || p.name.replace("=", ": "));
  document.getElementById("reasons").innerHTML = reasons.length
    ? reasons.map(x => "<li>" + x + "</li>").join("")
    : "<li>Nothing major, this customer looks safe</li>";
}

fetch("model.json")
  .then(r => r.json())
  .then(m => {
    MODEL = m;
    document.getElementById("form").addEventListener("input", update);
    document.getElementById("form").addEventListener("change", update);
    update();
  });
