# Privacy Shield 🔒

**Industrial-Grade Differential Privacy Data Anonymization Platform**

Privacy Shield is a full-stack data anonymization platform that protects sensitive CSV data using **differential privacy** — the same mathematically rigorous framework used by Apple, Google, and the US Census Bureau. It features a modern **Next.js web frontend**, a **FastAPI backend**, and a comprehensive analysis engine with utility scoring and re-identification risk assessment.

> Upload a CSV → Configure your privacy budget → Download a provably anonymous dataset with full reports — in seconds.

---

## ✨ Features

- **🔐 Mathematically Proven Privacy** — Built on differential privacy with Laplace, Gaussian, and randomized response mechanisms
- **📊 Preserve Data Utility** — Smart noise calibration per column type ensures aggregate statistics survive anonymization
- **🔍 Re-identification Risk Analysis** — Automated membership inference simulations and k-anonymity analysis
- **🗂️ Automatic Column Detection** — Auto-detects ages, monetary values, IDs, booleans, years, counts, and more
- **📋 Comprehensive Reports** — Column-by-column utility scores, mean preservation, std deviation analysis, and MAE
- **⚙️ Configurable Privacy Budget** — Fine-tune ε with purpose presets (General, ML Training, Analytics, QA Testing, Data Sharing)
- **🤖 AI-Enhanced Analysis** — Optional Gemini integration for semantic column classification
- **🌐 Modern Web Interface** — Next.js 16 + React 19 frontend with interactive charts and glassmorphism design
- **⚡ FastAPI Backend** — Async job processing with real-time progress tracking

---

## 🎯 What is Differential Privacy?

Differential privacy is a mathematical framework that ensures the presence or absence of any single individual's data doesn't significantly affect the output. Privacy Shield adds carefully calibrated random noise to make individual records provably unidentifiable while keeping aggregate statistics approximately intact.

**Key concept**: `ε` (epsilon) controls the privacy-utility tradeoff:

| Epsilon Range | Privacy Level | Best For |
|---------------|---------------|----------|
| ε ≤ 0.5 | Maximum Privacy | Public data sharing |
| 0.5 < ε ≤ 1.5 | Balanced | Analytics & ML training |
| ε > 1.5 | Maximum Utility | Internal QA testing |

---

## 🏗️ Architecture

Privacy Shield is a full-stack application with three layers:

```
┌─────────────────────────────────────────────┐
│  Frontend (Next.js 16 / React 19)           │
│  • Landing page with feature showcase       │
│  • CSV upload with drag-and-drop            │
│  • Interactive Recharts visualizations       │
│  • Structured report cards                  │
├─────────────────────────────────────────────┤
│  Backend (FastAPI + Uvicorn)                │
│  • Async job queue with progress tracking   │
│  • RESTful API (upload, status, results)    │
│  • CSV download endpoint                    │
├─────────────────────────────────────────────┤
│  Core Engine (Python)                       │
│  • Column type inference pipeline           │
│  • DP mechanisms (Laplace, Gaussian, RR)    │
│  • Privacy budget accounting                │
│  • Utility & risk metric computation        │
│  • AI semantic analysis (optional)          │
└─────────────────────────────────────────────┘
```

---

## 🚀 Quick Start

### Prerequisites

- **Python 3.10+**
- **Node.js 18+** and **npm**
- (Optional) Gemini API key for AI-enhanced column detection

### 1. Clone & Install Backend

```bash
git clone https://github.com/Prathamesh-Udoshi/privacy-shield.git
cd privacy_shield

# Create a virtual environment
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS/Linux

# Install Python dependencies
pip install -r requirements.txt
```

### 2. Install Frontend

```bash
cd frontend
npm install
cd ..
```

### 3. Configure Environment

```bash
# Copy the example env file
cp .env.example .env

# (Optional) Add your Gemini API key for AI-powered column detection
# GEMINI_API_KEY=your-key-here
```

### 4. Run the Application

**Terminal 1 — Backend:**
```bash
uvicorn backend.main:app --reload --port 8000
```

**Terminal 2 — Frontend:**
```bash
cd frontend
npm run dev
```

Then open **http://localhost:3000** in your browser.

---

## 🖥️ Web Interface

### Landing Page
The landing page explains what Privacy Shield does, how differential privacy works, and walks users through a 4-step guide:

1. **Upload Your CSV** — Drag and drop any CSV with sensitive data
2. **Configure Privacy Settings** — Choose a purpose preset or set ε manually
3. **Anonymize with One Click** — Differential privacy noise applied column-by-column
4. **Analyze & Download** — Review charts, reports, and download your safe CSV

### Results Dashboard

After anonymization, the results page provides:

- **Metric Cards** — Linkage risk level, average utility score, privacy budget usage, records processed
- **Column Type Detection** — Visual badges showing detected types (Age, Monetary, Boolean, ID, etc.)
- **📊 Analysis Tab** — Interactive bar charts for utility scores and relative error per column, plus a detailed statistics table
- **🔍 Data Preview Tab** — Side-by-side table comparing original vs. anonymized values with highlighted changes
- **📋 Reports Tab** — Structured utility preservation and risk assessment reports with color-coded metrics

---

## 📊 How It Works

### Column-Aware Noise

| Column Type | DP Mechanism | Sensitivity | Example |
|-------------|--------------|-------------|---------|
| Age | Bounded Laplace | Range-capped | Personal ages (0-120) |
| Year | Bounded Laplace | Range-capped | Birth year, model year |
| Numeric | Laplace / Gaussian | Range-capped | Continuous measurements |
| Monetary | Scaled Laplace | Range-capped | Currency amounts (auto-scaled) |
| Count | Discrete Laplace | 1 | Integer counts |
| Boolean | Randomized Response | — | True/false flags |
| ID / PK | MD5/SHA Hashing | — | Persistent identifiers |
| String | Masking / Hashing | — | Categorical data & PII |

### Laplace Mechanism

```
noise = -scale × sign(u) × ln(1 - 2|u|)
```

Where: `ε` = privacy parameter, `u` = uniform random (-0.5, 0.5), Scale = sensitivity / ε

### Privacy Budget Accounting

- Total budget (ε_total) is set by the user
- Each column consumes part of this budget
- If budget is exceeded → warning + operation skipped
- Final report shows budget utilization

### Smart Sensitivity & Small Data Safeguards

1. **Range-Adaptive Noise** — Pre-scans data ranges to scale noise appropriately
2. **Auto-Epsilon Tuning** — For datasets with <500 rows, automatically increases ε to prevent data destruction
3. **Non-Negative Constraints** — Enforces boundaries for values that can never be negative (Age, Price, Count)

---

## 🤖 AI Semantic Analysis (Optional)

Privacy Shield integrates with **Google Gemini models** for high-fidelity column classification:

- Add `GEMINI_API_KEY` to your `.env` file
- AI identifies columns like `Val_A` as internal IDs or `Amount_3` as currency
- Acts as a high-priority override for the heuristic inference engine

---

## 📈 Understanding the Reports

### Utility Preservation Report
Column-by-column breakdown with:
- **Mean Preservation** — Original vs. noisy mean with absolute/relative error
- **Std Deviation Change** — How much variance shifted
- **Mean Absolute Error (MAE)** — Average per-value distortion
- **Utility Score (0-100)** — Weighted composite (50% mean, 30% std, 20% MAE)
- **Overall Interpretation** — EXCELLENT (≥80), GOOD (≥60), FAIR (≥40), POOR (<40)

### Re-identification Risk Assessment
Multi-layered evaluation:
- **Membership Inference Simulation** — Distance-based linking attack measuring how many records can be re-identified
- **Uniqueness Reduction** — How much data entropy changed after anonymization
- **K-Anonymity Analysis** — Estimated k-value based on quasi-identifiers
- **Overall Risk Category** — LOW, MODERATE, or CRITICAL based on combined scoring

---

## 🔌 API Reference

The FastAPI backend exposes the following endpoints:

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/v1/upload` | Upload CSV + privacy params, returns `job_id` |
| `GET` | `/api/v1/jobs/{job_id}/status` | Poll job progress (status, progress %, message) |
| `GET` | `/api/v1/jobs/{job_id}/result` | Get full results (metrics, reports, preview data) |
| `GET` | `/api/v1/jobs/{job_id}/download` | Download the anonymized CSV file |

API documentation is available at **http://localhost:8000/docs** when the backend is running.

---

## 📁 Project Structure

```
privacy_shield/
├── frontend/                   # Next.js 16 web application
│   ├── app/
│   │   ├── page.tsx            # Landing page + upload tool
│   │   ├── results/[jobId]/
│   │   │   └── page.tsx        # Results dashboard
│   │   ├── globals.css         # Design system & styles
│   │   └── layout.tsx          # Root layout with fonts
│   ├── lib/
│   │   └── api.ts              # API client & types
│   └── package.json
├── backend/                    # FastAPI application
│   ├── main.py                 # App entry point + CORS
│   ├── routers/
│   │   └── anonymize.py        # Upload, status, result, download endpoints
│   ├── job_store.py            # In-memory job queue
│   └── schemas.py              # Pydantic response models
├── core/
│   └── anonymizer.py           # Main anonymization orchestrator
├── dp/                         # Differential privacy mechanisms
│   ├── laplace.py              # Vectorized Laplace mechanism
│   ├── gaussian.py             # Gaussian mechanism for (ε, δ)-DP
│   ├── budget.py               # Privacy budget tracking
│   └── mechanisms.py           # Range-adaptive DP strategies
├── metrics/                    # Analysis & reporting
│   ├── utility.py              # Statistical utility metrics
│   └── risk.py                 # Membership inference simulator
├── preprocessing/
│   └── pipeline.py             # Column type inference & preprocessing
├── ai/
│   └── semantic_analyzer.py    # Gemini-powered column classification
├── config/
│   └── loader.py               # YAML configuration handling
├── examples/
│   ├── users.csv               # Sample dataset
│   ├── housing.csv             # ML-scale dataset
│   └── policy.yaml             # Sample configuration
├── requirements.txt            # Python dependencies
├── .env                        # Environment variables
└── README.md
```

---

## ⚙️ Configuration

### YAML Policy File

Create a `policy.yaml` for fine-grained control:

```yaml
global_epsilon: 1.0

columns:
  age:
    method: bounded_laplace
    epsilon: 0.2
    min: 18
    max: 90

  purchase_amount:
    method: laplace
    epsilon: 0.4
    sensitivity: 1000.0

  login_count:
    method: discrete_laplace
    epsilon: 0.2

  is_active:
    method: randomized_response
    epsilon: 0.5
```

### Configuration Options

| Parameter | Description | Default |
|-----------|-------------|---------|
| `global_epsilon` | Total privacy budget | 1.0 |
| `epsilon` | Per-column privacy parameter | Auto-assigned |
| `method` | DP mechanism to use | Based on column type |
| `sensitivity` | Query sensitivity | Based on column type |
| `min`/`max` | Bounds for bounded mechanisms | — |
| `mask_type` | String masking method (`partial`/`hash`) | `partial` |

---

## ⚠️ Limitations & Notes

### Technical Limitations
- **CSV-only** — Designed for tabular data
- **Memory-bound** — Loads the entire dataset into memory
- **In-memory job store** — Jobs are lost on backend restart

### Privacy Limitations
- **No composition theorems** — Simple budget tracking, not advanced composition
- **Assumed sensitivities** — Uses rule-of-thumb sensitivities, not query-specific
- **No correlated columns** — Treats columns independently
- **Simulated risk** — Membership inference is a simulation, not a formal mathematical bound

### Best Practices
1. Start with a small ε (0.1–0.5) for strong privacy
2. Check utility scores before using anonymized data in pipelines
3. Monitor risk reports for unexpected patterns
4. Test on small datasets first, then scale up
5. Document your privacy parameters

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| Frontend | Next.js 16, React 19, TypeScript, Recharts, Lucide Icons |
| Backend | FastAPI, Uvicorn, Python 3.10+ |
| DP Engine | NumPy, Pandas, custom Laplace/Gaussian implementations |
| AI (optional) | Google Gemini API |
| Styling | Vanilla CSS with glassmorphism design system |

---

## 📄 License

This is a reference implementation for educational and development purposes. Adapt and modify as needed for your specific use case.

---

## 🙋 Support

For questions about differential privacy concepts or tool usage:
1. Check the example configurations in `examples/`
2. Review the utility and risk reports after anonymization
3. Start with small epsilon values and increase gradually
4. Consult the API docs at `/docs` when the backend is running

> **Remember**: Privacy is hard. When in doubt, consult privacy experts or use established DP libraries for production systems.