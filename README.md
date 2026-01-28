# Privacy Shield 🔒

**Complete Differential Privacy Data Protection Toolkit**

Privacy Shield is a comprehensive Python toolkit for protecting sensitive data using differential privacy. It provides both a **command-line interface** for batch processing and a **user-friendly web interface** for interactive use. The tool automatically analyzes any CSV file, intelligently infers data types, applies appropriate privacy mechanisms, and provides detailed reports on privacy guarantees and statistical utility preservation.

**Perfect for**: Testing pipelines, ML training, data sharing, compliance, and privacy-preserving analytics.

## 🎯 What is Differential Privacy?

Differential privacy is a mathematical framework for protecting individual privacy in datasets. It ensures that the presence or absence of any single individual's data doesn't significantly affect the results of queries on the dataset.

**Key concept**: Adding carefully calibrated random noise makes it statistically impossible to determine whether a particular individual's data was included in the analysis.

## 🏗️ Why Privacy Shield Exists

In 2026, privacy regulations are strict - you cannot use real customer data for testing, development, or analytics. Privacy Shield helps by:

- **Anonymizing data**: Making individual records unidentifiable through mathematical guarantees.
- **Preserving statistics**: Keeping aggregate insights (means, distributions) approximately intact through **Smart Sensitivity Scaling**.
- **AI-Enhanced Analysis**: Uses OpenAI GPT models to semantically understand complex datasets and sensitive columns.
- **Small Dataset Protection**: Automatically tunes privacy parameters for small samples (Iris/Titanic) to prevent data destruction.
- **Web interface**: User-friendly application with side-by-side **Privacy Impact Viewers**.
- **ML-ready output**: Consistent type-safe output ensured for seamless model retraining and deployment.

## 📊 How It Works

### Laplace Mechanism
Privacy Shield implements the **Laplace mechanism** manually (no external DP libraries):

```
noise = -scale × sign(u) × ln(1 - 2|u|)
```

Where:
- `ε` (epsilon) = privacy parameter (smaller = more privacy)
- `u` = uniform random variable (-0.5, 0.5)
- Scale = sensitivity / epsilon

### Privacy Budget Accounting
- You set a total privacy budget (ε_total)
- Each column consumes part of this budget
- If budget is exceeded → warning + operation skipped
- Final report shows budget utilization

### Column-Aware Noise

| Column Type | Strategy | Sensitivity | Use Case |
|-------------|----------|-------------|----------|
| Age | Bounded Laplace | range-capped | Personal ages (0-120) |
| Year | Bounded Laplace | range-capped | Years like model year, birth year |
| Numeric | Laplace/Gaussian | range-capped | Continuous measurements |
| Monetary | Scaled Laplace | range-capped | Currency amounts (Auto-scaled) |
| Count | Discrete Laplace | 1 | Integer counts |
| Boolean | Randomized Response | - | True/false flags |
| ID / PK | MD5/SHA Hashing | - | Persistent identifiers (PassengerId, UID) |
| String | Masking/Hashing | - | Categorical data & PII |

### 🤖 AI Semantic Analysis (NEW)

Privacy Shield now integrates with **OpenAI's GPT models** to provide high-fidelity data categorization.
- **How to use**: Add `OPENAI_API_KEY` to your `.env` file.
- **Benefit**: AI can identify columns like `Val_A` as internal IDs or `Amount_3` as currency, even when headers are cryptic. It acts as a high-priority override for our heuristic engine.

### ⚡ Smart Sensitivity & Small Data Safeguards

1. **Range-Adaptive Noise**: The tool pre-scans data ranges to scale noise. This prevents "Small Value Vaporization" (e.g., preventing a $7 fare from becoming $8000).
2. **Auto-Epsilon Tuning**: For datasets with **< 500 rows**, the tool automatically increases the privacy budget (typically to ε=4.0) to ensure the resulting data remains statistically useful.
3. **Non-Negative Constraints**: Automatically detects and enforces boundaries for measurements that can never be negative (Age, Price, Count).

## 🚀 Quick Start

### Basic Usage
```bash
# Anonymize a CSV file with default settings
python privacyshield.py --input users.csv --output safe_users.csv

# Use a configuration file
python privacyshield.py --input users.csv --output safe_users.csv --config policy.yaml

# Set custom global epsilon
python privacyshield.py --input users.csv --output safe_users.csv --epsilon 0.5
```

### Example Output
```
Privacy Shield v1.0
Input: examples/users.csv
Output: safe_users.csv
Global ε: 1.0

Reading input data...
Loaded 50 rows with 9 columns

Inferring column types...
  user_id: count
  name: string
  age: age
  gender: string
  location: string
  purchase_amount: monetary
  login_count: count
  is_active: boolean
  last_login_days: count

Applying differential privacy...
  Processing user_id (count)...
  Processing name (string)...
  Processing age (age)...
  Processing gender (string)...
  Processing location (string)...
  Processing purchase_amount (monetary)...
  Processing login_count (count)...
  Processing is_active (boolean)...
  Processing last_login_days (count)...

Privacy Budget Report
==================================================
Total Budget: ε = 1.000
Used Budget:  ε = 0.700
Remaining:    ε = 0.300
Utilization:  70.0%

Consumption Details:
  1. bounded_laplace on 'age': ε = 0.200
  2. scaled_laplace on 'purchase_amount': ε = 0.400
  3. discrete_laplace on 'login_count': ε = 0.200

Utility Preservation Report
==================================================
Column: age
  Sample Size: 50
  Mean Preservation:
    Original: 35.5
    Noisy:    35.6
    Error:    0.1 (0.3%)
  Utility Score: 98.7/100

Re-identification Risk Assessment
==================================================
Uniqueness Reduction: 45.2%
Overall Risk Assessment:
  Risk Level: LOW
  Interpretation: Low risk of re-identification
```

## 🌐 Web Interface

Privacy Shield includes a **Streamlit-powered web application** for interactive use:

### Starting the Web App
```bash
streamlit run streamlit_app.py
```

### Web Interface Features
- **🆔 ID Handling**: Automatic detection and hashing of primary keys/identifiers.
- **🔍 Privacy Impact Viewer**: Side-by-side row-level comparison of Original vs. Noisy data.
- **⚡ AI Status Dashboard**: Real-time indicator showing if OpenAI-powered detection is active.
- **📊 Robust Analysis**: Empirical risk scores calculated through membership inference simulations.
- **💾 Type-Safe Download**: Guaranteed numeric consistency for immediate ML use.

### Web Interface Workflow
1. **Upload CSV** → Automatic column analysis
2. **Configure Privacy** → Set epsilon values (0.1-5.0)
3. **Review Results** → See mechanisms applied to each column
4. **Download Output** → Get anonymized data with full reports

### Benefits of Web Interface
- **No Command Line Required**: Perfect for non-technical users
- **Visual Feedback**: See exactly what's being protected
- **Educational**: Learn about differential privacy concepts
- **Batch Processing**: Handle multiple files easily
- **Audit Trail**: Complete record of privacy decisions

## 🤖 Machine Learning with Anonymized Data

Privacy Shield includes a demonstration script showing how to use anonymized data for machine learning training:

### Running the ML Demo
```bash
python ml_training_demo.py
```

### ML Demo Features
- **Privacy-Preserving Training**: Train ML models on anonymized data
- **Utility Comparison**: Compare model performance on original vs anonymized data
- **Supported Algorithms**: Random Forest, Logistic Regression
- **Statistical Validation**: Measures accuracy preservation under privacy constraints

### Key Insights
- **ε Selection Guide**: Higher ε (less privacy) = better model accuracy
- **Feature Engineering**: Anonymized data maintains feature relationships
- **Production Ready**: Models trained on anonymized data are privacy-safe for deployment

### Example ML Results
```
Model Performance Comparison
==================================================

Random Forest:
  Original Accuracy: 0.875
  Anonymized Accuracy: 0.850
  Accuracy Change: -2.5%

Logistic Regression:
  Original Accuracy: 0.825
  Anonymized Accuracy: 0.800
  Accuracy Change: -2.5%
```

**Result**: Minimal utility loss while providing formal privacy guarantees!

## ⚙️ Configuration

Create a `policy.yaml` file:

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
| `min`/`max` | Bounds for bounded mechanisms | - |
| `mask_type` | String masking method (`partial`/`hash`) | `partial` |

## 📈 Understanding the Reports

### Privacy Budget Report
Shows how much of your total ε was consumed. Lower utilization = more privacy budget left for future operations.

### Utility Report
Measures how well aggregate statistics are preserved:
- **Utility Score**: 0-100 (higher = better preservation)
- **Mean Error**: How much the average changed
- **MAE**: Mean Absolute Error between original and noisy values

### Risk Assessment
A multi-layered evaluation of re-identification risk:
- **Membership Inference Simulation**: A distance-based "linking attack" that measures how many anonymized records can be correctly matched to original identities.
- **Uniqueness Reduction**: Measures how much the data entropy has increased.
- **K-Anonymity**: Estimated anonymity level based on quasi-identifiers.
- **Risk Level**: Labeled as LOW, MODERATE, or CRITICAL based on link success probability.

## 🛠️ Installation & Requirements

### Requirements
- Python 3.8+
- For CLI usage: Standard library only (csv, math, random, argparse, yaml)
- For Web interface: Streamlit, pandas, numpy

### Installation
```bash
# Clone or download the privacy_shield directory
cd privacy_shield

# Install dependencies
pip install -r requirements.txt

# Make executable (optional)
chmod +x privacyshield.py
```

### Dependencies
```bash
# Core dependencies (CLI)
pip install pyyaml

# Web interface dependencies
pip install streamlit pandas

# ML demonstration
pip install scikit-learn
```

### Requirements File
Create `requirements.txt`:
```
pyyaml>=6.0
streamlit>=1.28.0
pandas>=2.0.0
scikit-learn>=1.3.0
```

## 📁 Project Structure

```
privacy_shield/
├── streamlit_app.py     # Web-based interface
├── privacyshield.py     # Main CLI tool
├── ml_training_demo.py  # ML training demonstration (Titanic/Housing support)
├── requirements.txt     # Project dependencies
├── .env                 # API Key storage (Copy .env.example)
├── ai/
│   └── semantic_analyzer.py # OpenAI-powered semantic engine
├── dp/
│   ├── laplace.py       # Vectorized Laplace mechanism
│   ├── gaussian.py      # Gaussian mechanism for (ε, δ)-DP
│   ├── budget.py        # Privacy budget tracking
│   └── mechanisms.py    # Range-Adaptive DP strategies
├── metrics/
│   ├── utility.py       # Statistical utility metrics
│   └── risk.py          # Membership Inference Simulator
├── config/
│   └── loader.py        # YAML configuration handling
├── examples/
│   ├── users.csv        # Sample dataset
│   ├── housing.csv      # Large-scale ML dataset
│   └── policy.yaml      # Sample configuration
└── README.md
```

## ⚠️ Limitations & Important Notes

### Technical Limitations
1. **Not for production systems** - This is for testing/development only
2. **Heuristic risk assessment** - Not a formal privacy guarantee
3. **CSV-only** - Designed specifically for tabular data
4. **Memory-bound** - Loads entire dataset into memory

### Privacy Limitations
1. **No composition theorems** - Simple budget tracking, not advanced composition
2. **Assumed sensitivities** - Uses rule-of-thumb sensitivities, not query-specific
3. **No correlated columns** - Treats columns independently
4. **String handling** - Basic masking, not formal DP for text
5. **Simulated Risk** - The Membership Inference score is a simulation, not a formal mathematical ceiling of risk.

### When NOT to Use
- Production data processing
- High-stakes privacy decisions
- Large datasets (>100k rows)
- Correlated sensitive attributes
- Real-time systems

### Best Practices
1. **Start with small ε** (0.1-0.5) for high privacy
2. **Test utility metrics** before using in pipelines
3. **Monitor risk reports** for unexpected patterns
4. **Version control configs** like code
5. **Document your privacy parameters**

## 🔧 Development & Testing

### CLI Testing
```bash
# Test with example data
python privacyshield.py --input examples/users.csv --output test_output.csv

# Test with configuration
python privacyshield.py --input examples/users.csv --output test_output.csv --config examples/policy.yaml

# Test with custom epsilon
python privacyshield.py --input examples/users.csv --output test_output.csv --epsilon 0.1
```

### Web Interface Testing
```bash
# Start the web application
streamlit run streamlit_app.py

# Then visit http://localhost:8501 in your browser
```

### ML Training Demo
```bash
# Test anonymized data for machine learning
python ml_training_demo.py
```

### Understanding Test Results
- **Utility Score > 80**: Good statistical preservation
- **Risk Level LOW**: Acceptably private
- **Budget utilization < 90%**: Room for additional operations
- **Type Inference Accuracy**: All columns correctly classified
- **Web Interface**: Proper mechanism explanations and reports

## 🤝 Contributing

This tool follows engineering best practices:
- **Clean, typed Python** with clear docstrings
- **Modular architecture** for easy extension
- **Comprehensive error handling**
- **No over-engineering** - focused on the core use case

## 📄 License

This is a reference implementation for educational and testing purposes. Adapt and modify as needed for your specific use case.

## 🙋 Support

For questions about differential privacy concepts or tool usage:
1. Check the example configurations
2. Review the utility and risk reports
3. Start with small epsilon values and increase gradually
4. Test on small datasets first

Remember: **Privacy is hard**. When in doubt, consult privacy experts or use established DP libraries for production systems.