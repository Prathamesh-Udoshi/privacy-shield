# Privacy Shield 🔒

**Differential Privacy Data Anonymization Tool**

Privacy Shield is a lightweight Python tool that adds mathematically rigorous noise to CSV files using differential privacy (DP) mechanisms. It's designed for software testing pipelines where you need to work with realistic user data without exposing real customer information.

## 🎯 What is Differential Privacy?

Differential privacy is a mathematical framework for protecting individual privacy in datasets. It ensures that the presence or absence of any single individual's data doesn't significantly affect the results of queries on the dataset.

**Key concept**: Adding carefully calibrated random noise makes it statistically impossible to determine whether a particular individual's data was included in the analysis.

## 🏗️ Why Privacy Shield Exists

In 2026, privacy regulations are strict - you cannot use real customer data for testing, development, or analytics. Privacy Shield helps by:

- **Anonymizing data**: Making individual records unidentifiable
- **Preserving statistics**: Keeping aggregate insights (means, distributions) approximately intact
- **Being auditable**: Every privacy decision is transparent and configurable
- **Being lightweight**: No heavy ML libraries or cloud dependencies

## 📊 How It Works

### Laplace Mechanism
Privacy Shield implements the **Laplace mechanism** manually (no external DP libraries):

```
noise = (1/ε) × sign(u) × ln(1 - 2|u|)
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

| Column Type | Strategy | Sensitivity |
|-------------|----------|-------------|
| Age | Bounded Laplace | 1 |
| Monetary | Scaled Laplace | max - min |
| Counts | Discrete Laplace | 1 |
| Boolean | Randomized Response | - |
| Strings | Mask/Hash | - |

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
  ...

Privacy Budget Report
==================================================
Total Budget: ε = 1.000
Used Budget:  ε = 0.700
Remaining:    ε = 0.300
Utilization:  70.0%

Consumption Details:
  1. bounded_laplace on 'age': ε = 0.200
  2. laplace on 'purchase_amount': ε = 0.400
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
Heuristic estimate of re-identification risk:
- **Uniqueness Reduction**: How much the data became less unique
- **Risk Level**: LOW/MEDIUM/HIGH based on multiple factors
- **K-Anonymity**: Estimated anonymity level

## 🛠️ Installation & Requirements

### Requirements
- Python 3.8+
- Standard library only (csv, math, random, argparse, yaml)

### Installation
```bash
# Clone or download the privacy_shield directory
cd privacy_shield

# Make executable (optional)
chmod +x privacyshield.py
```

### Dependencies
```bash
pip install pyyaml  # For YAML configuration support
```

## 📁 Project Structure

```
privacy_shield/
├── privacyshield.py     # Main CLI tool
├── dp/
│   ├── laplace.py       # Laplace noise implementation
│   ├── budget.py        # Privacy budget tracking
│   └── mechanisms.py    # Column-aware DP strategies
├── metrics/
│   ├── utility.py       # Statistical utility metrics
│   └── risk.py          # Risk estimation heuristics
├── config/
│   └── loader.py        # YAML configuration handling
├── examples/
│   ├── users.csv        # Sample dataset
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

### Running Tests
```bash
# Test with example data
python privacyshield.py --input examples/users.csv --output test_output.csv

# Test with configuration
python privacyshield.py --input examples/users.csv --output test_output.csv --config examples/policy.yaml

# Test with custom epsilon
python privacyshield.py --input examples/users.csv --output test_output.csv --epsilon 0.1
```

### Understanding Test Results
- **Utility Score > 80**: Good statistical preservation
- **Risk Level LOW**: Acceptably private
- **Budget utilization < 90%**: Room for additional operations

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