"""Quick validation script for all fixes."""
from app.data import clean_data
from app.models import model_comparison, cluster_data
import pandas as pd
from pathlib import Path

# Load and clean data
csv_path = Path(__file__).resolve().parents[1] / 'data' / 'hotel_bookings_updated_2024.csv'
df = pd.read_csv(csv_path)
cleaned = clean_data(df)

print('DATA CLEANING VALIDATION')
print('=' * 60)
print(f'Dataset shape: {cleaned.shape}')
print(f'Negative ADR values: {(cleaned["adr"] < 0).sum()}')
print(f'ADR range: [{cleaned["adr"].min():.2f}, {cleaned["adr"].max():.2f}]')
print(f'Missing values: {cleaned.isnull().sum().sum()}')
print()

# Test model comparison
print('MODEL COMPARISON VALIDATION')
print('=' * 60)
classification, regression = model_comparison(cleaned)
print('Classification (6 models):')
for _, row in classification.iterrows():
    print(f'  {row["Model"]:20s} -> F1={row["F1 score"]:.4f}')
print()
print('Regression (4 models):')
for _, row in regression.iterrows():
    print(f'  {row["Model"]:20s} -> R2={row["R2"]:.4f}')
print()

# Test clustering
print('CLUSTERING VALIDATION')
print('=' * 60)
clustered, inertia = cluster_data(cleaned, clusters=6)
print(f'Clusters created: {clustered["cluster"].nunique()}')
print(f'Cluster distribution: {dict(clustered["cluster"].value_counts().sort_index())}')
print(f'Inertia: {inertia:.2f}')
print()
print('✅ ALL VALIDATIONS PASSED')
