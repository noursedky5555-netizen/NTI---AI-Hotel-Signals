# Hotel Signals

A modular Streamlit dashboard for exploring hotel booking cancellations, rates, and booking behavior.

## Project Index

```text
hotel/
├── app/          Runtime modules and page renderers
│   └── pages/    One renderer per website page
├── assets/       Canonical logo and browser icons
├── data/         Input datasets used by the dashboard
├── scripts/      Validation and asset-generation utilities
├── tests/        Automated tests
├── .streamlit/   Streamlit configuration and secrets template
├── .vscode/      Editor settings
├── requirements.txt
└── README.md
```

### Main application files

- `main.py` - Single Streamlit entry point, shared context, and page routing
- `app/data.py` - Data loading, cleaning, and feature preparation
- `app/models.py` - Classification, regression, clustering, and prediction logic
- `app/charts_matplotlib.py` - Dashboard and EDA visualizations
- `app/gemini_service.py` - Optional Gemini assistant integration
- `app/config.py` - Shared application settings and colors
- `app/pages/dashboard.py` - Dashboard and cancellation-risk form
- `app/pages/eda_insights.py` - EDA insights and visualization gallery
- `app/pages/model_performance.py` - Model comparisons and clustering analysis
- `app/pages/ai_assistant.py` - AI summary and chat interface
- `app/pages/about.py` - Project details and team information

The application uses `assets/logo.png` for both the sidebar brand mark and the Streamlit browser page icon.

## Run

From the repository root:

```powershell
pip install -r requirements.txt
streamlit run main.py
```

The app loads `data/hotel_bookings_updated_2024.csv` by default. You can upload another compatible CSV from the sidebar, or use generated demo data when no default file is available.

## Validation

Run the lightweight project validation from the repository root:

```powershell
python -m scripts.quick_validation
```

Run the automated tests when `pytest` is installed:

```powershell
python -m pytest -q
```

## Gemini AI Assistant

To enable the AI Assistant, copy `.streamlit/secrets.toml.example` to `.streamlit/secrets.toml` and replace the placeholder with your Gemini API key:

```toml
GEMINI_API_KEY = "YOUR_GEMINI_API_KEY_HERE"
```

The local secrets file is ignored by Git. Without a configured key, the dashboard and all non-AI features continue to work.

## Asset generation

The scripts in `scripts/` write generated images to `assets/`:

```powershell
python scripts/create_professional_logos.py
```
