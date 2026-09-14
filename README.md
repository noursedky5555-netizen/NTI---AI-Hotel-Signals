# Hotel Signals

A modular Streamlit dashboard for exploring hotel booking cancellations, rates, and booking behavior.

## Run

```powershell
pip install -r hotel/requirements.txt
streamlit run hotel/main.py
```

The app starts with generated demo data. Upload a hotel bookings CSV from the sidebar to analyze a real file. The file must include the numeric booking fields checked in `hotel/config.py`.
