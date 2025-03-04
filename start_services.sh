#!/bin/bash

# Start API server in the background
uvicorn src.api_server:app --host 0.0.0.0 --port 8000 &

# Start Streamlit dashboard
streamlit run src/dashboard.py --server.port 8501 --server.address 0.0.0.0
