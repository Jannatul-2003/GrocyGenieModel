---
title: Grocy Genie Lstm
emoji: 🚀
colorFrom: pink
colorTo: indigo
sdk: docker
app_file: app.py
pinned: false
license: apache-2.0
short_description: LSTM model for grocery demand prediction and inventory manag
---


## Quick Start

Follow these steps to get the application running on your local machine:

### Prerequisites

- Python 3.8 or higher
- Git

### Installation & Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/Jannatul-2003/GrocyGenieModel.git
   cd GrocyGenieModel
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   ```

3. **Activate the virtual environment**
   
   **Windows:**
   ```bash
   venv\Scripts\activate
   ```
   
   **macOS/Linux:**
   ```bash
   source venv/bin/activate
   ```

4. **Upgrade pip**
   ```bash
   pip install --upgrade pip
   ```

5. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```
   
   If you have a requirements.lock.txt file, also run:
   ```bash
   pip install -r requirements.lock.txt
   ```

6. **Run the application**
   
   **Option 1: Using Python directly**
   ```bash
   python app.py
   ```
   
   **Option 2: Using Uvicorn (recommended for FastAPI)**
   ```bash
   uvicorn app:app --reload
   ```

7. **Access the application**
   
   Open your browser and navigate to:
   - **Interactive API Documentation (Swagger UI):** http://127.0.0.1:8000/docs
   - **Alternative API Documentation (ReDoc):** http://127.0.0.1:8000/redoc
   - **Main Application:** http://127.0.0.1:8000


## License

This project is licensed under the Apache License 2.0 - see the LICENSE file for details.

## Configuration Reference

For more information about Hugging Face Spaces configuration, check out the [configuration reference](https://huggingface.co/docs/hub/spaces-config-reference).

[![Hugging Face Spaces](https://img.shields.io/badge/🤗%20View%20on-Hugging%20Face-blue)](https://huggingface.co/spaces/Jannatul03/grocy-genie-lstm)

## FastAPI

**https://jannatul03-grocy-genie-grocy-genie-lstm.hf.space/docs**