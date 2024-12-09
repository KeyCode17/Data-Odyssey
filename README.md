# Mochammad Daffa Putra Karyudi

---

# Streamlit Application Tutorial

This repository contains a Streamlit app, and this guide will walk you through the steps to set it up and run it using `streamlit run`.

## Prerequisites

Before running the app, ensure that you have the following prerequisites:

1. **Python 3.10**  
   Make sure that you have Python 3.10 or later installed. You can check your Python version by running:
   ```bash
   python --version
   ```

2. **Install Dependencies**  
   Ensure all required libraries are installed. Run:

   To install dependencies from `requirements.txt`, run:
   ```bash
   pip install -r requirements.txt
   ```

   This will install `Streamlit` and any other packages required for your application.

## Running the App

Follow these steps to run the app:

### 1. Clone the Repository (If Applicable)

If you haven't already cloned the repository, do so with:
```bash
git clone https://github.com/KeyCode17/Data-Odyssey.git
cd Data-Odyssey
```

### 2. Install the Dependencies

Ensure all required libraries are installed by running:
```bash
pip install -r requirements.txt
```

### 3. Run the Streamlit Application

In the terminal, navigate to the directory containing `app.py`. Then, run the following command to start the Streamlit application:
```bash
streamlit run app.py
```

### 4. Open the App in Your Browser

After running the above command, Streamlit will automatically open a new tab in your default web browser to show the application. If it doesn't open automatically, you can manually navigate to:
```
http://localhost:8501
```

### 5. Interact with the App

Once the app is running, you can interact with it directly through the web interface. Depending on the design of the app, you may be able to upload data, visualize results, and change parameters interactively.

### 6. Stop the App

To stop the app, return to the terminal and press `Ctrl+C`.

## Troubleshooting

### App Not Opening Automatically
If the app doesn't open in the browser, check for error messages in the terminal and ensure the correct port (`8501` by default) is not being blocked by a firewall.

### Dependency Issues
If you face any dependency issues, make sure that your environment is properly set up. You can try creating a fresh virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
pip install -r requirements.txt
```