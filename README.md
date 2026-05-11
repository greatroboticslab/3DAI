# Requirments
Docker

# Getting Started

## 1. Clone the repository
```bash
git clone https://github.com/greatroboticslab/3DAI
cd 3DAI
```

## 2. Set up a Python virtual environment

### Windows
```shell
python -m venv venv  # Creates a new virtual environment in the current folder named "venv"
venv\Scripts\activate  # Activates the virtual environment
```

### Linux / macOS
```bash
python3 -m venv .venv  # Same as Windows; the dot makes the folder hidden on Linux
source .venv/bin/activate # Activates the virtual environment
```

## 3. Install dependencies
```bash
pip install -r requirements.txt  # Installs all required packages
```

## 4. Create Environment File
```bash
cp enviroment.example .env
```

## 5. Start Database and API
```bash
docker compose up -d
```

## 7. Start Capture Tool
```bash
python3 capture_tool.py
```

## 8. Verify API is Running
[http://localhost:8000/docs](http://localhost:8000/docs)