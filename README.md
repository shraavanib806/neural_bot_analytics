# neural_bot Setup Guide

Follow these steps to set up and run the Neural Bot project.

## Prerequisites

- Python installed on your machine.
- A terminal or command prompt for running commands.

## Project Setup

### 1. Change to Project Directory

```sh
cd bactester_ui
```
### 2. Create Python Virtual Environment
```sh
python -m venv venv
```

### 3. Activate Virtual Environment
#### For Windows:
```sh
venv\Scripts\activate
```

#### For Linux:
```sh
source venv/bin/activate
```

### 4. Install Required Libraries
```sh
pip install -r requirements.txt
```

### 5. Install TA-Lib wheel 
#### first download talib wheel which is for your python version from this site : https://github.com/cgohlke/talib-build/releases
```sh
pip install talib_wheel_filepath
```
#### replace talib_wheel_filepath with actual filepath of downloaded talib wheel on you device.

### 7. Run the Server
```sh
streamlit run neural_bot_beta.py
```
