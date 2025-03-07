# FinAssist Installation Guide

## Setting up the Environment

### Prerequisites
- Python 3.8 or higher
- pip (Python package installer)

### Installation Steps

1. Clone the repository:
   ```bash
   git clone https://github.com/miladnasiri/Financial-AI-Agents-.git
   cd Financial-AI-Agents-
   ```

2. Create a virtual environment (recommended):
   ```bash
   python -m venv finassist-env
   ```

3. Activate the virtual environment:
   - On Windows:
     ```bash
     finassist-env\Scripts\activate
     ```
   - On macOS and Linux:
     ```bash
     source finassist-env/bin/activate
     ```

4. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Running the Application

1. Run the main Python script:
   ```bash
   python financial-ai-agent-project.py
   ```

2. For the visualization dashboard (if you have a frontend server):
   ```bash
   # This would depend on your frontend setup
   # For a React application, typically:
   npm install
   npm start
   ```
