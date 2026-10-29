#!/usr/bin/env bash
# Mac/Linux: bash setup.sh   (Windows: run the commands manually; activate with venv\Scripts\activate)
set -e
python -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
python -c "import surprise, pandas, sklearn, flask; print(\"Environment OK\")"
git init 2>/dev/null || true
git add . && git commit -m "Initial project structure" || true
