#!/bin/bash
cd "$(dirname "$0")"
python3 -m pip install --user -q -r requirements.txt
cd backend
python3 main.py
