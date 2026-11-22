#!/bin/bash
# Bibley 2 - double-click this to start.
cd "$(dirname "$0")"
clear
echo "=============================="
echo "  BIBLEY 2"
echo "=============================="

if [ ! -d venv ]; then
  echo "First-time setup. This takes a few minutes, one time only."
  echo
  python3 -m venv venv
  if [ ! -d venv ]; then
    echo "Something went wrong making the venv. Read the error shown above before trying setup again."
    read -r -p "Press enter to close."
    exit 1
  fi
  ./venv/bin/pip install --upgrade pip
  ./venv/bin/pip install -r requirements.txt
  if [ $? -ne 0 ]; then
    echo "Something went wrong installing parts. Read the error shown above before trying setup again."
    read -r -p "Press enter to close."
    exit 1
  fi
  echo
  echo "Setup done. Starting Bibley..."
fi

./venv/bin/python server.py
echo
read -r -p "Bibley stopped. Press enter to close."
