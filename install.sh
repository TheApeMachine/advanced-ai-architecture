#!/bin/bash

# Uninstall existing llama-cpp-python if present
pip uninstall llama-cpp-python --yes

# Install dependencies
pip install -r requirements.txt

CMAKE_ARGS="-DGGML_METAL=on" pip install llama-cpp-python