# Makefile

# Variables
PYTHON_FILE := app.py
NOTEBOOK_FILE := app.ipynb

# Default target
all: convert_to_notebook

# Convert Python file to Jupyter Notebook
convert_to_notebook: install_jupytext
	jupytext --to notebook $(PYTHON_FILE) -o $(NOTEBOOK_FILE)

# Clean up generated files
clean:
	rm -f $(NOTEBOOK_FILE)

# Phony targets
.PHONY: all install_jupytext convert_to_notebook clean
