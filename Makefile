NAME = dslr

VENV = .venv
PYTHON = $(VENV)/bin/python
PIP = $(VENV)/bin/pip
VENV_ERROR_LOG = /tmp/dslr_venv_error.log

TRAIN_DATA = datasets/dataset_train.csv
TEST_DATA = datasets/dataset_test.csv
WEIGHTS = weights.json
PREDICTIONS = houses.csv

.PHONY: all install describe histogram scatter pair train predict re clean fclean help

all: install

install:
	@set -e; \
	if [ ! -x "$(PIP)" ]; then \
		echo "Virtual environment is missing or incomplete. Recreating $(VENV)..."; \
		rm -rf $(VENV); \
		$(MAKE) --no-print-directory $(PYTHON); \
	fi
	$(PIP) install -r requirements.txt

$(PYTHON):
	@set -e; \
	if python3 -m venv $(VENV) >$(VENV_ERROR_LOG) 2>&1; then \
		rm -f $(VENV_ERROR_LOG); \
	else \
		if grep -Eqi "ensurepip is not|python3-venv" $(VENV_ERROR_LOG); then \
			echo "python3-venv is missing. Falling back to a user-space virtualenv setup..."; \
			if ! python3 -m virtualenv --version >/dev/null 2>&1; then \
				if python3 -m pip --version >/dev/null 2>&1; then \
					python3 -m pip install --user --upgrade virtualenv; \
				else \
					cat $(VENV_ERROR_LOG); \
					rm -f $(VENV_ERROR_LOG); \
					echo "Unable to create .venv without sudo: python3-venv and pip are both unavailable."; \
					exit 1; \
				fi; \
			fi; \
			python3 -m virtualenv $(VENV); \
			rm -f $(VENV_ERROR_LOG); \
		else \
			cat $(VENV_ERROR_LOG); \
			rm -f $(VENV_ERROR_LOG); \
			exit 1; \
		fi; \
	fi

describe:
	$(PYTHON) scripts/describe.py $(TRAIN_DATA)

histogram:
	$(PYTHON) scripts/histogram.py $(TRAIN_DATA)

scatter:
	$(PYTHON) scripts/scatter_plot.py $(TRAIN_DATA)

pair:
	$(PYTHON) scripts/pair_plot.py $(TRAIN_DATA)

train:
	$(PYTHON) scripts/logreg_train.py $(TRAIN_DATA)

predict:
	$(PYTHON) scripts/logreg_predict.py $(TEST_DATA) $(WEIGHTS)

re: fclean all

clean:
	rm -f $(PREDICTIONS)
	rm -f $(WEIGHTS)
	rm -rf visuals
	rm -rf __pycache__
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete

fclean: clean
	rm -rf .venv

help:
	@echo "Available targets:"
	@echo "  make install     - Install venv and Python dependencies"
	@echo "  make describe    - Display descriptive statistics from dataset_train.csv"
	@echo "  make histogram   - Generate histograms from dataset_train.csv"
	@echo "  make scatter     - Generate scatter plot(s) from dataset_train.csv"
	@echo "  make pair        - Generate pair plot from dataset_train.csv"
	@echo "  make train       - Train logistic regression and save weights"
	@echo "  make predict     - Predict houses from dataset_test.csv using weights.json"
	@echo "  make clean       - Remove generated files"
	@echo "  make re          - Clean and reinstall dependencies"
