NAME = dslr

VENV = .venv
PYTHON = $(VENV)/bin/python
PIP = $(VENV)/bin/pip

TRAIN_DATA = datasets/dataset_train.csv
TEST_DATA = datasets/dataset_test.csv
WEIGHTS = weights.json
PREDICTIONS = houses.csv

all: install

install:
	@test -d $(VENV) || python3 -m venv $(VENV)
	$(PIP) install -r requirements.txt

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
	@echo "  make install     - Install venv"
	@echo "  make describe    - Display descriptive statistics from dataset_train.csv"
	@echo "  make histogram   - Generate histograms from dataset_train.csv"
	@echo "  make scatter     - Generate scatter plot(s) from dataset_train.csv"
	@echo "  make pair        - Generate pair plot from dataset_train.csv"
	@echo "  make train       - Train logistic regression and save weights"
	@echo "  make predict     - Predict houses from dataset_test.csv using weights.json"
	@echo "  make clean       - Remove generated files"
	@echo "  make re          - Clean and reinstall dependencies"
