NAME = dslr

VENV = .venv
PYTHON = $(VENV)/bin/python
PIP = $(VENV)/bin/pip
VENV_ERROR_LOG = /tmp/dslr_venv_error.log

TRAIN_DATA = datasets/dataset_train.csv
TEST_DATA = datasets/dataset_test.csv
WEIGHTS = weights.json
TRAIN_ITERATIONS = 1000
TRAIN_ALPHA = 0.01
TRAIN_OUTPUT = $(WEIGHTS)
PREDICTIONS = houses.csv
ANIMATION_DATA = $(TRAIN_DATA)
ANIMATION_OUTPUT = visuals/logreg_train_weights.gif
ANIMATION_ITERATIONS = 40
ANIMATION_FRAME_STEP = 1
ANIMATION_MAX_PREVIEW_FRAMES = 250
ANIMATION_FIGURE_SCALE = 1.0
ANIMATION_GIF_FINAL_FRAME_HOLD_MS = 2000
KIVIAT_WEIGHTS = $(WEIGHTS)
KIVIAT_OUTPUT = visuals/kiviat_house_discipline_weights.png
KIVIAT_SMOOTH_POINTS_PER_SEGMENT = 10
ANALYSIS_LOG_TRAIN_DATA = datasets/dataset_analyse_log_train.csv
ANALYSIS_LOG_TRAIN_OUTPUT = weights_training.json
ANALYSIS_LOG_TRAIN_ITERATIONS = 1
ANALYSIS_LOG_TRAIN_ALPHA = 0.01
ANALYSIS_LOG_PREDICT_DATA = datasets/dataset_analyse_log_predict.csv
ANALYSIS_LOG_PREDICT_WEIGHTS = $(ANALYSIS_LOG_TRAIN_OUTPUT)
ANALYSIS_LOG_PREDICT_OUTPUT = houses_training.csv

.PHONY: all install describe histogram scatter pair train analysis_log_train predict analysis_log_predict animate kiviat re clean fclean help

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
	$(PYTHON) scripts/logreg_train.py $(TRAIN_DATA) \
		--alpha $(TRAIN_ALPHA) \
		--iterations $(TRAIN_ITERATIONS) \
		--out $(TRAIN_OUTPUT)

analysis_log_train: install
	$(PYTHON) scripts/logreg_train.py $(ANALYSIS_LOG_TRAIN_DATA) \
		--alpha $(ANALYSIS_LOG_TRAIN_ALPHA) \
		--iterations $(ANALYSIS_LOG_TRAIN_ITERATIONS) \
		--analysis-log \
		--out $(ANALYSIS_LOG_TRAIN_OUTPUT)

predict:
	$(PYTHON) scripts/logreg_predict.py $(TEST_DATA) $(WEIGHTS)

analysis_log_predict: install
	$(PYTHON) scripts/logreg_predict.py $(ANALYSIS_LOG_PREDICT_DATA) $(ANALYSIS_LOG_PREDICT_WEIGHTS) \
		--analysis-log \
		--out $(ANALYSIS_LOG_PREDICT_OUTPUT)

animate: install
	$(PYTHON) scripts/animate_logreg_train.py $(ANIMATION_DATA) \
		--iterations $(ANIMATION_ITERATIONS) \
		--frame-step $(ANIMATION_FRAME_STEP) \
		--max-preview-frames $(ANIMATION_MAX_PREVIEW_FRAMES) \
		--figure-scale $(ANIMATION_FIGURE_SCALE) \
		--gif-final-frame-hold-ms $(ANIMATION_GIF_FINAL_FRAME_HOLD_MS) \
		--save $(ANIMATION_OUTPUT) \

kiviat: install
	$(PYTHON) scripts/kiviat_house_discipline_weights.py $(KIVIAT_WEIGHTS) \
		--out $(KIVIAT_OUTPUT) \
		--smooth-points-per-segment $(KIVIAT_SMOOTH_POINTS_PER_SEGMENT)

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
	@echo "  make analysis_log_train - Train with verbose analysis logs on dataset_analyse_log_train.csv"
	@echo "  make predict     - Predict houses from dataset_test.csv using weights.json"
	@echo "  make analysis_log_predict - Predict with verbose analysis logs on dataset_analyse_log_predict.csv"
	@echo "  make animate     - Generate training animation GIF in visuals/"
	@echo "  make kiviat      - Generate Kiviat chart of discipline weights by house"
	@echo "  make clean       - Remove generated files"
	@echo "  make re          - Clean and reinstall dependencies"
