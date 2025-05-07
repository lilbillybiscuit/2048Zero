

PY     ?= python3
GAMES  ?= 100
DEPTH  ?= 3

.PHONY: minimax
minimax:            
	$(PY) main.py 2048-best --games $(GAMES) --depth $(DEPTH)

