
PY      ?= python3       
DEPTH   ?= 3                
GAMES   ?= 100             

.PHONY: help run best bench fmt lint clean

help:       
	@grep -E '^[a-zA-Z_-]+:.*##' $(MAKEFILE_LIST) | \
	  awk 'BEGIN{FS=":.*##"}{printf " \033[36m%-10s\033[0m %s\n", $$1, $$2}'

run:          
	$(PY) main.py 2048 --depth $(DEPTH)

best:       
	$(PY) main.py 2048-best --games $(GAMES) --depth $(DEPTH)

bench:       
	$(PY)-m minimax.run_bench --games $(GAMES) --depth $(DEPTH)

fmt:        
	$(PY) -m black .

lint:        
	flake8 .

clean:       
	find . -name '__pycache__' -prune -exec rm -r {} +
