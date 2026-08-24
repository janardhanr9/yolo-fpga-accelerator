# Reference model and RTL simulation.
#
#   make test     python reference-model tests
#   make sim      run every testbench in tb/
#   make wave     open the last VCD (needs gtkwave or surfer)
#   make lint     verilator's static checks, stricter than iverilog's
#   make clean

PY      := .venv/bin/python
IVERILOG:= iverilog -g2012 -Wall
BUILD   := build

TBS     := $(wildcard tb/tb_*.v)
OUTS    := $(patsubst tb/%.v,$(BUILD)/%.out,$(TBS))

.PHONY: test sim wave lint clean all
all: test sim

test:
	@$(PY) tests/run.py

# Each testbench is paired with the rtl/ module of the same name:
# tb/tb_line_buffer.v drives rtl/line_buffer.v.
$(BUILD)/tb_%.out: tb/tb_%.v rtl/%.v | $(BUILD)
	@$(IVERILOG) -o $@ $^

sim: $(OUTS)
	@for out in $(OUTS); do \
	  echo "--- $$out"; \
	  vvp $$out | grep -vE '^(VCD info|.*\$$finish)' || true; \
	done

lint:
	@for f in rtl/*.v; do \
	  echo "--- $$f"; \
	  verilator --lint-only -Wall $$f || true; \
	done

wave:
	@vcd=$$(ls -t *.vcd $(BUILD)/*.vcd 2>/dev/null | head -1); \
	if [ -z "$$vcd" ]; then echo "no VCD; run make sim first"; exit 1; fi; \
	if command -v surfer >/dev/null; then surfer $$vcd; \
	elif command -v gtkwave >/dev/null; then gtkwave $$vcd; \
	else echo "install a viewer: brew install --cask gtkwave  (or surfer)"; fi

$(BUILD):
	@mkdir -p $(BUILD)

clean:
	@rm -rf $(BUILD) *.vcd
	@find . -name __pycache__ -type d -prune -exec rm -rf {} +
