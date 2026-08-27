# Reference model and RTL simulation.
#
#   make test     python reference-model tests
#   make sim      run every testbench in tb/ under iverilog
#   make simv     same, under verilator (slower to build, stricter)
#   make lint     verilator's static checks only
#   make wave     open the last VCD (needs gtkwave or surfer)
#   make clean
#
# SystemVerilog, but restricted to what Icarus also accepts: no
# variable indexing into packed multi-dimensional arrays, and no named
# struct assignment patterns. Losing the fast simulator costs more than
# the nicer syntax is worth. Unpacked arrays index fine on both.

PY      := .venv/bin/python
IVERILOG:= iverilog -g2012 -Wall
BUILD   := build

# Only testbenches that actually contain a module -- the stubs for
# unwritten ones would otherwise print an empty entry each run.
TBS     := $(shell grep -l '^module' tb/tb_*.sv 2>/dev/null)
OUTS    := $(patsubst tb/%.sv,$(BUILD)/%.out,$(TBS))

.PHONY: test sim simv wave lint mutants clean all
all: test sim

test:
	@$(PY) tests/run.py

# Each testbench is paired with the rtl/ module of the same name:
# tb/tb_line_buffer.sv drives rtl/line_buffer.sv.
$(BUILD)/tb_%.out: tb/tb_%.sv rtl/%.sv | $(BUILD)
	@$(IVERILOG) -o $@ $^ 2>&1 | grep -v 'cannot be synthesized in an always_ff' || true

sim: $(OUTS)
	@for out in $(OUTS); do \
	  echo "--- $$out"; \
	  vvp $$out | grep -vE '^(VCD info|.*\$$finish)' || true; \
	done

# Verilator builds a real binary, so it catches races and width issues
# iverilog lets through -- worth running before you believe a PASS.
simv:
	@for tb in $(TBS); do \
	  name=$$(basename $$tb .sv); \
	  dut=rtl/$${name#tb_}.sv; \
	  echo "--- $$name"; \
	  verilator --binary -Wno-fatal --timing -Mdir $(BUILD)/$$name \
	    -o $$name $$tb $$dut >/dev/null 2>&1 \
	    && $(BUILD)/$$name/$$name 2>&1 | grep -E '^(PASS|FAIL| )' \
	    || echo "  build failed"; \
	done

# A testbench that has only ever passed proves nothing: you have watched
# it agree with a correct design, not disagree with a wrong one.
mutants:
	@$(PY) tests/mutants.py

lint:
	@for f in rtl/*.sv; do \
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
