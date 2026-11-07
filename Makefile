# SPDX-License-Identifier: AGPL-3.0-only OR LicenseRef-SnapKitty-Opaque-1.0
# Copyright (C) 2026 Ahmad Ali Parr / SnapKitty Research Lab

FPC        ?= fpc
FPCFLAGS   ?= -Mobjfpc -Sh -O2 -vew
NVCC       ?= /workspace/.tools/cuda/bin/nvcc
CUDA_ARCHS ?= 80 86 87 89 90
NVFLAGS    ?= -O3 -std=c++17 -Xcompiler -fPIC -shared -Xptxas -v

BUILD := build
BIN   := $(BUILD)/bin
UNITS := $(BUILD)/units
LIB   := $(BUILD)/libtmoe_cuda.so

SRC   := $(wildcard src/*.pas)
TESTS := test_pack test_gemv test_router test_switch test_swiglu test_moe test_gradcheck test_train test_checkpoint test_gpu
TOOLS := tmoe_train tmoe_generate tmoe_bench tmoe_cost

# highest arch also gets PTX so newer GPUs can JIT
LAST_ARCH := $(lastword $(CUDA_ARCHS))
GENCODE   := $(foreach a,$(CUDA_ARCHS),-gencode arch=compute_$(a),code=sm_$(a)) \
             -gencode arch=compute_$(LAST_ARCH),code=compute_$(LAST_ARCH)

.PHONY: all tools tests lib test bench train cost clean

all: tools tests

$(BIN) $(UNITS):
	mkdir -p $@

tools: $(addprefix $(BIN)/,$(TOOLS))
tests: $(addprefix $(BIN)/,$(TESTS))

$(BIN)/test_%: tests/test_%.pas tests/testutil.pas tests/testmats.pas $(SRC) | $(BIN) $(UNITS)
	$(FPC) $(FPCFLAGS) -Fusrc -Futests -FU$(UNITS) -FE$(BIN) $<

$(BIN)/tmoe_train: tools/tmoe_trainer.pas tools/tmoe_args.pas $(SRC) | $(BIN) $(UNITS)
	$(FPC) $(FPCFLAGS) -Fusrc -Futools -FU$(UNITS) -FE$(BIN) $< -o$@

$(BIN)/tmoe_%: tools/tmoe_%.pas tools/tmoe_args.pas $(SRC) | $(BIN) $(UNITS)
	$(FPC) $(FPCFLAGS) -Fusrc -Futools -FU$(UNITS) -FE$(BIN) $<

lib: $(LIB)

$(LIB): cuda/tmoe_kernels.cu cuda/tmoe_cuda.h | $(BIN)
	@command -v $(NVCC) >/dev/null 2>&1 || { echo "nvcc not found ($(NVCC)). The CUDA library was not built; CPU tests do not need it."; exit 1; }
	$(NVCC) $(NVFLAGS) $(GENCODE) -o $@ $< > $(BUILD)/ptxas.log 2>&1 || (cat $(BUILD)/ptxas.log; exit 1)
	@grep -E "spill|stack frame" $(BUILD)/ptxas.log | sort | uniq -c
	@echo "ptxas report: $(BUILD)/ptxas.log"

# tests run one after another; the GPU test skips when no device is present
test: tests $(BIN)/tmoe_cost
	@set -e; for t in $(TESTS); do echo "== $$t"; ./$(BIN)/$$t; done
	@echo "== tmoe_cost --check"; ./$(BIN)/tmoe_cost --check
	@python3 tests/ref_ternary.py

bench: $(BIN)/tmoe_bench $(BIN)/tmoe_cost
	./$(BIN)/tmoe_cost --check
	./$(BIN)/tmoe_bench --config tiny
	./$(BIN)/tmoe_bench --config small

cost: $(BIN)/tmoe_cost
	./$(BIN)/tmoe_cost

train: $(BIN)/tmoe_train $(BIN)/tmoe_generate
	./$(BIN)/tmoe_train --data data/tiny.txt --steps 400 --out $(BUILD)/tiny.ckpt
	./$(BIN)/tmoe_generate --ckpt $(BUILD)/tiny.ckpt --prompt "The keeper" --n 200

clean:
	rm -rf $(BUILD)
