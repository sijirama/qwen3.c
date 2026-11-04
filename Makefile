CC = clang
CFLAGS = -std=c11 -Wall -Wextra -Wpedantic -g

TARGET = qwen3
TEST_TARGET = test_qwen3
MODEL ?= models/Qwen3-0.6B-Q8_0.gguf

build: $(TARGET)

$(TARGET): main.c
	@echo "Building $@..."
	@$(CC) $(CFLAGS) $< -o $@

run: build
	@./$(TARGET) $(MODEL)

$(TEST_TARGET): test.c main.c
	@echo "Building tests..."
	@$(CC) $(CFLAGS) test.c -o $@

test: $(TEST_TARGET)
	@./$(TEST_TARGET) $(MODEL)

clean:
	@echo "Cleaning..."
	@rm -f $(TARGET) $(TEST_TARGET)

help:
	@echo "Available targets:"
	@echo "  build  Build the Qwen3 parser"
	@echo "  run    Build and run the executable"
	@echo "  test   Build and run loader tests"
	@echo "  clean  Remove build artifacts"
	@echo "  help   Show this help message"

.PHONY: build run test clean help
