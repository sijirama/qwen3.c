CC = clang
CFLAGS = -std=c11 -Wall -Wextra -Wpedantic -g

TARGET = qwen3

build: $(TARGET)

$(TARGET): main.c
	@echo "Building $@..."
	@$(CC) $(CFLAGS) $< -o $@

run: build
	@./$(TARGET)

clean:
	@echo "Cleaning..."
	@rm -f $(TARGET)

help:
	@echo "Available targets:"
	@echo "  build  Build the hello-world executable"
	@echo "  run    Build and run the executable"
	@echo "  clean  Remove build artifacts"
	@echo "  help   Show this help message"

.PHONY: build run clean help
