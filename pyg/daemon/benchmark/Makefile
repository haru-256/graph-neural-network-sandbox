.PHONY: profile
profile:
	@echo "Profiling..."
	uv run main.py

.PHONY: show-results
show-results:
	@echo "Showing results..."
	uv run python -m line_profiler $(PROFILE_FILE)

