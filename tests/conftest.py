import os

# Disable Gradio analytics/version check threads during tests.
os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")
os.environ.setdefault("GRADIO_ANALYTICS", "False")
