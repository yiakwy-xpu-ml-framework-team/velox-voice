"""Run the VeloxVoice SGLang-Omni-compatible ASR server.

Example:
    python -m veloxvoice.server --model-dir data/models/asr_model --port 8000
"""

from .api_server import main

if __name__ == "__main__":
    main()
