"""Launch VeloxVoice Audio Symbol Recognition."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
os.environ.setdefault("HF_HOME", "/root/.cache/huggingface")

from veloxvoice.server.api_server import create_app

app = create_app(
    model_dir=os.environ.get("MODEL_DIR", "data/models/asr_model"),
    music_model_dir=os.environ.get("MUSIC_MODEL_DIR") or None,
    base_model_dir=os.environ.get("BASE_MODEL_DIR") or None,
    device=os.environ.get("DEVICE", "cuda"),
)
