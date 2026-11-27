"""Launch VeloxVoice with the music tab (SheetSage2 + MERT-v2).

ASR is optional: pass MODEL_DIR to load the WeNet style ASR model as well; leave
it empty for a music-only server.

    MUSIC_MODEL_DIR=/path/to/SheetSage2 \
    BASE_MODEL_DIR=/path/to/MERT-v2-FullSong \
    [MODEL_DIR=/path/to/asr_checkpoint] \
    python -m uvicorn launch_music_server:app --host 0.0.0.0 --port 8000

MODEL_DIR is default to $ROOT/data/models/asr_model, with our advanced
Jev diarization module $ROOT/data/models/asr_model/modules/senseJev
"""

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
os.environ.setdefault("HF_HOME", os.path.expanduser("~/.cache/huggingface"))

from veloxvoice.server.api_server import create_app

app = create_app(
    # model_dir=os.environ.get("MODEL_DIR", "data/models/asr_model"),
    model_dir=os.environ.get("MODEL_DIR") or None,
    music_model_dir=os.environ.get("MUSIC_MODEL_DIR") or None,
    base_model_dir=os.environ.get("BASE_MODEL_DIR") or None,
    device=os.environ.get("DEVICE", "cuda"),
)
