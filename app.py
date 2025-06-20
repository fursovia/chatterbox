import io
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import torch
import torchaudio as ta
from loguru import logger

from chatterbox.tts import ChatterboxTTS


class TextToSpeechRequest(BaseModel):
    """Request body for text-to-speech generation."""
    text: str


# Global registry for loaded ML models
tts_models: dict[str, ChatterboxTTS] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """App lifespan handler that loads and later releases the TTS model."""
    # Enforce CUDA requirement
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA device is required to run Chatterbox TTS. No CUDA-capable GPU was detected."
        )

    device = "cuda"
    # Load the model once at startup
    tts_models["chatterbox"] = ChatterboxTTS.from_pretrained(device=device)

    # Yield control back to the application
    yield

    # Cleanup on shutdown
    tts_models.clear()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


app = FastAPI(title="Chatterbox TTS API", version="0.1.0", lifespan=lifespan)


@app.post("/text_to_speech", summary="Convert text to spoken audio", response_class=StreamingResponse)
async def text_to_speech(payload: TextToSpeechRequest):
    """Generate speech from text and return as a WAV audio stream."""
    model = tts_models["chatterbox"]

    # Measure generation time
    start_time = time.perf_counter()
    # Generate waveform (Tensor shape: [1, num_samples])
    wav_tensor = model.generate(payload.text)
    generation_time = time.perf_counter() - start_time
    logger.info(f"Generated speech in {generation_time:.3f} seconds for request text length={len(payload.text)}")

    # Serialize the tensor to a WAV byte stream in-memory.
    buffer = io.BytesIO()
    ta.save(buffer, wav_tensor, sample_rate=model.sr, format="wav")
    buffer.seek(0)

    # Stream the WAV bytes back to the client.
    headers = {"Content-Disposition": "inline; filename=speech.wav"}
    return StreamingResponse(buffer, media_type="audio/wav", headers=headers)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=8000)
