import os
import random
import string
import subprocess
import tempfile

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

import audio
import speech

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


def _random_name(ext: str) -> str:
    rnd = "".join(random.choices(string.ascii_uppercase + string.digits, k=10))
    return f"/tmp/{rnd}.{ext}"


def upload_webm(file: UploadFile) -> str:
    """Save an uploaded .webm file and convert it to WAV."""
    destination = _random_name("webm")
    with open(destination, "wb") as buffer:
        buffer.write(file.file.read())
    return audio.webp2wav(
        destination
    )  # function name in audio module is misleading but OK


def convert_mp3_to_wav(input_path: str) -> str:
    """Convert MP3 -> 16 kHz mono 16-bit PCM WAV using ffmpeg."""
    output_path = _random_name("wav")
    command = [
        "ffmpeg",
        "-i",
        input_path,
        "-acodec",
        "pcm_s16le",
        "-ar",
        "16000",
        "-ac",
        "1",
        output_path,
        "-y",
    ]
    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
        return output_path
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"FFmpeg conversion failed: {e.stderr}")
    except FileNotFoundError:
        raise RuntimeError("FFmpeg not found. Please install ffmpeg first.")


def save_uploaded_file(file: UploadFile, extension: str) -> str:
    """Save an uploaded file to /tmp and return the path."""
    destination = _random_name(extension)
    with open(destination, "wb") as buffer:
        buffer.write(file.file.read())
    return destination


def process_audio_file(file: UploadFile) -> str:
    """Return a WAV path for any supported upload (.webm/.mp3/.wav)."""
    original_filename = (file.filename or "").lower()
    print("filename:", original_filename)

    if original_filename.endswith(".webm"):
        # ✅ FIX: pass the UploadFile object, not the filename string
        return upload_webm(file)

    if original_filename.endswith(".mp3"):
        mp3_path = save_uploaded_file(file, "mp3")
        try:
            return convert_mp3_to_wav(mp3_path)
        finally:
            if os.path.exists(mp3_path):
                os.remove(mp3_path)

    if original_filename.endswith(".wav"):
        return save_uploaded_file(file, "wav")

    raise HTTPException(
        status_code=400,
        detail="Unsupported file format. Please upload .webm, .mp3, or .wav files only.",
    )


@app.post("/pronunciation")
async def api_analyze_pronunciation(
    file: UploadFile = File(...), expected_text: str = Form(...)
):
    print(
        "Running pronunciation model at:",
        __import__("datetime").datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    )

    wav_file = process_audio_file(file)
    try:
        sound = audio.load(wav_file)
        return speech.compare_audio_with_text(sound, expected_text)
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail="Something went wrong")
    finally:
        # Clean up the temp WAV so /tmp doesn't grow forever
        if os.path.exists(wav_file):
            os.remove(wav_file)


@app.post("/speech2text")
async def api_speech2text(file: UploadFile = File(...)):
    print(
        "Running stt model at:",
        __import__("datetime").datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    )

    # ✅ FIX: use the unified processor so mp3/wav also work here
    wav_file = process_audio_file(file)
    try:
        sound = audio.load(wav_file)
        return {"transcript": speech.transcribe(sound)}
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail="Something went wrong")
    finally:
        if os.path.exists(wav_file):
            os.remove(wav_file)


@app.post("/phonemes")
async def api_phonemes(text: str = Form(...)):
    try:
        phonemes, words = speech.get_phonemes_with_word_mapping(text)
        return {"phonemes": phonemes, "words": list(words.values())}
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail="Something went wrong")


@app.post("/tts")
async def api_tts(text: str = Form(...)):
    try:
        filename = audio.text2speech(text)
        return FileResponse(filename, media_type="audio/wav")
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail="Something went wrong")


@app.get("/wakeup")
async def wakeup():
    return {"status": "woke"}


@app.get("/")
async def home(request: Request):
    return templates.TemplateResponse(request=request, name="index.html", context={})
