import os
import random
import string
import subprocess

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

import audio
import speech

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


def upload_webp(file):
    tempname_random = "".join(
        random.choices(string.ascii_uppercase + string.digits, k=6)
    )
    destination = f"/tmp/{tempname_random}.webm"

    with open(destination, "wb") as buffer:
        buffer.write(file.file.read())

    return audio.webp2wav(destination)


def convert_mp3_to_wav(input_path):
    """Convert MP3 to WAV using ffmpeg"""
    tempname_random = "".join(
        random.choices(string.ascii_uppercase + string.digits, k=6)
    )
    output_path = f"/tmp/{tempname_random}.wav"

    try:
        command = [
            "ffmpeg",
            "-i",
            input_path,
            "-acodec",
            "pcm_s16le",  # 16-bit PCM
            "-ar",
            "16000",  # 16kHz sample rate
            "-ac",
            "1",  # mono channel
            output_path,
            "-y",  # overwrite output file if exists
        ]

        result = subprocess.run(command, check=True, capture_output=True, text=True)
        return output_path

    except subprocess.CalledProcessError as e:
        raise Exception(f"FFmpeg conversion failed: {e.stderr}")
    except FileNotFoundError:
        raise Exception("FFmpeg not found. Please install ffmpeg first.")


def save_uploaded_file(file, extension):
    """Save uploaded file to temporary location"""
    tempname_random = "".join(
        random.choices(string.ascii_uppercase + string.digits, k=6)
    )
    destination = f"/tmp/{tempname_random}.{extension}"

    with open(destination, "wb") as buffer:
        buffer.write(file.file.read())

    return destination


def process_audio_file(file):
    """Process uploaded audio file and convert to WAV if needed"""

    # Get the original filename
    original_filename = file.filename.lower()

    # Determine file type and process accordingly
    if original_filename.endswith(".webm"):
        # Save webm and convert using existing function
        wav_path = upload_webm(file)

    elif original_filename.endswith(".mp3"):
        # Save mp3 file temporarily
        mp3_path = save_uploaded_file(file, "mp3")
        try:
            # Convert mp3 to wav
            wav_path = convert_mp3_to_wav(mp3_path)
        finally:
            # Clean up temporary mp3 file
            if os.path.exists(mp3_path):
                os.remove(mp3_path)

    elif original_filename.endswith(".wav"):
        # Save wav file directly
        wav_path = save_uploaded_file(file, "wav")

    else:
        raise HTTPException(
            status_code=400,
            detail="Unsupported file format. Please upload .webm, .mp3, or .wav files only.",
        )

    return wav_path


@app.post("/pronunciation")
async def api_analyze_pronunciation(
    file: UploadFile = File(...), expected_text: str = Form(...)
):
    from datetime import datetime

    # Get current timestamp and format it
    formatted_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Print the formatted timestamp
    print("Running pronounciation model at:", formatted_timestamp)

    wav_file = process_audio_file(file)

    try:
        sound = audio.load(wav_file)
        return speech.compare_audio_with_text(sound, expected_text)
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail="Something went wrong")


@app.post("/speech2text")
async def api_speech2text(file: UploadFile = File(...)):
    from datetime import datetime

    # Get current timestamp and format it
    formatted_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Print the formatted timestamp
    print("Running stt model at:", formatted_timestamp)
    wav_file = upload_webp(file)
    try:
        sound = audio.load(wav_file)
        return {
            "transcript": speech.transcribe(sound),
        }
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail="Something went wrong")


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
