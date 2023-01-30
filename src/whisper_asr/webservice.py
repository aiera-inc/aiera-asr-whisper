import os
import typing
import uvicorn
import time
import json
import urllib
import base64

from fastapi import FastAPI, Form
from starlette.responses import Response

import torch
import whisper

from threading import Lock

app = FastAPI()

# prep the English-only model for primary transcription...
core_model_name = os.getenv("ASR_MODEL", "medium.en")
if ".en" not in core_model_name:
    core_model_name = f"{core_model_name}.en"

if core_model_name not in ["tiny.en", "base.en", "small.en", "medium.en"]:
    core_model_name = "medium.en"

# prep the non-English model for translation option...
translate_model_name = core_model_name.replace(".en", "")
if translate_model_name not in ["tiny", "base", "small", "medium"]:
    translate_model_name = "medium"

# load the models...
is_cuda = False
if torch.cuda.is_available():
    is_cuda = True
    core_model = whisper.load_model(core_model_name).cuda()
    translate_model = whisper.load_model(translate_model_name).cuda()

else:
    core_model = whisper.load_model(core_model_name)
    translate_model = whisper.load_model(translate_model_name)

# setup lock for model processing...
model_lock = Lock()


def time_ms():
    return time.time_ns() // 1_000_000


class PrettyJSONResponse(Response):
    media_type = "application/json"

    def render(self, content: typing.Any) -> bytes:
        return json.dumps(
            content,
            ensure_ascii=False,
            allow_nan=False,
            indent=4,
            separators=(", ", ": "),
        ).encode("utf-8")


@app.get("/status")
async def status():
    return {"status": "OK"}


@app.get("/transcribe/url", response_class=PrettyJSONResponse)
def transcribe_url(audio_url: str):
    start_time = time_ms()

    url_filename = audio_url.split("/")[-1]
    tmp_filename = f"{time_ms()}-{url_filename}.mp3"

    urllib.request.urlretrieve(audio_url, tmp_filename)

    audio = whisper.load_audio(tmp_filename)

    options_dict = {
        "language": "en"
    }

    with model_lock:
        result = core_model.transcribe(audio, **options_dict)

    if os.path.exists(tmp_filename):
        os.remove(tmp_filename)

    result["model"] = core_model_name
    result["using_cuda"] = is_cuda
    result["duration_ms"] = (time_ms() - start_time)

    return result


@app.post("/transcribe/bytes", response_class=PrettyJSONResponse)
def transcribe_bytes(audio_bytes: str = Form()):
    start_time = time_ms()

    audio_obj = base64.b64decode(audio_bytes)

    tmp_filename = f"{time_ms()}-bytes.mp3"

    tmp_file = open(tmp_filename, 'wb')
    tmp_file.write(audio_obj)
    tmp_file.close()

    audio = whisper.load_audio(tmp_filename)

    options_dict = {
        "language": "en"
    }

    with model_lock:
        result = core_model.transcribe(audio, **options_dict)

    if os.path.exists(tmp_filename):
        os.remove(tmp_filename)

    result["model"] = core_model_name
    result["using_cuda"] = is_cuda
    result["duration_ms"] = (time_ms() - start_time)

    return result


@app.get("/detect-language/url", response_class=PrettyJSONResponse)
def detect_language_url(audio_url: str):
    start_time = time_ms()

    url_filename = audio_url.split("/")[-1]
    tmp_filename = f"{time_ms()}-{url_filename}.mp3"

    urllib.request.urlretrieve(audio_url, tmp_filename)

    audio = whisper.load_audio(tmp_filename)
    audio = whisper.pad_or_trim(audio)
    mel = whisper.log_mel_spectrogram(audio).to(translate_model.device)

    with model_lock:
        _, probs = translate_model.detect_language(mel)

    language = max(probs, key=probs.get)

    if os.path.exists(tmp_filename):
        os.remove(tmp_filename)

    return {
        "model": translate_model_name,
        "language": language,
        "probabilities": probs,
        "duration_ms": (time_ms() - start_time)
    }


@app.get("/translate/url", response_class=PrettyJSONResponse)
def translate_url(audio_url: str):
    start_time = time_ms()

    url_filename = audio_url.split("/")[-1]
    tmp_filename = f"{time_ms()}-{url_filename}.mp3"

    urllib.request.urlretrieve(audio_url, tmp_filename)

    audio = whisper.load_audio(tmp_filename)

    with model_lock:
        result = translate_model.transcribe(audio, task="translate")

    if os.path.exists(tmp_filename):
        os.remove(tmp_filename)

    result["model"] = core_model_name
    result["using_cuda"] = is_cuda
    result["duration_ms"] = (time_ms() - start_time)

    return result


@app.post("/detect-language/bytes", response_class=PrettyJSONResponse)
def detect_language_bytes(audio_bytes: str = Form()):
    start_time = time_ms()

    audio_obj = base64.b64decode(audio_bytes)

    tmp_filename = f"{time_ms()}-bytes.mp3"

    tmp_file = open(tmp_filename, 'wb')
    tmp_file.write(audio_obj)
    tmp_file.close()

    audio = whisper.load_audio(tmp_filename)
    audio = whisper.pad_or_trim(audio)
    mel = whisper.log_mel_spectrogram(audio).to(translate_model.device)

    with model_lock:
        _, probs = translate_model.detect_language(mel)

    language = max(probs, key=probs.get)

    if os.path.exists(tmp_filename):
        os.remove(tmp_filename)

    return {
        "model": translate_model_name,
        "language": language,
        "probabilities": probs,
        "duration_ms": (time_ms() - start_time)
    }


@app.post("/translate/bytes", response_class=PrettyJSONResponse)
def translate_bytes(audio_bytes: str = Form()):
    start_time = time_ms()

    audio_obj = base64.b64decode(audio_bytes)

    tmp_filename = f"{time_ms()}-bytes.mp3"

    tmp_file = open(tmp_filename, 'wb')
    tmp_file.write(audio_obj)
    tmp_file.close()

    audio = whisper.load_audio(tmp_filename)

    with model_lock:
        result = translate_model.transcribe(audio, task="translate")

    if os.path.exists(tmp_filename):
        os.remove(tmp_filename)

    result["model"] = core_model_name
    result["using_cuda"] = is_cuda
    result["duration_ms"] = (time_ms() - start_time)

    return result


def start():
    uvicorn.run(app, host="0.0.0.0", port=9000, log_level="debug")
