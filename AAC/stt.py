import sounddevice as sd
import numpy as np
import whisper
import tempfile
import wave

# Load a lightweight model (tiny is fastest, small is better)
model = whisper.load_model("tiny")

def record_to_file(filename, duration=3, samplerate=16000):
    print("🎤 Speak now...")
    data = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, dtype="int16")
    sd.wait()
    with wave.open(filename, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(samplerate)
        wf.writeframes(data.tobytes())

def rawish_recognize():
    tmpfile = tempfile.mktemp(suffix=".wav")
    record_to_file(tmpfile, duration=3)

    # Whisper transcription
    result = model.transcribe(tmpfile, language="en", task="transcribe", fp16=False)
    print("Raw-ish output:", result["text"])

if __name__ == "__main__":
    rawish_recognize()
