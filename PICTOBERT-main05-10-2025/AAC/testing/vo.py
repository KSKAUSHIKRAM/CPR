from vosk import Model, KaldiRecognizer
import sounddevice as sd
import json

model = Model("vosk-model-small-en-us-0.15")
rec = KaldiRecognizer(model, 16000)

def callback(indata, frames, time, status):
    if status:
        print(status)

    # Convert numpy array to raw bytes
    if rec.AcceptWaveform(indata.tobytes()):
        print(json.loads(rec.Result()))
    else:
        print(json.loads(rec.PartialResult()))


with sd.RawInputStream(samplerate=16000, blocksize=8000, dtype='int16',
                       channels=1, callback=callback):
    print("🎤 Speak into the microphone...")
    import time
    while True:
        time.sleep(0.1)
