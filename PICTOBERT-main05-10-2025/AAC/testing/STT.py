import speech_recognition as sr
from vosk import Model, KaldiRecognizer
import json
import sounddevice as sd

# Load vosk fallback model (phoneme/small one is faster)
vosk_model = Model("model-small-en-us")

def recognize_speech_google_then_vosk():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("🎤 Speak now...")
        r.adjust_for_ambient_noise(source, duration=1)
        audio = r.listen(source)

    try:
        result = r.recognize_google(audio, language="en-IN", show_all=True)
        if isinstance(result, dict) and "alternative" in result:
            text = result["alternative"][0].get("transcript", "").strip().lower()
            return text
        else:
            raise sr.UnknownValueError

    except sr.UnknownValueError:
        print("⚠️ Google failed, trying Vosk fallback...")

        # Convert sr.AudioData to raw PCM
        pcm_data = audio.get_raw_data(convert_rate=16000, convert_width=2)
        rec = KaldiRecognizer(vosk_model, 16000)
        rec.AcceptWaveform(pcm_data)
        result_json = json.loads(rec.Result())
        text = result_json.get("text", "").strip()

        if text == "":
            text = "<raw-sound>"

        return text
