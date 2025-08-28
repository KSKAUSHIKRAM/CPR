import speech_recognition as sr
from vosk import Model, KaldiRecognizer
import pyaudio, json, os, requests

# Function to check internet
def is_internet_available():
    try:
        requests.get("http://www.google.com", timeout=3)
        return True
    except requests.RequestException:
        return False

# Offline Vosk recognition
def recognize_offline(language="en"):
    if language == "ta":
        model_path = "vosk-model-small-ta-0.22"
    else:
        model_path = "vosk-model-small-en-us-0.15"

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Vosk model not found at {model_path}")

    model = Model(model_path)
    rec = KaldiRecognizer(model, 16000)

    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=8000)
    stream.start_stream()

    print("🎤 [OFFLINE] Speak now...")
    while True:
        data = stream.read(4000, exception_on_overflow=False)
        if rec.AcceptWaveform(data):
            result = json.loads(rec.Result())
            return result.get("text", "")

# Online Google recognition
def recognize_online(language="en-IN"):
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("🎤 [ONLINE] Adjusting for ambient noise...")
        r.adjust_for_ambient_noise(source, duration=1)
        print("Listening now...")
        audio = r.listen(source, timeout=5, phrase_time_limit=10)

    try:
        if language == "ta":
            return r.recognize_google(audio, language="ta-IN")
        else:
            return r.recognize_google(audio, language="en-IN")
    except sr.UnknownValueError:
        return "[Unclear Speech]"
    except sr.RequestError:
        return "[Google API Error]"

# Main hybrid recognizer
def hybrid_speech_recognition(language="en"):
    if is_internet_available():
        print("✅ Internet available → Using Google Speech Recognition")
        return recognize_online(language)
    else:
        print("⚠️ No internet → Using Vosk offline recognition")
        return recognize_offline(language)

# Test run
if __name__ == "__main__":
    lang_choice = input("Enter language (en/ta): ").strip()
    text = hybrid_speech_recognition(lang_choice)
    print("📝 Recognized text:", text)
