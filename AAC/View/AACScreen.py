# =========================================================
# Imports
# =========================================================
from kivy.app import App
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.gridlayout import GridLayout
from kivy.uix.textinput import TextInput
from kivy.uix.image import AsyncImage
from kivy.uix.button import ButtonBehavior, Button
from kivy.uix.scrollview import ScrollView
from kivy.uix.label import Label
from kivy.clock import Clock
from kivy.graphics import Color, Rectangle
from kivy.core.window import Window
from kivy.uix.screenmanager import Screen
from kivy.config import Config
from kivy.properties import StringProperty
from kivy.metrics import dp
import os, re, io, time, json, wave, socket, shelve, pygame, requests, tempfile, subprocess, threading, audioop
import speech_recognition as sr
from gtts import gTTS
import spacy
Config.set('graphics', 'resizable', False)
Config.set('graphics', 'width', '1024')
Config.set('graphics', 'height', '600')
Window.size = (1024, 600)
Window.clearcolor = (1, 1, 1, 1)
# Project imports
from Control.Database_helper import Database_helper
from Control.Arasaac_help import Arasaac_help
from View.input_norm import get_suggestions
from pocketsphinx import Pocketsphinx, get_model_path


# =========================================================
# Window / Global Config
# =========================================================


categories = [
    "food", "animals", "clothes", "emotions", "body", "sports", "school", "family",
    "nature", "transport", "weather", "home", "health", "jobs", "colors", "toys"
]

nlp = spacy.load("en_core_web_sm")

# =========================================================
# NLP / Normalization Helpers
# =========================================================
def is_grammatically_valid(sent):
    doc = nlp(sent)
    tokens = [t.text.lower() for t in doc]
    tags   = [t.tag_ for t in doc]

    # ❌ "was" + VB
    for i, t in enumerate(tokens):
        if t == "was" and i+1 < len(tokens) and tags[i+1] == "VB":
            return False

    # ✅ allow "want + NOUN" or "want + VB"
    for i, t in enumerate(tokens):
        if t == "want" and i+1 < len(tokens):
            if tags[i+1] not in {"VB", "NN", "NNS"}:
                return False

    return True

WORD_RE = re.compile(r"[a-zA-Z0-9]+")
def all_tokens(sentence: str):
    return WORD_RE.findall(sentence.lower())

def normalize_input(text: str) -> str:
    """Lightweight spelling fixes/expansions for AAC inputs."""
    replacements = {
        "wa": ["want","was"], "wan": ["want"], "hav": ["have"], "giv": ["give"], 
        "tak": ["take"], "gat": ["get"], "goe": ["go"], "cum": ["come"], "plz": ["please"],
        "wat": ["water"], "mil": ["milk"], "ju": ["juice"], "jus": ["juice"], 
        "bred": ["bread"], "brd": ["bread"], "ric": ["rice"], "frut": ["fruit"], "snak": ["snack"],
        "clo": ["clothes"], "shoo": ["shoe"], "skul": ["school"], "hous": ["house"], "hom": ["home"],
        "hpy": ["happy"], "sadn": ["sad"], "angri": ["angry"], "luv": ["love"],
    }
    return " ".join([replacements.get(tok, [tok])[0] for tok in text.lower().split()])

FUNCTION_WORDS = {
    "i","to","the","a","please","me","can","have","is","am","are","on","in","at","for",
    "of","and","or","it","you","he","she","they","we","this","that","these","those",
    "with","my","your","his","her","their","our","want","need","go","give","get",
    "drink","eat","like","let","let's"
}
def content_tokens(sentence: str):
    return [t for t in WORD_RE.findall(sentence.lower()) if t not in FUNCTION_WORDS]

# =========================================================
# UI Components
# =========================================================
class SuggestionRow(BoxLayout):
    """Row: [horizontal pictos for ALL tokens]  sentence text    [OK]"""
    def __init__(self, sentence: str, ok_callback, **kwargs):
        super().__init__(orientation="horizontal", size_hint=(1, None), height=96, spacing=8, padding=(6, 6), **kwargs)
        self.sentence = sentence

        # Left: horizontal scroller of thumbnails (unlimited)
        self.pic_scroll = ScrollView(size_hint=(None, 1), width=dp(420),
                                     do_scroll_x=True, do_scroll_y=False, bar_width=dp(6))
        self.pic_strip  = BoxLayout(orientation="horizontal", size_hint=(None, 1), spacing=4, padding=(0,0))
        self.pic_strip.bind(minimum_width=self.pic_strip.setter("width"))
        self.pic_scroll.add_widget(self.pic_strip)
        self.add_widget(self.pic_scroll)

        # Middle: sentence text
        lbl = Label(text=sentence, color=(0,0,0,1), halign="left", valign="middle", size_hint=(1,1))
        lbl.bind(size=lambda *_: setattr(lbl, "text_size", lbl.size))
        self.add_widget(lbl)

        # Right: OK
        ok_btn = Button(text="OK", size_hint=(None, 1), width=dp(80),
                        background_color=(0, 0.6, 0, 1), color=(1,1,1,1))
        ok_btn.bind(on_press=lambda *_: ok_callback(sentence))
        self.add_widget(ok_btn)

    def set_pictos(self, items):
        """
        items = list[(label, url_or_empty)] for EVERY token (order preserved).
        If url is "", render a neutral text chip so the token is still visible.
        """
        self.pic_strip.clear_widgets()
        for label, url in items:
            if url:
                cell = BoxLayout(
                    orientation="vertical",
                    size_hint=(None, None),
                    size=(dp(56), self.height - 12),
                    padding=0,
                    spacing=2
                )
                thumb = AsyncImage(
                    source=url,
                    allow_stretch=True,
                    keep_ratio=True,
                    size_hint=(1, 0.75)
                )
                lbl = Label(
                    text=label.capitalize(),
                    color=(0,0,0,1),
                    size_hint=(1, 0.25),
                    font_size=dp(14),
                    halign="center", valign="middle"
                )
                lbl.bind(size=lambda *_: setattr(lbl, "text_size", lbl.size))
                cell.add_widget(thumb)
                cell.add_widget(lbl)
            else:
                chip = BoxLayout(orientation="vertical", size_hint=(None, 1),
                                 width=dp(56), padding=2, spacing=0)
                with chip.canvas.before:
                    Color(0.9, 0.9, 0.9, 1)
                    rect = Rectangle(size=chip.size, pos=chip.pos)
                chip.bind(size=lambda *_: setattr(rect, "size", chip.size),
                          pos=lambda *_: setattr(rect, "pos", chip.pos))
                txt = Label(text=label.capitalize(), color=(0,0,0,1),
                            halign="center", valign="middle")
                txt.bind(size=lambda *_: setattr(txt, "text_size", txt.size))
                chip.add_widget(txt)
                cell = chip
            self.pic_strip.add_widget(cell)

class ClickableImage(ButtonBehavior, BoxLayout):
    def __init__(self, label_text, img_url, callback=None, **kwargs):
        super().__init__(orientation='vertical', size_hint=(None, None), size=(100, 100), **kwargs)
        self.img = AsyncImage(source=img_url, allow_stretch=True, keep_ratio=True, size_hint=(1, 0.8))
        self.label = Label(text=label_text.capitalize(), size_hint=(1, 0.2),
                           color=(0,0,0,1), halign='center')
        self.label.bind(size=self.label.setter('text_size'))
        self.add_widget(self.img)
        self.add_widget(self.label)
        self.callback = callback

    def on_press(self):
        if self.callback:
            self.callback(self.label.text, self.img.source)

class ImageLabelButton_2(ButtonBehavior, BoxLayout):
    image_source = StringProperty('')
    text = StringProperty('')

    def __init__(self, img_source, text, callback=None, **kwargs):
        super().__init__(orientation='vertical', spacing=2, size_hint=(1, None), height=60, **kwargs)
        self.image_source = img_source
        self.text = text
        self.callback = callback

    def on_press(self):
        if self.callback:
            self.callback(None)

# =========================================================
# Main AAC Screen
# =========================================================
class AACScreen(Screen):
    """Main AAC interface: categories, suggestions, voice capture, TTS."""

    def __init__(self, location_label="Default", **kwargs):
        super().__init__(**kwargs)
        self.location_label = location_label
        self._built = False
        Clock.schedule_once(lambda dt: self.build_ui())  

    # -----------------------------------------------------
    # UI BUILDING
    # -----------------------------------------------------
    def build_ui(self):
        if self._built:
            return
        self._built = True

        self.listening = False
        self.r = None
        self.record_thread = None
        self.audio_buffer = None
        self.db_help = Database_helper()
        self.arasaac_func = Arasaac_help()
        self.g = ""

        # --- Root Layout ---
        root = FloatLayout()
        main = BoxLayout(orientation='horizontal')
        root.add_widget(main)
        self.add_widget(root)

        # --- Panels ---
        right_panel = BoxLayout(orientation='vertical', size_hint=(0.1, 1), spacing=5, padding=5)
        left_panel  = BoxLayout(orientation='vertical', size_hint=(0.9, 1))

        main.add_widget(left_panel)
        main.add_widget(right_panel)

        # --- Top Input Bar (Yellow) ---
        top_bar = BoxLayout(size_hint=(1, None), height=60, padding=5, spacing=5)
        with top_bar.canvas.before:
            Color(1, 1, 0.7, 1)
            self.rect_top = Rectangle(size=top_bar.size, pos=top_bar.pos)
        top_bar.bind(size=lambda _, val: setattr(self.rect_top, 'size', val),
                     pos=lambda _, val: setattr(self.rect_top, 'pos', val))

        self.text_input = TextInput(hint_text="Type...", multiline=False, size_hint=(0.85, 1))
        go_btn = Button(text="Go", size_hint=(0.15, 1))
        go_btn.bind(on_press=lambda x: self.on_go_clicked())

        top_bar.add_widget(self.text_input)
        top_bar.add_widget(go_btn)
        left_panel.add_widget(top_bar)

        # --- Result Grid (Light) ---
        scroll = ScrollView(size_hint=(1, None), height=100)
        with scroll.canvas.before:
            Color(1, 1, 0.9, 1)  # light cream
            self.rect_scroll = Rectangle(size=scroll.size, pos=scroll.pos)
        scroll.bind(size=lambda _, val: setattr(self.rect_scroll, 'size', val),
                    pos=lambda _, val: setattr(self.rect_scroll, 'pos', val))

        self.result_grid = GridLayout(rows=1, spacing=5, size_hint_x=None, height=100)
        self.result_grid.bind(minimum_width=self.result_grid.setter('width'))
        scroll.add_widget(self.result_grid)
        left_panel.add_widget(scroll)

        # --- Content Area (switches between Categories and Suggestions) ---
        self.content_area = BoxLayout(orientation='vertical', size_hint=(1, 1))
        with self.content_area.canvas.before:
            Color(0.8, 1, 0.8, 1)  # keep green always
            self.rect_content = Rectangle(size=self.content_area.size, pos=self.content_area.pos)
        self.content_area.bind(size=lambda _, v: setattr(self.rect_content, 'size', v),
                               pos=lambda _, v: setattr(self.rect_content, 'pos', v))
        left_panel.add_widget(self.content_area)

        # Category view (green): ScrollView holding image_grid
        self.image_grid = GridLayout(cols=8, spacing=10, padding=10, size_hint_y=None)
        self.image_grid.bind(minimum_height=self.image_grid.setter('height'))
        with self.image_grid.canvas.before:
            Color(0.8, 1, 0.8, 1)  # green
            self.rect_grid = Rectangle(size=self.image_grid.size, pos=self.image_grid.pos)
        self.image_grid.bind(size=lambda _, val: setattr(self.rect_grid, 'size', val),
                             pos=lambda _, val: setattr(self.rect_grid, 'pos', val))

        self.category_scroll = ScrollView(size_hint=(1, 1), do_scroll_x=False, do_scroll_y=True)
        self.category_scroll.add_widget(self.image_grid)

        # Suggestions view (also green): ScrollView holding sug_list
        self.sug_scroll = ScrollView(size_hint=(1, 1), do_scroll_x=False, do_scroll_y=True)
        with self.sug_scroll.canvas.before:
            Color(0.8, 1, 0.8, 1)   # green
            self.rect_sug = Rectangle(size=self.sug_scroll.size, pos=self.sug_scroll.pos)
        self.sug_scroll.bind(size=lambda _, val: setattr(self.rect_sug, 'size', val),
                             pos=lambda _, val: setattr(self.rect_sug, 'pos', val))

        self.sug_list = BoxLayout(orientation="vertical", spacing=6, padding=6, size_hint_y=None)
        self.sug_list.bind(minimum_height=self.sug_list.setter("height"))
        self.sug_scroll.add_widget(self.sug_list)

        # Initially show categories
        self.content_area.add_widget(self.category_scroll)

        # --- Bottom Taskbar (Black) ---
        self.taskbar = BoxLayout(orientation='horizontal', size_hint=(1, None), height=30)
        with self.taskbar.canvas.before:
            Color(0, 0, 0, 1)
            self.rect_task = Rectangle(size=self.taskbar.size, pos=self.taskbar.pos)
        self.taskbar.bind(size=lambda _, val: setattr(self.rect_task, 'size', val),
                          pos=lambda _, val: setattr(self.rect_task, 'pos', val))

        # Back button (hidden initially)
        self.back_btn = Button(
            text="Back",
            size_hint=(None, 1),
            width=80,
            background_normal='',
            background_down='',
            background_color=(1, 0, 0, 1),
            color=(1, 1, 1, 1)
        )

        self.back_btn.opacity = 0
        self.back_btn.disabled = True
        self.back_btn.bind(on_press=lambda x: self.show_all_categories())

        # Task label (clickable to go home)
        self.task_label = Button(
            text=f"Location: {self.location_label} | Time: {time.strftime('%H:%M:%S')}",
            size_hint=(1, 1),
            background_color=(0, 0, 0, 1),
            color=(1, 1, 1, 1),
            halign="left",
            valign="middle"
        )
        self.task_label.bind(on_press=self.go_to_firstscreen)

        Clock.schedule_interval(self.update_time, 1)

        self.taskbar.add_widget(self.back_btn)
        self.taskbar.add_widget(self.task_label)
        left_panel.add_widget(self.taskbar)

        # --- Right Buttons ---
        buttons_info = [
            ("View/icons/grid.png", "Browse", lambda _: self.show_all_categories()),
            ("View/icons/recommend.png", "Recommend", self.recommend),
            ("View/icons/mic.png", "Speak", self.start_voice_capture),
            ("View/icons/display.png", "Display", self.stop_and_recognize),
            ("View/icons/speak.jpg", "Listen", self.speak_text),
            ("View/icons/try.png", "Try Again", self.clear_all),
            ("View/icons/exit.png", "Exit", self.exit_app)
        ]
        for icon, text, method in buttons_info:
            btn = ImageLabelButton_2(icon, text, callback=method)
            right_panel.add_widget(btn)

        # Initial categories load
        Clock.schedule_once(lambda dt: self.load_categories())

    def on_pre_enter(self, *args):
        if not self._built:
            self.build_ui()
        self.task_label.text = f"Location: {self.location_label} | Time: {time.strftime('%H:%M:%S')}"
        self.text_input.text = ""
        self.result_grid.clear_widgets()
        self.image_grid.clear_widgets()
        self.load_categories()

    # -----------------------------------------------------
    # CATEGORY HANDLING
    # -----------------------------------------------------
    def load_categories(self):
        """Start async fetch of category icons; render on UI thread."""
        def _worker():
            items = []
            for cat in categories:
                url = self.arasaac_func.image_url(cat)
                items.append((cat, url))
            Clock.schedule_once(lambda dt: self._render_categories(items), 0)
        threading.Thread(target=_worker, daemon=True).start()

    def _render_categories(self, items):
        self.image_grid.clear_widgets()
        for cat, url in items:
            self.image_grid.add_widget(ClickableImage(cat, url, self.show_category))
        self.content_area.clear_widgets()
        self.content_area.add_widget(self.category_scroll)
        self.back_btn.opacity = 1
        self.back_btn.disabled = True

    def show_category(self, category, _):
        def _worker():
            pictos = self.arasaac_func.fetch_pictograms(category)
            Clock.schedule_once(lambda dt: self._render_category(category, pictos), 0)
        threading.Thread(target=_worker, daemon=True).start()

    def _render_category(self, category, pictos):
        self.image_grid.clear_widgets()
        for label, url in pictos:
            self.image_grid.add_widget(ClickableImage(label, url, self.add_to_result))
        self.content_area.clear_widgets()
        self.content_area.add_widget(self.category_scroll)
        self.back_btn.opacity = 1
        self.back_btn.disabled = False

    def show_all_categories(self, _=None):
        self.load_categories()
        self.content_area.clear_widgets()
        self.content_area.add_widget(self.category_scroll)
        self.back_btn.opacity = 1
        self.back_btn.disabled = True

    def add_to_result(self, label, url):
        widget = ClickableImage(label, url)
        widget.callback = lambda *_: self.result_grid.remove_widget(widget)
        self.result_grid.add_widget(widget)

    def process_text(self, text_source):
        if not text_source:
            return
        tokens = str(text_source).lower().split()
        self.result_grid.clear_widgets()
        for token in tokens:
            print(f"Token: {token}")
            url = self.arasaac_func.image_url(token)
            if url:
                widget = ClickableImage(token, url)
                widget.callback = lambda *_: self.result_grid.remove_widget(widget)
                self.result_grid.add_widget(widget)

    # -----------------------------------------------------
    # SUGGESTION FLOW
    # -----------------------------------------------------
    def on_go_clicked(self):
        text = self.text_input.text.strip()
        if not text:
            return
        self.sug_list.clear_widgets()
        threading.Thread(target=self._build_suggestions_async, args=(text,), daemon=True).start()

    def _build_suggestions_async(self, text: str):
            # 1. Normalize and tokenize the user's input
            norm_text = normalize_input(text)
            print(f"Normalized input: {norm_text}")
            toks = all_tokens(norm_text)
            print(f"Tokens: {toks}")
            pic_items = []
            for t in toks:
                print(f"User token: {t}")
                url = self.arasaac_func.image_url(t)
                pic_items.append((t, url))
            # 2. Prepare the user's input as the first suggestion row
            user_row = (norm_text, pic_items)

            # 3. Get AI suggestions as before
            try:
                sentences = get_suggestions(norm_text, top_k=12) or []
            except Exception as e:
                print(f"suggester error: {e}")
                sentences = []

            # ...existing filtering logic...
            valid_sentences = []
            for s in sentences:
                if len(all_tokens(s)) < 2:
                    continue
                if re.search(r"\b(was|is|are)\b$", s):
                    continue
                if "was milk" in s or "is milk" in s:
                    continue
                if not is_grammatically_valid(s):
                    continue
                valid_sentences.append(s)
            sentences = valid_sentences

            rows = [user_row]  # Start with the user's input at the top
            for s in sentences:
                toks = all_tokens(s)
                pic_items = []
                for t in toks:
                    url = self.arasaac_func.image_url(t)
                    pic_items.append((t, url))
                rows.append((s, pic_items))
            Clock.schedule_once(lambda dt: self._render_suggestion_rows(rows), 0)
    def _render_suggestion_rows(self, rows):
        self.sug_list.clear_widgets()
        for sentence, items in rows:
            row = SuggestionRow(sentence, ok_callback=self._accept_sentence)
            row.set_pictos(items)
            self.sug_list.add_widget(row)
        self.content_area.clear_widgets()
        self.content_area.add_widget(self.sug_scroll)
        self.back_btn.opacity = 1
        self.back_btn.disabled = False

    def _accept_sentence(self, sentence: str):
        self.result_grid.clear_widgets()
        toks = all_tokens(sentence)
        for t in toks:
            url = self.arasaac_func.image_url(t)
            if url:
                widget = ClickableImage(t, url)
            else:
                widget = ClickableImage(t, "")
                widget.img.source = ""
                with widget.canvas.before:
                    Color(0.9, 0.9, 0.9, 1)
                    bg = Rectangle(size=widget.size, pos=widget.pos)
                widget.bind(size=lambda *_: setattr(bg, "size", widget.size),
                            pos=lambda *_: setattr(bg, "pos", widget.pos))
            widget.callback = lambda *_: self.result_grid.remove_widget(widget)
            self.result_grid.add_widget(widget)
        self.text_input.text = sentence

    def go_to_firstscreen(self, *_):
        app = App.get_running_app()
        app.sm.current = "splash"

    # -----------------------------------------------------
    # VOICE CAPTURE / SPEECH RECOGNITION
    # -----------------------------------------------------
    def start_voice_capture(self, _=None):
        if self.listening:
            print("Already listening...")
            return

        self.r = sr.Recognizer()
        self.listening = True

        msg = "🎤 Speak now... press Display when done."
        print(msg)
        self.text_input.text = msg

        with sr.Microphone() as source:
            # Calibrate for background noise
            self.r.adjust_for_ambient_noise(source, duration=0.5)

            # Force thresholds (important for Raspberry Pi + gibberish)
            self.r.energy_threshold = 250          # tune 100–300
            self.r.dynamic_energy_threshold = False

            print(f"🎤 Listening... (threshold={self.r.energy_threshold})")
            audio = self.r.listen(source, phrase_time_limit=5)  # record up to 5s
            self.captured_audio = audio
            print("✅ Audio captured. Press Display to recognize.")

        self.listening = False


    def stop_and_recognize(self, _=None):
        if not hasattr(self, "captured_audio"):
            print("⚠️ No audio recorded. Please click Speak first.")
            return

        print("Stopping... recognizing now.")
        audio = self.captured_audio
        text = "<gibberish>"

        try:
            # Try Google first
            result = self.r.recognize_google(audio, language="en-IN", show_all=True)

            if isinstance(result, dict) and "alternative" in result:
                # Best guess
                text = result["alternative"][0].get("transcript", "").strip().lower()

                if not text:
                    text = "<gibberish>"

                print("\n🔎 Alternatives:")
                for alt in result["alternative"]:
                    print(" -", alt.get("transcript", ""))

            else:
                text = "<gibberish>"

        except sr.UnknownValueError:
            print("❌ Google couldn’t understand. Falling back to phonetic gibberish...")

            # --- Fallback: pocketsphinx for phonetic-like nonsense ---
            try:
                model_path = get_model_path()
                ps = Pocketsphinx(
                    hmm=model_path + "/en-us",
                    lm=model_path + "/en-us.lm.bin",
                    dict=model_path + "/cmudict-en-us.dict"
                )
                ps.decode(audio.get_wav_data())
                phonetic_text = ps.hypothesis()
                text = phonetic_text if phonetic_text else "<gibberish>"
                print("🎙 Gibberish (phonetic):", text)
            except Exception as e:
                print("Phonetic fallback failed:", e)
                text = "<gibberish>"

        except Exception as e:
            print(f"Speech recognition error: {e}")
            text = "<error>"

        print("✅ Final Output:", text)
        self.text_input.text = text
        self.process_text(text)
    # -----------------------------------------------------
    # TEXT TO SPEECH (TTS)
    # -----------------------------------------------------
    def speak_text(self, _):
        text = self.text_input.text.strip()
        if not text:
            return
        tmp = "tmp.mp3"

        def is_internet_available():
            try:
                requests.get("http://www.google.com", timeout=3)
                return True
            except requests.RequestException:
                return False

        try:
            if is_internet_available():
                print("Internet detected → Using gTTS")
                tts = gTTS(text=text, lang='en', slow=False)
                tts.save(tmp)
                pygame.mixer.init()
                pygame.mixer.music.load(tmp)
                pygame.mixer.music.play()
                while pygame.mixer.music.get_busy():
                    pygame.time.Clock().tick(10)
                pygame.mixer.quit()
                os.remove(tmp)
            else:
                print("No internet → Using eSpeak NG")
                subprocess.run(["espeak-ng", "-v", "en+f3", "-s", "140", "-p", "70", text])
        except Exception as e:
            print(f"TTS error: {e}")
        finally:
            self.db_help.insert(text, self.location_label)

    # -----------------------------------------------------
    # SYSTEM / APP UTILITIES
    # -----------------------------------------------------
    def recommend(self, _):
        def _worker():
            try:
                sentences = self.db_help.recommend(self.location_label)
                first_sentence = self.db_help.retrive_last_inserted(self.location_label)
                print(f"Last inserted sentence: {first_sentence}")

                if first_sentence:
                    sentence = first_sentence[0]
                    if sentence in sentences:
                        sentences.remove(sentence)   # remove existing occurrence
                    sentences.insert(0, sentence)   # add at zeroth index

            except Exception as e:
                print(f"recommendation error: {e}")
                sentences = []

            rows = []
            for s in sentences:
                toks = all_tokens(s)
                pic_items = []
                for t in toks:
                    url = self.arasaac_func.image_url(t)
                    pic_items.append((t, url))
                rows.append((s, pic_items))

            Clock.schedule_once(lambda dt: self._render_suggestion_rows(rows), 0)

        _worker()


    def clear_all(self, _): 
        self.text_input.text = ""
        self.result_grid.clear_widgets()
        self.content_area.clear_widgets()
        self.show_all_categories()

    def exit_app(self, _): 
        App.get_running_app().stop()

    def update_time(self, dt): 
        self.task_label.text = f"Location: {self.location_label} | Time: {time.strftime('%H:%M:%S')}"

    def go_to_firstscreen(self, *_):
        app = App.get_running_app()
        app.sm.current = "splash"
