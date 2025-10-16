# aac_screen.py
# =========================================================
# Imports
# =========================================================
from kivy.config import Config
from kivy.core.window import Window
Config.set('graphics', 'fullscreen', 'auto')  # Add this line
Config.set('graphics', 'borderless', '1')
Config.set('graphics', 'resizable', False)
Config.set('graphics', 'width', '1024')
Config.set('graphics', 'height', '600')
Window.clearcolor = (1, 1, 1, 1)
Config.write()
#Window.size = (1024, 600)

from difflib import SequenceMatcher
import chardet
import os
import sys
import re
import time
import threading
import subprocess
import csv
from PIL import Image, ImageDraw
from PIL import Image, ImageDraw

ICON_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "icons"))
if not os.path.exists(ICON_DIR):
    ICON_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "View", "icons"))
print("[INIT] ICON_DIR =", ICON_DIR)

PLACEHOLDER_PATH = os.path.join(ICON_DIR, "placeholder.png")
if not os.path.exists(PLACEHOLDER_PATH):
    os.makedirs(ICON_DIR, exist_ok=True)
    img = Image.new("RGB", (128, 128), (230, 230, 230))
    draw = ImageDraw.Draw(img)
    draw.text((40, 55), "N/A", fill=(0, 0, 0))
    img.save(PLACEHOLDER_PATH)
    print("[AUTO] Created placeholder at", PLACEHOLDER_PATH)
    
import requests
import pygame
import speech_recognition as sr
import spacy
from gtts import gTTS
from pocketsphinx import Pocketsphinx, get_model_path
from threading import Thread
from kivy.app import App
from kivy.clock import Clock
from kivy.core.audio import SoundLoader
from kivy.core.window import Window
from kivy.graphics import Color, Rectangle
from kivy.metrics import dp
from kivy.properties import StringProperty
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.button import ButtonBehavior
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.gridlayout import GridLayout
from kivy.uix.image import AsyncImage
from kivy.uix.label import Label
from kivy.uix.popup import Popup
from kivy.uix.scrollview import ScrollView
from kivy.uix.screenmanager import Screen
from kivy.uix.textinput import TextInput

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
# Project imports (your existing modules)
from Control.Database_helper import Database_helper
from Model.Arasaac_helper import Arasaac_helper

from View.input_norm import get_suggestions


# Kivy / Window config (as you had)
#Config.set('graphics', 'borderless', '1')   # no OS window frame
#Config.set('graphics', 'resizable', False)
#Config.set('graphics', 'width', '1024')
#Config.set('graphics', 'height', '600')
#Window.size = (1024, 600)
#Window.clearcolor = (1, 1, 1, 1)

# =========================================================
# Globals / NLP
# =========================================================
categories = [
    "food", "animals", "clothes", "emotions", "body", "sports", "school", "family",
    "nature", "transport", "weather", "home", "health", "jobs", "colors", "toys"
]
print("Loaded AACScreen from:", __file__)

nlp = spacy.load("en_core_web_sm")

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

from difflib import get_close_matches



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

class LoadingPopup(Popup):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.title = "Loading"
        self.size_hint = (None, None)
        self.size = (200, 150)
        self.auto_dismiss = False

        # Content
        box = BoxLayout(orientation='vertical', padding=20)
        box.add_widget(Label(text="Loading... Please wait"))
        self.content = box

class SuggestionRow(BoxLayout):
    def __init__(self, sentence: str, ok_callback, bg_color=(1,1,1,1), **kwargs):
        super().__init__(orientation="horizontal", size_hint=(1, None), height=96, spacing=8, padding=(6,6), **kwargs)
        self.sentence = sentence

        # Background color
        with self.canvas.before:
            from kivy.graphics import Color, Rectangle
            self.bg_color = Color(*bg_color)
            self.bg_rect = Rectangle(size=self.size, pos=self.pos)
        self.bind(size=self._update_rect, pos=self._update_rect)

        # Left: horizontal scroller of thumbnails
        self.pic_scroll = ScrollView(size_hint=(None, 1), width=dp(600),
                                     do_scroll_x=True, do_scroll_y=False, bar_width=dp(10))
        self.pic_strip  = BoxLayout(orientation="horizontal", size_hint=(None, 1), spacing=dp(10), padding=(dp(10), 0) )  # left/right padding inside the strip
        self.pic_strip.bind(minimum_width=self.pic_strip.setter("width"))
        self.pic_scroll.add_widget(self.pic_strip)
        self.add_widget(self.pic_scroll)

        # Add spacer between left (pictos) and middle (sentence)
        from kivy.uix.widget import Widget
        self.add_widget(Widget(size_hint_x=None, width=dp(40)))  # 🔸 20px horizontal space

        # Middle: sentence text
        lbl = Label(
            text=sentence,
            color=(0, 0, 0, 1),
            halign="left",
            valign="middle",
            size_hint=(1, 1),
            font_name="Roboto-Bold.ttf",
            font_size="25sp",
            padding=(dp(40), 0)   # ✅ add left padding (40px)
        )
        lbl.bind(size=lambda *_: setattr(lbl, "text_size", lbl.size))
        self.add_widget(lbl)


        # Right: OK button
        ok_btn = Button(text="OK", size_hint=(None, 1), width=dp(80),
                        background_color=(0, 0.6, 0, 1), color=(1,1,1,1),font_name="Roboto-Bold.ttf", font_size="25sp")
        ok_btn.bind(on_press=lambda *_: ok_callback(sentence))
        self.add_widget(ok_btn)

    def _update_rect(self, *args):
        self.bg_rect.pos = self.pos
        self.bg_rect.size = self.size


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
                    size=(dp(64), self.height - 6),   # slightly larger overall height
                    padding=(0, 2),
                    spacing=3
                )
                thumb = AsyncImage(
                    source=url,
                    allow_stretch=True,
                    keep_ratio=True,
                    size_hint=(1, 0.70)    # 70% of height reserved for image
                )
                lbl = Label(
                    text=label.capitalize(),
                    font_name="Roboto-Bold.ttf",
                    color=(0, 0, 0, 1),
                    size_hint=(1, 0.30),   # 30% reserved for label
                    font_size='22sp',
                    halign="center",
                    valign="top"           # anchor label at top to prevent clipping
                )
                lbl.bind(size=lambda *_: setattr(lbl, "text_size", (lbl.width, None)))
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
        self.img = AsyncImage(source=img_url or "", allow_stretch=True, keep_ratio=True, size_hint=(1, 0.8))
        self.label = Label(text=label_text.capitalize(), size_hint=(1, 0.2),
                           color=(0,0,0,1), halign='center')
        self.label.bind(size=self.label.setter('text_size'))
        self.add_widget(self.img)
        self.add_widget(self.label)
        self.callback = callback

    def on_press(self):
        if self.callback:
            # provide label text and image source
            self.callback(self.label.text, self.img.source)
            
class ImageLabelButton_2(ButtonBehavior, BoxLayout):
    image_source = StringProperty('')
    text = StringProperty('')

    def __init__(self, img_source, text, callback=None,height=500,font_size=10, **kwargs):
        super().__init__(orientation='vertical', spacing=2, size_hint=(1,0.6), **kwargs)
        self.height = height  # Set height here
        self.image_source = img_source
        self.text = text
        self.callback = callback
        self.clear_widgets()

        # Add the image (icon)
        self.img = AsyncImage(
            source=img_source,
            allow_stretch=True,
            keep_ratio=True,
            size_hint=(1, 0.7)  # 70% of button height
        )
        self.add_widget(self.img)
        self.lbl = Label(
            text=text,
            font_name="Roboto-Bold.ttf",
            size_hint=(1, 0.3),  # 30% of button height
            color=(0, 0, 0, 1),
            halign='center',
            valign='middle',
            font_size='25sp')
        self.lbl.bind(size=lambda *_: setattr(self.lbl, "text_size", self.lbl.size))
        self.add_widget(self.lbl)

    def on_press(self):
        if self.callback:
            # callback receives the widget (or None) as you had earlier
            self.callback(None)

class ImageButtonYN(ButtonBehavior, BoxLayout):
    """Dedicated image button for YES/NO dataset grid."""
    def __init__(self, label_text, img_url, callback=None, show_red_border=False, **kwargs):
        super().__init__(orientation='vertical',
                         size_hint=(None, None),
                         size=(150, 150),
                         spacing=4,
                         padding=4,
                         **kwargs)
        self.callback = callback
        self.label_text = label_text
        self.img_url = img_url
        self.show_red_border = show_red_border  # ✅ renamed from show_red_x

        # ---- Image widget ----
        try:
            if img_url and img_url.startswith("http"):
                img_source = img_url
            elif img_url and os.path.exists(img_url):
                img_source = img_url
            else:
                img_source = os.path.abspath(PLACEHOLDER_PATH)


            self.img = AsyncImage(
                source=img_source,
                allow_stretch=True,
                keep_ratio=True,
                size_hint=(1, 0.8)
            )
        except Exception as e:
            print(f"[IMG LOAD ERROR] {label_text}: {e}")
            self.img = AsyncImage(source="View/icons/placeholder.png")

        self.add_widget(self.img)

        # ✅ Red border (for NO grid)
        if self.show_red_border:
            with self.img.canvas.after:
                from kivy.graphics import Color, Line
                Color(1, 0, 0, 1)  # Red
                self.border = Line(rectangle=(self.img.x, self.img.y, self.img.width, self.img.height), width=2)
            self.img.bind(size=self._update_border, pos=self._update_border)

        # ---- Label widget ----
        # ---- Label widget ----
        self.lbl = Label(
            text=label_text.capitalize(),
            font_name="Roboto-Bold.ttf",
            color=(0, 0, 0, 1),
            halign='center',
            valign='middle',
            font_size='22sp',
            size_hint_y=None,     # ❗ make height manual, not proportional
        )
        # Bind to recalculate height based on text content
        self.lbl.bind(
            width=lambda *_: setattr(self.lbl, 'text_size', (self.lbl.width, None)),
            texture_size=lambda *_: setattr(self.lbl, 'height', self.lbl.texture_size[1] + 10)
        )
        self.add_widget(self.lbl)


    def _update_border(self, *_):
        """Ensure the red border follows image resize."""
        if hasattr(self, 'border'):
            self.border.rectangle = (self.img.x, self.img.y, self.img.width, self.img.height)

    def on_press(self):
        if self.callback:
            self.callback(self.label_text)
class SafeTextInput(TextInput):
    """Prevents duplicate keypresses from touchscreens by ignoring rapid repeats."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._last_key_time = 0
        self._last_key_char = None

    def insert_text(self, substring, from_undo=False):
        now = time.time()
        # Ignore same char if repeated too quickly (<0.08 s)
        if substring == self._last_key_char and (now - self._last_key_time) < 0.08:
            return
        self._last_key_char = substring
        self._last_key_time = now
        super().insert_text(substring, from_undo)


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
        self.arasaac_func = Arasaac_helper()

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

        self.text_input = SafeTextInput(
            hint_text="Type...",
            multiline=False,
            size_hint=(0.85, 1),
            font_name="Roboto-Bold.ttf",  # path to bold font
            font_size='30sp',   # adjust size as needed
            foreground_color=(0, 0, 0, 1),  # text color (black)
            hint_text_color=(0.5, 0.5, 0.5, 1)  # lighter gray hint
        )


        go_btn = Button(text="Go", size_hint=(0.15, 1),font_name="Roboto-Bold.ttf",  font_size='30sp')
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
        self.image_grid = GridLayout(cols=10,spacing=10, padding=10, size_hint_y=None)
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
            ("View/icons/grid.png", "Browse", lambda _: self.load_categories()),
            ("View/icons/recommend.png", "Recommend", self.recommend),
            ("View/icons/mic.png", "Speak", self.start_voice_capture),
            ("View/icons/display.png", "Display", self.stop_and_recognize),
            ("View/icons/speak.jpg", "Listen", self.speak_text),
            ("View/icons/try.png", "Try Again", self.clear_all),
            ("View/icons/exit.png", "Exit", self.exit_app)
        ]
        for icon, text, method in buttons_info:
            btn = ImageLabelButton_2(icon, text, callback=method,height=100)
            right_panel.add_widget(btn)

        # Initial categories load
        Clock.schedule_once(lambda dt: self.load_categories())

    def autocorrect_with_dataset(self, text):
        """
        Performs lightweight spelling correction using dataset vocabulary.
        Corrects tokens based on closest label or sentence word in dataset.
        """
        if not text:
            return text

        words = text.split()
        dataset_vocab = set()

        # Collect known words from labels + yes_sentences
        try:
            import csv, os
            dataset_path = os.path.join(os.path.dirname(__file__), "../dataset.csv")
            with open(dataset_path, "r", encoding="utf-8", errors="replace") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row.get("label"):
                        dataset_vocab.add(row["label"].lower())
                    if row.get("yes_sentence"):
                        dataset_vocab.update(row["yes_sentence"].lower().split())
        except Exception as e:
            print(f"[AUTOCORRECT] Failed to load dataset: {e}")

        corrected = []
        for w in words:
            if w in dataset_vocab:
                corrected.append(w)
            else:
                match = get_close_matches(w, dataset_vocab, n=1, cutoff=0.75)
                corrected.append(match[0] if match else w)

        corrected_text = " ".join(corrected)
        print(f"[AUTOCORRECT] '{text}' → '{corrected_text}'")
        return corrected_text

    
    
    def show_category_sentences_as_suggestions(self, category):
        """
        Display all sentences from a given category as pictogram suggestion rows.
        Each row shows pictograms + sentence + OK button (same as SuggestionRow).
        """
        entries = self.get_sentences_for_category(category)
        if not entries:
            self.display_message(f"No sentences found for category '{category}'.")
            return  # only return if truly empty

        # Prepare rows: (sentence, pictogram_items)
        rows = []
        for item in entries:
            sentence = item.get("yes_sentence", "").strip()
            if not sentence:
                continue
            tokens = all_tokens(sentence)
            pictos = [(t, self.arasaac_func.get_arasaac_image_url(t)) for t in tokens]
            rows.append((sentence, pictos))

        # Clear old UI and show suggestion-style layout
        self.sug_list.clear_widgets()
        for sentence, pictos in rows:
            row = SuggestionRow(sentence.capitalize(),
                                ok_callback=self._accept_sentence,
                                bg_color=(0.8, 1, 0.8, 1))  # greenish background
            row.set_pictos(pictos)
            self.sug_list.add_widget(row)

        self.content_area.clear_widgets()
        self.content_area.add_widget(self.sug_scroll)
        self.back_btn.opacity = 1
        self.back_btn.disabled = False
        print(f"[UI] Displayed {len(rows)} sentences for category '{category}'.")


    def on_pre_enter(self, *args):
        """Called automatically when AACScreen becomes visible."""
        if not self._built:
            self.build_ui()

        # Update header label
        self.task_label.text = f"Location: {self.location_label} | Time: {time.strftime('%H:%M:%S')}"

        # Reset text and result area
        self.text_input.text = ""
        self.result_grid.clear_widgets()

        # ✅ Automatically load YES/NO pictogram dataset
        #self.load_categories()

    def detect_category_from_text(self, recognized_text):
        """
        Detect category name from dataset.csv.
        Case-insensitive, tolerant to small typos, and robust against column name variations.
        Returns the detected category name (in lowercase) or None.
        """
        import csv, os
        from difflib import SequenceMatcher

        if not recognized_text:
            return None
        recognized_text = recognized_text.strip().lower()

        # --- Locate dataset ---
        dataset_path = os.path.join(os.path.dirname(__file__), "../dataset.csv")
        categories = set()

        try:
            with open(dataset_path, "r", encoding="utf-8", errors="replace") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    cat = (row.get("Category") or row.get("category") or "").strip().lower()
                    if cat:
                        categories.add(cat)
        except Exception as e:
            print(f"[CATEGORY LOAD ERROR] {e}")
            return None

        if not categories:
            print("[CATEGORY] ❌ No categories found in dataset.")
            return None

        # --- Direct / substring matching ---
        for cat in categories:
            if recognized_text == cat or recognized_text in cat or cat in recognized_text:
                print(f"[CATEGORY] ✅ Direct or substring match → '{cat}'")
                return cat

        # --- Fuzzy similarity matching (only if input is long enough) ---
        if len(recognized_text) > 4:
            best_cat, best_score = None, 0.0
            for cat in categories:
                score = SequenceMatcher(None, recognized_text, cat).ratio()
                if score > best_score:
                    best_cat, best_score = cat, score

            if best_cat and best_score >= 0.85:  # ⬆️ much stricter
                print(f"[CATEGORY] ✅ Fuzzy match → '{best_cat}' (score={best_score:.2f})")
                return best_cat





    def get_sentences_for_category(self, category_name):
        """Return all entries for the given category (case-insensitive)."""
        import csv, os
        dataset_path = os.path.join(os.path.dirname(__file__), "../dataset.csv")
        results = []
        if not category_name:
            return results

        try:
            with open(dataset_path, "r", encoding="utf-8", errors="replace") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    cat = (row.get("Category") or row.get("category") or "").strip().lower()
                    if cat == category_name.strip().lower():
                        results.append({
                            "label": (row.get("label") or "").strip(),
                            "yes_sentence": (row.get("yes_sentence") or "").strip()
                        })
            print(f"[DATASET] Found {len(results)} entries for '{category_name.lower()}'")
        except Exception as e:
            print(f"[DATASET ERROR] {e}")
        return results



    def detect_label_and_category_from_sentence(self, sentence):
        """
        Given a sentence like 'I want apple',
        detect which label (e.g., 'apple') exists in dataset.csv
        and return (label, category).
        """
        import csv, os
        from difflib import SequenceMatcher

        dataset_path = os.path.join(os.path.dirname(__file__), "../dataset.csv")
        sentence = (sentence or "").strip().lower()
        if not sentence:
            return None, None

        best_label, best_category, best_score = None, None, 0.0
        try:
            with open(dataset_path, "r", encoding="utf-8", errors="replace") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    label = (row.get("label") or "").strip().lower()
                    category = (row.get("Category") or row.get("category") or "").strip()
                    yes_sentence = (row.get("yes_sentence") or "").strip().lower()
                    if not label or not category:
                        continue

                    # Direct containment
                    # Exact whole-word match using regex
                    pattern = r"\b" + re.escape(label) + r"\b"
                    if re.search(pattern, sentence):
                        print(f"[LOOKUP] Direct label match: '{label}' in '{sentence}' → {category}")
                        return label, category

                    # Fuzzy similarity (fallback)
                    sim = SequenceMatcher(None, sentence, yes_sentence).ratio()
                    if sim > best_score:
                        best_score = sim
                        best_label, best_category = label, category
        except Exception as e:
            print(f"[LOOKUP ERROR] {e}")

        if best_label and best_category and best_score >= 0.6:
            print(f"[LOOKUP] Fuzzy match '{best_label}' → {best_category} ({best_score:.2f})")
            return best_label, best_category

        print(f"[LOOKUP] No match found for sentence '{sentence}'")
        return None, None


    def load_categories(self):
        """Display category folders based on dataset.csv."""
        import csv
        from collections import defaultdict

        candidate_paths = [
            os.path.join(os.path.dirname(__file__), "dataset.csv"),
            os.path.join(os.path.dirname(__file__), "..", "dataset.csv"),
            "/home/i7/Downloads/PICTOBERT-main05-10-2025/AAC/dataset.csv",
        ]
        csv_path = next((p for p in candidate_paths if os.path.exists(p)), None)
        if not csv_path:
            print("[ERROR] dataset.csv not found.")
            return

        print(f"[INFO] ✅ Using dataset.csv from: {csv_path}")
        self.content_area.clear_widgets()
        self.image_grid.clear_widgets()

        grouped = defaultdict(list)
        with open(csv_path, "r", encoding="utf-8", errors="replace") as f:
            reader = csv.DictReader(f)
            for row in reader:
                row = {k.strip().lower(): (v or "").strip() for k, v in row.items()}
                label = row.get("label", "")
                yes_sentence = row.get("yes_sentence", label)
                category = row.get("category", "Uncategorized")
                if label:
                    grouped[category].append({"label": label, "yes_sentence": yes_sentence})

        print(f"[INFO] Categories found: {list(grouped.keys())}")

        for category, items in grouped.items():
            icon_file = os.path.join(ICON_DIR, f"{category.lower().replace(' ', '_')}.png")
            if not os.path.exists(icon_file):
                icon_file = os.path.join(ICON_DIR, "folder.png")
                if not os.path.exists(icon_file):
                    icon_file = PLACEHOLDER_PATH

            btn = ImageButtonYN(
                category.capitalize(),
                os.path.abspath(icon_file),
                callback=lambda c=category, it=items: self.open_category_folder(c, it)
            )
            self.image_grid.add_widget(btn)

        self.content_area.add_widget(self.category_scroll)
        self.back_btn.opacity = 0
        self.back_btn.disabled = True





    def open_category_folder(self, category, entries):
        """
        Open a category folder, fetch pictograms from local or ARASAAC,
        and speak the yes_sentence when a pictogram is clicked.
        """
        from kivy.clock import Clock
        import threading

        # --- Clear previous content and prepare grid layout ---
        self.content_area.clear_widgets()
        grid = GridLayout(cols=10, spacing=10, padding=10, size_hint_y=None)
        grid.bind(minimum_height=grid.setter("height"))

        scroll = ScrollView(size_hint=(1, 1), do_scroll_y=True)
        scroll.add_widget(grid)
        self.content_area.add_widget(scroll)

        def load_items():
            """Background loader to avoid blocking UI"""
            for item in entries:
                label = item.get("label", "").strip()
                sentence = item.get("yes_sentence", label)
                if not label:
                    continue

                # ✅ hybrid cache (ARASAAC helper handles memory/disk fallback)
                tex, path = self.arasaac_func.get_cached_image(label, PLACEHOLDER_PATH)

                # add button to UI thread
                def add_btn(dt, lbl=label, sent=sentence, tex=tex, path=path):
                    try:
                        if tex:
                            from kivy.uix.image import Image
                            img_widget = Image(texture=tex, allow_stretch=True, keep_ratio=True)
                            container = BoxLayout(orientation='vertical',
                                                  size_hint=(None, None),
                                                  size=(150, 150))
                            container.add_widget(img_widget)
                            lbl_widget = Label(
                                text=lbl.capitalize(),
                                color=(0, 0, 0, 1),
                                size_hint=(1, None),
                                height=30,
                                font_name="Roboto-Bold.ttf",
                                font_size="20sp"
                            )
                            container.add_widget(lbl_widget)
                            container.bind(on_touch_down=lambda _, t, s=sent: self.speak_text_custom(s) if container.collide_point(*t.pos) else None)
                            grid.add_widget(container)
                        else:
                            btn = ImageButtonYN(
                                label_text=lbl.capitalize(),
                                img_url=path,
                                callback=lambda _: self.speak_text_custom(sent)
                            )
                            grid.add_widget(btn)
                    except Exception as e:
                        print(f"[UI ERROR] Could not add {lbl}: {e}")

                Clock.schedule_once(add_btn, 0)

            print(f"[INFO] Loaded {len(entries)} pictograms for category '{category}'")

        # --- Run pictogram loading in background thread ---
        threading.Thread(target=load_items, daemon=True).start()

        # --- Enable the Back button ---
        self.back_btn.opacity = 1
        self.back_btn.disabled = False
        self.back_btn.bind(on_press=lambda _: self.load_categories())





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
        self.g += label + " "
        print("G:", self.g)
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
            url = self.arasaac_func.get_arasaac_image_url(token)
            if url:
                widget = ClickableImage(token, url)
                widget.callback = lambda *_: self.result_grid.remove_widget(widget)
                self.result_grid.add_widget(widget)

    # -----------------------------------------------------
    # SUGGESTION RENDERER (sections)
    # -----------------------------------------------------
    def _render_suggestion_rows_sections(self, recent_rows, location_matched_rows, location_unmatched_rows,time_unmatched_rows):
        """
        Renders suggestion rows grouped into sections:
        - Recently Used
        - Location Matched
        - Location Unmatched
        """
        self.sug_list.clear_widgets()

        def add_section(title, rows, bg_color):
            if not rows:
                return
            # Section Label
            lbl = Label(
                text=title,
                size_hint=(1, None),
                height=32,
                color=(0, 0, 1, 1),
                halign="left", valign="middle"
            )
            lbl.bind(size=lambda *_: setattr(lbl, "text_size", lbl.size))
            self.sug_list.add_widget(lbl)

            # Rows with custom background
            for sentence, items in rows:
                row = SuggestionRow(sentence.capitalize(),
                                    ok_callback=self._accept_sentence,
                                    bg_color=bg_color)
                row.set_pictos(items)
                self.sug_list.add_widget(row)

        # Add sections with different bg colors
        add_section("Recently Used", recent_rows, (0.9, 0.9, 0.9, 1))   # light gray
        add_section("Location Matched", location_matched_rows, (1, 0.5, 0.8, 1))  # light green
        add_section("Location Unmatched", location_unmatched_rows, (1, 0.76, 0.71, 1))  # light orange
        add_section("Time Unmatched", time_unmatched_rows, (0.7, 0.6, 1, 1))  # light blue
        self.content_area.clear_widgets() 
        self.content_area.add_widget(self.sug_scroll) 
        self.back_btn.opacity = 1 
        self.back_btn.disabled = False
   
   
   
         # -----------------------------------------------------
    # Go Button (Category → Label → Exact → Tokenized pictograms)
    # -----------------------------------------------------
       # -----------------------------------------------------
    # Go Button: Unified and Stable AAC Input Pipeline
    # -----------------------------------------------------
    def on_go_clicked(self, _=None):
        """
        Robust AAC 'Go' logic:
        1️⃣ Exact category match → show all sentences of that category.
        2️⃣ Exact label match → show its sentence first, then others in same category.
        3️⃣ Exact sentence match → show it first, then others in same category.
        4️⃣ Else → tokenize and fetch pictograms directly.
        """

        text = (self.text_input.text or "").strip()
        if not text:
            return

        # STEP 0 — Normalize & Conditional Autocorrect
        norm_text = normalize_input(text)

        # Apply autocorrect only if input has more than one word
        if len(norm_text.split()) > 1:
            try:
                norm_text = self.autocorrect_with_dataset(norm_text)
            except AttributeError:
                norm_text = self.autocorrect_with_dataset(norm_text)
        else:
            print("[GO] Single-word input → Skipping autocorrect")

        norm_text = norm_text.lower().strip()
        print(f"[GO] Input: '{text}' → Normalized: '{norm_text}'")
        # ✅ Update textbox to show normalized/corrected text
        self.text_input.text = norm_text

        # ==========================================================
        # STEP 0.5 — Early pictogram retrieval + passive category detection
        # ==========================================================
        tokens = all_tokens(norm_text)
        if tokens:
            # Display pictograms for immediate feedback
            self.result_grid.clear_widgets()
            for token in tokens:
                url = self.arasaac_func.get_arasaac_image_url(token)
                if url:
                    widget = ClickableImage(token, url, callback=lambda *_: self.result_grid.remove_widget(widget))
                    self.result_grid.add_widget(widget)

            # Passive category detection (just log it, don't override logic)
            detected_category = self.detect_category_from_text(norm_text)
            if detected_category:
                print(f"[GO] Detected possible category → '{detected_category}' (will be handled in Step 1)")


        # Load dataset once
        dataset_path = os.path.join(os.path.dirname(__file__), "../dataset.csv")
        dataset_rows = []
        try:
            with open(dataset_path, "r", encoding="utf-8", errors="replace") as f:
                reader = csv.DictReader(f)
                dataset_rows = [row for row in reader if row.get("yes_sentence")]
        except Exception as e:
            print(f"[DATASET LOAD ERROR] {e}")

        # Extract all categories and labels
        all_categories = {r.get("category", "").lower() for r in dataset_rows if r.get("category")}
        all_labels = {r.get("label", "").lower() for r in dataset_rows if r.get("label")}

        # ==========================================================
        # STEP 1 — CATEGORY MATCH (case-insensitive)
        # ==========================================================
        detected_category = self.detect_category_from_text(norm_text)

        if detected_category:
            print(f"[GO] Detected possible category → '{detected_category}' (will be handled in Step 1)")

            # ✅ Fetch all category sentences
            entries = self.get_sentences_for_category(detected_category)
            if not entries:
                print(f"[GO] No entries found for category '{detected_category}'.")
                self.display_message(f"No sentences found for '{detected_category}'.")
                return

            # ✅ Show results (with pictograms + OK buttons)
            self.show_category_sentences_as_suggestions(detected_category)
            return  # <-- stops execution here, prevents going to tokenization


            # ✅ Prepare (sentence, pictograms) pairs
            rows = []
            for item in entries:
                sentence = item.get("yes_sentence", "").strip()
                if not sentence:
                    continue
                # Tokenize sentence and get pictograms for each word
                tokens = sentence.split()
                pictos = [(t, self.arasaac_func.get_arasaac_image_url(t) or PLACEHOLDER_PATH) for t in tokens]
                rows.append((sentence, pictos))

            # ✅ Clear old UI before showing new results
            self.sug_list.clear_widgets()
            for sentence, pictos in rows:
                row = SuggestionRow(sentence.capitalize(),
                                    ok_callback=self._accept_sentence,
                                    bg_color=(0.8, 1, 0.8, 1))  # soft green background
                row.set_pictos(pictos)
                self.sug_list.add_widget(row)

            # ✅ Display inside the scrollable suggestion area
            self.content_area.clear_widgets()
            self.content_area.add_widget(self.sug_scroll)
            self.back_btn.opacity = 1
            self.back_btn.disabled = False

            print(f"[UI] Displayed {len(rows)} category-based suggestions for '{category_name}'.")
            return



        # ==========================================================
        # STEP 2 — LABEL → CATEGORY mapping  (case-insensitive)
        # ==========================================================
        if not detected_category:  # only if no valid category found
            matched_label, matched_category = None, None

            for row in dataset_rows:
                lbl = (row.get("label") or "").strip()
                cat = (row.get("category") or row.get("Category") or "").strip()
                if lbl.lower() == norm_text.lower():
                    matched_label, matched_category = lbl, cat
                    break

            if matched_category:
                print(f"[GO] Label '{matched_label}' belongs to category '{matched_category}'")
                entries = self.get_sentences_for_category(matched_category.lower())
                if not entries:
                    self.display_message(f"No sentences found for category '{matched_category}'.")
                    return

                # ✅ Show all sentences from that category with pictograms + OK
                self.show_category_sentences_as_suggestions(matched_category.lower())
                return




        # ==========================================================
        # STEP 3 — SENTENCE → CATEGORY mapping (exact sentence match, prioritize matched)
        # ==========================================================
        if not detected_category and not matched_category:
            matched_sentence, matched_category = None, None

            # --- Find the exact sentence and its category ---
            for row in dataset_rows:
                sent = (row.get("yes_sentence") or "").strip()
                cat = (row.get("category") or row.get("Category") or "").strip()
                if sent.lower() == norm_text.lower():   # exact match (case-insensitive)
                    matched_sentence, matched_category = sent, cat
                    break

            if matched_category:
                print(f"[GO] Sentence match → '{matched_sentence}' → Category '{matched_category}'")

                # --- Fetch all sentences for that category ---
                entries = self.get_sentences_for_category(matched_category.lower())
                if not entries:
                    self.display_message(f"No sentences found for category '{matched_category}'.")
                    return

                # --- Reorder so matched sentence appears first ---
                primary = [e for e in entries if e.get("yes_sentence", "").strip().lower() == norm_text.lower()]
                others = [e for e in entries if e not in primary]
                ordered_entries = primary + others

                # --- Display sentences with pictograms + OK buttons ---
                rows = []
                for item in ordered_entries:
                    sentence = item.get("yes_sentence", "").strip()
                    if not sentence:
                        continue
                    tokens = sentence.split()
                    pictos = [(t, self.arasaac_func.get_arasaac_image_url(t) or PLACEHOLDER_PATH) for t in tokens]
                    rows.append((sentence, pictos))

                self.sug_list.clear_widgets()
                for sentence, pictos in rows:
                    row = SuggestionRow(sentence.capitalize(),
                                        ok_callback=self._accept_sentence,
                                        bg_color=(0.8, 1, 0.8, 1))
                    row.set_pictos(pictos)
                    self.sug_list.add_widget(row)

                self.content_area.clear_widgets()
                self.content_area.add_widget(self.sug_scroll)
                self.back_btn.opacity = 1
                self.back_btn.disabled = False

                print(f"[UI] Displayed {len(rows)} category-based suggestions for '{matched_category}' "
                      f"(with '{matched_sentence}' first).")
                return




        # ==========================================================
        # STEP 4 — FALLBACK: TOKENIZE & FETCH PICTOGRAMS
        # ==========================================================
        print(f"[GO] No dataset match → Tokenizing '{norm_text}'")
        tokens = all_tokens(norm_text)
        if not tokens:
            self.display_message("No valid tokens found.")
            return

        self.result_grid.clear_widgets()
        for token in tokens:
            url = self.arasaac_func.get_arasaac_image_url(token)
            if url:
                widget = ClickableImage(token, url, callback=lambda *_: self.result_grid.remove_widget(widget))
                self.result_grid.add_widget(widget)
            else:
                print(f"[PICTO] No pictogram for token '{token}'")

        if self.result_grid.children:
            self.content_area.clear_widgets()
            #self.content_area.add_widget(self.category_scroll)
            self.back_btn.opacity = 1
            self.back_btn.disabled = False
            print(f"[GO] Displayed pictograms for tokens: {tokens}")
        else:
            self.display_message("No pictograms found for the given input.")






    # -----------------------------------------------------
    # Async TF-IDF + AI Suggestion Builder (Phrase-aware)
    # -----------------------------------------------------
    def _build_suggestions_async(self, text: str):
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
        self.result_grid.clear_widgets()  # ensures grid is empty before new results

        norm_text = normalize_input(text)
        print(f"Normalized input: {norm_text}")
        toks = all_tokens(norm_text)
        user_pic_items = [(t, self.arasaac_func.get_arasaac_image_url(t)) for t in toks]

        # --- 1. PHRASE-LEVEL TF-IDF (1–3 gram)
        try:
            dataset_sentences = [row["yes_sentence"] for row in self.dataset_rows if row.get("yes_sentence")]
            vectorizer = TfidfVectorizer(ngram_range=(1, 3))
            tfidf_matrix = vectorizer.fit_transform(dataset_sentences)
            query_vec = vectorizer.transform([norm_text])
            scores = cosine_similarity(query_vec, tfidf_matrix).flatten()
            ranked_indices = scores.argsort()[::-1]
            tfidf_suggestions = [dataset_sentences[i] for i in ranked_indices[:5] if scores[i] > 0.1]
            print("TF-IDF (phrase-based) suggestions:", tfidf_suggestions)
        except Exception as e:
            print(f"[TF-IDF Error] {e}")
            tfidf_suggestions = []

        # --- 2. AI MODEL SUGGESTIONS
        try:
            ai_suggestions = get_suggestions(norm_text, top_k=10) or []
            print("AI suggestions:", ai_suggestions)
        except Exception as e:
            print(f"[AI suggester error] {e}")
            ai_suggestions = []

        # --- 3. Merge, Deduplicate & Filter ---
        sentences = list(dict.fromkeys(tfidf_suggestions + ai_suggestions))
        valid_sentences = []
        for s in sentences:
            if len(all_tokens(s)) < 2:
                continue
            if re.search(r"\b(was|is|are)\b$", s):
                continue
            if not is_grammatically_valid(s):
                continue
            valid_sentences.append(s)
        print("Filtered valid suggestions:", valid_sentences)

        # --- 4. Build pictogram rows ---
        rows = []
        seen = set()
        if valid_sentences:
            top_sentence = valid_sentences[0]
            top_pictos = [(t, self.arasaac_func.get_arasaac_image_url(t)) for t in all_tokens(top_sentence)]
            rows.append((top_sentence, top_pictos))
            seen.add(top_sentence.lower())

            if norm_text.lower() != top_sentence.lower():
                rows.append((norm_text, user_pic_items))
                seen.add(norm_text.lower())

            for s in valid_sentences[1:]:
                if s.lower() not in seen:
                    pictos = [(t, self.arasaac_func.get_arasaac_image_url(t)) for t in all_tokens(s)]
                    rows.append((s, pictos))
                    seen.add(s.lower())
        else:
            rows.append((norm_text, user_pic_items))

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
    # -----------------------------------------------------
    # Recommend Button (DB-driven only)
    # -----------------------------------------------------
    def recommend(self, _=None):
        # Show loading popup
        self.loading_popup = LoadingPopup()
        self.loading_popup.open()

        def _worker():
            try:
                recent = self.db_help.retrive_last_inserted(self.location_label)
                location_matched = self.db_help.matched_loc(self.location_label)
                location_unmatched = self.db_help.unmatched_loc(self.location_label)
                time_unmatched = self.db_help.unmatched_time(self.location_label)
            except Exception as e:
                print(f"recommendation error: {e}")
                recent, location_matched, location_unmatched = None, [], []

            # Prepare rows
            def process_rows(data_list):
                rows = []
                for s in data_list:
                    s = str(s)
                    pic_items = [(t, self.arasaac_func.get_arasaac_image_url(t)) for t in all_tokens(s)]
                    rows.append((s, pic_items))
                return rows

            recent_rows = process_rows([recent]) if recent else []
            location_matched_rows = process_rows(location_matched)
            location_unmatched_rows = process_rows(location_unmatched)
            time_unmatched_rows = process_rows(time_unmatched)

            # Debug
            print("recent_rows:", recent_rows)
            print("location_matched_rows:", location_matched_rows)
            print("location_unmatched_rows:", location_unmatched_rows)
            print("time_unmatched_rows:", time_unmatched_rows)

            # Schedule UI update
            def update_ui(dt):
                self._render_suggestion_rows_sections(
                    recent_rows, location_matched_rows, location_unmatched_rows,time_unmatched_rows
                )
                # Dismiss loading popup after UI update
                if self.loading_popup:
                    self.loading_popup.dismiss()

            Clock.schedule_once(update_ui, 0)

        Thread(target=_worker, daemon=True).start()
    def show_loading_popup(self):
        box = BoxLayout(orientation='vertical', padding=20)
        box.add_widget(Label(text="Loading... Please wait", font_name="Roboto-Bold.ttf", font_size="25sp"))
        popup = Popup(title="Loading", content=box,
                    size_hint=(None, None), size=(200, 150),
                    auto_dismiss=False)
        popup.open()
        return popup


    # -----------------------------------------------------
    # Accept sentence from SuggestionRow into result grid
    # -----------------------------------------------------
    def _accept_sentence(self, sentence: str):
        """
        When user clicks OK:
        - If input matches a category (e.g. 'food'), show all dataset sentences for that category.
        - Otherwise, do normal pictogram composition + AI suggestions.
        """
        sentence = (sentence or "").strip()
        if not sentence:
            return

        # --- Step 1: Detect if sentence corresponds to a known category ---
        category = self.detect_category_from_text(sentence)
        if category:
            print(f"[CATEGORY MODE] '{sentence}' detected as category → loading from dataset.csv")
            entries = self.get_sentences_for_category(category)
            if not entries:
                self.display_message(f"No sentences found for category '{category}'.")
                return

            # --- Step 2: Render dataset sentences as SuggestionRows ---
            self.sug_list.clear_widgets()
            for item in entries:
                sentence_text = item.get("yes_sentence", "").strip()
                if not sentence_text:
                    continue
                tokens = all_tokens(sentence_text)
                pictos = [(t, self.arasaac_func.get_arasaac_image_url(t)) for t in tokens]

                row = SuggestionRow(
                    sentence_text.capitalize(),
                    ok_callback=self._accept_sentence,
                    bg_color=(0.9, 1, 0.9, 1)
                )
                row.set_pictos(pictos)
                self.sug_list.add_widget(row)

            self.content_area.clear_widgets()
            self.content_area.add_widget(self.sug_scroll)
            self.back_btn.opacity = 1
            self.back_btn.disabled = False
            return  # ✅ Stop here – don’t call AI/TF-IDF suggester

        # --- Step 3: Fallback to normal pictogram composition if not category ---
        self.result_grid.clear_widgets()
        toks = all_tokens(sentence)
        for t in toks:
            url = self.arasaac_func.get_arasaac_image_url(t)
            if url:
                widget = ClickableImage(t, url)
            else:
                widget = ClickableImage(t, PLACEHOLDER_PATH)
            widget.callback = lambda *_: self.result_grid.remove_widget(widget)
            self.result_grid.add_widget(widget)

        self.text_input.text = sentence


    def display_message(self, text):
        popup = Popup(
            title="Message",
            content=Label(text=text),
            size_hint=(None, None), size=(400, 200)
        )
        popup.open()



    def get_sentence_from_pictos(self):
        labels = []
        for widget in self.result_grid.children[::-1]:
            if hasattr(widget, 'label'):
                labels.append(widget.label.text)
        sentence = " ".join(labels)
        print("Formed sentence:", sentence)
        return sentence

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
        self.content_area.clear_widgets()  # ensures grid is empty before new results

        if not hasattr(self, "captured_audio"):
            print("⚠️ No audio recorded. Please click Speak first.")
            return

        print("Stopping... recognizing now.")
        audio = self.captured_audio
        text = "<gibberish>"

        try:
            result = self.r.recognize_google(audio, language="en-IN", show_all=True)

            if isinstance(result, dict) and "alternative" in result:
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
        self.process_speech_result(text)
    # -----------------------------------------------------
    # TEXT TO SPEECH (TTS)
    # -----------------------------------------------------
    def speak_text(self, _=None):
        text = self.get_sentence_from_pictos()
        print("text:", text)
        if not text or text.strip() == "":
            popup = Popup(
                title="No pictograms selected",
                content=Label(text="Please select pictograms to form a sentence."),
                size_hint=(None, None), size=(400, 200)
            )
            popup.open()
            text = "Give some input"  # fallback text

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
                espeak_path = r"C:\Program Files\eSpeak NG\espeak-ng.exe"  # Update if needed
                subprocess.run([espeak_path, "-v", "en+f3", "-s", "140", "-p", "70", text])
        except Exception as e:
            print(f"TTS error: {e}")
        finally:
            if text != "Give some input":
                # Save to DB (your original call)
                try:
                    self.db_help.insert(text, self.location_label)
                except Exception as e:
                    print("DB insert error:", e)
    
    
    def process_speech_result(self, recognized_text):
        """
        Handles recognized speech and routes it through the normal logic.
        If no category is detected, falls back to TF-IDF suggestion flow.
        """
        if not recognized_text:
            print("[SPEECH] No text recognized.")
            return

        print(f"[SPEECH] Recognized: {recognized_text}")

        # Update the input field
        try:
            if hasattr(self, "input_field"):
                self.input_field.text = recognized_text
        except Exception as e:
            print(f"[SPEECH] Could not set input field text: {e}")

        # Try detecting category
        category = None
        try:
            category = self.detect_category_from_text(recognized_text)
        except Exception as e:
            print(f"[SPEECH] Category detection error: {e}")

        # ✅ If a category is found → show category pictograms
        if category:
            print(f"[SPEECH] Detected category → {category}")
            self.show_category_sentences_as_suggestions(category)
            return

        # ⚙️ Fallback → Normal TF-IDF flow with autocorrect
        print("[SPEECH] No category detected → Using TF-IDF suggestion flow")

        try:
            # Normalize + autocorrect
            norm_text = self.autocorrect_with_dataset(recognized_text)
            print(f"[SPEECH] Normalized + autocorrected text → {norm_text}")
           # self.result_grid.clear_widgets()  # ensures grid is empty before new results

            # ✅ Clear old category pictograms before displaying recognized text
            if hasattr(self, "result_grid"):
                self.result_grid.clear_widgets()

            # ✅ Now display the recognized text in display area (top panel)
            try:
                if hasattr(self, "display_text"):
                    self.display_text.text = norm_text
                else:
                    print(f"[DISPLAY] Recognized: {norm_text}")
            except Exception as e:
                print(f"[DISPLAY] Failed to set display text: {e}")

            # ✅ Then build suggestions (TF-IDF, etc.)
            self.process_text(norm_text)
            threading.Thread(target=self._build_suggestions_async, args=(norm_text,), daemon=True).start()
            

        except Exception as e:
            print(f"[SPEECH] Error during fallback suggestion: {e}")



    def speak_text_custom(self, sentence: str):
        """Speak the given sentence directly (used for YES/NO grid)."""
        import pygame
        import os
        from gtts import gTTS
        import requests
        import subprocess

        if not sentence or sentence.strip() == "":
            print("[TTS] Empty sentence — skipping.")
            return

        print(f"[TTS] Speaking custom sentence: {sentence}")
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
                tts = gTTS(text=sentence, lang='en', slow=False)
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
                subprocess.run(["espeak-ng", "-v", "en+f3", "-s", "140", "-p", "70", sentence])
        except Exception as e:
            print(f"[TTS ERROR] {e}")

        # ✅ Optionally save this utterance to the DB
        try:
            self.db_help.insert(sentence, self.location_label)
        except Exception as e:
            print("[DB Insert Error in speak_text_custom]", e)


    # -----------------------------------------------------
    # SYSTEM / APP UTILITIES
    # -----------------------------------------------------
    def clear_all(self, _=None):
        self.text_input.text = ""
        self.result_grid.clear_widgets()
        self.content_area.clear_widgets()
        self.show_all_categories()

    def exit_app(self, _=None):
        App.get_running_app().stop()

    def update_time(self, dt):
        self.task_label.text = f"Location: {self.location_label} | Time: {time.strftime('%H:%M:%S')}"

    def go_to_firstscreen(self, *_):
        """Go back to the first (splash) screen."""
        app = App.get_running_app()
        app.sm.current = "splash"


