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
Window.allow_vkeyboard = False          # global disable of Kivy soft keyboard
Window.softinput_mode = "below_target"  # prevent overlay push
from View.evaluation_monitor import EvaluationMonitor
from View.input_norm import ai_normalize_input
import json,os,sys,re,time,threading,subprocess,csv
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
from pydrive2.auth import GoogleAuth
from pydrive2.drive import GoogleDrive
from pydrive2.auth import RefreshError
# Add these imports at the top with your other imports
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

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
from kivy.uix.anchorlayout import AnchorLayout

from kivy.uix.gridlayout import GridLayout
from kivy.uix.image import AsyncImage,Image
from kivy.uix.label import Label
from kivy.uix.popup import Popup
from kivy.uix.scrollview import ScrollView
from kivy.uix.screenmanager import Screen
from kivy.uix.textinput import TextInput
from kivy.uix.dropdown import DropDown
from kivy.uix.spinner import Spinner

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
# Project imports (your existing modules)
from Control.Database_helper import Database_helper
from Model.Arasaac_helper import Arasaac_helper

from View.input_norm import get_suggestions
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


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
            # always create a new cell each loop
            if url:
                cell = BoxLayout(
                    orientation="vertical",
                    size_hint=(None, None),
                    size=(dp(175), self.height - 2),
                    padding=(0, 2),
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
                    font_name="Roboto-Bold.ttf",
                    color=(0, 0, 0, 1),
                    font_size="22sp",
                    halign="center",
                    valign="middle",
                    size_hint=(1, 0.25),
                    text_size=(dp(95), None),
                    shorten=False,
                )
                lbl.bind(size=lambda *_: setattr(lbl, "text_size", (lbl.width, None)))

                cell.add_widget(thumb)
                cell.add_widget(lbl)

            else:
                # create a separate BoxLayout for text-only chips
                cell = BoxLayout(
                    orientation="vertical",
                    size_hint=(None, None),
                    size=(dp(95), self.height - 4),
                    padding=(4, 2)
                )

                txt = Label(
                    text=label.capitalize(),
                    font_name="Roboto-Bold.ttf",
                    color=(0, 0, 0, 1),
                    halign="center",
                    valign="middle",
                    font_size="18sp",
                    text_size=(dp(90), None)
                )
                txt.bind(size=lambda *_: setattr(txt, "text_size", (txt.width, None)))
                cell.add_widget(txt)

            # ✅ Important: add AFTER cell is fully built
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
                size_hint=(1, 0.85)
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
            font_size='26sp',
            size_hint_y=None,     # ❗ make height manual, not proportional
            shorten=True,          # Prevent wrapping into two lines
            shorten_from='right'   # Truncate if label too long
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
        self.add_btn = None  # Will appear only after DALL·E fallback
        self.result_grid = GridLayout(cols=6, spacing=8, padding=8, size_hint_y=None)
        self.result_grid.bind(minimum_height=self.result_grid.setter('height'))
        #self.evaluator = EvaluationMonitor(interval=60)  # evaluate every 2 minutes
        #self.evaluator.start()

        Clock.schedule_once(lambda dt: self.build_ui())

    # -----------------------------------------------------
    # UI BUILDING
    # -----------------------------------------------------
    def build_ui(self):
        self.file_id=None
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

        self.text_input = TextInput(
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
        scroll = ScrollView(size_hint=(1, None), height=220)
        with scroll.canvas.before:
            Color(1, 1, 0.9, 1)  # light cream
            self.rect_scroll = Rectangle(size=scroll.size, pos=scroll.pos)
        scroll.bind(size=lambda _, val: setattr(self.rect_scroll, 'size', val),
                    pos=lambda _, val: setattr(self.rect_scroll, 'pos', val))

        self.result_grid = GridLayout(rows=1, spacing=5, size_hint_x=None, height=250)
        self.result_grid.bind(minimum_width=self.result_grid.setter('width'))
        
        
        # === RESULT GRID + ADD BUTTON (Add button anchored to top-right) ===
        result_row = AnchorLayout(anchor_x='right', anchor_y='top', size_hint=(1, None), height=250)

        # Inner layout: scrollable pictogram grid (fills left area)
        grid_row = BoxLayout(orientation='horizontal', size_hint=(1, 1), spacing=10, padding=(10, 5))
        scroll = ScrollView(size_hint=(0.85, 1))
        scroll.add_widget(self.result_grid)
        grid_row.add_widget(scroll)

        # Add button setup (top-right)
        self.add_btn = Button(
            text="Add",
            size_hint=(None, None),
            size=(180, 60),  # compact like Go button
            background_color=(0.25, 0.6, 0.25, 1),
            color=(1, 1, 1, 1),
            font_size="30sp",
            bold=True
        )
        self.add_btn.opacity = 0
        self.add_btn.disabled = True

        # Anchor the button inside the same layout
        grid_row.add_widget(self.add_btn)

        # Add to result row container
        result_row.add_widget(grid_row)
        left_panel.add_widget(result_row)


        # --- Content Area (switches between Categories and Suggestions) ---
        self.content_area = BoxLayout(orientation='vertical', size_hint=(1, 1))
        with self.content_area.canvas.before:
            Color(0.8, 1, 0.8, 1)  # keep green always
            self.rect_content = Rectangle(size=self.content_area.size, pos=self.content_area.pos)
        self.content_area.bind(size=lambda _, v: setattr(self.rect_content, 'size', v),
                               pos=lambda _, v: setattr(self.rect_content, 'pos', v))
        left_panel.add_widget(self.content_area)

        # Category view (green): ScrollView holding image_grid
        self.image_grid = GridLayout(cols=10, spacing=15, padding=15, size_hint_y=None)

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
        #self.evaluator.attach_dashboard_overlay(root)
        # Initial categories load
        Clock.schedule_once(lambda dt: self.load_categories())

    def _reposition_add_button(self, *_):
        """Keeps Add button aligned below Go even when layout resizes."""
        if hasattr(self, "go_button") and hasattr(self, "add_btn"):
            try:
                self.add_btn.x = self.go_button.x
                self.add_btn.y = self.go_button.y - (self.add_btn.height + 10)
            except Exception:
                pass


    def show_add_button_for_dalle_result(self, label, img_path, sentence):
        """
        Display DALL·E generated image with its sentence (centered text) and Add button
        inside the existing result_grid layout.
        """
        print("Label:", label)
        from kivy.uix.button import Button
        from kivy.uix.label import Label

        # Container for each DALL·E result (vertical so it looks like other pictos)
        container = BoxLayout(
            orientation='vertical',
            size_hint=(None, None),
            size=(200, 220),
            spacing=5,
            padding=5
        )

        # 🖼️ Image widget
        img_widget = AsyncImage(
            source=img_path,
            allow_stretch=True,
            keep_ratio=True,
            size_hint=(1, 0.7)
        )

        # 💬 Sentence label (auto-wrap, centered)
        sentence_widget = Label(
            text=sentence.capitalize(),
            size_hint=(1, 0.2),
            halign='center',
            valign='middle',
            color=(0, 0, 0, 1),
            font_size='18sp',
            font_name='Roboto-Bold.ttf'
        )
        sentence_widget.bind(
            size=lambda *_: setattr(sentence_widget, 'text_size', (sentence_widget.width, None))
        )

        # ➕ Add button
        add_button = Button(
            text='Add',
            size_hint=(1, 0.1),
            background_color=(0.2, 0.6, 1, 1),
            font_size='18sp',
            bold=True
        )

        # Add to container
        container.add_widget(img_widget)
        container.add_widget(sentence_widget)
        container.add_widget(add_button)

        # ✅ Show inside the existing result grid (not floating at bottom)
        self.result_grid.add_widget(container)

        # 🔊 Speak text on tap
        def on_speak_touch(instance, touch):
            if instance.collide_point(*touch.pos):
                self.speak_text_custom(sentence)
        img_widget.bind(on_touch_down=on_speak_touch)
        sentence_widget.bind(on_touch_down=on_speak_touch)

        # ➕ Add button behavior (opens category popup)
        add_button.bind(on_press=lambda *_: self.prompt_category_popup(label, img_path, sentence))
    def get_drive(self):
        """
        Returns an authenticated GoogleDrive instance.
        Uses:
        - client_secrets.json (downloaded from Google Cloud)
        - mycreds.json (saved credentials with refresh_token)
        """
        import os
        from pydrive2.auth import GoogleAuth
        from pydrive2.drive import GoogleDrive
        from google.auth.exceptions import RefreshError  # <-- from google.auth
        view_dir = os.path.dirname(os.path.abspath(__file__))

        CLIENT_SECRETS = os.path.join(view_dir, "client_secrets.json")
        CREDENTIALS_FILE = os.path.join(view_dir, "mycreds.json")

        # All settings are passed directly here – no settings.yaml, no _module issues
        settings = {
            "client_config_backend": "file",
            "client_config_file": CLIENT_SECRETS,

            "save_credentials": True,
            "save_credentials_backend": "file",
            "save_credentials_file": CREDENTIALS_FILE,

            # Drive full access
            "oauth_scope": ["https://www.googleapis.com/auth/drive"],

            # Important: ask for refresh_token
            "get_refresh_token": True,
        }

        gauth = GoogleAuth(settings=settings)

        # Load existing credentials if present
        if os.path.exists(CREDENTIALS_FILE):
            gauth.LoadCredentialsFile(CREDENTIALS_FILE)
        else:
            gauth.credentials = None

        # 🔹 First time: no credentials yet
        if gauth.credentials is None:
            print("[DRIVE AUTH] No saved credentials, running LocalWebserverAuth()")
            gauth.LocalWebserverAuth()

        # 🔹 Have credentials but token may be expired
        elif gauth.access_token_expired:
            print("[DRIVE AUTH] Access token expired, trying Refresh()")
            try:
                gauth.Refresh()
            except RefreshError:
                print("[DRIVE AUTH] Refresh failed or no refresh_token. Re-authenticating...")
                gauth.LocalWebserverAuth()
        else:
            print("[DRIVE AUTH] Using existing valid credentials")
            gauth.Authorize()

        # Save credentials (should now include refresh_token)
        gauth.SaveCredentialsFile(CREDENTIALS_FILE)

        return GoogleDrive(gauth)



 		
    def upload_to_drive(self, local_path):
        drive = self.get_drive()
        try:
            print(f"[UPLOAD] Uploading '{local_path}' to Google Drive...")
            f = drive.CreateFile({
                "title": os.path.basename(local_path),
                "parents": [{"id": '1X1ya6OLQA9SBIcicaaceBAukaV94RpOB'}]
            })
            f.SetContentFile(local_path)
            f.Upload()
            self.file_id = f.get("id")

            if not self.file_id:
                print("[UPLOAD ERROR] file_id is None! Upload failed.")
                return None

            print(f"[UPLOAD SUCCESS] ID = {self.file_id}")
            return f"https://drive.google.com/uc?export=view&id={self.file_id}"

        except Exception as e:
            print("[UPLOAD EXCEPTION]", e)



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
        Display all matched pictograms and their full AAC sentences
        inside the main result grid (not the suggestion list).
        """
        entries = self.get_sentences_for_category(category)
        if not entries:
            self.display_message(f"No sentences found for category '{category}'.")
            return

        # Clear the existing result grid before showing new ones
        self.result_grid.clear_widgets()

        from kivy.uix.boxlayout import BoxLayout
        from kivy.uix.label import Label
        from kivy.uix.button import Button

        for item in entries:
            sentence = item.get("yes_sentence", "").strip()
            label = item.get("label", "").strip()
            if not sentence or not label:
                continue

            # Get cached pictogram
            tex, img_path = self.arasaac_func.get_cached_image(label, PLACEHOLDER_PATH)

            # 🔹 Container for one pictogram entry (vertical style)
            container = BoxLayout(
                orientation='vertical',
                size_hint=(None, None),
                size=(180, 220),
                spacing=5,
                padding=5
            )

            # 🖼️ Image
            img_widget = AsyncImage(
                source=img_path,
                allow_stretch=True,
                keep_ratio=True,
                size_hint=(1, 0.65)
            )

            # 🏷️ Label (short)
            label_widget = Label(
                text=label.capitalize(),
                halign='center',
                valign='middle',
                color=(0, 0, 0, 1),
                font_size='18sp',
                bold=True,
                size_hint=(1, 0.15)
            )
            label_widget.bind(
                size=lambda instance, value: setattr(instance, 'text_size', (instance.width, None))
            )

            # 💬 Sentence (full AAC sentence)
            sentence_widget = Label(
                text=sentence,
                halign='center',
                valign='middle',
                color=(0, 0, 0, 1),
                font_size='16sp',
                size_hint=(1, 0.15)
            )
            sentence_widget.bind(
                size=lambda instance, value: setattr(instance, 'text_size', (instance.width, None))
            )

            # ➕ Add button
            add_button = Button(
                text="Add",
                size_hint=(1, 0.12),
                background_color=(0.2, 0.6, 1, 1),
                font_size='16sp',
                bold=True
            )
            add_button.bind(
                on_press=lambda _btn, s=sentence: self._accept_sentence(s)
            )

            # Add all widgets to the pictogram box
            container.add_widget(img_widget)
            container.add_widget(label_widget)
            container.add_widget(sentence_widget)
            container.add_widget(add_button)

            # ✅ Add to existing result grid
            self.result_grid.add_widget(container)

        print(f"[UI] Displayed {len(entries)} results for category '{category}' inside result grid.")


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
        from collections import defaultdict
        import csv, os

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
        all_categories = set()

        with open(csv_path, "r", encoding="utf-8", errors="replace") as f:
            reader = csv.DictReader(f)
            for row in reader:
                row = {
                    (k.strip().lower() if k else ""): (v or "").strip()
                    for k, v in row.items()
                    if k
                }
                label = row.get("label", "")
                yes_sentence = row.get("yes_sentence", label)
                category = row.get("category", "Uncategorized")

                if category:
                    grouped[category].append({"label": label, "yes_sentence": yes_sentence})
                    all_categories.add(category)

        # ✅ build unique category list AFTER the loop
        self.existing_categories = sorted(all_categories)
        print(f"[INFO] Categories found: {self.existing_categories}")

        # Display category icons
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
        add_category_btn = Button(
                text="+ Add Category",
                size_hint=(None, None),
                size=(150, 150),
                background_color=(0.2, 0.8, 0.2, 1),  # Green background
                color=(1, 1, 1, 1),  # White text
                font_name="Roboto-Bold.ttf",
                font_size="18sp",
                bold=True
            )
        add_category_btn.bind(on_press=lambda _: self.add_new_category())
        self.image_grid.add_widget(add_category_btn)
        self.content_area.add_widget(self.category_scroll)
        self.back_btn.opacity = 0
        self.back_btn.disabled = True
    def show_add_category_popup(self):

        """Show popup to add a new category with DALL-E generated icon."""
        from kivy.uix.boxlayout import BoxLayout
        from kivy.uix.textinput import TextInput
        from kivy.uix.button import Button
        from kivy.uix.label import Label
        from kivy.uix.popup import Popup
        
        layout = BoxLayout(orientation="vertical", spacing=10, padding=20)
        
        # Title label
        title_label = Label(
            text="Add New Category",
            size_hint=(0.5, None),
            height=40,
            font_size="20sp",
            bold=True,
            color=(0, 0, 0, 1)
        )
        layout.add_widget(title_label)
        
        # Category name input
        name_input = TextInput(
            hint_text="Enter category name (e.g., 'vehicles', 'furniture')",
            size_hint=(1, None),
            height=50,
            font_size="18sp",
            multiline=False
        )
        layout.add_widget(name_input)
        
        # Buttons layout
        button_layout = BoxLayout(orientation="horizontal", size_hint=(1, None), height=50, spacing=10)
        
        cancel_btn = Button(
            text="Cancel",
            size_hint=(0.5, 1),
            background_color=(0.8, 0.2, 0.2, 1),
            font_size="16sp"
        )
        
        create_btn = Button(
            text="Create Category",
            size_hint=(0.5, 1),
            background_color=(0.2, 0.8, 0.2, 1),
            font_size="16sp"
        )
        
        button_layout.add_widget(cancel_btn)
        button_layout.add_widget(create_btn)
        layout.add_widget(button_layout)
        
        # Create popup
        popup = Popup(
            title="Add New Category",
            content=layout,
            size_hint=(0.8, None),
            height=250,
            auto_dismiss=False
        )
    
        def cancel_action(_):
            popup.dismiss()
        
        def create_action(_):
            category_name = name_input.text.strip()
            if not category_name:
                self.show_toast("Please enter a category name.")
                return
            
            # Close popup and show loading
            popup.dismiss()
            self.create_category_with_dalle(category_name)
        
        cancel_btn.bind(on_press=cancel_action)
        create_btn.bind(on_press=create_action)
        
        popup.open()
    def create_category_with_dalle(self, category_name):
            
        """Generate DALL-E icon for new category and save to icon directory."""
        import os
        import re
        import base64
        import tempfile
        import threading
        import csv
        from kivy.clock import Clock
        
        def show_loading():
            loading_popup = LoadingPopup()
            loading_popup.content.children[0].text = f"Generating icon for '{category_name}'..."
            loading_popup.open()
            return loading_popup
        
        def worker():
            loading_popup = None
            try:
                # Show loading popup
                Clock.schedule_once(lambda dt: show_loading(), 0)
                
                # Generate DALL-E image
                from openai import OpenAI
                client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
                
                dalle_prompt = (
                    f"3D cartoon folder icon for '{category_name}' category, "
                    f"colorful, simple, clean design, no text, "
                    f"pictogram style, high contrast, suitable for AAC application in 150dpi"
                )
                
                print(f"[DALL-E] Generating icon for category: {category_name}")
                resp = client.images.generate(
                    model="dall-e-3",
                    prompt=dalle_prompt,
                    size="1024x1024",
                    response_format="b64_json",
                    n=1,
                )
                
                # Decode base64 image
                b64_data = resp.data[0].b64_json
                image_data = base64.b64decode(b64_data)
                
                # Create proper filename (without "dalle" prefix)
                safe_name = re.sub(r"[^a-z0-9_]+", "_", category_name.lower().strip())
                icon_filename = f"{safe_name}.png"
                icon_path = os.path.join(ICON_DIR, icon_filename)
                
                # Ensure ICON_DIR exists
                os.makedirs(ICON_DIR, exist_ok=True)
                
                # Save the image
                with open(icon_path, "wb") as f:
                    f.write(image_data)
                
                print(f"[ICON CREATED] Saved category icon: {icon_path}")
                
                # ✅ ADD ONLY CATEGORY NAME TO DATASET.CSV (empty label and yes_sentence)
                dataset_path = os.path.join(os.path.dirname(__file__), "../dataset.csv")
                
                # Check if file exists and has header
                file_exists = os.path.exists(dataset_path)
                
                # Append to dataset
                with open(dataset_path, "a", newline='', encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=["label", "yes_sentence", "category"])
                    
                    # Add header if file is new
                    if not file_exists:
                        writer.writeheader()
                    
                    # Add entry with ONLY category name (label and yes_sentence are empty)
                    writer.writerow({
                        "label": "",  # ✅ Empty
                        "yes_sentence": "",  # ✅ Empty  
                        "category": category_name.capitalize()  # ✅ Only category name
                    })
                
                print(f"[DATASET UPDATED] Added category '{category_name}' with empty label and sentence")
                
                def update_ui(dt):
                    # Add category to existing categories list
                    if hasattr(self, 'existing_categories'):
                        if category_name.capitalize() not in self.existing_categories:
                            self.existing_categories.append(category_name.capitalize())
                            self.existing_categories.sort()
                    
                    # Dismiss loading popup
                    if hasattr(self, 'loading_popup') and self.loading_popup:
                        self.loading_popup.dismiss()
                    
                    # Reload categories to show the new one
                    self.load_categories()
                    
                    # Show success message
                    self.show_toast(f"Category '{category_name}' created successfully!")
                
                Clock.schedule_once(update_ui, 0)
                
            except Exception as e:
                print(f"[DALL-E ERROR] Failed to create category icon: {e}")
                
                def show_error(dt):
                    if hasattr(self, 'loading_popup') and self.loading_popup:
                        self.loading_popup.dismiss()
                    self.show_toast(f"Failed to create category: {str(e)}")
                
                Clock.schedule_once(show_error, 0)
        
        # Run in background thread
        threading.Thread(target=worker, daemon=True).start()

    def add_new_category(self, spinner=None):
        """Enhanced method to add category with image upload/generation capability."""
        from kivy.uix.textinput import TextInput
        from kivy.uix.button import Button
        from kivy.uix.boxlayout import BoxLayout
        from kivy.uix.popup import Popup
        from kivy.uix.label import Label
        from kivy.uix.filechooser import FileChooserListView
        import tempfile
        import os
        import shutil
        
        layout = BoxLayout(orientation="vertical", spacing=10, padding=10)
        
        # Category name input
        layout.add_widget(Label(text="Enter the new category name:", size_hint=(1, None), height=25, halign="left"))
        name_input = TextInput(
            hint_text="Enter new category name (e.g., 'vehicles', 'furniture')",
            size_hint=(1, None),
            height=40,
            multiline=False
        )
        layout.add_widget(name_input)
        
        # Upload and Generate buttons
        image_btn_layout = BoxLayout(orientation="horizontal", size_hint=(1, None), height=50, spacing=10)
        
        upload_btn = Button(
            text="Upload",
            size_hint=(0.5, 1),
            background_color=(0.2, 0.4, 0.8, 1)
        )
        
        generate_btn = Button(
            text="Generate",
            size_hint=(0.5, 1),
            background_color=(0.8, 0.4, 0.2, 1)
        )
        
        image_btn_layout.add_widget(upload_btn)
        image_btn_layout.add_widget(generate_btn)
        layout.add_widget(image_btn_layout)
        
        # Image display box
        image_display = AsyncImage(
            size_hint=(1, None),
            height=200,
            allow_stretch=True,
            keep_ratio=True
        )
        layout.add_widget(image_display)
        
        # Submit and Cancel buttons
        control_layout = BoxLayout(orientation="horizontal", size_hint=(1, None), height=50, spacing=10)
        
        submit_btn = Button(
            text="Submit",
            size_hint=(0.5, 1),
            background_color=(0.2, 0.8, 0.2, 1)
        )
        
        cancel_btn = Button(
            text="Cancel",
            size_hint=(0.5, 1),
            background_color=(0.8, 0.2, 0.2, 1)
        )
        
        control_layout.add_widget(submit_btn)
        control_layout.add_widget(cancel_btn)
        layout.add_widget(control_layout)
        
        # Create popup
        popup = Popup(
            title="Add New Category",
            content=layout,
            size_hint=(0.8, None),
            height=500,
            auto_dismiss=False
        )
        
        # ✅ USE A DICTIONARY TO MAKE IT MUTABLE ACROSS FUNCTIONS
        image_data = {"path": None}  # This will be shared across all nested functions
        
        def upload_image(_):
            """Open file chooser for image upload"""
            # Create file chooser popup
            file_layout = BoxLayout(orientation="vertical", spacing=10, padding=10)
            
            file_chooser = FileChooserListView(
                path=os.path.expanduser("~"),
                filters=["*.png", "*.jpg", "*.jpeg", "*.PNG", "*.JPG", "*.JPEG"]
            )
            
            file_btn_layout = BoxLayout(orientation="horizontal", size_hint=(1, None), height=50, spacing=10)
            
            select_btn = Button(text="Select", size_hint=(0.5, 1), background_color=(0.2, 0.8, 0.2, 1))
            cancel_file_btn = Button(text="Cancel", size_hint=(0.5, 1), background_color=(0.8, 0.2, 0.2, 1))
            
            file_btn_layout.add_widget(select_btn)
            file_btn_layout.add_widget(cancel_file_btn)
            
            file_layout.add_widget(file_chooser)
            file_layout.add_widget(file_btn_layout)
            
            file_popup = Popup(
                title="Select Image",
                content=file_layout,
                size_hint=(0.8, 0.8),
                auto_dismiss=False
            )
            
            def select_file(_):
                if file_chooser.selection:
                    selected_file = file_chooser.selection[0]
                    
                    # Copy to temp location
                    category_name = name_input.text.strip()
                    safe_name = re.sub(r"[^a-z0-9_]+", "_", category_name.lower()) if category_name else "category"
                    ext = os.path.splitext(selected_file)[1]
                    temp_path = os.path.join(tempfile.gettempdir(), f"uploaded_cat_{safe_name}{ext}")
                    
                    try:
                        shutil.copy2(selected_file, temp_path)
                        # ✅ UPDATE THE SHARED DICTIONARY
                        image_data["path"] = temp_path
                        image_display.source = image_data["path"]
                        print(f"[DEBUG] Uploaded category image saved to: {temp_path}")
                        self.show_toast("Image uploaded successfully!")
                    except Exception as e:
                        self.show_toast(f"Upload failed: {str(e)}")
                        
                    file_popup.dismiss()
                else:
                    self.show_toast("Please select a file.")
            
            def cancel_file_selection(_):
                file_popup.dismiss()
            
            select_btn.bind(on_press=select_file)
            cancel_file_btn.bind(on_press=cancel_file_selection)
            file_popup.open()
        
        def generate_image(_):
            """Generate DALL-E image for category"""
            category_name = name_input.text.strip()
            if not category_name:
                self.show_toast("Please enter a category name first.")
                return
                
            # Show loading
            generate_btn.text = "Generating..."
            generate_btn.disabled = True
            
            def dalle_worker():
                try:
                    from openai import OpenAI
                    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
                    
                    dalle_prompt = (
                        f"3D cartoon folder icon for '{category_name}' category, "
                        f"colorful, simple, clean design, no text, "
                        f"pictogram style, high contrast, suitable for AAC application"
                    )
                    
                    resp = client.images.generate(
                        model="dall-e-3",
                        prompt=dalle_prompt,
                        size="1024x1024",
                        response_format="b64_json",
                        n=1,
                    )
                    
                    # Decode and save image
                    import base64
                    import re
                    b64_data = resp.data[0].b64_json
                    image_data_bytes = base64.b64decode(b64_data)
                    
                    safe_name = re.sub(r"[^a-z0-9_]+", "_", category_name.lower())
                    temp_path = os.path.join(tempfile.gettempdir(), f"dalle_cat_{safe_name}.png")
                    
                    with open(temp_path, "wb") as f:
                        f.write(image_data_bytes)
                    
                    # ✅ UPDATE THE SHARED DICTIONARY
                    image_data["path"] = temp_path
                    print(f"[DEBUG] Generated category image saved to: {temp_path}")
                    
                    def update_ui(dt):
                        image_display.source = image_data["path"]
                        generate_btn.text = "Generate"
                        generate_btn.disabled = False
                        self.show_toast("Category icon generated successfully!")
                        
                    Clock.schedule_once(update_ui, 0)
                    
                except Exception as e:
                    print(f"[DALLE ERROR] {e}")
                    def show_error(dt):
                        self.show_toast(f"Generation failed: {str(e)}")
                        generate_btn.text = "Generate"
                        generate_btn.disabled = False
                        
                    Clock.schedule_once(show_error, 0)
            
            import threading
            threading.Thread(target=dalle_worker, daemon=True).start()
        
        def submit_category(_):
            """Submit the new category"""
            category_name = name_input.text.strip()
            
            # ✅ DEBUG PRINT TO CHECK CURRENT STATE
            print(f"[DEBUG SUBMIT] Category name: {category_name}")
            print(f"[DEBUG SUBMIT] Image path: {image_data['path']}")
            
            # Validation
            if not category_name:
                self.show_toast("Please enter a category name.")
                return
                
            # Check if category already exists
            if hasattr(self, 'existing_categories') and category_name.capitalize() in self.existing_categories:
                self.show_toast(f"Category '{category_name}' already exists.")
                return
                
            # ✅ CHECK THE SHARED DICTIONARY
            if not image_data["path"] or not os.path.exists(image_data["path"]):
                self.show_toast("Please generate or upload an image.")
                print(f"[DEBUG] Image validation failed. Path: {image_data['path']}")
                return
            
            # Save category icon to permanent location
            try:
                import re
                safe_name = re.sub(r"[^a-z0-9_]+", "_", category_name.lower())
                icon_filename = f"{safe_name}.png"
                icon_path = os.path.join(ICON_DIR, icon_filename)
                
                # Ensure ICON_DIR exists
                os.makedirs(ICON_DIR, exist_ok=True)
                
                # Copy from temp to permanent location
                shutil.copy2(image_data["path"], icon_path)
                print(f"[ICON SAVED] Category icon saved to: {icon_path}")
                
                # ✅ ADD CATEGORY TO DATASET.CSV
                dataset_path = os.path.join(os.path.dirname(__file__), "../dataset.csv")
                file_exists = os.path.exists(dataset_path)
                
                with open(dataset_path, "a", newline='', encoding="utf-8") as f:
                    import csv
                    writer = csv.DictWriter(f, fieldnames=["label", "yes_sentence", "category"])
                    
                    # Add header if file is new
                    if not file_exists:
                        writer.writeheader()
                    
                    # Add entry with ONLY category name (empty label and yes_sentence)
                    writer.writerow({
                        "label": "",  # ✅ Empty
                        "yes_sentence": "",  # ✅ Empty  
                        "category": category_name.capitalize()  # ✅ Only category name
                    })
                
                print(f"[DATASET UPDATED] Added category '{category_name}' to dataset.csv")
                
                # Update spinner if provided
                if spinner and category_name.capitalize() not in spinner.values:
                    spinner.values.append(category_name.capitalize())
                    spinner.text = category_name.capitalize()
                
                # Add to existing categories list
                if hasattr(self, 'existing_categories'):
                    if category_name.capitalize() not in self.existing_categories:
                        self.existing_categories.append(category_name.capitalize())
                        self.existing_categories.sort()
                
                self.show_toast(f"Category '{category_name}' created successfully!")
                
                # Close popup
                popup.dismiss()
                
                # Refresh categories view
                if hasattr(self, 'load_categories'):
                    self.load_categories()
                    
            except Exception as e:
                self.show_toast(f"Save failed: {str(e)}")
                print(f"[ERROR] Category creation failed: {e}")
        
        def cancel_popup(_):
            """Cancel and close popup"""
            popup.dismiss()
        
        # Bind button events
        upload_btn.bind(on_press=upload_image)
        generate_btn.bind(on_press=generate_image)
        submit_btn.bind(on_press=submit_category)
        cancel_btn.bind(on_press=cancel_popup)
        
        popup.open()

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

            # ✅ ADD "ADD ITEM" BUTTON TO CATEGORY (after all items are loaded)
            def add_category_add_button(dt):
                add_item_btn = Button(
                    text="+ Add Item",
                    size_hint=(None, None),
                    size=(150, 150),
                    background_color=(0.2, 0.8, 0.2, 1),  # Green background
                    color=(1, 1, 1, 1),  # White text
                    font_name="Roboto-Bold.ttf",
                    font_size="18sp",
                    bold=True
                )
                add_item_btn.bind(on_press=lambda _: self.show_add_item_to_category_popup(category))
                grid.add_widget(add_item_btn)
                print(f"[UI] Added 'Add Item' button to category '{category}'")

            Clock.schedule_once(add_category_add_button, 0.1)  # Slight delay to ensure items are loaded first
            print(f"[INFO] Loaded {len(entries)} pictograms for category '{category}'")

        # --- Run pictogram loading in background thread ---
        threading.Thread(target=load_items, daemon=True).start()

        # --- Enable the Back button ---
        self.back_btn.opacity = 1
        self.back_btn.disabled = False
        self.back_btn.bind(on_press=lambda _: self.load_categories())

    def show_add_item_to_category_popup(self, category):
        """Show popup to add a new item directly to the specified category."""
        from kivy.uix.filechooser import FileChooserListView
        import tempfile
        import os
        import shutil
        
        layout = BoxLayout(orientation="vertical", spacing=10, padding=10)
        
        # Title showing which category we're adding to
        title_label = Label(
            text=f"Add Item to '{category.capitalize()}' Category",
            size_hint=(1, None),
            height=40,
            font_size="20sp",
            bold=True,
            color=(0, 0, 0, 1)
        )
        layout.add_widget(title_label)
        
        # Label input
        layout.add_widget(Label(text="Item Name:", size_hint=(1, None), height=25, halign="left"))
        label_input = TextInput(
            hint_text="Enter the name of the item (e.g., 'apple', 'car')",
            size_hint=(1, None), 
            height=44,
            multiline=False
        )
        layout.add_widget(label_input)
        
        # Sentence input
        layout.add_widget(Label(text="Sentence:", size_hint=(1, None), height=25, halign="left"))
        sentence_input = TextInput(
            hint_text="Enter the full sentence (e.g., 'I want apple')",
            size_hint=(1, None), 
            height=44,
            multiline=False
        )
        layout.add_widget(sentence_input)
        
        # Image generation/upload buttons
        image_btn_layout = BoxLayout(orientation="horizontal", size_hint=(1, None), height=50, spacing=10)
        
        generate_btn = Button(
            text="Generate Image",
            size_hint=(0.5, 1),
            background_color=(0.8, 0.4, 0.2, 1)
        )
        
        upload_btn = Button(
            text="Upload Image",
            size_hint=(0.5, 1),
            background_color=(0.2, 0.4, 0.8, 1)
        )
        
        image_btn_layout.add_widget(generate_btn)
        image_btn_layout.add_widget(upload_btn)
        layout.add_widget(image_btn_layout)
        
        # Image display box
        image_display = AsyncImage(
            size_hint=(1, None),
            height=200,
            allow_stretch=True,
            keep_ratio=True
        )
        layout.add_widget(image_display)
        
        # Control buttons
        control_layout = BoxLayout(orientation="horizontal", size_hint=(1, None), height=50, spacing=10)
        
        cancel_btn = Button(
            text="Cancel",
            size_hint=(0.5, 1),
            background_color=(0.8, 0.2, 0.2, 1)
        )
        
        submit_btn = Button(
            text="Add to Category",
            size_hint=(0.5, 1),
            background_color=(0.2, 0.8, 0.2, 1)
        )
        
        control_layout.add_widget(cancel_btn)
        control_layout.add_widget(submit_btn)
        layout.add_widget(control_layout)
        
        # Create popup
        popup = Popup(
            title=f"Add Item to {category}",
            content=layout,
            size_hint=(0.9, None),
            height=600,
            auto_dismiss=False
        )
        
        # ✅ USE A DICTIONARY TO MAKE IT MUTABLE ACROSS FUNCTIONS
        image_data = {"path": None}  # This will be shared across all nested functions
        
        def generate_image(_):
            """Generate DALL-E image based on label"""
            label_text = label_input.text.strip()
            if not label_text:
                self.show_toast("Please enter an item name first.")
                return
                
            # Show loading
            generate_btn.text = "Generating..."
            generate_btn.disabled = True
            
            def dalle_worker():
                try:
                    from openai import OpenAI
                    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
                    
                    dalle_prompt = (
                        f"3D cartoon icon for '{label_text}', "
                        f"colorful, simple, clean design, no text, "
                        f"pictogram style, high contrast, suitable for AAC application"
                    )
                    
                    resp = client.images.generate(
                        model="dall-e-3",
                        prompt=dalle_prompt,
                        size="1024x1024",
                        response_format="b64_json",
                        n=1,
                    )
                    
                    # Decode and save image
                    import base64
                    import re
                    b64_data = resp.data[0].b64_json
                    image_data_bytes = base64.b64decode(b64_data)
                    
                    safe_name = re.sub(r"[^a-z0-9_]+", "_", label_text.lower())
                    temp_path = os.path.join(tempfile.gettempdir(), f"dalle_{safe_name}.png")
                    
                    with open(temp_path, "wb") as f:
                        f.write(image_data_bytes)
                    
                    # ✅ UPDATE THE SHARED DICTIONARY
                    image_data["path"] = temp_path
                    print(f"[DEBUG] Generated image saved to: {temp_path}")
                    
                    def update_ui(dt):
                        image_display.source = image_data["path"]
                        generate_btn.text = "Generate Image"
                        generate_btn.disabled = False
                        self.show_toast("Image generated successfully!")
                        
                    Clock.schedule_once(update_ui, 0)
                    
                except Exception as e:
                    print(f"[DALLE ERROR] {e}")
                    def show_error(dt):
                        self.show_toast(f"Generation failed: {str(e)}")
                        generate_btn.text = "Generate Image"
                        generate_btn.disabled = False
                        
                    Clock.schedule_once(show_error, 0)
            
            import threading
            threading.Thread(target=dalle_worker, daemon=True).start()
        
        def upload_image(_):
            """Open file chooser for image upload"""
            # Create file chooser popup
            file_layout = BoxLayout(orientation="vertical", spacing=10, padding=10)
            
            file_chooser = FileChooserListView(
                path=os.path.expanduser("~"),
                filters=["*.png", "*.jpg", "*.jpeg", "*.PNG", "*.JPG", "*.JPEG"]
            )
            
            file_btn_layout = BoxLayout(orientation="horizontal", size_hint=(1, None), height=50, spacing=10)
            
            select_btn = Button(text="Select", size_hint=(0.5, 1), background_color=(0.2, 0.8, 0.2, 1))
            cancel_file_btn = Button(text="Cancel", size_hint=(0.5, 1), background_color=(0.8, 0.2, 0.2, 1))
            
            file_btn_layout.add_widget(select_btn)
            file_btn_layout.add_widget(cancel_file_btn)
            
            file_layout.add_widget(file_chooser)
            file_layout.add_widget(file_btn_layout)
            
            file_popup = Popup(
                title="Select Image",
                content=file_layout,
                size_hint=(0.8, 0.8),
                auto_dismiss=False
            )
            
            def select_file(_):
                if file_chooser.selection:
                    selected_file = file_chooser.selection[0]
                    
                    # Copy to temp location
                    label_text = label_input.text.strip()
                    safe_name = re.sub(r"[^a-z0-9_]+", "_", label_text.lower()) if label_text else "uploaded"
                    ext = os.path.splitext(selected_file)[1]
                    temp_path = os.path.join(tempfile.gettempdir(), f"uploaded_{safe_name}{ext}")
                    
                    try:
                        shutil.copy2(selected_file, temp_path)
                        # ✅ UPDATE THE SHARED DICTIONARY
                        image_data["path"] = temp_path
                        image_display.source = image_data["path"]
                        print(f"[DEBUG] Uploaded image saved to: {temp_path}")
                        self.show_toast("Image uploaded successfully!")
                    except Exception as e:
                        self.show_toast(f"Upload failed: {str(e)}")
                        
                    file_popup.dismiss()
                else:
                    self.show_toast("Please select a file.")
            
            def cancel_file_selection(_):
                file_popup.dismiss()
            
            select_btn.bind(on_press=select_file)
            cancel_file_btn.bind(on_press=cancel_file_selection)
            file_popup.open()
        
        def submit_data(_):
            """Submit the new item to the category"""
            final_label = label_input.text.strip()
            final_sentence = sentence_input.text.strip()
            
            # ✅ DEBUG PRINT TO CHECK CURRENT STATE
            print(f"[DEBUG SUBMIT] Label: {final_label}")
            print(f"[DEBUG SUBMIT] Sentence: {final_sentence}")
            print(f"[DEBUG SUBMIT] Image path: {image_data['path']}")
            
            # Validation
            if not final_label:
                self.show_toast("Please enter an item name.")
                return
                
            if not final_sentence:
                self.show_toast("Please enter a sentence.")
                return
                
            # ✅ CHECK THE SHARED DICTIONARY
            if not image_data["path"] or not os.path.exists(image_data["path"]):
                self.show_toast("Please generate or upload an image.")
                print(f"[DEBUG] Image validation failed. Path: {image_data['path']}")
                return
            
            # Save to dataset
            try:
                self.update_metadata_and_dataset(final_label, image_data["path"], final_sentence, category)
                self.show_toast(f"'{final_label}' added to '{category}' successfully!")
                
                # Clean up and close
                popup.dismiss()
                
                # Refresh the current category view
                entries = self.get_sentences_for_category(category)
                self.open_category_folder(category, entries)
                    
            except Exception as e:
                self.show_toast(f"Save failed: {str(e)}")
        
        def cancel_popup(_):
            """Cancel and close popup"""
            popup.dismiss()
        
        # Bind button events
        generate_btn.bind(on_press=generate_image)
        upload_btn.bind(on_press=upload_image)
        submit_btn.bind(on_press=submit_data)
        cancel_btn.bind(on_press=cancel_popup)
        
        popup.open()



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

    def add_to_result(self, label, url, sentence=None):
        """
        Adds a matched pictogram (image + label + full sentence)
        into the visible result grid. Keeps the same card layout.
        """
        from kivy.uix.boxlayout import BoxLayout
        from kivy.uix.label import Label
        from kivy.uix.button import Button

        if not sentence:
            sentence = label.capitalize()

        box = BoxLayout(
            orientation='vertical',
            size_hint=(None, None),
            size=(180, 220),
            spacing=5,
            padding=5
        )

        img_widget = AsyncImage(
            source=url,
            allow_stretch=True,
            keep_ratio=True,
            size_hint=(1, 0.65)
        )

        label_widget = Label(
            text=label.capitalize(),
            halign='center',
            valign='middle',
            color=(0, 0, 0, 1),
            font_size='18sp',
            bold=True,
            size_hint=(1, 0.15)
        )
        label_widget.bind(size=lambda i, v: setattr(i, 'text_size', (i.width, None)))

        sentence_widget = Label(
            text=sentence,
            halign='center',
            valign='middle',
            color=(0, 0, 0, 1),
            font_size='16sp',
            size_hint=(1, 0.15)
        )
        sentence_widget.bind(size=lambda i, v: setattr(i, 'text_size', (i.width, None)))

        remove_btn = Button(
            text="Remove",
            size_hint=(1, 0.12),
            background_color=(1, 0.4, 0.4, 1),
            font_size='15sp'
        )
        remove_btn.bind(on_press=lambda *_: self.result_grid.remove_widget(box))

        box.add_widget(img_widget)
        box.add_widget(label_widget)
        box.add_widget(sentence_widget)
        box.add_widget(remove_btn)

        self.result_grid.add_widget(box)

        # Optional: tap image or sentence to speak
        def on_tap_speak(instance, touch):
            if instance.collide_point(*touch.pos):
                try:
                    if hasattr(self, 'speak_text_custom'):
                        self.speak_text_custom(sentence)
                    else:
                        self.speak_text(sentence)
                except Exception:
                    pass

        img_widget.bind(on_touch_down=on_tap_speak)
        sentence_widget.bind(on_touch_down=on_tap_speak)

        self.current_sentence = sentence
        print(f"[UI] Added '{label}' with sentence '{sentence}' to result grid.")

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
   
    def prompt_category_popup(self, label, img_path, sentence):
        """
        Shows popup after Add button click with existing image display.
        - If label is single word → ask for a full sentence.
        - If sentence is multiword → ask for a label.
        """
        from kivy.uix.filechooser import FileChooserListView
        import tempfile
        import os
        import shutil
        
        # Detect if single word or multiword
        is_single_word = len(label.strip().split()) == 1
        
        layout = BoxLayout(orientation="vertical", spacing=10, padding=10)
        
        # Category selection section
        cat_layout = BoxLayout(orientation="horizontal", size_hint=(1, None), height=44, spacing=10)
        
        cat_spinner = Spinner(
            text="Select Category",
            values=self.existing_categories,
            size_hint=(0.7, 1)
        )
        
        add_cat_btn = Button(
            text="+ New Category", 
            size_hint=(0.3, 1),
            background_color=(0.2, 0.6, 1, 1)
        )
        
        cat_layout.add_widget(cat_spinner)
        cat_layout.add_widget(add_cat_btn)
        layout.add_widget(cat_layout)
        
        # ✅ CONDITIONAL INPUT BASED ON SINGLE/MULTI WORD
        if is_single_word:
            # Single word: Label readonly, ask for sentence
            layout.add_widget(Label(text="Label:", size_hint=(1, None), height=25, halign="left"))
            label_input = TextInput(
                text=label,
                readonly=False,  # ✅ Read-only for single word
                size_hint=(1, None), 
                height=44,
                multiline=False
            )
            layout.add_widget(label_input)
            
            layout.add_widget(Label(text="Sentence:", size_hint=(1, None), height=25, halign="left"))
            sentence_input = TextInput(
                hint_text="Enter a full sentence for this word",
                size_hint=(1, None), 
                height=44,
                multiline=False
            )
            layout.add_widget(sentence_input)
            
        else:
            # Multi word: Sentence readonly, ask for label
            layout.add_widget(Label(text="Sentence:", size_hint=(1, None), height=25, halign="left"))
            sentence_input = TextInput(
                text=sentence,
                readonly=False,  # ✅ Read-only for multi word
                size_hint=(1, None), 
                height=44,
                multiline=False
            )
            layout.add_widget(sentence_input)
            
            layout.add_widget(Label(text="Label:", size_hint=(1, None), height=25, halign="left"))
            label_input = TextInput(
                hint_text="Enter a short label for this sentence",
                size_hint=(1, None), 
                height=44,
                multiline=False
            )
            layout.add_widget(label_input)
        
        # Image generation/upload buttons
        image_btn_layout = BoxLayout(orientation="horizontal", size_hint=(1, None), height=50, spacing=10)
        
        generate_btn = Button(
            text="Generate",
            size_hint=(0.5, 1),
            background_color=(0.8, 0.4, 0.2, 1)
        )
        
        upload_btn = Button(
            text="Upload",
            size_hint=(0.5, 1),
            background_color=(0.2, 0.4, 0.8, 1)
        )
        
        image_btn_layout.add_widget(generate_btn)
        image_btn_layout.add_widget(upload_btn)
        layout.add_widget(image_btn_layout)
        
        # ✅ IMAGE DISPLAY BOX (Show existing image immediately)
        image_display = AsyncImage(
            source=img_path if img_path else "",  # ✅ Show existing image
            size_hint=(1, None),
            height=200,
            allow_stretch=True,
            keep_ratio=True
        )
        layout.add_widget(image_display)
        
        # Control buttons
        control_layout = BoxLayout(orientation="horizontal", size_hint=(1, None), height=50, spacing=10)
        
        cancel_btn = Button(
            text="Cancel",
            size_hint=(0.5, 1),
            background_color=(0.8, 0.2, 0.2, 1)
        )
        
        submit_btn = Button(
            text="Submit",
            size_hint=(0.5, 1),
            background_color=(0.2, 0.8, 0.2, 1)
        )
        
        control_layout.add_widget(cancel_btn)
        control_layout.add_widget(submit_btn)
        layout.add_widget(control_layout)
        
        # Create popup
        popup = Popup(
            title="Add to Dataset",
            content=layout,
            size_hint=(0.9, None),
            height=600,
            auto_dismiss=False
        )
        
        # Variables to track current image
        current_image_path = img_path  # ✅ Start with existing image
        
        def generate_image(_):
            """Generate NEW DALL-E image (only if user explicitly requests)"""
            nonlocal current_image_path
            
            # Get the current label text (whether readonly or editable)
            current_label = label_input.text.strip()
            if not current_label:
                self.show_toast("Please enter a label first.")
                return
                
            # Show loading
            generate_btn.text = "Generating..."
            generate_btn.disabled = True
            
            def dalle_worker():
                try:
                    from openai import OpenAI
                    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
                    
                    dalle_prompt = (
                        f"3D cartoon icon for '{current_label}', "
                        f"colorful, simple, clean design, no text, "
                        f"pictogram style, high contrast, suitable for AAC application"
                    )
                    
                    resp = client.images.generate(
                        model="dall-e-3",
                        prompt=dalle_prompt,
                        size="1024x1024",
                        response_format="b64_json",
                        n=1,
                    )
                    
                    # Decode and save image
                    import base64
                    import re
                    b64_data = resp.data[0].b64_json
                    image_data = base64.b64decode(b64_data)
                    
                    safe_name = re.sub(r"[^a-z0-9_]+", "_", current_label.lower())
                    temp_path = os.path.join(tempfile.gettempdir(), f"dalle_new_{safe_name}.png")
                    
                    with open(temp_path, "wb") as f:
                        f.write(image_data)
                    
                    current_image_path = temp_path
                    
                    def update_ui(dt):
                        image_display.source = current_image_path
                        generate_btn.text = "Generate"
                        generate_btn.disabled = False
                        self.show_toast("New image generated successfully!")
                        
                    Clock.schedule_once(update_ui, 0)
                    
                except Exception as e:
                    print(f"[DALLE ERROR] {e}")
                    def show_error(dt):
                        self.show_toast(f"Generation failed: {str(e)}")
                        generate_btn.text = "Generate"
                        generate_btn.disabled = False
                        
                    Clock.schedule_once(show_error, 0)
            
            import threading
            threading.Thread(target=dalle_worker, daemon=True).start()
        
        def upload_image(_):
            """Open file chooser for image upload"""
            nonlocal current_image_path
            
            # Create file chooser popup
            file_layout = BoxLayout(orientation="vertical", spacing=10, padding=10)
            
            file_chooser = FileChooserListView(
                path=os.path.expanduser("~"),
                filters=["*.png", "*.jpg", "*.jpeg", "*.PNG", "*.JPG", "*.JPEG"]
            )
            
            file_btn_layout = BoxLayout(orientation="horizontal", size_hint=(1, None), height=50, spacing=10)
            
            select_btn = Button(text="Select", size_hint=(0.5, 1), background_color=(0.2, 0.8, 0.2, 1))
            cancel_file_btn = Button(text="Cancel", size_hint=(0.5, 1), background_color=(0.8, 0.2, 0.2, 1))
            
            file_btn_layout.add_widget(select_btn)
            file_btn_layout.add_widget(cancel_file_btn)
            
            file_layout.add_widget(file_chooser)
            file_layout.add_widget(file_btn_layout)
            
            file_popup = Popup(
                title="Select Image",
                content=file_layout,
                size_hint=(0.8, 0.8),
                auto_dismiss=False
            )
            
            def select_file(_):
                if file_chooser.selection:
                    selected_file = file_chooser.selection[0]
                    
                    # Copy to temp location
                    current_label = label_input.text.strip()
                    safe_name = re.sub(r"[^a-z0-9_]+", "_", current_label.lower()) if current_label else "uploaded"
                    ext = os.path.splitext(selected_file)[1]
                    temp_path = os.path.join(tempfile.gettempdir(), f"uploaded_{safe_name}{ext}")
                    
                    try:
                        shutil.copy2(selected_file, temp_path)
                        current_image_path = temp_path
                        image_display.source = current_image_path
                        self.show_toast("Image uploaded successfully!")
                    except Exception as e:
                        self.show_toast(f"Upload failed: {str(e)}")
                        
                    file_popup.dismiss()
                else:
                    self.show_toast("Please select a file.")
            
            def cancel_file_selection(_):
                file_popup.dismiss()
            
            select_btn.bind(on_press=select_file)
            cancel_file_btn.bind(on_press=cancel_file_selection)
            file_popup.open()
        
        def show_new_category_popup(_):
            """Show popup for creating new category with image"""
            popup.dismiss()  # Close current popup
            
            new_cat_layout = BoxLayout(orientation="vertical", spacing=10, padding=10)
            
            # Category name input
            new_cat_layout.add_widget(Label(text="Category Name:", size_hint=(1, None), height=25, halign="left"))
            cat_name_input = TextInput(
                hint_text="Enter category name",
                text=label,
                size_hint=(1, None),
                height=44,
                multiline=False
            )
            new_cat_layout.add_widget(cat_name_input)
            
            # Image buttons for new category
            new_cat_img_layout = BoxLayout(orientation="horizontal", size_hint=(1, None), height=50, spacing=10)
            
            new_upload_btn = Button(text="Upload", size_hint=(0.5, 1), background_color=(0.2, 0.4, 0.8, 1))
            new_generate_btn = Button(text="Generate", size_hint=(0.5, 1), background_color=(0.8, 0.4, 0.2, 1))
            
            new_cat_img_layout.add_widget(new_upload_btn)
            new_cat_img_layout.add_widget(new_generate_btn)
            new_cat_layout.add_widget(new_cat_img_layout)
            
            # ✅ IMAGE DISPLAY BOX - This will show the generated/uploaded image
            new_cat_image = AsyncImage(
                source=img_path if img_path else "",
                size_hint=(1, None),
                height=200,
                allow_stretch=True,
                keep_ratio=True
            )
            new_cat_layout.add_widget(new_cat_image)
            
            # Control buttons for new category
            new_cat_controls = BoxLayout(orientation="horizontal", size_hint=(1, None), height=50, spacing=10)
            
            new_cat_submit = Button(text="Submit", size_hint=(0.5, 1), background_color=(0.2, 0.8, 0.2, 1))
            new_cat_cancel = Button(text="Cancel", size_hint=(0.5, 1), background_color=(0.8, 0.2, 0.2, 1))
            
            new_cat_controls.add_widget(new_cat_submit)
            new_cat_controls.add_widget(new_cat_cancel)
            new_cat_layout.add_widget(new_cat_controls)
            
            # ✅ CREATE THE NEW POPUP
            new_cat_popup = Popup(
                title="Create New Category",
                content=new_cat_layout,
                size_hint=(0.8, None),
                height=500,
                auto_dismiss=False
            )
            
            # ✅ TRACK IMAGE PATH
            new_cat_image_data = {"path": None}
            
            # ✅ UPLOAD FUNCTIONALITY
            def new_cat_upload(_):
                """Handle image upload for new category"""
                from kivy.uix.filechooser import FileChooserListView
                import os, tempfile, shutil, re
                
                file_layout = BoxLayout(orientation="vertical", spacing=10, padding=10)
                
                file_chooser = FileChooserListView(
                    path=os.path.expanduser("~"),
                    filters=["*.png", "*.jpg", "*.jpeg", "*.PNG", "*.JPG", "*.JPEG"]
                )
                
                file_btn_layout = BoxLayout(orientation="horizontal", size_hint=(1, None), height=50, spacing=10)
                
                select_btn = Button(text="Select", size_hint=(0.5, 1), background_color=(0.2, 0.8, 0.2, 1))
                cancel_file_btn = Button(text="Cancel", size_hint=(0.5, 1), background_color=(0.8, 0.2, 0.2, 1))
                
                file_btn_layout.add_widget(select_btn)
                file_btn_layout.add_widget(cancel_file_btn)
                
                file_layout.add_widget(file_chooser)
                file_layout.add_widget(file_btn_layout)
                
                file_popup = Popup(
                    title="Select Image",
                    content=file_layout,
                    size_hint=(0.8, 0.8),
                    auto_dismiss=False
                )
                
                def select_file(_):
                    if file_chooser.selection:
                        selected_file = file_chooser.selection[0]
                        
                        category_name = cat_name_input.text.strip()
                        safe_name = re.sub(r"[^a-z0-9_]+", "_", category_name.lower()) if category_name else "category"
                        ext = os.path.splitext(selected_file)[1]
                        temp_path = os.path.join(tempfile.gettempdir(), f"uploaded_newcat_{safe_name}{ext}")
                        
                        try:
                            shutil.copy2(selected_file, temp_path)
                            new_cat_image_data["path"] = temp_path
                            new_cat_image.source = temp_path  # ✅ Display the uploaded image
                            print(f"[DEBUG] Uploaded image displayed: {temp_path}")
                            self.show_toast("Image uploaded successfully!")
                        except Exception as e:
                            self.show_toast(f"Upload failed: {str(e)}")
                            
                        file_popup.dismiss()
                    else:
                        self.show_toast("Please select a file.")
                
                def cancel_file_selection(_):
                    file_popup.dismiss()
                
                select_btn.bind(on_press=select_file)
                cancel_file_btn.bind(on_press=cancel_file_selection)
                file_popup.open()
            
            # ✅ GENERATE FUNCTIONALITY
            def new_cat_generate(_):
                """Generate DALL-E image for new category"""
                category_name = cat_name_input.text.strip()
                if not category_name:
                    self.show_toast("Please enter category name first.")
                    return
                    
                new_generate_btn.text = "Generating..."
                new_generate_btn.disabled = True
                
                def dalle_worker():
                    try:
                        from openai import OpenAI
                        import base64
                        import re
                        import tempfile
                        import os
                        
                        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
                        
                        dalle_prompt = (
                            f"3D cartoon folder icon for '{category_name}' category, "
                            f"colorful, simple, clean design, no text, "
                            f"pictogram style, high contrast, suitable for AAC application"
                        )
                        
                        resp = client.images.generate(
                            model="dall-e-3",
                            prompt=dalle_prompt,
                            size="1024x1024",
                            response_format="b64_json",
                            n=1,
                        )
                        
                        b64_data = resp.data[0].b64_json
                        image_data = base64.b64decode(b64_data)
                        
                        safe_name = re.sub(r"[^a-z0-9_]+", "_", category_name.lower())
                        temp_path = os.path.join(tempfile.gettempdir(), f"dalle_newcat_{safe_name}.png")
                        
                        with open(temp_path, "wb") as f:
                            f.write(image_data)
                        
                        new_cat_image_data["path"] = temp_path
                        print(f"[DEBUG] Generated image saved to: {temp_path}")
                        
                        def update_ui(dt):
                            new_cat_image.source = temp_path  # ✅ Display the generated image
                            new_generate_btn.text = "Generate"
                            new_generate_btn.disabled = False
                            self.show_toast("Category icon generated!")
                            print(f"[DEBUG] Generated image displayed in dialog: {temp_path}")
                            
                        Clock.schedule_once(update_ui, 0)
                        
                    except Exception as e:
                        print(f"[DALLE ERROR] {e}")
                        def show_error(dt):
                            self.show_toast(f"Generation failed: {str(e)}")
                            new_generate_btn.text = "Generate"
                            new_generate_btn.disabled = False
                            
                        Clock.schedule_once(show_error, 0)
                
                import threading
                threading.Thread(target=dalle_worker, daemon=True).start()
            
            # ✅ SUBMIT FUNCTIONALITY
            def submit_new_category(_):
                """Submit the new category"""
                import os
                category_name = cat_name_input.text.strip()
                new_cat_image_data["path"]=img_path
                print(f"[DEBUG SUBMIT] Category: {category_name}")
                print(f"[DEBUG SUBMIT] Image path: {new_cat_image_data['path']}")
                
                if not category_name:
                    self.show_toast("Please enter category name.")
                    return
                    
                if not new_cat_image_data["path"] or not os.path.exists(new_cat_image_data["path"]):
                    self.show_toast("Please upload or generate an image.")
                    return
                
                try:
                    # Save to permanent location and update dataset
                    import re
                    import shutil
                    import csv
                    import os
                    
                    safe_name = re.sub(r"[^a-z0-9_]+", "_", category_name.lower())
                    icon_filename = f"{safe_name}.png"
                    icon_path = os.path.join(ICON_DIR, icon_filename)
                    
                    os.makedirs(ICON_DIR, exist_ok=True)
                    shutil.copy2(new_cat_image_data["path"], icon_path)
                    
                    # Add to dataset
                    dataset_path = os.path.join(os.path.dirname(__file__), "../dataset.csv")
                    file_exists = os.path.exists(dataset_path)
                    
                    with open(dataset_path, "a", newline='', encoding="utf-8") as f:
                        writer = csv.DictWriter(f, fieldnames=["label", "yes_sentence", "category"])
                        if not file_exists:
                            writer.writeheader()
                        writer.writerow({
                            "label": "",
                            "yes_sentence": "",
                            "category": category_name.capitalize()
                        })
                    
                    # Update category spinner in the original popup
                    if hasattr(self, 'existing_categories'):
                        if category_name.capitalize() not in self.existing_categories:
                            self.existing_categories.append(category_name.capitalize())
                            self.existing_categories.sort()
                    
                    # ✅ UPDATE THE ORIGINAL POPUP'S SPINNER
                    cat_spinner.values.append(category_name.capitalize())
                    cat_spinner.text = category_name.capitalize()
                    
                    self.show_toast(f"Category '{category_name}' created!")
                    new_cat_popup.dismiss()
                    self.load_categories()  # Refresh categories list
                    # ✅ RE-OPEN THE ORIGINAL POPUP
                    
                    
                except Exception as e:
                    self.show_toast(f"Failed to create category: {str(e)}")
            
            # ✅ CANCEL FUNCTIONALITY
            def cancel_new_category(_):
                """Cancel new category creation and return to original popup"""
                new_cat_popup.dismiss()
                popup.open()  # ✅ Re-open the original popup
            
            # ✅ BIND ALL BUTTON EVENTS (This was missing in your code!)
            new_upload_btn.bind(on_press=new_cat_upload)
            new_generate_btn.bind(on_press=new_cat_generate)
            new_cat_submit.bind(on_press=submit_new_category)
            new_cat_cancel.bind(on_press=cancel_new_category)
            
            # ✅ OPEN THE NEW POPUP
            new_cat_popup.open()
        
        def submit_data(_):
            """Submit the form data"""
            chosen_cat = cat_spinner.text.strip()
            
            # ✅ Get final values based on single/multi word mode
            if is_single_word:
                final_label = label_input.text.strip()  # readonly, should be original label
                final_sentence = sentence_input.text.strip()  # user input
            else:
                final_label = label_input.text.strip()  # user input
                final_sentence = sentence_input.text.strip()  # readonly, should be original sentence
            
            # Validation
            if not chosen_cat or chosen_cat == "Select Category":
                self.show_toast("Please select or create a category.")
                return
                
            if not final_label:
                self.show_toast("Please enter a label.")
                return
                
            if not final_sentence:
                if is_single_word:
                    self.show_toast("Please enter a sentence for this word.")
                else:
                    self.show_toast("Please enter a label for this sentence.")
                return
                
            if not current_image_path:
                self.show_toast("No image available. Please generate or upload an image.")
                return
            
            # Save to dataset
            try:
                self.update_metadata_and_dataset(final_label, current_image_path, final_sentence, chosen_cat)
                self.show_toast("Item added successfully!")
                
                # Clean up and close
                popup.dismiss()
                
                # Refresh categories
                if hasattr(self, "load_categories"):
                    self.load_categories()
                    
            except Exception as e:
                self.show_toast(f"Save failed: {str(e)}")
        
        def cancel_popup(_):
            """Cancel and close popup"""
            popup.dismiss()
        
        # Bind button events
        generate_btn.bind(on_press=generate_image)  # ✅ Only generates NEW image on request
        upload_btn.bind(on_press=upload_image)
        add_cat_btn.bind(on_press=show_new_category_popup)
        submit_btn.bind(on_press=submit_data)
        cancel_btn.bind(on_press=cancel_popup)
        
        popup.open()

    def update_metadata_and_dataset(self, label, img_path, sentence, category):
        """
        Adds entry to dataset.csv and metadata_drive.json after DALL·E fallback confirmation.
        """
        import os, csv, json

        # === Update metadata_drive.json ===
        json_path = os.path.join(os.path.dirname(__file__), "../Model/metadata_drive.json")
        data = {}
        if os.path.exists(json_path):
            with open(json_path, "r", encoding="utf-8") as f:
                try:
                    data = json.load(f)
                except json.JSONDecodeError:
                    data = {}

        next_id = max([v.get("pic_id", 0) for v in data.values()] + [0]) + 1

        file_url = self.upload_to_drive(img_path)
        if file_url:
            url = file_url
            print(f"[DRIVE UPLOAD] Uploaded '{label}' to Drive: {file_url}")
        else:
            print(f"[DRIVE UPLOAD ERROR] Upload failed, falling back to local path for '{label}'")
            url = os.path.abspath(img_path)

        data[label] = {
            "filename": os.path.basename(img_path),
            "pic_id": next_id,
            "url": url,   # ✅ safe URL
        }

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)
        print(f"[JSON UPDATED] Added '{label}' to metadata_drive.json")

        # === Append to dataset.csv ===
        dataset_path = os.path.join(os.path.dirname(__file__), "../dataset.csv")
        file_exists = os.path.exists(dataset_path)

        if file_exists:
            with open(dataset_path, "rb+") as f:
                f.seek(0, os.SEEK_END)
                if f.tell() > 0:
                    f.seek(-1, os.SEEK_END)
                    if f.read(1) != b"\n":
                        f.write(b"\n")

        with open(dataset_path, "a", newline='', encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["label", "yes_sentence", "category"])
            if not file_exists:
                writer.writeheader()
            writer.writerow({
                "label": label.strip(),
                "yes_sentence": sentence.strip(),
                "category": category.strip().capitalize()
            })
        print(f"[CSV UPDATED] Added {label} → ({sentence}, {category})")

        # Optional refresh
        if hasattr(self, "load_categories"):
            self.load_categories()



        # Hide Add button after saving
        # Hide the Add button after saving
        if hasattr(self, "add_btn") and self.add_btn:
            try:
                self.result_grid.remove_widget(self.add_btn)
                self.add_btn = None
            except Exception:
                pass


    def show_toast(self, message):
        from kivy.uix.label import Label
        from kivy.uix.popup import Popup
        Popup(title="Info", content=Label(text=message),
              size_hint=(0.6, None), height=150).open()

         # -----------------------------------------------------
    # Go Button (Category → Label → Exact → Tokenized pictograms)
    # -----------------------------------------------------
       # -----------------------------------------------------
    # Go Button: Unified and Stable AAC Input Pipeline
    # -----------------------------------------------------
    def on_go_clicked(self, _=None):
        """
        Final precise AAC 'Go' logic with loading popup and background processing
        - Single-word: category → label → fuzzy category → AI → DALL·E
        - Multi-word: sentence → AI-normalized sentence → DALL·E
        """
        text = (self.text_input.text or "").strip().lower()
        if not text:
            return
        self.matched_label=None

        # ✅ Show loading popup immediately
        self.show_go_loading_popup()
        
        # ✅ Run the actual processing in background thread
        def background_worker():
            try:
                self.process_go_input_with_loading(text)
            except Exception as e:
                print(f"[GO ERROR] {e}")
                Clock.schedule_once(lambda dt: self.dismiss_go_loading_popup(), 0)
                Clock.schedule_once(lambda dt: self.show_toast("Processing failed"), 0)
        
        # Start background processing
        import threading
        threading.Thread(target=background_worker, daemon=True).start()

    def process_go_input_with_loading(self, text):
        """
        Your existing on_go_clicked logic but with status updates and thread safety
        """
        import os, csv, re, tempfile, base64
        from difflib import get_close_matches, SequenceMatcher
        start_time = time.time()
        
        words = text.split()
        is_single_word = len(words) == 1
        
        # ✅ Clear result grid on main thread
        Clock.schedule_once(lambda dt: self.result_grid.clear_widgets(), 0)
        
        # ==========================================================
        # STEP 1 — Load dataset (with status update)
        # ==========================================================
        self.update_go_status("Loading dataset...")
        
        dataset_path = os.path.join(os.path.dirname(__file__), "../dataset.csv")
        dataset_rows = []
        try:
            with open(dataset_path, "r", encoding="utf-8") as f:
                import csv
                reader = csv.DictReader(f)
                # normalize case for all relevant fields
                for r in reader:
                    cat = (r.get("category") or "").strip().lower()
                    lbl = (r.get("label") or "").strip().lower()
                    sent = (r.get("yes_sentence") or "").strip()
                    if sent:
                        dataset_rows.append({
                            "category": cat,
                            "label": lbl,
                            "yes_sentence": sent
                        })
        except Exception as e:
            print("[DATASET LOAD ERROR]", e)
            Clock.schedule_once(lambda dt: self.dismiss_go_loading_popup(), 0)
            return

        # collect lowercased sets for quick match
        categories = {r["category"] for r in dataset_rows if r["category"]}
        labels = {r["label"] for r in dataset_rows if r["label"]}

        def show_category_sentences(category_name):
            """Helper: display all sentences belonging to a category."""
            
            # ✅ ALWAYS get all entries for the category first
            all_category_entries = [r for r in dataset_rows if r.get("category", "").lower() == category_name]
            
            if not self.matched_label:
                # No specific label - show all entries in category
                entries = all_category_entries
            else:
                # Specific label matched - show label matches first, then others
                entry_label_match = [r for r in all_category_entries if r.get("label", "").lower() == self.matched_label]
                entry_others = [r for r in all_category_entries if r.get("label", "").lower() != self.matched_label]
                entries = entry_label_match + entry_others
            
            def update_ui(dt):
                self.sug_list.clear_widgets()
                for r in entries:
                    sentence = r.get("yes_sentence", "").strip()
                    label = r.get("label", "").strip().lower()
                    if not sentence:
                        continue
                    _, path = self.arasaac_func.get_cached_image(label, PLACEHOLDER_PATH)
                    if not path:
                        url = self.arasaac_func.get_arasaac_image_url(label)
                        path = url if url else PLACEHOLDER_PATH
                    pictos = [(label, path)]
                    row = SuggestionRow(sentence.capitalize(),
                                        ok_callback=self._accept_sentence,
                                        bg_color=(0.8, 1, 0.8, 1))
                    row.set_pictos(pictos)
                    self.sug_list.add_widget(row)
                self.content_area.clear_widgets()
                self.content_area.add_widget(self.sug_scroll)
                self.back_btn.opacity, self.back_btn.disabled = 1, False
                print(f"[UI] Displayed {len(entries)} sentences for category '{category_name}'")
                self.dismiss_go_loading_popup()
            
            Clock.schedule_once(update_ui, 0)

        # ==========================================================
        # CASE 1 — SINGLE-WORD INPUT
        # ==========================================================
        if is_single_word:
            self.update_go_status("Processing single word...")
            word = text.strip().lower()
            matched_category = None
            self.matched_label = None

            # ---------- Normalize dataset categories and labels ----------
            normalized_categories = {c.strip().lower() for c in categories if c}
            normalized_labels = {l.strip().lower() for l in labels if l}

            # ---------- 1️⃣ Exact category match (case-insensitive) ----------
            if word in normalized_categories:
                self.update_go_status("Found category match...")
                matched_category = word
                print(f"[CATEGORY MATCH] '{word}' (case-insensitive)")
                show_category_sentences(matched_category)
                return  # ✅ stop here — do NOT go to AI normalization

            # ---------- 2️⃣ Exact label match (case-insensitive) ----------
            if word in normalized_labels:
                self.update_go_status("Found label match...")
                for row in dataset_rows:
                    if row["label"].strip().lower() == word:
                        matched_category = row["category"].strip().lower()
                        self.matched_label = word
                        break
                if matched_category:
                    print(f"[LABEL MATCH] '{word}' → Category '{matched_category}' (case-insensitive)")
                    show_category_sentences(matched_category)
                    return  # ✅ stop here — do NOT go to AI normalization

            # ---------- 3️⃣ Fuzzy category match (handles plurals) ----------
            self.update_go_status("Checking fuzzy matches...")
            close_cat = get_close_matches(word, list(normalized_categories), n=1, cutoff=0.8)
            if not close_cat and word.endswith("s"):
                singular = word[:-1]
                close_cat = get_close_matches(singular, list(normalized_categories), n=1, cutoff=0.8)
            if close_cat:
                matched_category = close_cat[0]
                print(f"[FUZZY CATEGORY MATCH] '{word}' ≈ '{matched_category}'")
                show_category_sentences(matched_category)
                return  # ✅ stop here — do NOT go to AI normalization

            # ---------- 4️⃣ Dictionary check ----------
            self.update_go_status("Checking dictionary...")
            
            def word_dictionary(word):
                """Check if word exists in WordNet dictionary."""
                try:
                    from nltk.corpus import wordnet
                    word = word.lower()
                    return bool(wordnet.synsets(word))
                except:
                    return False
                    
            if word_dictionary(word):
                print(f"[DICTIONARY CHECK] '{word}' found in WordNet dictionary.")
                norm_word = word
            else:
                print(f"[DICTIONARY CHECK] '{word}' NOT found in WordNet dictionary.")
                # ---------- AI normalization ----------
                self.update_go_status("AI processing...")
                print(f"[AI NORMALIZATION - 1 word] Triggered for '{word}' (no category/label match)")
                try:
                    norm_word = ai_normalize_input(
                        word,
                        location=self.location_label,
                        time_phase=getattr(self, "time_phase", "unspecified"),
                    ).lower().strip()
                    
                    import re
                    norm_word = re.sub(r'["\'\.\,\;\:\!\?]', '', norm_word)  # Remove quotes, periods, punctuation
                    norm_word = re.sub(r'\s+', ' ', norm_word).strip()        # Remove extra spaces
                    print(f"[AI CLEANED] '{word}' → '{norm_word}'")
                    # Update text input on main thread
                    Clock.schedule_once(lambda dt: setattr(self.text_input, 'text', norm_word), 0)
                except Exception as e:
                    print("[AI NORMALIZER ERROR]", e)
                    norm_word = word
                    import re
                    norm_word = re.sub(r'["\'\.\,\;\:\!\?]', '', norm_word)
                    norm_word = re.sub(r'\s+', ' ', norm_word).strip()

            # ---------- Check corrected word ----------
            if norm_word != word:
                self.update_go_status("Checking corrected word...")
                print(f"[AI→CORRECTED] '{word}' → '{norm_word}'")
                # re-run checks for corrected word
                if norm_word in normalized_categories:
                    print(f"[AI→CATEGORY MATCH] '{norm_word}'")
                    show_category_sentences(norm_word)
                    return

                if norm_word in normalized_labels:
                    for row in dataset_rows:
                        if row["label"].strip().lower() == norm_word:
                            matched_category = row["category"].strip().lower()
                            print(f"[AI→LABEL MATCH] '{norm_word}' → Category '{matched_category}'")    
                            self.matched_label = norm_word
                            break
                    if matched_category:
                        print(f"[AI→LABEL MATCH] '{norm_word}' → Category '{matched_category}'")
                        show_category_sentences(matched_category)
                        return

            # ---------- 5️⃣ DALL·E fallback ----------
            self.update_go_status("Generating image...")
            print(f"[DALL·E] No dataset match even after AI normalization → generating image for '{norm_word}'")
            try:
                from openai import OpenAI
                client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
                dalle_prompt = (
                    f"3D cartoon icon, pictogram style, "
                    f"no text, minimal, high contrast; concept: '{norm_word}'."
                )
                resp = client.images.generate(
                    model="dall-e-3",
                    prompt=dalle_prompt,
                    size="1024x1024",
                    response_format="b64_json",
                    n=1,
                )
                b64 = resp.data[0].b64_json
                safe = re.sub(r"[^a-z0-9_]+", "_", norm_word)
                out_path = os.path.join(tempfile.gettempdir(), f"dalle_{safe}.png")

                with open(out_path, "wb") as f:
                    f.write(base64.b64decode(b64))
                
                # ✅ Update UI on main thread
                def update_dalle_ui(dt):
                    self.show_add_button_for_dalle_result(label=norm_word, img_path=out_path, sentence=norm_word)
                    self.dismiss_go_loading_popup()
                
                Clock.schedule_once(update_dalle_ui, 0)
                return
                
            except Exception as e:
                print("[DALL·E ERROR]", e)
                Clock.schedule_once(lambda dt: self.dismiss_go_loading_popup(), 0)
                return

        # ==========================================================
        # CASE 2 — MULTI-WORD INPUT (FINAL)
        # ==========================================================
        else:
            self.update_go_status("Processing multi-word input...")
            text_lower = text.lower().strip()

            matched_sentence = None
            matched_label = None
            matched_category = None
            best_score = 0.0
            forced_match = None

            # ==========================================================
            # STEP 1 — FORCE TOP MATCH IF INPUT IS PRESENT IN DATASET
            # ==========================================================
            for row in dataset_rows:
                sent = (row.get("yes_sentence") or "").strip().lower()
                if not sent:
                    continue

                # ✅ HARD PRESENCE MATCH (TOP PRIORITY)
                if text_lower in sent:
                    print("Hard Presence")
                    print(f"[FORCED MATCH] Input present in dataset sentence: '{sent}'")
                    forced_match = {
                        "sentence": sent,
                        "label": (row.get("label") or "").lower(),
                        "category": (row.get("category") or "").lower(),
                    }
                    best_score = 1.5
                    break

                # Normal similarity fallback
                else:
                    self.update_go_status("AI sentence correction...")
                    try:
                        self.mul_norm_text = ai_normalize_input(
                            text_lower,
                            location=self.location_label,
                            time_phase=getattr(self, "time_phase", "unspecified"),
                        ).lower().strip()
                        #Clock.schedule_once(lambda dt: setattr(self.text_input, "text", norm_text), 0)
                        #process_text_input()
                    except Exception as e:
                        print("[AI NORMALIZER ERROR]", e)
                        self.mul_norm_text = text_lower
                    
                    score = SequenceMatcher(None, self.mul_norm_text, sent).ratio()
                    print("score:",score)
                    if score < best_score:
                        print("score inside if:",score)
                        print("inside")
                        best_score = score
                        matched_sentence = sent
                        matched_label = (row.get("label") or "").lower()
                        matched_category = (row.get("category") or "").lower()
                        
                #self.text_input.text=norm_text
                break

            # ==========================================================
            # STEP 2 — TF-IDF (ONLY IF NO FORCED MATCH)
            # ==========================================================
            if not forced_match:
                self.update_go_status("Checking sentence similarity...")
                try:
                    sentences = [
                        r.get("yes_sentence", "").strip().lower()
                        for r in dataset_rows if r.get("yes_sentence")
                    ]

                    if sentences:
                        vectorizer = TfidfVectorizer()
                        tfidf_matrix = vectorizer.fit_transform(sentences)
                        tfidf_input = vectorizer.transform([text_lower])
                        cosine_scores = cosine_similarity(tfidf_input, tfidf_matrix)[0]

                        best_tfidf_idx = cosine_scores.argmax()
                        best_tfidf_score = cosine_scores[best_tfidf_idx]

                        if best_tfidf_score > best_score:
                            best_score = best_tfidf_score
                            matched_sentence = sentences[best_tfidf_idx]
                            matched_label = (dataset_rows[best_tfidf_idx].get("label") or "").lower()
                            matched_category = (dataset_rows[best_tfidf_idx].get("category") or "").lower()

                except Exception as e:
                    print("[TF-IDF ERROR]", e)

            # ==========================================================
            # STEP 3 — APPLY FORCED MATCH (OVERRIDES ALL)
            # ==========================================================
            if forced_match:
                matched_sentence = forced_match["sentence"]
                matched_label = forced_match["label"]
                matched_category = forced_match["category"]
                best_score = 1.0

    # ==========================================================
    # STEP 4 — SHOW TOP MATCH + CATEGORY SENTENCES (MODIFIED)
    # ==========================================================
            if best_score >= 0.7 and matched_sentence:

                def update_sentence_ui(dt):
                    # --- TOP RESULT ---
                    _, path = self.arasaac_func.get_cached_image(matched_label, PLACEHOLDER_PATH)
                    if not path:
                        url = self.arasaac_func.get_arasaac_image_url(matched_label)
                        path = url if url else PLACEHOLDER_PATH

                    widget = ClickableImage(
                        label_text=matched_label.capitalize(),
                        img_url=path,
                        callback=lambda *_: self.speak_text(matched_sentence)
                    )
                    self.result_grid.add_widget(widget)

                    # --- CATEGORY SENTENCES BELOW (SORTED WITH MATCHED FIRST) ---
                    self.sug_list.clear_widgets()

                    # ✅ COLLECT AND SORT ALL CATEGORY ENTRIES
                    all_category_entries = []
                    matched_entries = []
                    other_entries = []

                    for row in dataset_rows:
                        if (row.get("category") or "").lower() != matched_category:
                            continue

                        sentence = (row.get("yes_sentence") or "").strip()
                        label = (row.get("label") or "").lower()
                        
                        if not sentence:
                            continue

                        # ✅ CHECK IF THIS IS THE MATCHED SENTENCE
                        if sentence.lower().strip() == matched_sentence.lower().strip():
                            matched_entries.append((sentence, label))
                        else:
                            other_entries.append((sentence, label))

                    # ✅ COMBINE: MATCHED SENTENCES FIRST, THEN OTHERS
                    sorted_entries = matched_entries + other_entries

                    print(f"[UI] Displaying {len(matched_entries)} matched + {len(other_entries)} other sentences")

                    # ✅ RENDER IN SORTED ORDER
                    for i, (sentence, label) in enumerate(sorted_entries):
                        _, path = self.arasaac_func.get_cached_image(label, PLACEHOLDER_PATH)
                        if not path:
                            url = self.arasaac_func.get_arasaac_image_url(label)
                            path = url if url else PLACEHOLDER_PATH

                        # ✅ HIGHLIGHT THE MATCHED SENTENCE WITH DIFFERENT COLOR
                        if i < len(matched_entries):  # This is the matched sentence
                            bg_color = (0.8, 1, 0.8, 1)  # Lighter green for matched
                            print(f"[MATCHED] '{sentence}' displayed at top")
                        else:
                            bg_color = (0.9, 1, 0.9, 1)  # Regular green for others

                        sug = SuggestionRow(
                            sentence,
                            ok_callback=self._accept_sentence,
                            bg_color=bg_color
                        )
                        sug.set_pictos([(label, path)])
                        self.sug_list.add_widget(sug)

                    self.content_area.clear_widgets()
                    self.content_area.add_widget(self.sug_scroll)
                    self.back_btn.opacity, self.back_btn.disabled = 1, False
                    self.dismiss_go_loading_popup()

                Clock.schedule_once(update_sentence_ui, 0)
                return

            # ==========================================================
            # STEP 5 — AI NORMALIZATION (ONLY IF NOTHING MATCHED)
            # ==========================================================
            self.update_go_status("AI sentence correction...")
            
            """try:
                self.mul_norm_text = ai_normalize_input(
                    text_lower,
                    location=self.location_label,
                    time_phase=getattr(self, "time_phase", "unspecified"),
                ).lower().strip()
                print("normalized text:", self.mul_norm_text)
                #Clock.schedule_once(lambda dt: setattr(self.text_input, "text", self.mul_norm_text), 0)
                
            except Exception as e:
                print("[AI NORMALIZER ERROR]", e)
                self.mul_norm_text = text_lower"""
                
            Clock.schedule_once(lambda dt: setattr(self.text_input, "text", self.mul_norm_text), 0)
            best_score = 0.0
            forced_match = None

            for row in dataset_rows:
                sent = (row.get("yes_sentence") or "").strip().lower()
                if not sent:
                    continue

                if self.mul_norm_text in sent:
                    forced_match = {
                        "sentence": sent,
                        "label": (row.get("label") or "").lower(),
                        "category": (row.get("category") or "").lower(),
                    }
                    best_score = 1.0
                    break

                score = SequenceMatcher(None, self.mul_norm_text, sent).ratio()
                if score > best_score:
                    best_score = score
                    matched_sentence = sent
                    matched_label = (row.get("label") or "").lower()
                    matched_category = (row.get("category") or "").lower()

            if forced_match:
                matched_sentence = forced_match["sentence"]
                matched_label = forced_match["label"]
                matched_category = forced_match["category"]
                best_score = 1.0

            if best_score >= 0.85 and matched_sentence:

                def update_ai_match_ui(dt):
                    _, path = self.arasaac_func.get_cached_image(matched_label, PLACEHOLDER_PATH)
                    if not path:
                        url = self.arasaac_func.get_arasaac_image_url(matched_label)
                        path = url if url else PLACEHOLDER_PATH

                    widget = ClickableImage(
                        label_text=matched_label.capitalize(),
                        img_url=path,
                        callback=lambda *_: self.speak_text(matched_sentence)
                    )
                    self.result_grid.add_widget(widget)

                    # CATEGORY LIST
                    self.sug_list.clear_widgets()
                    for row in dataset_rows:
                        if (row.get("category") or "").lower() != matched_category:
                            continue

                        sentence = (row.get("yes_sentence") or "").strip()
                        label = (row.get("label") or "").lower()


                        _, path = self.arasaac_func.get_cached_image(label, PLACEHOLDER_PATH)
                        if not path:
                            url = self.arasaac_func.get_arasaac_image_url(label)
                            path = url if url else PLACEHOLDER_PATH

                        sug = SuggestionRow(
                            sentence,
                            ok_callback=self._accept_sentence,
                            bg_color=(0.9, 1, 0.9, 1)
                        )
                        sug.set_pictos([(label, path)])
                        self.sug_list.add_widget(sug)

                    self.content_area.clear_widgets()
                    self.content_area.add_widget(self.sug_scroll)
                    self.back_btn.opacity, self.back_btn.disabled = 1, False
                    self.dismiss_go_loading_popup()
                Clock.schedule_once(lambda dt: setattr(self.text_input, "text", self.mul_norm_text), 0)
                Clock.schedule_once(update_ai_match_ui, 0)
                return

            # ==========================================================
            # STEP 6 — DALL·E FALLBACK
            # ==========================================================

            # --- DALL·E fallback ---
            self.update_go_status("Generating image for sentence...")
            print(f"[DALL·E] No dataset match after AI normalization → generating image for '{self.mul_norm_text}'")
            try:
                from openai import OpenAI
                client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
                dalle_prompt = (
                    f"3D colorful icon, pictogram style, "
                    f"no text, minimal, high contrast; concept: generate an cartoon 3d icon for '{self.mul_norm_text}'."
                )
                resp = client.images.generate(
                    model="dall-e-3",
                    prompt=dalle_prompt,
                    size="1024x1024",
                    response_format="b64_json",
                    n=1,
                )
                b64 = resp.data[0].b64_json
                safe = re.sub(r"[^a-z0-9_]+", "_", self.mul_norm_text)
                out_path = os.path.join(tempfile.gettempdir(), f"dalle_{safe}.png")
                
                with open(out_path, "wb") as f:
                    f.write(base64.b64decode(b64))
                
                # ✅ Update UI on main thread
                def update_multi_dalle_ui(dt):
                    self.show_add_button_for_dalle_result(label=self.mul_norm_text, img_path=out_path, sentence=self.mul_norm_text)
                    self.dismiss_go_loading_popup()
                
                Clock.schedule_once(update_multi_dalle_ui, 0)
                return
                
            except Exception as e:
                print("[DALL·E ERROR]", e)
                Clock.schedule_once(lambda dt: self.dismiss_go_loading_popup(), 0)
                return


    def show_go_loading_popup(self):
        from kivy.uix.popup import Popup
        from kivy.uix.boxlayout import BoxLayout
        from kivy.uix.label import Label
        from kivy.uix.progressbar import ProgressBar
        from kivy.clock import Clock

        if hasattr(self, "go_loading_popup") and self.go_loading_popup:
            return

        layout = BoxLayout(orientation="vertical", padding=20, spacing=15)
        layout.add_widget(Label(text="Processing your input...", font_size="20sp"))

        self.go_progress_bar = ProgressBar(max=100, value=0)
        layout.add_widget(self.go_progress_bar)

        self.go_status_label = Label(text="Initializing...")
        layout.add_widget(self.go_status_label)

        self.go_loading_popup = Popup(
            title="Please wait",
            content=layout,
            size_hint=(0.45, None),
            height=200,
            auto_dismiss=False
        )
        self.go_loading_popup.open()

        def animate(dt):
            if hasattr(self, "go_loading_popup") and self.go_loading_popup:
                try:
                    current_value = self.go_progress_bar.value
                    self.go_progress_bar.value = (current_value + 2) % 100
                except Exception as e:
                    print(f"[PROGRESS ANIMATION ERROR] {e}")
                    return False  # Stop animation on error
            else:
                return False  # Stop animation if popup is gone


        self._go_anim = Clock.schedule_interval(animate, 0.1)


    def dismiss_go_loading_popup(self):
        from kivy.clock import Clock
        #Clock.schedule_once(lambda dt: setattr(self.text_input, "text", self.mul_norm_text), 0)
        if hasattr(self, "_go_anim"):
            Clock.unschedule(self._go_anim)

        if hasattr(self, "go_loading_popup") and self.go_loading_popup:
            self.go_loading_popup.dismiss()
            self.go_loading_popup = None


    def update_go_status(self, message):
        from kivy.clock import Clock
        if hasattr(self, "go_status_label"):
            Clock.schedule_once(lambda dt: setattr(self.go_status_label, "text", message), 0)


    


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
        # --- Show loading popup ---
        self.loading_popup = LoadingPopup()
        self.loading_popup.open()

        def _worker():
            from openai import OpenAI
            import re, time

            # ========== 1️⃣ FETCH DATABASE-BASED RECOMMENDATIONS ==========
            try:
                recent = self.db_help.retrive_last_inserted(self.location_label)
                location_matched = self.db_help.matched_loc(self.location_label)
                location_unmatched = self.db_help.unmatched_loc(self.location_label)
                time_unmatched = self.db_help.unmatched_time(self.location_label)
            except Exception as e:
                print(f"[RECOMMEND] Database retrieval error: {e}")
                recent, location_matched, location_unmatched, time_unmatched = None, [], [], []

            # --- Helper: Convert sentences into pictogram rows ---
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

            # ========== 2️⃣ AI-GENERATED AAC RECOMMENDATIONS ==========
            ai_rows = []
            try:
                location = self.location_label or "home"

                # Derive time phase dynamically
                hour = int(time.strftime("%H"))
                if 5 <= hour < 12:
                    time_phase = "morning"
                elif 12 <= hour < 16:
                    time_phase = "midday"
                elif 16 <= hour < 19:
                    time_phase = "afternoon"
                elif 19 <= hour < 22:
                    time_phase = "evening"
                else:
                    time_phase = "night"

                # Construct dynamic prompt
                prompt = f"""
                Generate 10 short, simple sentences commonly used by AAC users
                for daily communication at {location.lower()} in the {time_phase}.
                Each sentence should be natural, clear, and context-appropriate
                (e.g., greetings, needs, daily actions).
                Output as a numbered list.
                """

                print(f"[AI PROMPT] Location={location}, TimePhase={time_phase}")
                client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.7
                )

                output = response.choices[0].message.content
                print("[AI RAW OUTPUT]", output)

                # Extract clean sentences
                sentences = re.findall(r"\d+\.\s*(.+)", output)
                if not sentences:
                    sentences = [line.strip() for line in output.split("\n") if line.strip()]

                # --- Build pictogram rows + cache into DB ---
                for s in sentences:
                    clean_sentence = s.strip()
                    pic_items = [(t, self.arasaac_func.get_arasaac_image_url(t)) for t in all_tokens(clean_sentence)]
                    ai_rows.append((clean_sentence, pic_items))

                    # Insert into local DB for reuse (as AI-generated entry)
                    try:
                        self.db_help.insert(clean_sentence, location)
                        print(f"[AI→DB] Cached: '{clean_sentence}' @ {location}")
                    except Exception as e:
                        print(f"[AI→DB] Insert error for '{clean_sentence}': {e}")

            except Exception as e:
                print("[AI RECOMMENDATION ERROR]", e)
                ai_rows = []

            # ========== 3️⃣ UPDATE UI (on main thread) ==========
            def update_ui(dt):
                # Render DB-based sections
                self._render_suggestion_rows_sections(
                    recent_rows,
                    location_matched_rows,
                    location_unmatched_rows,
                    time_unmatched_rows
                )

                # Append AI-based section
                if ai_rows:
                    lbl = Label(
                        text="AI Recommendation",
                        size_hint=(1, None),
                        height=32,
                        color=(0, 0, 1, 1),
                        halign="left", valign="middle"
                    )
                    #lbl.bind(size=lambda *_: setattr(lbl, "text_size", lbl.size))
                    #self.sug_list.add_widget(lbl)

                    lbl.bind(size=lambda *_: setattr(lbl, "text_size", lbl.size))
                    self.sug_list.add_widget(lbl)

                    for sentence, pictos in ai_rows:
                        row = SuggestionRow(
                            sentence.capitalize(),
                            ok_callback=self._accept_sentence,
                            bg_color=(0.8, 0.9, 1, 1)  # light blue tone
                        )
                        row.set_pictos(pictos)
                        self.sug_list.add_widget(row)

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
    def show_listening_popup(self):
        """Show popup with microphone interface during voice capture."""
        from kivy.uix.boxlayout import BoxLayout
        from kivy.uix.label import Label
        from kivy.uix.button import Button
        from kivy.uix.popup import Popup
        from kivy.uix.image import Image
        
        # Main layout
        layout = BoxLayout(orientation="vertical", spacing=20, padding=20)
        
        # Microphone icon
        mic_icon = Image(
            source="View/icons/mic.png",  # Your existing mic icon
            size_hint=(None, None),
            size=(80, 80),
            allow_stretch=True,
            keep_ratio=True
        )
        # Center the mic icon
        mic_layout = BoxLayout(size_hint=(1, None), height=100)
        mic_layout.add_widget(Label())  # Left spacer
        mic_layout.add_widget(mic_icon)
        mic_layout.add_widget(Label())  # Right spacer
        layout.add_widget(mic_layout)
        
        # Listening status label
        self.listening_label = Label(
            text="🎤 Listening...",
            size_hint=(1, None),
            height=50,
            font_size="24sp",
            font_name="Roboto-Bold.ttf",
            color=(0, 0, 0, 1),
            halign="center"
        )
        layout.add_widget(self.listening_label)
        
        # Text recognized area
        self.recognized_text_label = Label(
            text="Speak now...",
            size_hint=(1, None),
            height=80,
            font_size="18sp",
            color=(0, 1, 0, 1),
            halign="center",
            valign="middle",
            text_size=(None, None)
        )
        layout.add_widget(self.recognized_text_label)
        
        # Stop button
        stop_button = Button(
            text="Stop Listening",
            size_hint=(1, None),
            height=60,
            background_color=(0.8, 0.2, 0.2, 1),
            color=(1, 1, 1, 1),
            font_size="20sp",
            font_name="Roboto-Bold.ttf"
        )
        stop_button.bind(on_press=self.stop_listening_from_popup)
        layout.add_widget(stop_button)
        
        # Create popup
        self.listening_popup = Popup(
            title="Voice Recognition",
            content=layout,
            size_hint=(0.6, None),  # 60% width, auto height
            height=400,
            auto_dismiss=False
        )
        
        self.listening_popup.open()


    def start_voice_capture(self, _=None):
        """Start voice capture and show listening popup."""
        import speech_recognition as sr
        import threading
        
        if self.listening:
            print("Already listening...")
            return

        # Show the listening popup
        self.show_listening_popup()
        
        # Start listening in background thread
        def listen_worker():
            try:
                self.r = sr.Recognizer()
                self.listening = True
                
                with sr.Microphone() as source:
                    # Update popup status
                    Clock.schedule_once(lambda dt: setattr(self.listening_label, 'text', '🎤 Adjusting for noise...'), 0)
                    
                    # Calibrate for background noise
                    self.r.adjust_for_ambient_noise(source, duration=1.0)
                    
                    # Update popup status
                    Clock.schedule_once(lambda dt: setattr(self.listening_label, 'text', '🎤 Listening...'), 0)
                    Clock.schedule_once(lambda dt: setattr(self.recognized_text_label, 'text', 'Speak clearly now...'), 0)
                    
                    # Force thresholds
                    self.r.energy_threshold = 250
                    self.r.dynamic_energy_threshold = False
                    
                    print(f"🎤 Listening... (threshold={self.r.energy_threshold})")
                    
                    # Listen for audio
                    audio = self.r.listen(source, phrase_time_limit=10)  # 10 seconds max
                    self.captured_audio = audio
                    
                    # Update status - processing
                    Clock.schedule_once(lambda dt: setattr(self.listening_label, 'text', '⏳ Processing...'), 0)
                    Clock.schedule_once(lambda dt: setattr(self.recognized_text_label, 'text', 'Recognizing speech...'), 0)
                    
                    # Recognize speech
                    try:
                        # Try Google Speech Recognition
                        result = self.r.recognize_google(audio, language="en-IN", show_all=False)
                        recognized_text = result.strip() if result else "<no speech>"
                        
                        print(f"✅ Recognized: {recognized_text}")
                        
                        # Update popup with result
                        Clock.schedule_once(lambda dt: setattr(self.listening_label, 'text', '✅ Recognition Complete'), 0)
                        Clock.schedule_once(lambda dt: setattr(self.recognized_text_label, 'text', f'Recognized: "{recognized_text}"'), 0)
                        
                        # Update main text input
                        Clock.schedule_once(lambda dt: setattr(self.text_input, 'text', recognized_text), 0)
                        
                        # Auto-close popup after 2 seconds and process result
                        def finish_recognition(dt):
                            if hasattr(self, 'listening_popup'):
                                self.listening_popup.dismiss()
                            self.process_speech_result(recognized_text)
                        
                        Clock.schedule_once(finish_recognition, 2.0)
                        
                    except sr.UnknownValueError:
                        # Could not understand audio
                        Clock.schedule_once(lambda dt: setattr(self.listening_label, 'text', '❌ Could not understand'), 0)
                        Clock.schedule_once(lambda dt: setattr(self.recognized_text_label, 'text', 'Please try speaking again...'), 0)
                        
                    except sr.RequestError as e:
                        # API error
                        Clock.schedule_once(lambda dt: setattr(self.listening_label, 'text', f'❌ API Error: {e}'), 0)
                        Clock.schedule_once(lambda dt: setattr(self.recognized_text_label, 'text', 'Speech service unavailable'), 0)
                        
            except Exception as e:
                print(f"[VOICE CAPTURE ERROR] {e}")
                Clock.schedule_once(lambda dt: setattr(self.listening_label, 'text', f'❌ Error: {str(e)}'), 0)
                
            finally:
                self.listening = False
        
        # Start listening in background
        threading.Thread(target=listen_worker, daemon=True).start()

    def stop_listening_from_popup(self, _=None):
        """Stop listening and close popup."""
        self.listening = False
        
        if hasattr(self, 'listening_popup'):
            self.listening_popup.dismiss()
        
        # If we have captured audio, process it
        if hasattr(self, 'captured_audio') and self.captured_audio:
            self.stop_and_recognize()
        else:
            print("No audio captured to process.")
            self.show_toast("No speech was captured.")

    def stop_and_recognize(self, _=None):
        if hasattr(self, 'listening_popup') and self.listening_popup:
            self.listening_popup.dismiss()
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
        # ✅ Use current sentence if no pictograms are selected
        if (not text or text.strip() == "") and hasattr(self, "current_sentence"):
            text = self.current_sentence

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
        Handles recognized speech and displays results in the suggestion area (bottom)
        instead of the top result grid.
        """
        if not recognized_text:
            print("[SPEECH] No text recognized.")
            return

        print(f"[SPEECH] Recognized: {recognized_text}")

        try:
            # Normalize + autocorrect
            norm_text = self.autocorrect_with_dataset(recognized_text)
            print(f"[SPEECH] Normalized + autocorrected text → {norm_text}")
            
            # ✅ DON'T clear result_grid - keep it as is
            # self.result_grid.clear_widgets()  # ❌ Remove this line

            # ✅ Update text input with recognized text
            self.text_input.text = norm_text

            # ✅ Clear suggestion area and display speech results there
            self.sug_list.clear_widgets()
            
            # Check if it's a category
            category = self.detect_category_from_text(norm_text)
            
            if category:
                print(f"[SPEECH] Detected category → {category}")
                # Show category sentences in suggestion area
                entries = self.get_sentences_for_category(category)
                
                for item in entries:
                    sentence = item.get("yes_sentence", "").strip()
                    label = item.get("label", "").strip()
                    if not sentence:
                        continue
                        
                    # Get pictogram
                    _, path = self.arasaac_func.get_cached_image(label, PLACEHOLDER_PATH)
                    if not path:
                        url = self.arasaac_func.get_arasaac_image_url(label)
                        path = url if url else PLACEHOLDER_PATH
                    
                    pictos = [(label, path)]
                    
                    row = SuggestionRow(
                        sentence.capitalize(),
                        ok_callback=self._accept_sentence,
                        bg_color=(0.9, 1, 0.9, 1)  # Light green for speech results
                    )
                    row.set_pictos(pictos)
                    self.sug_list.add_widget(row)
            else:
                # Not a category - show individual word pictograms in suggestion rows
                tokens = norm_text.split()
                
                # Create a suggestion row for the full sentence
                pictos = []
                for token in tokens:
                    _, path = self.arasaac_func.get_cached_image(token, PLACEHOLDER_PATH)
                    if not path:
                        url = self.arasaac_func.get_arasaac_image_url(token)
                        path = url if url else PLACEHOLDER_PATH
                    pictos.append((token, path))
                
                # Add the recognized sentence as a suggestion row
                row = SuggestionRow(
                    norm_text.capitalize(),
                    ok_callback=self._accept_sentence,
                    bg_color=(0.8, 0.9, 1, 1)  # Light blue for speech input
                )
                row.set_pictos(pictos)
                self.sug_list.add_widget(row)
            
            # ✅ Switch to suggestion view (bottom area)
            self.content_area.clear_widgets()
            self.content_area.add_widget(self.sug_scroll)
            self.back_btn.opacity = 1
            self.back_btn.disabled = False
            
            print(f"[SPEECH] Results displayed in suggestion area")

        except Exception as e:
            print(f"[SPEECH] Error during processing: {e}")



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
        
        # 🔹 Derive and store time phase once
        hour = int(time.strftime("%H"))
        if 5 <= hour < 12:
            self.time_phase = "morning"
        elif 12 <= hour < 16:
            self.time_phase = "midday"
        elif 16 <= hour < 19:
            self.time_phase = "afternoon"
        elif 19 <= hour < 22:
            self.time_phase = "evening"
        else:
            self.time_phase = "night"


    def go_to_firstscreen(self, *_):
        """Go back to the first (splash) screen."""
        app = App.get_running_app()
        app.sm.current = "splash"


