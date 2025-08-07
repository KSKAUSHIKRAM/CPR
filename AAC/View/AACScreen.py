from kivy.app import App
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.gridlayout import GridLayout
from kivy.uix.textinput import TextInput
from kivy.uix.image import AsyncImage, Image
from kivy.uix.button import ButtonBehavior, Button
from kivy.uix.scrollview import ScrollView
from kivy.uix.label import Label
from kivy.clock import Clock
from kivy.graphics import Color, RoundedRectangle, Rectangle
from kivy.core.window import Window
import threading
import os
import pygame
from gtts import gTTS
import requests
import speech_recognition as sr
import shelve
import time
from kivy.properties import StringProperty
from Control.Controller import  insert
# --- Configuration ---
Window.size = (1024, 600)
Window.clearcolor = (1, 1, 1, 1)

categories = [
    "food", "animals", "clothes", "emotions", "body", "sports", "school", "family",
    "nature", "transport", "weather", "home", "health", "jobs", "colors", "toys"
]
CACHE_DB = "cache_urls.db"

# --- Helper functions ---
def get_arasaac_image_url(word):
    with shelve.open(CACHE_DB) as cache:
        if word in cache:
            return cache[word]
        url = f"https://api.arasaac.org/api/pictograms/en/search/{word}"
        try:
            response = requests.get(url, timeout=5)
            data = response.json()
            if data and isinstance(data, list):
                pic_id = data[0]["_id"]
                img_url = f"https://static.arasaac.org/pictograms/{pic_id}/{pic_id}_500.png"
                cache[word] = img_url
                return img_url
        except Exception as e:
            print(f"API error: {e}")
    return ""

def fetch_pictograms(category):
    url = f"https://api.arasaac.org/api/pictograms/en/search/{category}"
    pictos = []
    try:
        response = requests.get(url, timeout=5)
        data = response.json()
        for item in data[:16]:
            label = item.get("keywords", [{}])[0].get("keyword", category)
            pic_id = item["_id"]
            img_url = f"https://static.arasaac.org/pictograms/{pic_id}/{pic_id}_500.png"
            pictos.append((label, img_url))
    except Exception as e:
        print(f"Fetch pictograms failed: {e}")
    return pictos

# --- Custom Widgets ---
class ClickableImage(ButtonBehavior, BoxLayout):
    def __init__(self, label_text, img_url, callback=None, bg_color=(0.8,1,0.8,1), **kwargs):
        super().__init__(orientation='vertical', size_hint=(None, None), size=(100, 100), **kwargs)
        with self.canvas.before:
            Color(*bg_color)
            self.bg = RoundedRectangle(size=self.size, pos=self.pos, radius=[10])
        self.img = AsyncImage(source=img_url, allow_stretch=True, keep_ratio=True, size_hint=(1, 0.8))
        self.label = Label(text=label_text.capitalize(), size_hint=(1, 0.2),
                           color=(0,0,0,1), halign='center')
        self.label.bind(size=self.label.setter('text_size'))
        self.add_widget(self.img)
        self.add_widget(self.label)
        self.callback = callback
        self.bind(pos=self._update_bg, size=self._update_bg)

    def _update_bg(self, *args):
        self.bg.size = self.size
        self.bg.pos = self.pos

    def on_press(self):
        if self.callback:
            self.callback(self.label.text, self.img.source)

class ImageLabelButton(ButtonBehavior, BoxLayout):
    image_source = StringProperty('')
    text = StringProperty('')

    def __init__(self, img_source, text, callback=None, **kwargs):
        super().__init__(orientation='vertical', spacing=2, size_hint=(1, None), height=80,**kwargs)
        self.image_source = img_source
        self.text = text
        self.callback = callback

        # Image takes 70% height
        img = Image(
            source=img_source if os.path.exists(img_source) else 'View/icons/placeholder.png',
            size_hint=(1, 0.2),
            allow_stretch=True,
        keep_ratio=True,
        size_hint_y=0.6
        )
        #img.size_hint=(1,0.4)
        lbl = Label(
            text=text,
            size_hint=(1, 0.3),
            halign='center',
            valign='middle',
            color=(0, 0, 0, 1),
            font_size='14sp',
            size_hint_y=0.4
        )
        #lbl.size_hint=(1,0.6)
        lbl.bind(size=lbl.setter('text_size'))

        self.add_widget(img)
        self.add_widget(lbl)

    def on_press(self):
        if self.callback:
            self.callback(None)



# --- Main App ---
class AACApp(App):
    def __init__(self, location_label, **kwargs):
        super().__init__(**kwargs)
        self.location_label = location_label

    def build(self):
        root = FloatLayout()
        main = BoxLayout(orientation='horizontal')

        # --- Left panel ---
        self.left = BoxLayout(orientation='vertical', spacing=0, padding=0)
        with self.left.canvas.before:
            Color(0.95, 0.95, 0.95, 1)
            self.left_bg = Rectangle()
        self.left.bind(pos=self.update_left_bg, size=self.update_left_bg)

        self.text_input = TextInput(hint_text="Type...", multiline=False, size_hint=(1,None), height=60)
        self.text_input.bind(on_text_validate=self.on_enter)

        scroll = ScrollView(size_hint=(1,None), height=100)
        self.result_grid = GridLayout(rows=1, spacing=1, size_hint_x=None, height=150)
        self.result_grid.bind(minimum_width=self.result_grid.setter('width'))
        scroll.add_widget(self.result_grid)
        with scroll.canvas.before:
            Color(1, 1, 0.8, 1)
            self.display_bg = RoundedRectangle(radius=[10])
        scroll.bind(pos=self.update_display_bg, size=self.update_display_bg)

        self.image_grid = GridLayout(cols=8, spacing=[10,15], size_hint=(1,1), padding=[10,25,10,0])
        with self.image_grid.canvas.before:
            Color(0.8, 1, 0.8, 1)
            self.category_bg = RoundedRectangle(radius=[10])
        self.image_grid.bind(pos=self.update_category_bg, size=self.update_category_bg)

        self.left.add_widget(self.text_input)
        self.left.add_widget(scroll)
        self.left.add_widget(self.image_grid)

        # --- Taskbar ---
        self.taskbar = BoxLayout(orientation='horizontal', size_hint=(1,None), height=30, padding=5, spacing=5)
        with self.taskbar.canvas.before:
            Color(0.2,0.2,0.2,1)
            self.taskbar_bg = Rectangle()
        self.taskbar.bind(pos=self.update_taskbar_bg, size=self.update_taskbar_bg)

        self.back_button = Button(
            text='Back', size_hint=(None,1), width=60, background_normal='', background_color=(1,0,0,1),
            color=(1,1,1,1)
        )
        self.back_button.bind(on_press=self.go_back)
        self.back_button.opacity = 0
        self.back_button.disabled = True

        self.task_label = Label(text=f"Location: {self.location_label} | Time: {time.strftime('%H:%M:%S')}",
                                color=(1,1,1,1), halign='center')
        Clock.schedule_interval(self.update_time, 1)

        self.taskbar.add_widget(self.back_button)
        self.taskbar.add_widget(self.task_label)
        self.left.add_widget(self.taskbar)

        # --- Right panel with new order ---
        self.right = BoxLayout(orientation='vertical',
    size_hint_x=None,
    height=50,
    width=120,  # enough width for icons + text
    spacing=10,
    padding=[5,10,5,10]
)
        with self.right.canvas.before:
            Color(1, 0.8, 0.9, 1)
            self.right_bg = Rectangle()
        self.right.bind(pos=self.update_right_bg, size=self.update_right_bg)

        buttons_info = [
            ("View/icons/grid.png", "Browse", lambda _: self.show_all_categories()),
            ("View/icons/mic.png", "Speak", self.start_voice_thread),
            ("View/icons/recommend.png", "Recommend", self.recommend),
            ("View/icons/try.png", "Try Again", self.clear_all),
            ("View/icons/display.png", "Display", self.process_text),
            ("View/icons/speak.jpg", "Listen", self.speak_text),
            ("View/icons/exit.png", "Exit", self.exit_app)
        ]
        num_buttons=len(buttons_info)
        for icon, text, method in buttons_info:
            btn = ImageLabelButton(icon, text, callback=method)
            btn.size_hint_y=1/num_buttons
            self.right.add_widget(btn)

        main.add_widget(self.left)
        main.add_widget(self.right)
        root.add_widget(main)

        threading.Thread(target=self.load_categories, daemon=True).start()
        return root

    # --- Background updates ---
    def update_left_bg(self, *args): self.left_bg.pos = self.left.pos; self.left_bg.size = self.left.size
    def update_display_bg(self, *args): self.display_bg.pos = self.result_grid.parent.pos; self.display_bg.size = self.result_grid.parent.size
    def update_category_bg(self, *args): self.category_bg.pos = self.image_grid.pos; self.category_bg.size = self.image_grid.size
    def update_right_bg(self, *args): self.right_bg.pos = self.right.pos; self.right_bg.size = self.right.size
    def update_taskbar_bg(self, *args): self.taskbar_bg.pos = self.taskbar.pos; self.taskbar_bg.size = self.taskbar.size

    # --- Functionality ---
    def on_enter(self, instance): self.process_text(None)
    def process_text(self, _):
        tokens = self.text_input.text.lower().split()
        self.result_grid.clear_widgets()
        for token in tokens:
            url = get_arasaac_image_url(token)
            if url:
                widget = ClickableImage(token, url, bg_color=(1,1,0.8,1))
                widget.callback = lambda *_: self.result_grid.remove_widget(widget)
                self.result_grid.add_widget(widget)
        insert(tokens,self.location_label)
    def load_categories(self):
        for cat in categories:
            Clock.schedule_once(lambda dt, cat=cat: self.add_placeholder(cat))
        for idx, cat in enumerate(categories):
            url = get_arasaac_image_url(cat)
            Clock.schedule_once(lambda dt, idx=idx, url=url: self.update_icon(idx, url))
    def add_placeholder(self, cat):
        self.image_grid.add_widget(ClickableImage(cat, "", self.show_category, bg_color=(0.8,1,0.8,1)))
    def update_icon(self, idx, url):
        widget = self.image_grid.children[::-1][idx]
        widget.img.source = url
    def show_category(self, category, _):
        pictos = fetch_pictograms(category)
        self.image_grid.clear_widgets()
        for label, url in pictos:
            self.image_grid.add_widget(ClickableImage(label, url, self.add_to_result, bg_color=(0.8,1,0.8,1)))
        self.back_button.opacity = 1
        self.back_button.disabled = False
    def show_all_categories(self, _=None):
        self.go_back(None)
    def go_back(self, instance):
        self.image_grid.clear_widgets()
        threading.Thread(target=self.load_categories, daemon=True).start()
        self.back_button.opacity = 0
        self.back_button.disabled = True
    def add_to_result(self, label, url):
        widget = ClickableImage(label, url, bg_color=(1,1,0.8,1))
        widget.callback = lambda *_: self.result_grid.remove_widget(widget)
        self.result_grid.add_widget(widget)
    def recommend(self, _): pass
    def start_voice_thread(self, _): threading.Thread(target=self.capture_voice, daemon=True).start()
    def capture_voice(self):
        r = sr.Recognizer()
        with sr.Microphone() as src:
            audio = r.listen(src)
        try:
            text = r.recognize_google(audio)
            Clock.schedule_once(lambda dt: setattr(self.text_input, 'text', text))
        except Exception as e:
            print(f"Voice error: {e}")
    def speak_text(self, _):
        text = self.text_input.text.strip()
        if text:
            try:
                tts = gTTS(text=text, lang='en', slow=False)
                tmp = "tmp.mp3"
                tts.save(tmp)
                pygame.mixer.init()
                pygame.mixer.music.load(tmp)
                pygame.mixer.music.play()
                while pygame.mixer.music.get_busy():
                    pygame.time.Clock().tick(10)
                pygame.mixer.quit()
                os.remove(tmp)
            except Exception as e:
                print(f"TTS error: {e}")
    def clear_all(self, _):
        self.text_input.text = ""
        self.result_grid.clear_widgets()
    def exit_app(self, _): App.get_running_app().stop()
    def update_time(self, dt):
        self.task_label.text = f"Location: {self.location_label} | Time: {time.strftime('%H:%M:%S')}"

# --- Entry point ---
def run_screen1(location_label):
    AACApp(location_label).run()