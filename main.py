from kivy.app import App
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.gridlayout import GridLayout
from kivy.uix.textinput import TextInput
from kivy.uix.button import Button
from kivy.clock import Clock
from kivy.uix.label import Label
from kivy.uix.scrollview import ScrollView
from kivy.graphics import Color, Rectangle
from kivy.core.window import Window
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.uix.anchorlayout import AnchorLayout
from kivy.uix.behaviors import ButtonBehavior
from kivy.uix.image import Image
import threading
import os
from arasaac_utils import *
import pygame
from gtts import gTTS
import requests
import speech_recognition as sr
from ClickableImage import *
from Virtualkeyboard import * 

Window.size = (1024, 600)

categories = [
    "food", "animals", "clothes", "emotions", "body", "sports", "school", "family",
    "nature", "transport", "weather", "home", "health", "jobs", "colors", "toys"
]
class AACApp(Screen):
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        self.showing_category = False
        root = FloatLayout()  # ✅ Root: allows overlay
        # Main left/right panel as before
        main_panel = BoxLayout(orientation='horizontal', size_hint=(1, 1))
        # Left panel
        left = BoxLayout(orientation='vertical', size_hint=(0.75, 1), spacing=5)
        self.text_input = TextInput(hint_text="Type...", multiline=False, size_hint=(2, 0.1), font_size='18sp')
        self.text_input.bind(on_text_validate=self.on_enter)
        self.text_input.bind(focus=self.on_focus)

        scroll = ScrollView(size_hint=(1.5, 0.2))
        self.result_grid = GridLayout(rows=1, spacing=5, size_hint_x=None, height=130)
        self.result_grid.bind(minimum_width=self.result_grid.setter('width'))
        scroll.add_widget(self.result_grid)

        self.image_grid = GridLayout(cols=4, rows=4, spacing=5, size_hint=(2, 1))
        self.back_button = Button(text='⬅ Back', font_size='18sp', size_hint=(1, None), height=40,
                                  background_color=(0.33,0.33,0.33,1), color=(1,1,1,1))
        self.back_button.bind(on_press=self.go_back)
        self.back_button.opacity = 0
        self.back_button.disabled = True

        left.add_widget(self.text_input)
        left.add_widget(scroll)
        left.add_widget(self.back_button)
        left.add_widget(self.image_grid)


        # Right panel
        right = BoxLayout(orientation='vertical', size_hint=(0.25, 1), spacing=10, padding=10)
        for label, callback in [
            ("Mic", self.start_voice_thread),
            ("Retry", self.on_retry_press),
            ("Speak", self.on_speak_press),
            ("Display", self.process_input)
        ]:
            btn = Button(text=label, font_size='20sp', size_hint=(1, None), height=80)
            btn.bind(on_press=callback)
            right.add_widget(btn)

        main_panel.add_widget(left)
        main_panel.add_widget(right)

        root.add_widget(main_panel)

        # ✅ Create keyboard but keep hidden initially
        self.virtual_keyboard = VirtualKeyboard(self.text_input)
        self.virtual_keyboard.opacity = 0
        self.virtual_keyboard.disabled = True
        root.add_widget(self.virtual_keyboard)

        self.call_populate_async()
        self.add_widget(root)

    def on_focus(self, instance, value):
        """ Show/hide keyboard based on focus """
        if value:  # Focused
            self.virtual_keyboard.opacity = 1
            self.virtual_keyboard.disabled = False
        else:      # Unfocused
            self.virtual_keyboard.opacity = 0
            self.virtual_keyboard.disabled = True

    def call_populate_async(self):
        self.show_loading_spinner()
        threading.Thread(target=self.populate_category_grid, daemon=True).start()

    def populate_category_grid(self):
        widgets_data = []

        for cat in categories:
            url = get_arasaac_image_url(cat)  # This is safe in background
            widgets_data.append((cat, url))

        # Now ask Kivy to build widgets on the main thread
        Clock.schedule_once(lambda dt: self.create_widgets_on_main_thread(widgets_data))
    def show_loading_spinner(self):
        #self.image_grid.clear_widgets()
        #self.loading_image = Image(source='loading.gif')  # Can be static or animated
        #self.image_grid.add_widget(self.loading_image)
        self.image_grid.clear_widgets()
        anchor = AnchorLayout(anchor_x='center', anchor_y='center',size_hint=(2,1))
        self.loading_image = Image(
            source='loading.gif',
            size_hint=(None, None),
            size=(150, 150),  # You can change this size as needed
        )
        anchor.add_widget(self.loading_image)
        self.image_grid.add_widget(anchor)

    def create_widgets_on_main_thread(self, widgets_data):
        widgets = []

        for cat, url in widgets_data:
            widget = ClickableImage(cat, url, self.update_category_images)  # Now safe
            widgets.append(widget)

        self.update_grid(widgets)

    def update_grid(self, widgets):
        self.image_grid.clear_widgets()
        for widget in widgets:
            self.image_grid.add_widget(widget)

    def update_category_images(self, category, _):
        self.image_grid.clear_widgets()
        pictos = fetch_pictograms(category)
        for label, url in pictos:
            widget = ClickableImage(label, url, self.add_result_widget)
            self.image_grid.add_widget(widget)
        self.back_button.opacity = 1
        self.back_button.disabled = False
        self.showing_category = True

    def go_back(self, instance):
        self.image_grid.clear_widgets()
        self.call_populate_async()
        self.back_button.opacity = 0
        self.back_button.disabled = True
        self.showing_category = False

    def on_enter(self):
        #print("Entered:", self.text_input.text)
        print("AAC screen loaded")

    def add_result_widget(self, word, url):
        widget = ClickableImage(word, url)
        widget.callback = lambda *args: self.remove_result_widget(widget)
        self.result_grid.add_widget(widget)

    def remove_result_widget(self, widget):
        self.result_grid.remove_widget(widget)

    def process_input(self, instance):
        text = self.text_input.text
        tokens = tokenize_text(text)
        self.result_grid.clear_widgets()
        for token in tokens:
            url = get_arasaac_image_url(token)
            if url:
                self.add_result_widget(token, url)

    def start_voice_thread(self, instance):
        threading.Thread(target=self.capture_voice, daemon=True).start()

    def capture_voice(self):
        r = sr.Recognizer()
        with sr.Microphone() as source:
            print("Say something...")
            audio = r.listen(source)
        try:
            text = r.recognize_google(audio)
            Clock.schedule_once(lambda dt: setattr(self.text_input, 'text', text))
        except Exception as e:
            print(f"Voice error: {e}")

    def on_retry_press(self, instance):
        self.text_input.text = ""
        self.result_grid.clear_widgets()

    def on_speak_press(self, instance):
        text = self.text_input.text.strip()
        if text:
            speak = gTTS(text=text, lang='en', slow=False)
            speak.save("tmp.mp3")
            pygame.mixer.init()
            pygame.mixer.music.load("tmp.mp3")
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy():
                pygame.time.Clock().tick(10)
            pygame.mixer.quit()
            os.remove("tmp.mp3")

    def exit(self, instance):
        App.get_running_app().stop()

