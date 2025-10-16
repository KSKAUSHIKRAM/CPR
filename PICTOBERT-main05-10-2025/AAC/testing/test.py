# =========================================================
# Imports
# =========================================================
from kivy.config import Config
from kivy.core.window import Window
Config.set('graphics', 'borderless', '1')
Config.set('graphics', 'resizable', '0')
Config.set('graphics', 'width', '1024')
Config.set('graphics', 'height', '600')
Window.clearcolor = (1, 1, 1, 1)
Config.write()

import os
import pandas as pd
import threading
from kivy.app import App
from kivy.clock import Clock
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

# =========================================================
# Load dataset with ARASAAC URLs
# =========================================================
df = pd.read_csv("fixed_duplicates.csv")

# Create a category → words mapping from your dataset
categories = {
    "food": ["bread", "rice", "milk", "juice"],
    "animals": ["dog", "cat", "cow"],
    "clothes": ["shirt", "pants", "shoes"],
    # Add more categories if needed
}

# Function to get ARASAAC URL from the dataset
def get_arasaac_url(word):
    row = df[df['word'].str.lower() == word.lower()]
    if not row.empty:
        return row.iloc[0]['ARASAAC_URL']
    return None

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
        box = BoxLayout(orientation='vertical', padding=20)
        box.add_widget(Label(text="Loading... Please wait"))
        self.content = box

class ClickableImage(ButtonBehavior, BoxLayout):
    def __init__(self, label_text, img_url, callback=None, **kwargs):
        super().__init__(orientation='vertical', size_hint=(None, None), size=(100, 100), **kwargs)
        self.img = AsyncImage(source=img_url or "", allow_stretch=True, keep_ratio=True, size_hint=(1, 0.8))
        self.label = Label(text=label_text.capitalize(), size_hint=(1, 0.2), color=(0,0,0,1), halign='center')
        self.label.bind(size=self.label.setter('text_size'))
        self.add_widget(self.img)
        self.add_widget(self.label)
        self.callback = callback

    def on_press(self):
        if self.callback:
            self.callback(self.label.text, self.img.source)

# =========================================================
# AAC Screen
# =========================================================
class AACScreen(Screen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.page_size = 8  # pictograms per row
        self.category_words = []  # words in current category
        self.current_page = 0
        self._built = False
        Clock.schedule_once(lambda dt: self.build_ui())

    def build_ui(self):
        if self._built:
            return
        self._built = True

        root = FloatLayout()
        self.add_widget(root)

        # Main horizontal layout
        main = BoxLayout(orientation='horizontal')
        root.add_widget(main)

        # Left panel (categories + pictograms)
        self.left_panel = BoxLayout(orientation='vertical', size_hint=(0.9, 1))
        main.add_widget(self.left_panel)

        # Right panel (buttons)
        self.right_panel = BoxLayout(orientation='vertical', size_hint=(0.1, 1))
        main.add_widget(self.right_panel)

        # --- Category scroll area ---
        self.image_grid = GridLayout(cols=self.page_size, spacing=10, padding=10, size_hint_y=None, height=150)
        self.image_grid.bind(minimum_width=self.image_grid.setter('width'))

        self.category_scroll = ScrollView(size_hint=(1, None), height=150, do_scroll_x=True, do_scroll_y=False)
        self.category_scroll.add_widget(self.image_grid)
        self.left_panel.add_widget(self.category_scroll)

        # --- Pagination buttons ---
        self.page_controls = BoxLayout(size_hint=(1, None), height=40, spacing=5, padding=5)
        self.prev_btn = Button(text="Previous", on_press=lambda *_: self.change_page(-1))
        self.next_btn = Button(text="Next", on_press=lambda *_: self.change_page(1))
        self.page_controls.add_widget(self.prev_btn)
        self.page_controls.add_widget(self.next_btn)
        self.left_panel.add_widget(self.page_controls)

        # --- Load category buttons ---
        for cat in categories.keys():
            btn = Button(text=cat.capitalize(), size_hint=(1, None), height=40)
            btn.bind(on_press=lambda btn, c=cat: self.load_category(c))
            self.right_panel.add_widget(btn)

    # Load words of selected category
    def load_category(self, category):
        self.category_words = categories.get(category, [])
        self.current_page = 0
        self.display_page()

    # Display pictograms for current page
    def display_page(self):
        self.image_grid.clear_widgets()
        start = self.current_page * self.page_size
        end = start + self.page_size
        for word in self.category_words[start:end]:
            url = get_arasaac_url(word)
            widget = ClickableImage(word, url)
            self.image_grid.add_widget(widget)
        # Enable/disable buttons
        self.prev_btn.disabled = self.current_page == 0
        self.next_btn.disabled = end >= len(self.category_words)

    # Change page
    def change_page(self, direction):
        self.current_page += direction
        self.display_page()

# =========================================================
# Run app
# =========================================================
class AACApp(App):
    def build(self):
        from kivy.uix.screenmanager import ScreenManager
        sm = ScreenManager()
        sm.add_widget(AACScreen(name='aac'))
        return sm

if __name__ == '__main__':
    AACApp().run()
