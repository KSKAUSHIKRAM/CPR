from kivy.uix.behaviors import ButtonBehavior
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.image import AsyncImage
from kivy.uix.label import Label
from kivy.graphics import Color, Rectangle
from kivy.uix.gridlayout import GridLayout
from kivy.uix.button import Button

class ClickableImage(ButtonBehavior, BoxLayout):
    def __init__(self, label_text, img_url, callback=None, **kwargs):
        super().__init__(orientation='vertical', size_hint=(None, None), size=(120, 120), **kwargs)
        self.label_text = label_text
        self.img_url = img_url
        self.callback = callback

        self.img = AsyncImage(source=img_url, allow_stretch=True, keep_ratio=True, size_hint=(1, 0.8))
        self.label = Label(text=label_text.capitalize(), size_hint=(1, 0.2), color=(0, 0, 0, 1), halign='center')
        self.label.bind(size=self.label.setter('text_size'))

        self.add_widget(self.img)
        self.add_widget(self.label)

        with self.canvas.before:
            Color(1, 1, 1, 1)
            self.bg = Rectangle(size=self.size, pos=self.pos)
        self.bind(size=self._update_bg, pos=self._update_bg)

    def _update_bg(self, *args):
        self.bg.size = self.size
        self.bg.pos = self.pos

    def on_press(self):
        if self.callback:
            self.callback(self.label_text, self.img_url)

