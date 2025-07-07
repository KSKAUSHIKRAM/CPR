import threading
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.textinput import TextInput
from kivy.uix.button import Button
from kivy.uix.image import AsyncImage
from kivy.core.window import Window  # <--- add this
from kivy.clock import Clock
from arasaac_utils import get_arasaac_image_url, tokenize_text
from kivy.core.window import Window
from kivy.uix.anchorlayout import AnchorLayout
import speech_recognition as sr

Window.size=(800,600)
class MyApp(App):
    def build(self):
        Window.clearcolor = (1, 1, 1, 1)  # <--- white background
        self.layout = BoxLayout(orientation='vertical', padding=10, spacing=20)

        self.text_input = TextInput(
            hint_text="Type something",
            multiline=False,
            size_hint=(None, None),
            size=(300, 50),
            foreground_color=(0, 0, 0, 1),       # black text
            background_color=(0.9, 0.9, 0.9, 1), # light grey background
            cursor_color=(1, 0, 0, 1),           # red cursor
            hint_text_color=(0.5, 0.5, 0.5, 1),  # grey hint
            font_size=18
        )
        top_container = AnchorLayout(
        anchor_x='left', 
        anchor_y='top',
        size_hint_y=None,
        height=350
    )
        
        top_container.add_widget(self.text_input)
        

        self.button1 = Button(text="Print Text", size_hint_y=(1,None), height=40)
        self.button1.bind(on_press=self.process_input)
        self.button2 = Button(text="Speech", size_hint_y=(1,None), height=30)
        self.button2.bind(on_press=self.voice_recognition_threaded)
        self.result_area = BoxLayout(orientation='horizontal', size_hint_y=(1,None), height=250, spacing=10)

        self.layout.add_widget(top_container)
        self.layout.add_widget(self.button1)
        self.layout.add_widget(self.button2)
        self.layout.add_widget(self.result_area)
        return self.layout
    def process_input_threaded(self, instance):
        threading.Thread(target=self.process_input, daemon=True).start()

    def process_input(self, instance):
        text = self.text_input.text
        tokens = tokenize_text(text)
        Clock.schedule_once(lambda dt:self.result_area.clear_widgets())

        for token in tokens:
            img_url = get_arasaac_image_url(token)
            if img_url:
                Clock.schedule_once(lambda dt, url=img_url: self.result_area.add_widget(AsyncImage(source=url)))
    def voice_recognition_threaded(self,instance):
        threading.Thread(target=self.voice_recognition,daemon=True).start()
    def voice_recognition(self):
        r=sr.Recognizer()
        with sr.Microphone() as source:
            print("Say Something")
            audio=r.listen(source)
        try:
            text=r.recognize_google(audio)
            print(f"You said:{text}")
            Clock.schedule_once(lambda dt: self.text_input.setter('text')(self.text_input, text))
 
        except sr.UnknownValueError:
            print("Could not understand audio")
    

if __name__ == "__main__":
    MyApp().run()
