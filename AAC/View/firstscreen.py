from kivy.uix.behaviors import ButtonBehavior
from kivy.app import App
from kivy.properties import StringProperty
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.screenmanager import Screen,ScreenManager,FadeTransition
from kivy.lang import Builder
from kivy.uix.label import Label
from kivy.config import Config

class ImageLabelButton(ButtonBehavior, BoxLayout):
    image_source = StringProperty('')
    text = StringProperty('')

    def __init__(self, image_source='', text='', on_select=None, **kwargs):
        super().__init__(**kwargs)
        self.image_source = image_source
        self.text = text
        self.on_select = on_select

    def on_press(self):
        if self.on_select:
            self.on_select(self.text)

# ✅ Screens
class SplashScreen(Screen):
    def on_kv_post(self, base_widget):
        places = [
            ('Home', 'View/icons/Home.png'),
            ('Hospital / Clinic', 'View/icons/hospital.jpg'),
            ('Shop / Grocery', 'View/icons/shop.png'),
            ('Work / volunteering', 'View/icons/office.png'),
            ('Restaurant / café', 'View/icons/hotel.png'),
            ('Religious place', 'View/icons/religion.jpg'),
            ('Outside (Other)', 'View/icons/outside.jpg')
        ]
        row = self.ids.location_row
        for label, icon in places:
            btn = ImageLabelButton(image_source=icon, text=label, on_select=self.select_location)
            row.add_widget(btn)

    def select_location(self, location_label):
        print(f"Selected location: {location_label}")
        self.manager.get_screen('main').location_label = location_label
        self.manager.current = 'main'

class MainScreen(Screen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.location_label = None

    def on_enter(self):
        if hasattr(self, 'launched') and self.launched:
            return
        self.launched = True

        from View import AACScreen
        location = self.location_label or "Unknown"
        App.get_running_app().stop()
        AACScreen.run_screen1(location)
    def get_location(self):
        return self.location_label

# ✅ Load KV file after all class definitions
Builder.load_file("View/Design.kv")


        