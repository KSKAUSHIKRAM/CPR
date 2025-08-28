from kivy.uix.behaviors import ButtonBehavior
from kivy.properties import StringProperty
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.screenmanager import Screen
from kivy.lang import Builder

# ------------------------
# Image + Label Button
# ------------------------
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


# ------------------------
# SplashScreen
# ------------------------
class SplashScreen(Screen):
    def on_kv_post(self, base_widget):
        """Called after KV is applied, safe to access ids"""
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
        row.clear_widgets()

        for label, icon in places:
            btn = ImageLabelButton(
                image_source=icon,
                text=label,
                on_select=self.select_location
            )
            row.add_widget(btn)

    def select_location(self, location_label):
        """Handle location selection"""
        print(f"Selected location: {location_label}")
        aac_screen = self.manager.get_screen('aac')
        aac_screen.location_label = location_label
        aac_screen._built = False  # reset if AAC screen needs rebuild
        self.manager.current = 'aac'


# ------------------------
# Load Design.kv
# ------------------------
# Important: only load rules (no <ScreenManager> in KV, no duplicate screens!)
Builder.load_file("View/Design.kv")
