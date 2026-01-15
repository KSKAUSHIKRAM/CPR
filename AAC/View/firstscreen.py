from kivy.uix.behaviors import ButtonBehavior
from kivy.properties import StringProperty
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.screenmanager import Screen
from kivy.lang import Builder
from kivy.core.window import Window
from kivy.uix.image import AsyncImage
from kivy.uix.label import Label




class ImageLabelButton(ButtonBehavior, BoxLayout):
    image_source = StringProperty('')
    text = StringProperty('')

    def __init__(self, image_source='', text='', on_select=None, **kwargs):
        super().__init__(orientation='vertical', spacing=5, **kwargs)
        self.image_source = image_source
        self.text = text
        self.on_select = on_select

        # Dynamically estimate width for up to 8–10 icons
        total_icons = 8
        available_width = Window.width - (total_icons * 20)
        w = available_width / total_icons
        h = Window.height * 0.22

        # --- Image ---
        self.img = AsyncImage(
            source=self.image_source,
            allow_stretch=True,
            keep_ratio=True,
            size_hint=(1, None),
            height=h * 0.7
        )
        self.add_widget(self.img)

        # --- Label (supports multi-line text) ---
        self.lbl = Label(
            text=self.text,
            font_name="Roboto-Bold.ttf",
            size_hint=(1, None),
            height=h * 0.3,  # bottom 30%
            color=(0, 0, 0, 1),
            halign='center',
            valign='middle',
            text_size=(w, None),
            font_size=str(int(Window.height * 0.022)) + "sp"
        )
        self.lbl.bind(size=lambda *_: setattr(self.lbl, "text_size", self.lbl.size))
        self.add_widget(self.lbl)

        # Set button size
        self.size_hint = (None, None)
        self.size = (w, h)

        # Recalculate if window resized
        Window.bind(size=self._resize)

    def _resize(self, *_):
        """Recalculate sizes dynamically when window changes."""
        total_icons = 8
        available_width = Window.width - (total_icons * 20)
        w = available_width / total_icons
        h = Window.height * 0.22

        self.size = (w, h)
        self.img.height = h * 0.7
        self.lbl.height = h * 0.3
        self.lbl.font_size = str(int(Window.height * 0.022)) + "sp"
        self.lbl.text_size = (w, None)

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
            ('Outside (Other)', 'View/icons/outside.jpg'),
            ('Exit', 'View/icons/exit.png')
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
        if location_label == 'Exit':
            App.get_running_app().stop()
            return
        print(f"Selected location: {location_label}")
        aac_screen = self.manager.get_screen('aac')
        aac_screen.location_label = location_label
        aac_screen._built = False  # reset if AAC screen needs rebuild
        self.manager.current = 'aac'


# ------------------------
# Load Design.kv
# ------------------------
Builder.load_file("View/Design.kv")
