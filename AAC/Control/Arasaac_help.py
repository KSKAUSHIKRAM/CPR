from Model.Arasaac_helpers import Arasaac_helper

class Arasaac_help:
    def image_url(self, word):
        return Arasaac_helper.get_arasaac_image_url(word)

    def fetch_pictograms(self, category):
        return Arasaac_helper.fetch_pictograms(category)