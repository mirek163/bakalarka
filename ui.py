import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
from tkinter import filedialog
from functools import partial
from PIL import ImageTk, Image
import os
import glob
from main import GAN


class GANUI:
    def __init__(self):
        # ------------------------------------------------------
        # Nastavení počátečních proměných
        # ------------------------------------------------------
        noise_type = 'random'  # Výchozí typ šumu
        epoch_number = 100000  # Výchozí počet epoch
        self.gan = GAN(noise_type)

        # ------------------------------------------------------
        # Nastavení okna
        # ------------------------------------------------------

        self.window = tk.Tk()
        self.window.title("Generátor obrázků pomocí GAN")
        self.window.geometry("1000x600")  # Nastavení počáteční velikosti okna

        # Horní panel s výběrem šumu a vstupním polem pro počet epoch
        top_bar = ttk.Frame(self.window, padding="10")
        top_bar.pack(side="top", fill="x")

        noise_label = ttk.Label(top_bar, text="Typ šumu:")
        noise_label.pack(side="left")

        noise_options = ['random', 'perlin', 'simplex']
        noise_var = tk.StringVar(value=noise_type)

        noise_dropdown = ttk.Combobox(top_bar, textvariable=noise_var, values=noise_options)
        noise_dropdown.pack(side="left")

        epoch_label = ttk.Label(top_bar, text="Počet epoch:")
        epoch_label.pack(side="left")

        epoch_entry = ttk.Entry(top_bar)
        epoch_entry.insert(tk.END, str(epoch_number))
        epoch_entry.pack(side="left")

        generate_button = ttk.Button(top_bar, text="Generovat",
                                     command=partial(self.generate_images, noise_var, epoch_entry))
        generate_button.pack(side="left")

        delete_button = ttk.Button(top_bar, text="Smazat obrázky", command=partial(self.delete_images, noise_var))
        delete_button.pack(side="left")

        # Label pro zobrazení obrázku
        self.image_label = ttk.Label(self.window)
        self.image_label.pack()

    def generate_images(self, noise_var, epoch_entry):
        """
        Generuje obrázky pomocí GAN na základě vybraného typu šumu a počtu epoch.
        """
        noise_type = noise_var.get() # Zadané udaje od uživatele v aplikaci
        epoch_number = int(epoch_entry.get())

        # Načtení vah
        weights_path = f"data/weights/{noise_type}/weights.h5"
        self.gan.generator.load_weights(weights_path) # zavolání funkce v main

        # Generování obrázků
        self.gan.sample_images(epoch_number, noise_type)
        image_path = f"data/output/{noise_type}/{epoch_number}.png" # zavolání funkce v main

        self.display_image(image_path)

    def delete_images(self, noise_var):
        """
        Smaže všechny vygenerované obrázky pro aktuální typ šumu. ( Pro nějake ulehčení )
        """
        noise_type = noise_var.get()
        folder_path = f"data/output/{noise_type}"
        files = glob.glob(f"{folder_path}/*.png")
        for file in files:
            os.remove(file)

    def display_image(self, image_path):
        """
        Zobrazí obrázek na uživatelském rozhraní.

        Parametry:
        - image_path: Cesta k obrázku.
        """
        image = Image.open(image_path) # Otevře obrázek v 'image_path' pomocí knihovny PIL a vytvoří objekt Image

        photo = ImageTk.PhotoImage(image) # Vytvoří Tkinter-kompatibilní objekt obrázku z objektu Image knihovny PIL

        self.image_label.configure(image=photo)  # Nastaví atribut 'image' widgetu self.image_label na zobrazení obrázku

        # Uloží odkaz na objekt PhotoImage přiřazením ho do atributu 'image' widgetu self.image_label.
        # Tím se zajistí, že objekt obrázku nebude odstraněn systémem pro správu paměti a zůstane v paměti
        self.image_label.image = photo

    def run(self):
        self.window.mainloop()


if __name__ == '__main__':
    # vytvoří instanci třídy GANUI a spustí její metodu run(), která zobrazí GUI.
    gan_ui = GANUI()
    gan_ui.run()
