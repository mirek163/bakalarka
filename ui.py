import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
from tkinter import filedialog
from tkinter import *
from functools import partial
from PIL import ImageTk, Image
import os
import glob
from main import GAN,WEIGHT_PATH, OUTPUT_PATH


class GANUI:
    def __init__(self):
        # ------------------------------------------------------
        # Nastavení počátečních proměných
        # ------------------------------------------------------
        noise_type = 'simplex'  # Výchozí typ šumu
        epoch_number = 30000  # Výchozí počet epoch
        self.gan = GAN()

        # ------------------------------------------------------
        # Nastavení okna
        # ------------------------------------------------------
        self.window = tk.Tk()
        self.window.title("Generátor obrázků pomocí GAN")
        self.window.geometry("1000x600")  # Nastavení počáteční velikosti okna
        # Vytvoření canvasu
        canvas = tk.Canvas(self.window)
        canvas.pack(side="left", fill="both", expand=True)

        # Připojení scrollbaru k canvasu
        scrollbar = ttk.Scrollbar(self.window, orient="vertical", command=canvas.yview)
        scrollbar.pack(side="right", fill="y")

        # Nastavení scrollbaru na canvas
        canvas.configure(yscrollcommand=scrollbar.set)

        # Vytvoření rámce uvnitř canvasu pro umístění prvků UI
        inner_frame = ttk.Frame(canvas)
        canvas.create_window((0, 0), window=inner_frame, anchor="nw")

        # Horní panel s výběrem šumu a vstupním polem pro počet epoch
        top_bar = ttk.Frame(inner_frame, padding="10")
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

        generate_button = ttk.Button(top_bar, text="Generovat", command=partial(self.generate_images, noise_var, epoch_entry))
        generate_button.pack(side="left")

        delete_button = ttk.Button(top_bar, text="Smazat obrázky", command=partial(self.delete_images, noise_var))
        delete_button.pack(side="left")


        # Label pro zobrazení šumu
        self.old_image_label = Label(inner_frame)
        self.old_image_label.pack(side="top", pady=10, padx=170)

        # Label pro zobrazení datasetu
        self.dataset_image_label = Label(inner_frame)
        self.dataset_image_label.pack(side="top",pady=10, padx=170)

        # Label pro zobrazení nového obrázku
        self.new_image_label = Label(inner_frame)
        self.new_image_label.pack(side="top",pady=10, padx=170)


        # Přidání funkcionality scrollbaru k canvasu
        inner_frame.bind("<Configure>", lambda event: canvas.configure(scrollregion=canvas.bbox("all")))

        # Spuštění posunování canvasu pomocí kolečka myši
        canvas.bind_all("<MouseWheel>", lambda event: canvas.yview_scroll(int(-1 * (event.delta / 120)), "units"))


    def generate_images(self, noise_var, epoch_entry):
        """
        Generuje obrázky pomocí GAN na základě vybraného typu šumu a počtu epoch.
        """
        noise_type = noise_var.get() # Zadané udaje od uživatele v aplikaci
        epoch_number = int(epoch_entry.get())

        # Načtení vah


        weights_dir = WEIGHT_PATH + noise_type + '/'
        weights_path_generator = os.path.join(weights_dir, 'weights_g%d.h5' % epoch_number)
        self.gan.generator.load_weights(weights_path_generator) # zavolání funkce v main

        # Generování obrázků
        self.gan.sample_images(epoch_number,UI=True, noise_type=noise_type,mode='generate')
        old_image_path = f"{OUTPUT_PATH}{noise_type}/o_{epoch_number}.png"
        new_image_path = f"{OUTPUT_PATH}{noise_type}/n_{epoch_number}.png"
        dataset_image_path = f"{OUTPUT_PATH}{noise_type}/d_{epoch_number}.png"

        self.display_image(old_image_path,new_image_path,dataset_image_path)
        #self.display_image(new_image_path,dataset_image_path)

    def delete_images(self, noise_var):
        """
        Smaže všechny vygenerované obrázky pro aktuální typ šumu. ( Pro nějake ulehčení )
        """
        noise_type = noise_var.get()
        folder_path = f"data/output/{noise_type}"
        files = glob.glob(f"{folder_path}/*.png")
        for file in files:
            os.remove(file)

    def display_image(self, old_image_path, new_image_path, dataset_image_path):

   # def display_image(self,new_image_path,dataset_image_path):
        """
        Zobrazí obrázek na uživatelském rozhraní.

        Parametry:
        - image_path: Cesta k obrázku.
        """
        old_image = Image.open(old_image_path)
        new_image = Image.open(new_image_path) # Otevře obrázek v 'image_path' pomocí knihovny PIL a vytvoří objekt Image
        dataset_image = Image.open(dataset_image_path) # Otevře obrázek v 'image_path' pomocí knihovny PIL a vytvoří objekt Image


        old_photo = ImageTk.PhotoImage(old_image) # Vytvoří Tkinter-kompatibilní objekt obrázku z objektu Image knihovny PIL
        dataset_photo = ImageTk.PhotoImage(dataset_image)
        new_photo = ImageTk.PhotoImage(new_image) # Vytvoří Tkinter-kompatibilní objekt obrázku z objektu Image knihovny PIL

        self.old_image_label.configure(image=old_photo)  # Nastaví atribut 'image' widgetu self.image_label na zobrazení obrázku
        self.dataset_image_label.configure(image=dataset_photo)  # Nastaví atribut 'image' widgetu self.image_label na zobrazení obrázku
        self.new_image_label.configure(image=new_photo)  # Nastaví atribut 'image' widgetu self.image_label na zobrazení obrázku


        # Uloží odkaz na objekt PhotoImage přiřazením ho do atributu 'image' widgetu self.image_label.
        # Tím se zajistí, že objekt obrázku nebude odstraněn systémem pro správu paměti a zůstane v paměti
        self.old_image_label.image = old_photo
        self.dataset_image_label.image = dataset_photo
        self.new_image_label.image = new_photo


    def run(self):
        self.window.mainloop()


if __name__ == '__main__':
    # vytvoří instanci třídy GANUI a spustí její metodu run(), která zobrazí GUI.
    gan_ui = GANUI()
    gan_ui.run()
