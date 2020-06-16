from PIL import Image
import glob

#read image
karakter = 'gha'
folder = 'data_train/aksara/'+karakter+'/*.png'
images = []
for filename in glob.glob(folder):
    image = Image.open(filename)
    images.append(image)

binary =[]
for (img) in images:
    pixels = img.load()
    width, height = img.size
    all_pixels = []
    for x in range(width):
        for y in range(height):
            hpixel = pixels[x,y]
            img_gray = (0.2989 * hpixel[0]) + (0.5870 * hpixel[1]) + (0.1140 * hpixel[2]) #konversi ke grayscale

            if img_gray >= 110:
                all_pixels.append(1)
            else:
                all_pixels.append(0)

    konversi = {0: 255,
                1: 0}

    data_isi = [konversi[letter] for letter in all_pixels]

    image = Image.new("1", img.size)
    image.putdata(data_isi)
    flip = image.transpose(Image.FLIP_TOP_BOTTOM)
    ftranspose = flip.transpose(Image.ROTATE_270)
    binary.append(ftranspose)

for (i,new) in enumerate(binary):
    new.save('{}{}{}'.format('data_train/aksara_biner/'+karakter+'/'+karakter,i+1,'.png'))


"""
img_binary =[]
img_new = ImageMath.eval('255-(invert)', invert=image)

img = img.resize((768,768)) #>masih bisa (ukuran asli 768x768)
img = img.resize((50,50),Image.ANTIALIAS)
img = img.resize((600,600)) >mulai kepotong

pixel_values = list(img_new.getdata())
img_new.save()
print(pixel_values)
img_new.save('data_train/gabiner.png')
all_pixels = '\n'.join([','.join(map(str,item)) for item in all_pixels])
print(all_pixels) #menampilkan pixel binary
"""

