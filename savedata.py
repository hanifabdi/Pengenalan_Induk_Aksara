from PIL import Image
import glob

karakter = 'gha'
folder = 'data_train/aksara/'+karakter+'/*.png'
images = []
for filename in glob.glob(folder):
    image = Image.open(filename)
    imgs = image.resize((768, 768))
    images.append(imgs)

for (i,new) in enumerate(images):
    new.save('{}{}{}'.format('data_train/aksara/'+karakter+'/'+karakter,i+1,'.png'))