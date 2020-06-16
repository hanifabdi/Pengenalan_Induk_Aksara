from PIL import Image
import glob

karakter = 'gha'
folder = 'data_test/aksara_biner/'+karakter+'/*.png'
R = 1
size = 64
lbp = 'LBP_R1'
imgs = []
for filename in glob.glob(folder):
    im = Image.open(filename)
    img = im.resize((size+(R * 2), size+(R * 2)))
    imgs.append(img)

img_new = []
for (image) in imgs:
    width = image.size[0]
    height = image.size[1]
    pixels2 = list(image.getdata())
    pixels = [pixels2[i * width:(i + 1) * width] for i in range(height)]
    LBP_value = []
    for i in range(R, width - R):
        for j in range(R, height - R):
            center = pixels[i][j]
            top_left = pixels[i - R][j - R]
            top_up = pixels[i - R][j]
            top_right = pixels[i - R][j + R]
            right = pixels[i][j + R]
            bottom_right = pixels[i + R][j + R]
            bottom_down = pixels[i + R][j]
            bottom_left = pixels[i + R][j - R]
            left = pixels[i][j - R]
            val = [top_left, top_up, top_right, right, bottom_right, bottom_down, bottom_left, left]

            vals = []
            for x in val:
                if (x > center):
                    vals.append(1)
                else:
                    vals.append(0)

            weights = [1, 2, 4, 8, 16, 32, 64, 128]
            LBP = 0
            for a in range(0, len(vals)):
                LBP += weights[a] * vals[a]
            LBP_value.append(LBP)

    new_image = Image.new('1', (width - (R * 2), height - (R * 2)))
    new_image.putdata(LBP_value)
    img_new.append(new_image)

for (i,new) in enumerate(img_new):
    new.save('{}{}{}'.format('data_test/'+lbp+'/'+karakter+'/'+karakter,i+1,'.png'))