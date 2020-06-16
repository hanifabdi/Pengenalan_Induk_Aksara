from PIL import Image

image = Image.open('data_train/gha14.png').convert('L')
width = image.size[0]
height = image.size[1]
patterns = []
"""print(width)
print(height)
#print(pixels)
pix = image.load()
"""
pixels2 = list(image.getdata())
pixels = [pixels2[i * width:(i + 1) * width] for i in range(height)]
R = 1
LBP_value = []
#362+400
for i in range(R, width-R):
    for j in range(R, height-R):
        center = pixels[i] [j]
        top_left = pixels[i - R] [j - R]
        top_up = pixels[i] [j - R]
        top_right = pixels[i + R] [j - R]
        right = pixels[i + R] [j]
        left = pixels[i - R] [j]
        bottom_left = pixels[i- R] [j + R]
        bottom_right = pixels[i + R] [j + R]
        bottom_down = pixels[i] [j + R]
        val = [top_left, top_up, top_right, right, bottom_right,bottom_down, bottom_left, left]
        """print(center)
        print(val)"""
        vals = []
        for x in val:
            if (x > center):
                vals.append(1)
            else:
                vals.append(0)
        #print(vals)
        weights = [1, 2, 4, 8, 16, 32, 64, 128]
        LBP = 0
        for a in range(0, len(vals)):
            LBP += weights[a] * vals[a]
        #print(LBP)
        LBP_value.append(LBP)
#print(LBP_value)

new_image = Image.new('1', (width-(R*2), height-(R*2)))
new_image.putdata(LBP_value)
"""flip = new_image.transpose(Image.FLIP_TOP_BOTTOM)
imgs = flip.transpose(Image.ROTATE_270)"""
#pixel_values = list(new_image.getdata())
#print(pixel_values)
new_image.save("output_binerR2.png")
new_image.show()







