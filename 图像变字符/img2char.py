# coding=utf-8

# import cv2
# import matplotlib.pyplot as plt
# img_original=cv2.imread('/Users/chenjia/Desktop/IMG_5857.JPG'  ,0)
# img_original[img_original<=40] = 255
# cv2.imwrite('1.jpg', img_original)


from PIL import Image
IMG = './1.png' #设置图片文件
WIDTH = 150 #设置字符画的宽
HEIGHT = 80 #设置字符画的高
OUTPUT = 'output5.txt'  #设置存放字符画的文本文件
ascii_char = list("$@B%8&WM#*oahkbdpqwmZO0QLCJUYXzcvunxrjft/\|()1{}[]?-_+~<>i!lI;:,\"^`'. ")    #设置显示的字符集
def get_char(r,g,b,alpha = 256):
    if alpha == 0:
        return ' '
    length = len(ascii_char)
    gray = int(0.2126 * r + 0.7152 * g + 0.0722 * b)
    unit = (255.0 + 1)/length
    return ascii_char[int(gray/unit)]
if __name__ == '__main__':
    im = Image.open(IMG)
    im = im.resize((WIDTH,HEIGHT), Image.NEAREST)
    txt = ""
    for i in range(HEIGHT):
        for j in range(WIDTH):
            g = im.convert('RGBA').getpixel((j, i)) 
            txt += get_char(*g)
        txt += '\n'
    print(txt)
    with open(OUTPUT,'w') as f:
        f.write(txt)