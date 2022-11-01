import PyQt5

from src.LPImage.LPImagePlus import LPImagePlus

from src.LPImage.LPImage import LPImage

if __name__ == "__main__":

    print("License Plate Recognition")

    seq = 1
    size = 9
    images = []
    for i in range(seq, seq + size - 1):
        # img = LPImage('../img/' + str(i) + '.jpg')
        img = LPImagePlus('../img/' + str(i) + '.jpg')
        img.process()
        images.append(img)
    # img = LPImage('../img/12.jpg')
    # img.process()
