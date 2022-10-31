import PyQt5

from src.LPImage.LPImage import LPImage

if __name__ == "__main__":

    print("License Plate Recognition")

    # for i in range(10, 21):
    #     img = LPImage('../img/' + str(i) + '.jpg')
    #     img.process()

    img = LPImage('../img/12.jpg')
    img.process()
