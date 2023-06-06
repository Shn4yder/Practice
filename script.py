import sys
import argparse
import cv2
import numpy as np
import colorsys


def createParser ():

    parser = argparse.ArgumentParser()

    parser.add_argument('-p', '--path', default='test.jpg', help='Путь к изображению')
    
    parser.add_argument('-info', action='store_const', const=True, help='Информация о изображении')
    parser.add_argument('-px', nargs=2, help='RGB/HSV указанного пикселя')
    parser.add_argument('-save_gr', help='Конвертация изображнеия в ч/б формат и сохранение по указанному пути')
    parser.add_argument('-hist_eq', action='store_const', const=True, help='Операция histogram equalization')
    parser.add_argument('-morph', action='store_const', const=True, help='Морфологические операции')
    parser.add_argument('-thresh', action='store_const', const=True, help='Некоторые виды бинаризации')
    parser.add_argument('-segment', action='store_const', const=True, help='Сегментация с помомщью watershed')

    return parser


if __name__ == '__main__':
    parser = createParser()
    namespace = parser.parse_args(sys.argv[1:])

    path = namespace.path.replace('\\', r'\\')

    img = cv2.imread(path)
    if namespace.info:
        height, width, layers = img.shape[:]
        print("Информация о изображении:\nРазмер", width, "х", height, "Слои", layers)
    if namespace.px:
        b, g, r = list(img[int(namespace.px[0]), int(namespace.px[1])])
        print("RGB пикселя : ", r, g, b)
        h, s, v = colorsys.rgb_to_hsv(r/255, g/255, b/255)
        h, s, v = (int(h * 360), int(s * 100), int(v * 100))
        print('HSV : ', h, s, v)
    if namespace.save_gr:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cv2.imwrite(namespace.save_gr, img)
    if namespace.hist_eq:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        equalized = cv2.equalizeHist(img)

        cv2.namedWindow("original", cv2.WINDOW_NORMAL)
        cv2.namedWindow("equalized", cv2.WINDOW_NORMAL)
        cv2.imshow("original", img)
        cv2.imshow("equalized", equalized)
    if namespace.morph:
        kernel = np.ones((5, 5), np.uint8)
        erosion = cv2.erode(img, kernel)
        dilation = cv2.dilate(img, kernel)
        opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
        closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)

        cv2.namedWindow("erosion", cv2.WINDOW_NORMAL)
        cv2.namedWindow("dilation", cv2.WINDOW_NORMAL)
        cv2.namedWindow("opening", cv2.WINDOW_NORMAL)
        cv2.namedWindow("closing", cv2.WINDOW_NORMAL)
        cv2.namedWindow("original", cv2.WINDOW_NORMAL)
        cv2.imshow("erosion", erosion)
        cv2.imshow("dilation", dilation)
        cv2.imshow("opening", opening)
        cv2.imshow("closing", closing)
        cv2.imshow("original", img)
    if namespace.thresh:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        th_ad_mean = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
        th_ad_gauss = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        ret, otsu = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        ret, th_bin = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
        ret, th_trunc = cv2.threshold(img, 127, 255, cv2.THRESH_TRUNC)
        ret, th_zero = cv2.threshold(img, 127, 255, cv2.THRESH_TOZERO)

        cv2.namedWindow("adaptive (mean_c)", cv2.WINDOW_NORMAL)
        cv2.namedWindow("adaptive (gaussian_c)", cv2.WINDOW_NORMAL)
        cv2.namedWindow("Otsu's", cv2.WINDOW_NORMAL)
        cv2.namedWindow("binary", cv2.WINDOW_NORMAL)
        cv2.namedWindow("truncate", cv2.WINDOW_NORMAL)
        cv2.namedWindow("threshold to zero", cv2.WINDOW_NORMAL)
        cv2.imshow("adaptive (mean_c)", th_ad_mean )
        cv2.imshow("adaptive (gaussian_c)", th_ad_gauss)
        cv2.imshow("Otsu's", otsu)
        cv2.imshow("binary", th_bin)
        cv2.imshow("truncate", th_trunc)
        cv2.imshow("threshold to zero", th_zero)
    if namespace.segment:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        kernel = np.ones((3, 3), np.uint8)
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
        sure_bg = cv2.dilate(opening, kernel, iterations=3)
        dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
        ret, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)

        sure_fg = np.uint8(sure_fg)
        unknown = cv2.subtract(sure_bg, sure_fg)

        ret, markers = cv2.connectedComponents(sure_fg)

        markers = markers + 1
        markers[unknown == 255] = 0

        markers = cv2.watershed(img, markers)
        img[markers == -1] = [255, 0, 0]
        cv2.imshow("Segmentation", img)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


