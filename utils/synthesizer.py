import os
import random
import numpy as np
import cv2 as cv
import imutils


def check_holes_overlap(xy, hsws, cx, cy, hs, ws, overlap_ratio=0.7):
    if len(xy) == 0:
        return False
    for i in range(len(xy)):
        _cx, _cy = xy[i]
        _hs, _ws = hsws[i]
        if (abs(cx - _cx) < ((ws + _ws) // 2)*overlap_ratio) and (abs(cy - _cy) < ((hs + _hs) // 2)*overlap_ratio):
            return True
    return False
 
def sowing_seed(seed, cx, cy, ws, hs, background):
    mask = np.zeros_like(background)
    t_xmin = cx - ws // 2
    t_ymin = cy - hs // 2
    t_xmax = t_xmin + ws
    t_ymax = t_ymin + hs

    mask[t_ymin: t_ymax, t_xmin: t_xmax, :] = seed
    gmask = cv.cvtColor(mask, cv.COLOR_BGR2GRAY)
    contours, _ = cv.findContours(gmask, 1, 2)
    max_contour = max(contours, key=cv.contourArea)

    center, axes, angle = cv.fitEllipse(max_contour)
    ellipse = (center, (axes[0]-3, axes[1]-2), angle)
    # breakpoint()

    x, y, w, h = cv.boundingRect(max_contour)
    xmin = cx - w // 2
    ymin = cy - h // 2
    xmax = xmin + w
    ymax = ymin + h

    # cv.drawContours(background, [max_contour], 0, (0, 0, 0), cv.FILLED)
    cv.ellipse(background, ellipse, (0, 0, 0), -1)
    return cv.add(background, mask), xmin, ymin, xmax, ymax

def random_crop_background(background, max_crop=10):
    top, down, left, right = [random.randint(0, max_crop) for _ in range(4)]
    h, w = background.shape[:2]
    background = background[top: h-down, left: w-right, :]
    return background

def random_brightness_background(background):
    if random.random() > 0.4:
        alpha = random.random() + random.randrange(1, 3)
        beta = random.randrange(0, 100)
        background = cv.convertScaleAbs(background, alpha=alpha, beta=beta)
    return background

def random_process_coin(coin, scale_ratio):
    h, w = coin.shape[:2]

    # random_brightness
    if random.random() > 0.5:
        alpha = random.random() + random.randrange(1, 2)
        beta = random.randrange(20, 70)
        coin = cv.convertScaleAbs(coin, alpha=alpha, beta=beta)
        x = w//2
        y = h//2
        r = int((x+y)//2)
        mask = np.zeros((h, w), dtype=np.uint8)
        cv.circle(mask, (x, y), r, (255), -1)
        coin = cv.bitwise_and(coin, coin, mask=mask)

    # random scale
    h = h/scale_ratio
    w = w/scale_ratio
    if random.random() > 0.5:
        scale_x = random.random()/20
        scale_y = random.random()/20
        sign = 1 if random.random() > 0.5 else -1
        h = h + sign*h*scale_y
        sign = 1 if random.random() > 0.5 else -1
        w = w + sign*w*scale_x
    h, w = int(h), int(w)
    coin = cv.resize(coin, (w, h))

    # random rotate
    angle = random.randrange(-180, 180)
    coin = imutils.rotate_bound(coin, angle)

    return coin    

class CoinImageSynthesizer():
    def __init__(self, background_list, coin_list, label_dict, min_coin_per_image=10,
                 max_coin_per_image=15):
        self.background_list = background_list
        self.coin_list = coin_list
        self.label_dict = label_dict
        self.min_coin_per_image = min_coin_per_image
        self.max_coin_per_image = max_coin_per_image

    def parse_coin_name(self, fname):
        basename = os.path.basename(fname)
        basename = basename.split('.')[0].split('_')[0]
        return self.label_dict[basename]

    def generate_backgound(self):
        background_name = random.choice(self.background_list)
        background = cv.imread(background_name)
        background = random_crop_background(background)
        background = random_brightness_background(background)
        return background

    def generate_coin(self, scale_ratio):
        coin_name = random.choice(self.coin_list)
        label = self.parse_coin_name(coin_name)
        coin = cv.imread(coin_name)
        coin = random_process_coin(coin, scale_ratio)
        return coin, label

    def synthesize(self, num_image, output_path, annotation_file, name_prefix='a'):
        f = open(annotation_file, 'w')
        for cnt_img in range(num_image):
            print(f'Generating image {cnt_img}')
            # random background
            background = self.generate_backgound()
            H, W = background.shape[:2]

            num_coin = random.randrange(self.min_coin_per_image, self.max_coin_per_image+1)
            cnt_coin = 0
            patient = 0

            xy = []
            boxes = []
            hw = []
            scale_ratio = np.random.uniform(1.0, 2.5)
            while cnt_coin < num_coin:
                # random coin
                coin, label = self.generate_coin(scale_ratio)
                h, w = coin.shape[:2]

                # random position x and y
                x = random.randrange(w//2, W-w//2)
                y = random.randrange(h//2, H-h//2)

                if check_holes_overlap(xy, hw, x, y, h, w):
                    patient += 1
                    if patient > 50:
                        break
                    continue

                # sowing seed
                background, xmin, ymin, xmax, ymax = sowing_seed(coin, x, y, w, h, background)
                
                # update
                cnt_coin += 1
                xy.append([x, y])
                hw.append([h, w])
                boxes.append([xmin, ymin, xmax, ymax, label])

            # save background with sowed seed to file
            image_name = f'{name_prefix}_{cnt_img}.jpg'
            image_path = os.path.join(output_path, image_name)
            cv.imwrite(image_path, background)
            
            # write bboxes to annotation file
            text = image_name + ' '
            for box in boxes:
                text += ','.join(list(map(str, box)))
                text += ' '
            text = text.strip() + '\n'
            f.write(text)
        f.close()    
        print('Done')