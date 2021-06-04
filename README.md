# Counting Real Money by Fake Money Data

## 1. Preprocess coin images
Original coin images

<img src="https://github.com/oattao/CountingKoreanCoin/blob/master/data/coin_images/10h.jpg" width="108" height="144" /> <img src="https://github.com/oattao/CountingKoreanCoin/blob/master/data/coin_images/10t.jpg" width="108" height="144" /> <img src="https://github.com/oattao/CountingKoreanCoin/blob/master/data/coin_images/50h.jpg" width="108" height="144" /> <img src="https://github.com/oattao/CountingKoreanCoin/blob/master/data/coin_images/50t.jpg" width="108" height="144" /> <img src="https://github.com/oattao/CountingKoreanCoin/blob/master/data/coin_images/100h.jpg" width="108" height="144" /> <img src="https://github.com/oattao/CountingKoreanCoin/blob/master/data/coin_images/100t.jpg" width="108" height="144" /> <img src="https://github.com/oattao/CountingKoreanCoin/blob/master/data/coin_images/500h.jpg" width="108" height="144" /> <img src="https://github.com/oattao/CountingKoreanCoin/blob/master/data/coin_images/500t.jpg" width="108" height="144" />

```console
python make_seed.py 
      --input_path=data/coin_images 
      --output_path=data/coin_seeds 
      --thresh=120 --morph_iteration=3
```

After run tha above command, seed coin images are obtained:

<img src="https://github.com/oattao/CountingKoreanCoin/blob/master/data/coin_seeds/10h.jpg" width="150" height="150" /> <img src="https://github.com/oattao/CountingKoreanCoin/blob/master/data/coin_seeds/10t.jpg" width="150" height="150" /> <img src="https://github.com/oattao/CountingKoreanCoin/blob/master/data/coin_seeds/50h.jpg" width="150" height="150" /> <img src="https://github.com/oattao/CountingKoreanCoin/blob/master/data/coin_seeds/50t.jpg" width="150" height="150" /> 

<img src="https://github.com/oattao/CountingKoreanCoin/blob/master/data/coin_seeds/100h.jpg" width="150" height="150" /> <img src="https://github.com/oattao/CountingKoreanCoin/blob/master/data/coin_seeds/100t.jpg" width="150" height="150" /> <img src="https://github.com/oattao/CountingKoreanCoin/blob/master/data/coin_seeds/500h.jpg" width="150" height="150" /> <img src="https://github.com/oattao/CountingKoreanCoin/blob/master/data/coin_seeds/500t.jpg" width="150" height="150" />

## 2. Create fake image data for training object detection model
### Background images + Coin images -> Image with coins

Some background images

<img src="https://github.com/oattao/CountingKoreanCoin/blob/master/data/backgrounds/bg%20(1).jpg" width="300" height="200" /> <img src="https://github.com/oattao/CountingKoreanCoin/blob/master/data/backgrounds/bg%20(5).jpg" width="300" height="200" />

Some coin image

<img src="https://github.com/oattao/CountingKoreanCoin/blob/master/data/coin_seeds/10h.jpg" width="150" height="150" /> <img src="https://github.com/oattao/CountingKoreanCoin/blob/master/data/coin_seeds/10t.jpg" width="150" height="150" /> <img src="https://github.com/oattao/CountingKoreanCoin/blob/master/data/coin_seeds/50h.jpg" width="150" height="150" /> <img src="https://github.com/oattao/CountingKoreanCoin/blob/master/data/coin_seeds/50t.jpg" width="150" height="150" /> 

<img src="https://github.com/oattao/CountingKoreanCoin/blob/master/data/coin_seeds/100h.jpg" width="150" height="150" /> <img src="https://github.com/oattao/CountingKoreanCoin/blob/master/data/coin_seeds/100t.jpg" width="150" height="150" /> <img src="https://github.com/oattao/CountingKoreanCoin/blob/master/data/coin_seeds/500h.jpg" width="150" height="150" /> <img src="https://github.com/oattao/CountingKoreanCoin/blob/master/data/coin_seeds/500t.jpg" width="150" height="150" />

Making synthetic data to practice fault detection on image

Using opencv to make synthetic data

Using object detection to detect fault on images.
