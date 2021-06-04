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

After running the above command, seed coin images are obtained:

<img src="https://github.com/oattao/CountingKoreanCoin/blob/master/data/coin_seeds/10h.jpg" width="100" height="100" /> <img src="https://github.com/oattao/CountingKoreanCoin/blob/master/data/coin_seeds/10t.jpg" width="100" height="100" /> <img src="https://github.com/oattao/CountingKoreanCoin/blob/master/data/coin_seeds/50h.jpg" width="100" height="100" /> <img src="https://github.com/oattao/CountingKoreanCoin/blob/master/data/coin_seeds/50t.jpg" width="100" height="100" /> 

<img src="https://github.com/oattao/CountingKoreanCoin/blob/master/data/coin_seeds/100h.jpg" width="100" height="100" /> <img src="https://github.com/oattao/CountingKoreanCoin/blob/master/data/coin_seeds/100t.jpg" width="100" height="100" /> <img src="https://github.com/oattao/CountingKoreanCoin/blob/master/data/coin_seeds/500h.jpg" width="100" height="100" /> <img src="https://github.com/oattao/CountingKoreanCoin/blob/master/data/coin_seeds/500t.jpg" width="100" height="100" />

## 2. Create fake image data for training object detection model
### Background images + Coin images -> Image with coins

Some background images

<img src="https://github.com/oattao/CountingKoreanCoin/blob/master/data/backgrounds/bg%20(1).jpg" width="300" height="200" /> <img src="https://github.com/oattao/CountingKoreanCoin/blob/master/data/backgrounds/bg%20(5).jpg" width="300" height="200" /> <img src="https://github.com/oattao/CountingKoreanCoin/blob/master/data/backgrounds/bg%20(18).jpg" width="300" height="200" />  <img src="https://github.com/oattao/CountingKoreanCoin/blob/master/data/backgrounds/bg%20(14).jpg" width="300" height="200" />  <img src="https://github.com/oattao/CountingKoreanCoin/blob/master/data/backgrounds/bg%20(12).jpg" width="300" height="200" /> <img src="https://github.com/oattao/CountingKoreanCoin/blob/master/data/backgrounds/bg%20(3).jpg" width="300" height="200" />

Now we have backgrounds and seed coins, we will synthesize fake data by randomly sawing seed coins on backgrounds.
Run the command:
```console
python synthesize.py --background_path=data/background
                     --seed_path=data/coin_seeds
                     --num_images=400
```

We will obtains 400 fake images like this:

<img src="https://github.com/oattao/CountingKoreanCoin/blob/master/data/toshow/t_207.jpg" width="300" height="220" /> <img src="https://github.com/oattao/CountingKoreanCoin/blob/master/data/toshow/t_356.jpg" width="300" height="220" /> <img src="https://github.com/oattao/CountingKoreanCoin/blob/master/data/toshow/t_213.jpg" width="300" height="220" />

and the corresponding bounding box annotation:
```console
t_207.jpg 847,530,1024,706,7 576,751,755,928,6 406,197,612,402,5 110,233,273,403,6 666,178,869,380,4 325,512,498,684,7 1040,155,1258,379,3 132,659,299,828,7 1120,433,1289,606,6
t_213.jpg 990,496,1104,609,5 565,585,662,681,7 684,391,811,518,2 121,514,258,649,1 1185,826,1303,945,4 1106,303,1231,428,2 502,700,597,795,6 1168,613,1298,744,2 211,96,353,237,1 541,196,667,322,2 1107,115,1241,250,1 369,662,486,780,5
t_356.jpg 657,175,849,368,1 241,431,412,600,2 425,678,615,872,0 990,306,1185,502,1 262,104,396,239,6 1057,670,1199,810,6 818,555,1014,751,1 173,793,314,935,6 442,367,577,502,7
```

## 3. Convert image data into TFrecord format for training with Tensorflow Object Detection API

```console
python make_tfrecord.py
```

## 4. Training 
```console
python train_model.py --model_dir=models/intraining_models/ssd 
                      --pipeline_config_path=config/pipeline_ssd.config
```
## 5. After training, test with read images:

```console
python count_coint.py --image_path=data/real_coin_images/c.jpg
```
### The output
<img src="https://github.com/oattao/CountingKoreanCoin/blob/master/data/toshow/output1.jpg" width="220" height="300" />
