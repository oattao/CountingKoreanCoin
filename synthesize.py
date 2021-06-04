import os
import glob
import shutil
import argparse
from config.config import label_dict
from utils.synthesizer import CoinImageSynthesizer

parser = argparse.ArgumentParser()
parser.add_argument('--background_path', default='./data/backgrounds')
parser.add_argument('--seed_path', default='./data/coin_seeds')
parser.add_argument('--data_type', default='train')
parser.add_argument('--num_images', type=int)

args = parser.parse_args()

background_path = args.background_path
coin_path = args.seed_path
data_type = args.data_type
prefix = data_type[0]
num_images = args.num_images
annotation_file = f'./data/synthetic_images/annotation_{data_type}.txt'
output_path = f'./data/synthetic_images/{data_type}'

if os.path.exists(output_path):
    shutil.rmtree(output_path)
os.mkdir(output_path)

background_list = glob.glob(f'{background_path}/*.jpg')
coin_list = glob.glob(f'{coin_path}/*.jpg')

faker = CoinImageSynthesizer(background_list=background_list, coin_list=coin_list,
                              label_dict=label_dict)
faker.synthesize(num_images, output_path, annotation_file, prefix)
