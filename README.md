# NgNet

NgNet is a object detector based on [KittiBox](https://github.com/MarvinTeichmann/KittiBox).

## Comparision

Statistics generated from running 'evaluate.py' on VGG, Resnet50, and Resnet100 models.

|           	| train easy 	| train moderate 	| train hard 	| val easy 	| val moderate 	| val hard 	| speed (msec) 	| speed (fps) 	| post (msec) 	|
|-----------	|------------	|----------------	|------------	|----------	|--------------	|----------	|--------------	|-------------	|-------------	|
| VGG       	| 93.25%     	| 84.10%         	| 69.09%     	| 94.27%   	| 86.32%       	| 70.78%   	| 104.4554     	| 9.5735      	| 3.5804      	|
| Resnet50  	| 98.46%     	| 91.50%         	| 79.41%     	| 96.78%   	| 86.47%       	| 72.33%   	| 59.9431      	| 16.6825     	| 3.1012      	|
| Resnet100 	| 98.56%     	| 93.58%         	| 81.19%     	| 96.01%   	| 89.13%       	| 75.02%   	| 98.4383      	| 10.1506     	| 2.9213      	|

Difference:

|                  	| train easy 	| train moderate 	| train hard 	| val easy 	| val moderate 	| val hard 	| speed 	|
|------------------	|------------	|----------------	|------------	|----------	|--------------	|----------	|-------	|
| Resnet50 vs VGG  	| 5.21%      	| 7.4%           	| 10.35%     	| 2.51%    	| 0.15%        	| 1.55%    	| x1.74 	|
| Resnet100 vs VGG 	| 5.31%      	| 9.48%          	| 12.1%      	| 1.74%    	| 2.81%        	| 4.24%    	| x1.06 	|

## Requirements

The code requires Tensorflow 1.0 as well as the following python libraries: 

* matplotlib
* numpy
* Pillow
* scipy
* runcython
* imageio
* opencv

Those modules can be installed using: `pip install -r requirements.txt`.

## Installation

Read [KittiBox README](https://github.com/nghiattran/ngnet/blob/master/KittiBox_README.md) for detailed installation.

## Demos

## Udacity-Didi Challenge

Generate data from [Udacity CrowdAI and AUTTI](https://github.com/udacity/self-driving-car/tree/master/annotations)
using [vod-converter](https://github.com/nghiattran/vod-converter). Note: this converter is a fork from
[umautobots vod-converter](https://github.com/umautobots/vod-converter) and it contains some changes to make it work
for this repositoty.

# Acknowledge

This project started out as a fork of [KittiBox](https://github.com/MarvinTeichmann/KittiBox).