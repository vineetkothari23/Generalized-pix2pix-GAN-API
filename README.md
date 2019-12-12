# pix2pix GAN in pytorch
Image to Image translation modules, including dataloader setup, vanilla pix to pix model, cloud training and test apis in Pytorch
## About:
This is a pytorch implementation of paired image-to-image translation, i.e. converting one image to another type by analysing the relation amongst the two images.
The paper was published in *year* by *Authors*. 
Citations:

## Broader look out:
Example: Converting images with uppercase letters to lowercase letters.
Denoising images.
Aerial images to simplified map images.
Animation and images here:


### Architecture
### Prerequisites
### Usage

#### Installation
#### Requirements
- Python 3.5+
- PyTorch 0.3.0
 
#### Directory structure
The directory structure is followed as 
```
.
├── ...
├── version_no                    # Version of different models and training process
│   ├── model          # saved model checkpoint files
│   ├── report         # reporting of final training, validation loss and other metrics
│   └── output          # Output directory
│       └── epoch                    # Storing the training epoch images
├── data                    # Dataset of images (Optional)
├── res                # Resources directory
│    └── Helvetica                    # Font file to generate paired images for training (optional) 
└── ...
```

#### Train/ test
1. Clone the repository
```
$ git clone https://github.com/MicroprocessorX069/Generalized-pix2pix-GAN-API.git
$ cd Generalized-pix2pix-GAN-API
```
3. Train
(i) Train
```
$ python python train.py --root_dir "./" --version "1.0" --batch_size 64 --input_width 32 --input_height 32 
```
(ii) Test
'''
$ python python test.py --root_dir "./" --version "1.0" 
'''
4. Enjoy the results
```
$ cd output/epoch
or
$ cd report
```

#### Using a pretrained model weights
Download the model weights as .ckpt file in "./model/" and hit the same commands to train and test with the correct root directory.

## Results
![Training gif](link)
## Implementation details
- [Theoritical details](docs/CONTRIBUTING.md)
- [Modules](docs/CONTRIBUTING.md)
- [Data](docs/CONTRIBUTING.md)
- [Architecture](docs/CONTRIBUTING.md)
- [Code structure](docs/CONTRIBUTING.md)
- [Distributed training](docs/CONTRIBUTING.md)
- [Docker](docs/CONTRIBUTING.md)

## Docker
## Related projects
## Acknowledgements


