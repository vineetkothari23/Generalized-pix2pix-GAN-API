from parameters import get_parameters
from train import train_process

if __name__ =='__main__':
  config=get_parameters()
  train_process(config)
  