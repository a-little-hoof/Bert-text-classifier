from model import Config,EncoderRNN,AttnDecoderRNN
from train import trainIters
from utils import prepareData
import random

###
config = Config()

###data
input_lang,output_lang,pairs = prepareData()
print(random.choices(pairs))

###build model
encoder = EncoderRNN(config,input_lang.n_words, config.hidden_size).to(config.device)
attn_decoder = AttnDecoderRNN(config,config.hidden_size, output_lang.n_words, dropout_p=0.1).to(config.device)

###train
trainIters(input_lang,output_lang,pairs,config,encoder, attn_decoder, 75000, print_every=5000)