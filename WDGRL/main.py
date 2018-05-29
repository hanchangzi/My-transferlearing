"""Main script for ARDA."""

from core.test import  test
from core.adapt import train
from misc import params
from misc.utils import get_data_loader, init_model, init_random_seed
from models import Classifier, Discriminator, Generator

# from core.pretrain import train
if __name__ == '__main__':
    # init random seed
    init_random_seed(params.manual_seed)

    # load dataset
    src_data_loader = get_data_loader(params.src_dataset)
    src_data_loader_test = get_data_loader(params.src_dataset, train=False)
    tgt_data_loader = get_data_loader(params.tgt_dataset)
    tgt_data_loader_test = get_data_loader(params.tgt_dataset, train=False)
    # # init models
    classifier = init_model(net=Classifier(),
                            restore=params.c_model_restore)
    generator = init_model(net=Generator(),
                           restore=params.g_model_restore)
    critic = init_model(net=Discriminator(input_dims=params.d_input_dims,
                                          hidden_dims=params.d_hidden_dims,
                                          output_dims=params.d_output_dims),
                        restore=params.d_model_restore)
    train(classifier, generator, critic, src_data_loader, tgt_data_loader)
