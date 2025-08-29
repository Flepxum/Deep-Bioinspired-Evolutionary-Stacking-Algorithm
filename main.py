# -*- coding: utf-8 -*-
import warnings

from config import get_config
from data_loader import get_data_loader
from meta_svm import data_save, svm
from trainer import Trainer
from utils import prepare_dirs, save_config, combine_list


def main(config):
    # ignore warning
    warnings.filterwarnings("ignore")

    # ensure directories are setup
    prepare_dirs(config)

    # ensure reproducibility
    # torch.manual_seed(config.random_seed)
    kwargs = {}

    if config.use_gpu:
        # torch.cuda.manual_seed_all(config.random_seed)
        kwargs = {'num_workers': config.num_workers, 'pin_memory': config.pin_memory}
        # torch.backends.cudnn.deterministic = True
    
    train_loader, valid_loader, meta_valid_set = get_data_loader(
        config.data_dir, config.batch_size,
        config.random_seed, False, **kwargs
    ) 

    probabilities_list1 = []
    probabilities_list2 = []
    probabilities_list3 = []
    y_true_list = []
    valid_y_true_list = []

    k = 5
    flag = 1
    # first
    print("first:")
    valid_probabilities_list = []
    # instantiate trainer
    for i in range(k):
        temp_train = Trainer(config)
        # either train
        data_loader = train_loader[i], valid_loader[i], meta_valid_set  
        if config.is_train:
            save_config(config)
            probabilities, y_true = temp_train.train(data_loader) 
            for i in probabilities:
                probabilities_list1.append(i)
            losses, accs, valid_report, probabilities, y_true = temp_train.validate(epoch=0, flag=1)
            valid_probabilities_list.append(probabilities)
            if flag == 1:
                for i in y_true:
                    valid_y_true_list.append(i)
                flag = 0
            del temp_train
        else:
            # trainer.test()
            pass

    new = []
    for i in range(1, len(valid_probabilities_list)):
        for j in range(len(valid_probabilities_list[0])):
            valid_probabilities_list[0][j] += valid_probabilities_list[i][j]
    for i in valid_probabilities_list[0]:
        i = i / k
        new.append(i)
    # second
    print("second:")
    valid_probabilities_list = []
    for i in range(k):
        temp_train = Trainer(config)
        # either train
        data_loader = train_loader[i], valid_loader[i], meta_valid_set

        if config.is_train:
            save_config(config)
            probabilities, y_true = temp_train.train(data_loader)
            for i in probabilities:
                probabilities_list2.append(i)
            losses, accs, valid_report, probabilities, y_true = temp_train.validate(epoch=0, flag=1)
            valid_probabilities_list.append(probabilities)
        del temp_train
    else:
        # trainer.test()
        pass

    new2 = []
    for i in range(1, len(valid_probabilities_list)):
        for j in range(len(valid_probabilities_list[0])):
            valid_probabilities_list[0][j] += valid_probabilities_list[i][j]
    for i in valid_probabilities_list[0]:
        i = i / k
        new2.append(i)
    # third
    print("third:")
    valid_probabilities_list = []
    for i in range(k):
        temp_train = Trainer(config)
        # either train
        data_loader = train_loader[i], valid_loader[i], meta_valid_set
        if config.is_train:
            save_config(config)
            probabilities, y_true = temp_train.train(data_loader)
            for i in probabilities:
                probabilities_list3.append(i)
            for i in y_true:
                y_true_list.append(i)
            losses, accs, valid_report, probabilities, y_true = temp_train.validate(epoch=0, flag=1)
            valid_probabilities_list.append(probabilities)
        del temp_train
    else:
        # trainer.test()
        pass

    new3 = []
    for i in range(1, len(valid_probabilities_list)):
        for j in range(len(valid_probabilities_list[0])):
            valid_probabilities_list[0][j] += valid_probabilities_list[i][j]
    for i in valid_probabilities_list[0]:
        i = i / k
        new3.append(i)

    probabilities_list4 = combine_list(probabilities_list1, probabilities_list2, probabilities_list3)
    new4 = combine_list(new, new2, new3)
    data_save(probabilities_list4, y_true_list, new4, valid_y_true_list)
    svm()

if __name__ == '__main__':
    config, unparsed = get_config()
    # print(config)
    # config.data_dir = './data/Multimodal-cervical'
    # config.batch_size = 2
    # config.num_classes = 3
    # config.epochs = 20
    # print(config)
    main(config)
