import os
import shutil
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from torch.autograd import Variable
from tqdm import tqdm

from densenet import dcanet121
from shannon_entropy import EntropyLoss
from tensorboard_logger import configure, log_value
from utils import accuracy, AverageMeter

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


class Trainer(object):
    """
    Trainer encapsulates all the logic necessary for
    training the MobileNet Model.

    All hyperparameters are provided by the user in the
    config file.
    """
    has_executed = False

    def __init__(self, config):  # , data_loader
        """
        Construct a new Trainer instance.

        Args
        ----
        - config: object containing command line arguments.
        - data_loader: data iterator
        """
        self.config = config

        self.num_classes = config.num_classes

        # training params
        self.epochs = config.epochs
        self.start_epoch = 0
        self.momentum = config.momentum
        self.lr = config.init_lr
        self.weight_decay = config.weight_decay
        self.nesterov = config.nesterov
        self.gamma = config.gamma
        # misc params
        self.use_gpu = config.use_gpu
        self.best = config.best
        self.ckpt_dir = config.ckpt_dir
        self.logs_dir = config.logs_dir
        self.counter = 0
        self.lr_patience = config.lr_patience
        self.train_patience = config.train_patience
        self.use_tensorboard = config.use_tensorboard
        self.resume = config.resume
        self.print_freq = config.print_freq
        self.model_name = config.save_name

        self.model_num = config.model_num
        self.independent = config.independent
        self.models = []
        self.optimizers = []
        self.schedulers = []

        self.loss_kl = nn.KLDivLoss(reduction='batchmean')
        self.loss_ce = nn.CrossEntropyLoss()
        self.best_valid_accs = [0.] * self.model_num

        if Trainer.has_executed == False:
            Trainer.has_executed = True
            # configure tensorboard logging
            if self.use_tensorboard:
                tensorboard_dir = self.logs_dir + self.model_name
                print('[*] Saving tensorboard logs to {}'.format(tensorboard_dir))
                if not os.path.exists(tensorboard_dir):
                    os.makedirs(tensorboard_dir)
                configure(tensorboard_dir)

        for i in range(self.model_num):
            model = dcanet121(weights='DenseNet121_Weights.DEFAULT', n=i)
            num_ftrs = model.classifier.in_features
            model.classifier = nn.Linear(num_ftrs, self.num_classes)
            # print(model)
            if self.use_gpu:
                model.cuda()

            self.models.append(model)

            optimizer = optim.Adam(model.parameters(), lr=self.lr)
            self.optimizers.append(optimizer)
        for i in range(self.model_num):
            print('[*] Number of parameters of one model: {:,}'.format(
                sum([p.data.nelement() for p in self.models[i].parameters()])))
            param_dtype = next(self.models[i].parameters()).dtype
            print("Parameter data type:", param_dtype)

    def train(self, data_loader):
        """
        Train the model on the training set.

        A checkpoint of the model is saved after each epoch
        and if the validation accuracy is improved upon,
        a separate ckpt is created for use on the test set.
        """

        # data params
        if self.config.is_train:
            self.train_loader = data_loader[0]
            self.valid_loader = data_loader[1]
            self.meta_valid_loader = data_loader[2]
            self.num_train = len(self.train_loader.dataset)
            self.num_valid = len(self.valid_loader.dataset)
            self.num_meta_valid = len(self.meta_valid_loader.dataset)
        else:
            self.test_loader = data_loader
            self.num_test = len(self.test_loader.dataset)
        self.num_classes = self.config.num_classes

        # load the most recent checkpoint
        if self.resume:
            self.load_checkpoint(best=False)

        print("\n[*] Train on {} samples, validate on {} samples, meta_validate on {} samples".format(
            self.num_train, self.num_valid, self.num_meta_valid)
        )

        for epoch in range(self.start_epoch, self.epochs):
            print(
                '\nEpoch: {}/{} - LR: {:.6f}'.format(
                    epoch + 1, self.epochs, self.optimizers[0].param_groups[0]['lr'], )
            )

            # train for 1 epoch
            train_losses, train_accs = self.train_one_epoch(epoch)

            for scheduler in self.schedulers:
                scheduler.step()

                # evaluate on validation set
            flag = 0
            valid_losses, valid_accs, valid_report, probabilities, y_true = self.validate(epoch, flag)

            for i in range(self.model_num):
                is_best = valid_accs[i].avg > self.best_valid_accs[i]
                msg1 = "model_{:d}: train loss: {:.4f} - train acc: {:.4f} "
                msg2 = "- val loss: {:.4f} - val acc: {:.4f}"
                if is_best:
                    # self.counter = 0
                    msg2 += " [*]"
                msg = msg1 + msg2
                print(msg.format(i + 1, train_losses[i].avg, train_accs[i].avg, valid_losses[i].avg, valid_accs[i].avg))

                # check for improvement
                # if not is_best:
                # self.counter += 1
                # if self.counter > self.train_patience:
                # print("[!] No improvement in a while, stopping training.")
                # return
                self.best_valid_accs[i] = max(valid_accs[i].avg, self.best_valid_accs[i])
                self.save_checkpoint(i,
                                     {'epoch': epoch + 1,
                                      'model_state': self.models[i].state_dict(),
                                      'optim_state': self.optimizers[i].state_dict(),
                                      'best_valid_acc': self.best_valid_accs[i],
                                      }, is_best
                                     )
            for i in range(len(valid_report)):
                print(valid_report[i])
        return probabilities, y_true

    def train_one_epoch(self, epoch):
        """
        Train the model for 1 epoch of the training set.

        An epoch corresponds to one full pass through the entire
        training set in successive mini-batches.

        This is used by train() and should not be called manually.
        """
        batch_time = AverageMeter()
        kl_losses = []
        losses = []
        accs = []
        a = 0.001
        criterion = EntropyLoss()

        for i in range(self.model_num):
            self.models[i].train()
            kl_losses.append(AverageMeter())
            losses.append(AverageMeter())
            accs.append(AverageMeter())
        tic = time.time()
        with tqdm(total=self.num_train) as pbar:
            for i, (images, labels) in enumerate(self.train_loader):
                if self.use_gpu:
                    images, labels = images.cuda(), labels.cuda()
                images, labels = Variable(images), Variable(labels)

                # forward pass
                outputs = []
                for model in self.models:
                    outputs.append(model(images))

                for i in range(self.model_num):
                    ce_loss = self.loss_ce(outputs[i], labels)
                    xn_loss = criterion(outputs[i])
                    kl_loss = 0
                    if not self.independent:
                        for j in range(self.model_num):
                            if i != j:
                                kl_loss += self.loss_kl(F.log_softmax(outputs[i], dim=1),
                                                        F.softmax(Variable(outputs[j]), dim=1))

                    # loss = ce_loss + kl_loss / (self.model_num - 1) if not self.independent else ce_loss
                    loss = a * xn_loss + ce_loss + kl_loss / (self.model_num - 1) if not self.independent else ce_loss

                    # measure accuracy and record loss
                    prec = accuracy(outputs[i].data, labels.data, topk=(1,))[0]

                    if not self.independent:
                        kl_losses[i].update(kl_loss.item(), images.size()[0])

                    losses[i].update(loss.item(), images.size()[0])
                    accs[i].update(prec.item(), images.size()[0])

                    # compute gradients and update SGD
                    self.optimizers[i].zero_grad()
                    loss.backward()
                    self.optimizers[i].step()

                # # Training Strategy : a model only learn from the other who performs better, otherwise the kl loss won't be added with its ce loss
                # ce_loss1 = self.loss_ce(outputs[0], labels)
                # ce_loss2 = self.loss_ce(outputs[1], labels)
                #
                # best = 0 if ce_loss1 <= ce_loss2 else 1
                #
                # for i in range(self.model_num):
                #     ce_loss = self.loss_ce(outputs[i], labels)
                #     kl_loss = 0
                #
                #     for j in range(self.model_num):
                #         if i!=j:
                #             kl_loss += self.loss_kl(F.log_softmax(outputs[i], dim = 1),
                #                                     F.softmax(Variable(outputs[j]), dim=1))
                #     loss = (ce_loss + kl_loss / (self.model_num - 1)) if i != best else ce_loss
                #
                #     # measure accuracy and record loss
                #     prec = accuracy(outputs[i].data, labels.data, topk=(1,))[0]
                #     kl_losses[i].update(kl_loss.item(), images.size()[0])
                #     losses[i].update(loss.item(), images.size()[0])
                #     accs[i].update(prec.item(), images.size()[0])
                #
                #     # compute gradients and update SGD
                #     self.optimizers[i].zero_grad()
                #     loss.backward()
                #     self.optimizers[i].step()

                # measure elapsed time
                toc = time.time()
                batch_time.update(toc - tic)

                if not self.independent:
                    pbar.set_description(
                        (
                            "{:.1f}s - model1_loss: {:.4f} - model1_acc: {:.4f} - model1_kl_loss: {:.4f}- model2_kl_loss: {:.4f}".format(
                                (toc - tic), losses[0].avg, accs[0].avg, kl_losses[0].avg, kl_losses[1].avg
                            )
                        )
                    )
                else:
                    pbar.set_description(
                        (
                            "{:.1f}s - model1_loss: {:.4f} - model1_acc: {:.4f} - model1_kl_loss: {:.4f}".format(
                                (toc - tic), losses[0].avg, accs[0].avg, kl_losses[0].avg
                            )
                        )
                    )

                self.batch_size = images.shape[0]
                pbar.update(self.batch_size)
                del images
                del labels
                # log to tensorboard
                if self.use_tensorboard:
                    iteration = epoch * len(self.train_loader) + i
                    for i in range(self.model_num):
                        log_value('train_loss_%d' % (i + 1), losses[i].avg, iteration)
                        log_value('train_acc_%d' % (i + 1), accs[i].avg, iteration)

            return losses, accs

    def validate(self, epoch, flag, sns=None):
        """
        Evaluate the model on the validation set.
        """
        losses = []
        accs = []
        y_true = []
        y_pred = []
        valid_report = []
        probabilities = []
        probabilities_copy = []

        for i in range(self.model_num):
            y_pred.append([])

        for i in range(self.model_num):
            self.models[i].eval()
            losses.append(AverageMeter())
            accs.append(AverageMeter())

        if flag == 0:
            data_set = self.valid_loader
        if flag == 1:
            data_set = self.meta_valid_loader

        for i, (images, labels) in enumerate(data_set):
            if self.use_gpu:
                images, labels = images.cuda(), labels.cuda()
            images, labels = Variable(images), Variable(labels)

            # forward pass
            outputs = []
            for model in self.models:
                outputs.append(model(images))

            for i in range(self.model_num):
                ce_loss = self.loss_ce(outputs[i], labels)
                kl_loss = 0
                for j in range(self.model_num):
                    if i != j:
                        kl_loss += self.loss_kl(F.log_softmax(outputs[i], dim=1),
                                                F.softmax(Variable(outputs[j]), dim=1))

                loss = ce_loss + kl_loss / (self.model_num - 1) if not self.independent else ce_loss

                # measure accuracy and record loss
                prec = accuracy(outputs[i].data, labels.data, topk=(1,))[0]
                losses[i].update(loss.item(), images.size()[0])
                accs[i].update(prec.item(), images.size()[0])

                # record predictions for final result statistics
                _, pred = outputs[i].data.topk(1, 1, True, True)
                pred = pred.t()
                y_pred[i].append(pred[0].cpu())

                if i == 0:
                    raw_probability = F.softmax(outputs[i].cpu(), dim=1)
                    probability = raw_probability.detach().numpy()
                    probabilities.append(probability)

            y_true.append(labels.data.cpu())
            del images
            del labels
        y_true = torch.cat(y_true, dim=0)

        for i in probabilities:
            for j in i:
                probabilities_copy.append(j)

        for i in range(self.model_num):
            y_pred[i] = torch.cat(y_pred[i], dim=0)
            valid_report.append(classification_report(y_true, y_pred[i], zero_division=0, digits=4))
            cm = confusion_matrix(y_true, y_pred[i])
            print(f'model{i + 1}_Confusion Matrix:')
            print(cm)
            print('\n')
        if flag == 1:
            for i in range(len(valid_report)):
                print(valid_report[i])

        # log to tensorboard for every epoch
        if self.use_tensorboard:
            for i in range(self.model_num):
                log_value('valid_loss_%d' % (i + 1), losses[i].avg, epoch + 1)
                log_value('valid_acc_%d' % (i + 1), accs[i].avg, epoch + 1)

        return losses, accs, valid_report, probabilities_copy, y_true

    def test(self):
        """
        Test the model on the held-out test data.
        This function should only be called at the very
        end once the model has finished training.
        """
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()

        # load the best checkpoint
        if self.resume:
            self.load_checkpoint(best=self.best)
            self.model.eval()

        for i, (images, labels) in enumerate(self.test_loader):
            if self.use_gpu:
                images, labels = images.cuda(), labels.cuda()
            images, labels = Variable(images), Variable(labels)

        # forward pass
        outputs = self.model(images)
        loss = self.loss_fn(outputs, labels)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(outputs.data, labels.data, topk=(1, 5))
        losses.update(loss.item(), images.size()[0])
        top1.update(prec1.item(), images.size()[0])
        top5.update(prec5.item(), images.size()[0])

        print(
            '[*] Test loss: {:.4f}, top1_acc: {:.4f}%, top5_acc: {:.4f}%'.format(
                losses.avg, top1.avg, top5.avg)
        )

    def save_checkpoint(self, i, state, is_best):
        self.created__ = """
        Save a copy of the model so that it can be loaded at a future
        date. This function is used when the model is being evaluated
        on the test data.

        If this model has reached the best validation accuracy thus
        far, a seperate file with the suffix `best` is created.
        """
        # print("[*] Saving model to {}".format(self.ckpt_dir))
        filename = self.model_name + str(i + 1) + '_ckpt.pth.tar'
        ckpt_path = os.path.join(self.ckpt_dir, filename)
        torch.save(state, ckpt_path)

        if is_best:
            filename = self.model_name + str(i + 1) + '_model_best.pth.tar'
            shutil.copyfile(
                ckpt_path, os.path.join(self.ckpt_dir, filename)
            )
