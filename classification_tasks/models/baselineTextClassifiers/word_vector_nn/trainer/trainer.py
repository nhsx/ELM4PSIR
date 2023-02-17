import time
from typing import Dict, Optional

import numpy as np
import torch
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.common import (
    AverageMeter,
    adjust_learning_rate,
    clip_gradient,
    save_checkpoint,
)
from utils.tensorboard import TensorboardWriter

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Trainer:
    """
    Training pipeline

    Parameters
    ----------
    num_epochs : int
        We should train the model for __ epochs

    start_epoch : int
        We should start training the model from __th epoch

    train_loader : DataLoader
        DataLoader for training data

    model : nn.Module
        Model

    model_name : str
        Name of the model

    loss_function : nn.Module
        Loss function (cross entropy)

    optimizer : optim.Optimizer
        Optimizer (Adam)

    lr_decay : float
        A factor in interval (0, 1) to multiply the learning rate with

    dataset_name : str
        Name of the dataset

    word_map : Dict[str, int]
        Word2id map

    grad_clip : float, optional
        Gradient threshold in clip gradients

    print_freq : int
        Print training status every __ batches

    checkpoint_path : str, optional
        Path to the folder to save checkpoints

    checkpoint_basename : str, optional, default='checkpoint'
        Basename of the checkpoint

    tensorboard : bool, optional, default=False
        Enable tensorboard or not?

    log_dir : str, optional
        Path to the folder to save logs for tensorboard
    """

    def __init__(
        self,
        num_epochs: int,
        start_epoch: int,
        train_loader: DataLoader,
        valid_loader: DataLoader,
        model: nn.Module,
        model_name: str,
        loss_function: nn.Module,
        optimizer,
        lr_decay: float,
        dataset_name: str,
        word_map: Dict[str, int],
        grad_clip=Optional[None],
        print_freq: int = 100,
        checkpoint_path: Optional[str] = None,
        checkpoint_basename: str = "checkpoint",
        tensorboard: bool = False,
        log_dir: Optional[str] = None,
    ) -> None:
        self.num_epochs = num_epochs
        self.start_epoch = start_epoch
        self.train_loader = train_loader
        self.valid_loader = valid_loader

        self.model = model
        self.model_name = model_name
        self.loss_function = loss_function
        self.optimizer = optimizer
        self.lr_decay = lr_decay

        self.dataset_name = dataset_name
        self.word_map = word_map
        self.print_freq = print_freq
        self.grad_clip = grad_clip

        self.checkpoint_path = checkpoint_path
        self.checkpoint_basename = checkpoint_basename

        # setup visualization writer instance
        self.writer = TensorboardWriter(log_dir, tensorboard)
        self.len_epoch = len(self.train_loader)

    def train(self, epoch: int) -> None:
        """
        Train an epoch

        Parameters
        ----------
        epoch : int
            Current number of epoch
        """
        self.model.train()  # training mode enables dropout

        batch_time = AverageMeter()  # forward prop. + back prop. time per batch
        data_time = AverageMeter()  # data loading time per batch
        losses = AverageMeter(tag="loss", writer=self.writer)  # cross entropy loss
        accs = AverageMeter(tag="acc", writer=self.writer)  # accuracies

        start = time.time()

        # print(f"Length of train data loader is: {len(self.train_loader)}")
        # batches
        for i, batch in enumerate(tqdm(self.train_loader)):
            data_time.update(time.time() - start)

            if self.model_name in ["han"]:
                documents, sentences_per_document, words_per_sentence, labels = batch

                documents = documents.to(
                    device
                )  # (batch_size, sentence_limit, word_limit)
                sentences_per_document = sentences_per_document.squeeze(1).to(
                    device
                )  # (batch_size)
                words_per_sentence = words_per_sentence.to(
                    device
                )  # (batch_size, sentence_limit)
                labels = labels.squeeze(1).to(device)  # (batch_size)

                # forward
                scores, _, _ = self.model(
                    documents, sentences_per_document, words_per_sentence
                )  # (n_documents, n_classes),
                # (n_documents, max_doc_len_in_batch, max_sent_len_in_batch),
                # (n_documents, max_doc_len_in_batch)

            else:
                # print(f"length of batch is: {len(batch)}")
                # print(f"batch is: {batch}")
                sentences, words_per_sentence, labels = batch

                sentences = sentences.to(device)  # (batch_size, word_limit)
                words_per_sentence = words_per_sentence.squeeze(1).to(
                    device
                )  # (batch_size)
                labels = labels.squeeze(1).to(device)  # (batch_size)

                # for torchtext
                # sentences = batch.text[0].to(device)  # (batch_size, word_limit)
                # words_per_sentence = batch.text[1].to(device)  # (batch_size)
                # labels = batch.label.to(device)  # (batch_size)

                scores = self.model(
                    sentences, words_per_sentence
                )  # (batch_size, n_classes)

            # calc loss
            loss = self.loss_function(scores, labels)  # scalar

            # backward
            self.optimizer.zero_grad()
            loss.backward()

            # clip gradients
            if self.grad_clip is not None:
                clip_gradient(self.optimizer, self.grad_clip)

            # update weights
            self.optimizer.step()

            # find accuracy
            _, predictions = scores.max(dim=1)  # (n_documents)
            correct_predictions = torch.eq(predictions, labels).sum().item()
            accuracy = correct_predictions / labels.size(0)

            # set step for tensorboard
            step = (epoch - 1) * self.len_epoch + i
            self.writer.set_step(step=step, mode="train")

            # keep track of metrics
            batch_time.update(time.time() - start)
            losses.update(loss.item(), labels.size(0))
            accs.update(accuracy, labels.size(0))

            start = time.time()

            # print training status
            if i % self.print_freq == 0:
                print(
                    "Epoch: [{0}][{1}/{2}]\t"
                    "Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                    "Data Load Time {data_time.val:.3f} ({data_time.avg:.3f})\t"
                    "Loss {loss.val:.4f} ({loss.avg:.4f})\t"
                    "Accuracy {acc.val:.3f} ({acc.avg:.3f})".format(
                        epoch,
                        i,
                        len(self.train_loader),
                        batch_time=batch_time,
                        data_time=data_time,
                        loss=losses,
                        acc=accs,
                    )
                )

    def evaluate(self, epoch) -> None:
        # track metrics

        # batch_time = AverageMeter()  # forward prop. + back prop. time per batch
        # data_time = AverageMeter()  # data loading time per batch
        losses = AverageMeter(tag="loss", writer=self.writer)  # cross entropy loss
        # accs = AverageMeter(tag="acc", writer=self.writer)  # accuracies
        f1_macro = AverageMeter(
            tag="f1_macro", writer=self.writer
        )  # cross entropy loss
        recall_macro = AverageMeter(
            tag="recall_macro", writer=self.writer
        )  # accuracies
        # precision_macro = AverageMeter(
        #     tag="precision_macro", writer=self.writer
        # )  # cross entropy loss
        # roc_macro = AverageMeter(tag="roc_macro", writer=self.writer)  # accuracies

        # set model to eval mode
        self.model.eval()

        # evaluate in batches
        with torch.no_grad():
            all_scores = []
            all_labels = []
            all_preds = []
            for i, batch in enumerate(tqdm(self.valid_loader, desc="Evaluating")):
                if self.model_name in ["han"]:
                    (
                        documents,
                        sentences_per_document,
                        words_per_sentence,
                        labels,
                    ) = batch

                    documents = documents.to(
                        device
                    )  # (batch_size, sentence_limit, word_limit)
                    sentences_per_document = sentences_per_document.squeeze(1).to(
                        device
                    )  # (batch_size)
                    words_per_sentence = words_per_sentence.to(
                        device
                    )  # (batch_size, sentence_limit)
                    labels = labels.squeeze(1).to(device)  # (batch_size)

                    # forward
                    scores, word_alphas, sentence_alphas = self.model(
                        documents, sentences_per_document, words_per_sentence
                    )  # (n_documents, n_classes),
                    # (n_documents, max_doc_len_in_batch, max_sent_len_in_batch),
                    # (n_documents, max_doc_len_in_batch)

                else:
                    sentences, words_per_sentence, labels = batch

                    sentences = sentences.to(device)  # (batch_size, word_limit)
                    words_per_sentence = words_per_sentence.squeeze(1).to(
                        device
                    )  # (batch_size)
                    labels = labels.squeeze(1).to(device)  # (batch_size)

                    # for torchtext
                    # sentences = batch.text[0].to(device)  # (batch_size, word_limit)
                    # words_per_sentence = batch.text[1].to(device)  # (batch_size)
                    # labels = batch.label.to(device)  # (batch_size)

                    scores = self.model(
                        sentences, words_per_sentence
                    )  # (batch_size, n_classes)

                # calc loss
                loss = self.loss_function(scores, labels)  # scalar

                # accuracy
                _, predictions = scores.max(dim=1)  # (n_documents)
                # correct_predictions = torch.eq(predictions, labels).sum().item()
                # accuracy = correct_predictions / labels.size(0)

                # apply softmax to logits and get the positive class probability
                positive_class_probas = torch.nn.functional.softmax(scores, dim=1)[:, 1]
                # print(f"positive class probs shape: {positive_class_probas.shape}")

                # append to lists
                all_scores.append(positive_class_probas.cpu().tolist())
                all_labels.append(labels.cpu().tolist())
                all_preds.append(predictions.cpu().tolist())

                # print(f"all scores: {all_scores}\n\n")
                # print(f"all labels: {all_labels}\n\n")
                # print(f"all preds: {all_preds}")

                # # apply softmax to scores for metric calc
                # # the handling of roc_auc score differs for binary and multi class
                # if len(class_labels) > 2:
                #     scores.append(
                #       torch.nn.functional.softmax(out_predictions).cpu().tolist()
                #      )
                # # append probas
                # else:
                #     scores.append(
                #       torch.nn.functional.softmax(out_predictions)[1].cpu().tolist()
                #     )

                # # get predictied labels
                # predictions.append(
                #   np.argmax(out_predictions.to('cpu').detach().numpy(), axis = -1)
                # )

            # clunky - combine lists into one list
            all_scores = np.concatenate(all_scores)
            all_labels = np.concatenate(all_labels)
            all_preds = np.concatenate(all_preds)

            # get sklearn based metrics
            acc = balanced_accuracy_score(all_labels, all_preds)
            f1_weighted = f1_score(all_labels, all_preds, average="weighted")
            f1_macro = f1_score(all_labels, all_preds, average="macro")
            prec_weighted = precision_score(all_labels, all_preds, average="weighted")
            prec_macro = precision_score(all_labels, all_preds, average="macro")
            recall_weighted = recall_score(all_labels, all_preds, average="weighted")
            recall_macro = recall_score(all_labels, all_preds, average="macro")
            # roc for binary requires the probabilities for the greatest class
            # i.e. class 1

            roc_auc_weighted = roc_auc_score(all_labels, all_scores, average="weighted")
            roc_auc_macro = roc_auc_score(all_labels, all_scores, average="macro")

            # print('\n * VALID ACCURACY - %.1f percent\n' % (accs.avg * 100))
            print("Valid metrics: ")
            print(f"accuracy: {acc}")
            print(f"f1 macro: {f1_macro}")
            print(f"roc macro: {roc_auc_macro}")

            # set up step
            self.writer.set_step(step=epoch, mode="valid")
            # write different metrics to tensorboard
            self.writer.add_scalar("f1_macro", f1_macro)
            self.writer.add_scalar("f1_weighted", f1_weighted)
            self.writer.add_scalar("accuracy", acc)
            self.writer.add_scalar("precision_macro", prec_macro)
            self.writer.add_scalar("precision_weighted", prec_weighted)
            self.writer.add_scalar("recall_macro", recall_macro)
            self.writer.add_scalar("recall_weighted", recall_weighted)
            self.writer.add_scalar("roc_auc_macro", roc_auc_macro)
            self.writer.add_scalar("roc_auc_weighted", roc_auc_weighted)

            # keep track of metrics

            losses.update(loss.item(), labels.size(0))
            # return validation accuracy for monitoring/saving best checkpoint
            return acc

    def run_train(self):
        start = time.time()
        running_val_acc = 0
        # epochs
        for epoch in range(self.start_epoch, self.num_epochs):
            # trian an epoch
            self.train(epoch=epoch)

            # time per epoch
            epoch_time = time.time() - start
            print(
                "Epoch: [{0}] finished, time consumed: {epoch_time:.3f}".format(
                    epoch, epoch_time=epoch_time
                )
            )

            # decay learning rate every epoch
            adjust_learning_rate(self.optimizer, self.lr_decay)

            # TODO add evaluation on validation dataset here
            print("Running evaluation!")
            val_accuracy = self.evaluate(epoch=epoch)

            # TODO adjust below to only save if a metric increases
            if val_accuracy > running_val_acc:
                print("Validation accuracy improved - saving checkpoint!")
                # save checkpoint
                if self.checkpoint_path is not None:
                    save_checkpoint(
                        epoch=epoch,
                        model=self.model,
                        model_name=self.model_name,
                        optimizer=self.optimizer,
                        dataset_name=self.dataset_name,
                        word_map=self.word_map,
                        checkpoint_path=self.checkpoint_path,
                        checkpoint_basename=self.checkpoint_basename,
                    )

            # now set the running val acc to the most recent returned validation
            # accuracy
            running_val_acc = val_accuracy

            start = time.time()
