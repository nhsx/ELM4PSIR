import os

import models
import torch
import torch.backends.cudnn as cudnn
from datasets import load_data
from torch import nn, optim
from trainer import Trainer

from utils import load_checkpoint, parse_opt, save_config

cudnn.benchmark = True  # set to true only if inputs to model are fixed size; otherwise
# lot of computational overhead
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_trainer(config):
    # load a checkpoint
    if config.checkpoint is not None:
        # load data
        train_loader = load_data(config, "train", False)
        valid_loader = load_data(config, "valid")
        model, optimizer, word_map, start_epoch = load_checkpoint(
            config.checkpoint, device
        )
        print("\nLoaded checkpoint from epoch %d.\n" % (start_epoch - 1))

    # or initialize model
    else:
        start_epoch = 0

        # load data
        train_loader, embeddings, emb_size, word_map, n_classes, vocab_size = load_data(
            config, "train", True
        )
        valid_loader = load_data(config, "valid")

        model = models.make(
            config=config,
            n_classes=n_classes,
            vocab_size=vocab_size,
            embeddings=embeddings,
            emb_size=emb_size,
        )

        optimizer = optim.Adam(
            params=filter(lambda p: p.requires_grad, model.parameters()), lr=config.lr
        )

    # loss functions
    loss_function = nn.CrossEntropyLoss()

    # move to device
    model = model.to(device)
    loss_function = loss_function.to(device)

    # change the log and ckpt dir paths based on whether pretrained embds were used and
    # whether these were finetuned or not
    if config.fine_tune_word_embeddings:
        # update ckpt and logdir based on the embed and rnn hidden size
        log_dir = f"{config.log_dir}/emb_{config.emb_size}_hidden_{config.rnn_size}/"

        # checkpoint dir
        checkpoint_path = (
            f"{config.checkpoint_path}/emb_{config.emb_size}_hidden_{config.rnn_size}/"
        )
    else:
        # update ckpt and logdir based on the embed and rnn hidden size
        log_dir = (
            f"{config.log_dir}/frozen_emb/emb_{config.emb_size}_hidden_"
            f"{config.rnn_size}/"
        )

        # checkpoint dir
        checkpoint_path = (
            f"{config.checkpoint_path}/frozen_emb/emb_{config.emb_size}_hidden_"
            f"{config.rnn_size}/"
        )
    # make sure the ckpt save dir is made

    if not os.path.exists(f"{checkpoint_path}"):
        os.makedirs(f"{checkpoint_path}")

    # TODO save the config alongside the checkpoints
    save_config(config, log_dir)

    # set up the trainer
    trainer = Trainer(
        num_epochs=config.num_epochs,
        start_epoch=start_epoch,
        train_loader=train_loader,
        valid_loader=valid_loader,
        model=model,
        model_name=config.model_name,
        loss_function=loss_function,
        optimizer=optimizer,
        lr_decay=config.lr_decay,
        dataset_name=config.dataset,
        word_map=word_map,
        grad_clip=config.grad_clip,
        print_freq=config.print_freq,
        checkpoint_path=checkpoint_path,
        checkpoint_basename=config.checkpoint_basename,
        tensorboard=config.tensorboard,
        log_dir=log_dir
        # log_dir = config.log_dir
    )

    return trainer


if __name__ == "__main__":
    config = parse_opt()
    print(f"config args are: {config.__dict__}")
    trainer = set_trainer(config)
    trainer.run_train()
