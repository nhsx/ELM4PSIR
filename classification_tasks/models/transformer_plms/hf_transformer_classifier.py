import itertools
from typing import Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn as nn
from loguru import logger
from matplotlib.font_manager import FontProperties
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AdamW,
    AutoModel,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)
from transformers import (
    get_constant_schedule_with_warmup,
    get_linear_schedule_with_warmup,
)
from transformers.modeling_outputs import (
    SequenceClassifierOutput,
)
from transformers.optimization import Adafactor


def plot_confusion_matrix(cm, class_names):
    """
    Returns a matplotlib figure containing the plotted confusion matrix.

    Args:
       cm (array, shape = [n, n]): a confusion matrix of integer classes
       class_names (array, shape = [n]): String names of the integer classes

    credit: (https://towardsdatascience.com/
            exploring-confusion-matrix-evolution-on-tensorboard-e66b39f4ac12)

    """

    font = FontProperties()
    font.set_family("serif")
    font.set_name("Times New Roman")
    font.set_style("normal")

    figure = plt.figure(figsize=(8, 8))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)

    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    print(f"inside CM tick marks are: {tick_marks}")
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    # Normalize the confusion matrix.
    cm = np.around(cm.astype("float") / cm.sum(axis=1)[:, np.newaxis], decimals=2)

    # Use white text if squares are dark; otherwise black.
    threshold = cm.max() * 0.95

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        color = "white" if cm[i, j] > threshold else "black"
        plt.text(j, i, cm[i, j], horizontalalignment="center", color=color)

    plt.tight_layout()
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    # figure.savefig(f'experiments/{model}/test_mtx.png')

    return figure


# data class
class IncidentDataset(Dataset):
    def __init__(
        self,
        data: pd.DataFrame,
        tokenizer: AutoTokenizer,
        max_token_len: int = 512,
        mode="train",
        label_col="label",
    ):
        self.tokenizer = tokenizer
        self.data = data
        self.max_token_len = max_token_len
        self.mode = mode

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int):
        data_row = self.data.iloc[index]

        all_text = data_row["text"]
        labels = data_row["label"]
        # encoding = self.tokenizer.encode_plus(
        #   all_text,
        #   add_special_tokens=True,
        #   max_length=self.max_token_len,
        #   return_token_type_ids=False,
        #   padding="max_length",
        #   truncation=True,
        #   return_attention_mask=True,
        #   return_tensors='pt',
        # )
        encoding = self.tokenizer(
            all_text,
            padding="max_length",
            truncation=True,
            max_length=self.max_token_len,
            return_tensors="pt",
        )

        # TODO  - implement a balancing of the dataset - i.e. have around 50-50 split
        # of labels

        return dict(
            input_ids=encoding["input_ids"].flatten(),
            attention_mask=encoding["attention_mask"].flatten(),
            labels=torch.tensor(labels),
        )


# data module class - wrapped around pytorch lightning data module
class IncidentDataModule(pl.LightningDataModule):
    def __init__(
        self, train_df, valid_df, test_df, tokenizer, batch_size=2, max_token_len=512
    ):
        super().__init__()
        self.batch_size = batch_size
        self.train_df = train_df
        self.valid_df = valid_df
        self.test_df = test_df
        self.tokenizer = tokenizer
        self.max_token_len = max_token_len

        logger.warning(f"size of training dataset: {len(train_df)} ")
        logger.warning(f"size of validation dataset: {len(valid_df)} ")
        logger.warning(f"size of test dataset: {len(test_df)} ")

    def setup(self, stage=None):
        self.train_dataset = IncidentDataset(
            self.train_df, self.tokenizer, self.max_token_len
        )

        self.valid_dataset = IncidentDataset(
            self.valid_df, self.tokenizer, self.max_token_len
        )
        self.test_dataset = IncidentDataset(
            self.test_df, self.tokenizer, self.max_token_len
        )

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.valid_dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size)


# transformer model base


class IncidentModel(pl.LightningModule):
    def __init__(
        self,
        model,
        num_labels,
        bert_hidden_dim=768,
        classifier_hidden_dim=768,
        n_training_steps=None,
        n_warmup_steps=None,
        dropout=0.2,
        weight_classes=False,
        ce_class_weights=torch.tensor([0.5, 1.5]),
        class_labels=None,
        reinit_n_layers=2,
        nr_frozen_epochs=0,
        nr_frozen_layers=0,
        encoder_learning_rate=2e-5,
        classifier_learning_rate=3e-5,
        optimizer="AdamW",
        cache_dir=None,
        model_type="autoforsequence",
    ):
        super().__init__()
        logger.warning(
            (
                f"Building model based on following architecture. {model} and number "
                f"of labels: {num_labels}"
            )
        )

        self.save_hyperparameters()

        self.num_labels = num_labels
        # this will determine if we use the transformers autoforsequenceclassification
        # class or a custom version with our own classifier
        self.model_type = model_type

        if self.model_type == "autoforsequence":
            logger.warning("Will be using AutoModelForSequenceClassification!")
            self.model = AutoModelForSequenceClassification.from_pretrained(
                f"{model}",
                cache_dir=cache_dir,
                num_labels=self.num_labels,
                return_dict=True,
            )
        elif self.model_type == "customclassifier":
            logger.warning(
                (
                    "Will be using based AutoModel with our own classification head! "
                    "Be warned this can cause problems for many frameworks that "
                    "expect transformer based classes in their purest form"
                )
            )
            self.model = AutoModel.from_pretrained(
                f"{model}", cache_dir=cache_dir, return_dict=True
            )
            # can also add the automodelforsequence classification here - may be easier
            # for later use
            # self.model = AutoModelForSequenceClassification.from_pretrained(
            #   f"{model}",
            #   cache_dir = cache_dir,
            #   return_dict=True
            # )
            # nn.Identity does nothing if the dropout is set to None

            self.classifier = nn.Sequential(
                nn.Linear(bert_hidden_dim, classifier_hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout) if dropout is not None else nn.Identity(),
                nn.Linear(classifier_hidden_dim, num_labels),
            )
        else:
            raise NotImplementedError

        self.class_labels = class_labels
        # reinitialize n layers
        self.reinit_n_layers = reinit_n_layers
        if reinit_n_layers > 0:
            logger.warning(
                f"Re-initializing the last {reinit_n_layers} layers of encoder"
            )
            self._do_reinit()
        # if we want to bias loss based on class sample sizes
        if weight_classes:
            self.criterion = nn.CrossEntropyLoss(weight=ce_class_weights)
        else:
            self.criterion = nn.CrossEntropyLoss()

        self.n_training_steps = n_training_steps
        self.n_warmup_steps = n_warmup_steps
        self.nr_frozen_epochs = nr_frozen_epochs
        self.nr_frozen_layers = nr_frozen_layers
        self.encoder_learning_rate = encoder_learning_rate
        self.classifier_learning_rate = classifier_learning_rate
        self.optimizer = optimizer

        # freeze if you wanted
        if self.nr_frozen_epochs > 0:
            logger.warning(
                (
                    "Freezing the PLM i.e. the encoder - will just be tuning the "
                    "classification head!"
                )
            )
            # self.freeze_encoder() # replace with freeze_n_layers
            # if nr_frozen_layers > -1 then freeze n layers otherwise freeze all
            self.freeze_n_layers(self.nr_frozen_layers)
        else:
            self._frozen = False

    def _do_reinit(self):
        # re-init pooler
        self.model.pooler.dense.weight.data.normal_(
            mean=0.0, std=self.model.config.initializer_range
        )
        self.model.pooler.dense.bias.data.zero_()
        for param in self.model.pooler.parameters():
            param.requires_grad = True

        # re-init last n layers
        for n in range(self.reinit_n_layers):
            self.model.encoder.layer[-(n + 1)].apply(self._init_weight_and_bias)

    def _init_weight_and_bias(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(
                mean=0.0, std=self.model.config.initializer_range
            )
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def unfreeze_encoder(self) -> None:
        """un-freezes the encoder layer."""
        if self._frozen:
            print("Model was frozen so will try unfreezing!")
            if self.model_type == "autoforsequence":
                for param in self.model.base_model.parameters():
                    param.requires_grad = True
                # the name of the PLM component depends on the architecture/pretrained
                # model
                # if "roberta" in self.model.name_or_path:

                #     for param in self.model.roberta.parameters(): # can maybe replace
                # with self.model.base_model? and avoid this if
                # roberta or bert business?
                #         param.requires_grad = True

                # elif "bert" in self.model.name_or_path:
                #     for param in self.model.bert.parameters():
                #         param.requires_grad = True

                # else:
                #     raise NotImplementedError
            else:
                for param in self.model.parameters():
                    param.requires_grad = True
        self._frozen = False

    def freeze_encoder(self) -> None:
        """freezes the encoder layer."""

        if self.model_type == "autoforsequence":
            for param in self.model.base_model.parameters():
                param.requires_grad = False
            # the name of the PLM component depends on the architecture/pretrained
            # model
            # if "roberta" in self.model.name_or_path:

            #     for param in self.model.roberta.parameters():
            #         param.requires_grad = False

            # elif "bert" in self.model.name_or_path:
            #     for param in self.model.bert.parameters():
            #         param.requires_grad = False
            # else:
            #     raise NotImplementedError

        else:
            for param in self.model.parameters():
                param.requires_grad = False
        self._frozen = True

    def freeze_n_layers(model, freeze_layer_count=0) -> None:
        """freeze N last layers of a transformer model"""
        # first freeze the embedding layer - we do this regardless
        for param in model.base_model.embeddings.parameters():
            param.requires_grad = False
        # if the freeze layer count is 0 - do nothing and leave requires_grad = True

        if freeze_layer_count > model.config.num_hidden_layers:
            print(
                f"""The freeze_layer_count provided:{freeze_layer_count}
            is higher than the number of layers the model has: {model.config.num_hidden_layers}!
            """
            )
        else:
            if freeze_layer_count != 0:
                if freeze_layer_count != -1:
                    # if freeze_layer_count == -1, we freeze all of em
                    # otherwise we freeze the first `freeze_layer_count` encoder layers
                    for layer in model.base_model.encoder.layer[:freeze_layer_count]:
                        for param in layer.parameters():
                            param.requires_grad = False
                else:
                    for layer in model.base_model.encoder.layer:
                        for param in layer.parameters():
                            param.requires_grad = False

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], SequenceClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`,
        *optional*):
            Labels for computing the token classification loss. Indices should be in
            `[0, ..., config.num_labels - 1]`.
        """

        return_dict = return_dict if return_dict is not None else True

        # outputs = self.model(input_ids, attention_mask, return_dict = return_dict)

        if self.model_type == "autoforsequence":
            outputs = self.model(
                input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                labels=labels,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            return outputs

        elif self.model_type == "customclassifier":
            outputs = self.model(
                input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

            # obtaining the last layer hidden states of the Transformer
            last_hidden_state = (
                outputs.last_hidden_state
            )  # shape: (batch_size, seq_length, bert_hidden_dim)

            #         or can use the output pooler : output = self.classifier(
            # output.pooler_output
            # )
            # As I said, the CLS token is in the beginning of the sequence. So, we grab
            # its representation
            # by indexing the tensor containing the hidden representations
            CLS_token_state = last_hidden_state[:, 0, :]
            # passing this representation through our custom head
            logits = self.classifier(CLS_token_state)

            loss = 0
            if labels is not None:
                loss = self.criterion(logits, labels)

            if not return_dict:
                raise NotImplementedError
                # output = (logits,) + outputs[2:] # this is from original
                # modelling_roberta - it may break the pytorch training setup in
                # its current form
                # return ((loss,) + output) if loss is not None else output
                # return loss, logits

            return SequenceClassifierOutput(
                loss=loss,
                logits=logits,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
            )
        else:
            raise NotImplementedError

    def training_step(self, batch, batch_idx):
        # input_ids = batch["input_ids"]
        # attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        # just pass all to model
        outputs = self(**batch)
        loss = outputs.loss
        logits = outputs.logits
        self.log("train_loss", loss, prog_bar=True, logger=True)
        return {"loss": loss, "predictions": logits.detach(), "labels": labels.detach()}

    def validation_step(self, batch, batch_idx):
        # input_ids = batch["input_ids"]
        # attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        # print(f"labels in val step are: {labels} with shape: {labels.shape}")
        outputs = self(**batch)
        # print(f"Inside val step logits shape: {outputs.logits.shape}")
        loss = outputs.loss
        logits = outputs.logits
        self.log("val_loss", loss, prog_bar=True, logger=True)
        return {"loss": loss, "predictions": logits.detach(), "labels": labels.detach()}

    def test_step(self, batch, batch_idx):
        # input_ids = batch["input_ids"]
        # attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        outputs = self(**batch)
        loss = outputs.loss
        logits = outputs.logits
        self.log("test_loss", loss, prog_bar=True, logger=True)
        return {"loss": loss, "predictions": logits.detach(), "labels": labels.detach()}

    def validation_epoch_end(self, outputs):
        logger.warning("on validation epoch end")

        # get class labels
        class_labels = self.class_labels

        labels = []
        predictions = []
        scores = []
        for output in outputs:
            for out_labels in output["labels"].to("cpu").detach().numpy():
                labels.append(out_labels)
            for out_predictions in output["predictions"]:
                # the handling of roc_auc score differs for binary and multi class
                if len(class_labels) > 2:
                    scores.append(
                        torch.nn.functional.softmax(out_predictions).cpu().tolist()
                    )
                # append probas
                else:
                    # just get the proba for the 1 class as this is the format required
                    # for sklearn roc calc for binary
                    scores.append(
                        torch.nn.functional.softmax(out_predictions)[1].cpu().tolist()
                    )

                # get predictied labels
                predictions.append(
                    np.argmax(out_predictions.to("cpu").detach().numpy(), axis=-1)
                )

            # use softmax to normalize, as the sum of probs should be 1

        # get epoch loss
        batch_losses = [x["loss"] for x in outputs]  # This part
        epoch_loss = torch.stack(batch_losses).mean()

        logger.info(f"labels are: {labels} with the set being: {set(labels)}")
        logger.info(
            f"predictions are: {predictions} with the set being : {set(predictions)}"
        )

        logger.warning(f"min of labels is: {min(labels)} and max is: {max(labels)}")
        logger.warning(
            f"min of predictions is: {min(predictions)} and max is: {max(predictions)}"
        )
        # get sklearn based metrics
        acc = balanced_accuracy_score(labels, predictions)
        f1_weighted = f1_score(labels, predictions, average="weighted")
        f1_macro = f1_score(labels, predictions, average="macro")
        prec_weighted = precision_score(labels, predictions, average="weighted")
        prec_macro = precision_score(labels, predictions, average="macro")
        recall_weighted = recall_score(labels, predictions, average="weighted")
        recall_macro = recall_score(labels, predictions, average="macro")

        #  roc_auc  - only really good for binary classification but can try for
        # multiclass too pytorch lightning runs a sanity check and roc_score fails
        # if not all classes appear...
        try:
            if len(class_labels) > 2:
                roc_auc_weighted = roc_auc_score(
                    labels, scores, average="weighted", multi_class="ovr"
                )
                roc_auc_macro = roc_auc_score(
                    labels, scores, average="macro", multi_class="ovr"
                )
            else:
                roc_auc_weighted = roc_auc_score(labels, scores, average="weighted")
                roc_auc_macro = roc_auc_score(labels, scores, average="macro")
        except ValueError:
            logger.warning(
                (
                    "roc_scores not calculated due to value error - caused by not all "
                    "classes present in batch"
                )
            )
            roc_auc_weighted = 0
            roc_auc_macro = 0

        # print(f"scores are: {scores}")
        # print(f"labels are: {labels}")
        # print(f"predictions are: {predictions}")
        # print(f"roc scores: {roc_auc_macro}")
        # get confusion matrix

        logger.warning(f"class labels just before cm : {class_labels}")
        cm = confusion_matrix(labels, predictions)
        logger.warning(f"Raw cm : {cm} with shape: {cm.shape}")

        # make plot
        cm_figure = plot_confusion_matrix(cm, class_labels)

        # log this for monitoring
        self.log("monitor_balanced_accuracy", acc)
        self.log("monitor_roc_auc", roc_auc_macro)

        logger.warning(f"current epoch : {self.current_epoch}")

        # log to tensorboard
        self.logger.experiment.add_figure(
            "valid/confusion_matrix", cm_figure, self.current_epoch
        )
        self.logger.experiment.add_scalar(
            "valid/balanced_accuracy", acc, self.current_epoch
        )
        self.logger.experiment.add_scalar(
            "valid/prec_weighted", prec_weighted, self.current_epoch
        )
        self.logger.experiment.add_scalar(
            "valid/prec_macro", prec_macro, self.current_epoch
        )
        self.logger.experiment.add_scalar(
            "valid/f1_weighted", f1_weighted, self.current_epoch
        )
        self.logger.experiment.add_scalar(
            "valid/f1_macro", f1_macro, self.current_epoch
        )
        self.logger.experiment.add_scalar(
            "valid/recall_weighted", recall_weighted, self.current_epoch
        )
        self.logger.experiment.add_scalar(
            "valid/recall_macro", recall_macro, self.current_epoch
        )
        self.logger.experiment.add_scalar(
            "valid/roc_auc_weighted", roc_auc_weighted, self.current_epoch
        )
        self.logger.experiment.add_scalar(
            "valid/roc_auc_macro", roc_auc_macro, self.current_epoch
        )
        self.logger.experiment.add_scalar("valid/loss", epoch_loss, self.current_epoch)

    def test_epoch_end(self, outputs):
        # get class labels
        class_labels = self.class_labels

        labels = []
        predictions = []
        scores = []
        for output in outputs:
            for out_labels in output["labels"].to("cpu").detach().numpy():
                labels.append(out_labels)
            for out_predictions in output["predictions"]:
                # the handling of roc_auc score differs for binary and multi class
                if len(class_labels) > 2:
                    scores.append(
                        torch.nn.functional.softmax(out_predictions).cpu().tolist()
                    )
                # append probas
                else:
                    scores.append(
                        torch.nn.functional.softmax(out_predictions)[1].cpu().tolist()
                    )

                # get predictied labels
                predictions.append(
                    np.argmax(out_predictions.to("cpu").detach().numpy(), axis=-1)
                )

            # use softmax to normalize, as the sum of probs should be 1
        # get sklearn based metrics
        acc = balanced_accuracy_score(labels, predictions)
        f1_weighted = f1_score(labels, predictions, average="weighted")
        f1_macro = f1_score(labels, predictions, average="macro")
        prec_weighted = precision_score(labels, predictions, average="weighted")
        prec_macro = precision_score(labels, predictions, average="macro")
        recall_weighted = recall_score(labels, predictions, average="weighted")
        recall_macro = recall_score(labels, predictions, average="macro")

        #  roc_auc  - only really good for binaryy classification but can try for
        # multiclass too
        if len(class_labels) > 2:
            roc_auc_weighted = roc_auc_score(
                labels, scores, average="weighted", multi_class="ovr"
            )
            roc_auc_macro = roc_auc_score(
                labels, scores, average="macro", multi_class="ovr"
            )
        else:
            roc_auc_weighted = roc_auc_score(labels, scores, average="weighted")
            roc_auc_macro = roc_auc_score(labels, scores, average="macro")

        # get confusion matrix
        cm = confusion_matrix(labels, predictions)

        # make plot
        cm_figure = plot_confusion_matrix(cm, class_labels)

        # log this for monitoring
        self.log("monitor_balanced_accuracy", acc)

        logger.warning(f"current epoch : {self.current_epoch}")

        # log to tensorboard
        self.logger.experiment.add_figure(
            "test/confusion_matrix", cm_figure, self.current_epoch
        )
        self.logger.experiment.add_scalar(
            "test/balanced_accuracy", acc, self.current_epoch
        )
        self.logger.experiment.add_scalar(
            "test/prec_weighted", prec_weighted, self.current_epoch
        )
        self.logger.experiment.add_scalar(
            "test/prec_macro", prec_macro, self.current_epoch
        )
        self.logger.experiment.add_scalar(
            "test/f1_weighted", f1_weighted, self.current_epoch
        )
        self.logger.experiment.add_scalar("test/f1_macro", f1_macro, self.current_epoch)
        self.logger.experiment.add_scalar(
            "test/recall_weighted", recall_weighted, self.current_epoch
        )
        self.logger.experiment.add_scalar(
            "test/recall_macro", recall_macro, self.current_epoch
        )
        self.logger.experiment.add_scalar(
            "test/roc_auc_weighted", roc_auc_weighted, self.current_epoch
        )
        self.logger.experiment.add_scalar(
            "test/roc_auc_macro", roc_auc_macro, self.current_epoch
        )

    def configure_optimizers(self):
        """Sets different Learning rates for different parameter groups."""

        if self.model_type == "autoforsequence":
            # if "roberta" in self.model.name_or_path:
            #     parameters = [
            #         {"params": self.model.classifier.parameters()},
            #         {
            #             "params": self.model.roberta.parameters(),
            #             "lr": self.encoder_learning_rate,
            #         },
            #     ]
            # elif "bert" in self.model.name_or_path:
            #     parameters = [
            #         {"params": self.model.classifier.parameters()},
            #         {
            #             "params": self.model.bert.parameters(),
            #             "lr": self.encoder_learning_rate,
            #         },
            #     ]

            parameters = [
                {"params": self.model.classifier.parameters()},
                {
                    "params": self.model.base_model.parameters(),
                    "lr": self.encoder_learning_rate,
                },
            ]

            # else:
            #     raise NotImplementedError

        else:
            parameters = [
                {"params": self.classifier.parameters()},
                {
                    "params": self.model.parameters(),
                    "lr": self.encoder_learning_rate,
                },
            ]

        if self.optimizer == "adamw":
            optimizer = AdamW(parameters, lr=self.classifier_learning_rate)
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=self.n_warmup_steps,
                num_training_steps=self.n_training_steps,
            )
        elif self.optimizer == "adafactor":
            optimizer = Adafactor(
                parameters,
                lr=self.classifier_learning_rate,
                relative_step=False,
                scale_parameter=False,
                warmup_init=False,
            )
            scheduler = get_constant_schedule_with_warmup(
                optimizer, num_warmup_steps=self.n_warmup_steps
            )
        else:
            raise NotImplementedError

        return dict(
            optimizer=optimizer, lr_scheduler=dict(scheduler=scheduler, interval="step")
        )

    def on_epoch_end(self):
        """Pytorch lightning hook"""
        logger.warning(
            (
                f"On epoch {self.current_epoch}. Number of frozen epochs is: "
                f"{self.nr_frozen_epochs}"
            )
        )
        if self.current_epoch + 1 >= self.nr_frozen_epochs:
            logger.warning("unfreezing PLM(encoder)")
            self.unfreeze_encoder()
