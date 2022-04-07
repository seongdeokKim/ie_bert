from copy import deepcopy

import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim
from seqeval.metrics import classification_report, f1_score


class Trainer:
    def __init__(self, config):
        self.config = config
        self.best_model = None
        self.best_loss = np.inf
        self.best_accuracy = 0.0

    def check_best(self, model, current_loss):
        if current_loss <= self.best_loss: # If current epoch returns lower validation loss,
            self.best_loss = current_loss  # Update lowest validation loss.
            self.best_model = deepcopy(model.state_dict()) # Update best model weights.

    def train(
            self,
            model, optimizer, scheduler,
            train_loader, valid_loader,
            index_to_label,
            device,
    ):

        for epoch in range(self.config.n_epochs):

            # Put the model into training mode.
            model.train()
            # Reset the total loss for this epoch.
            total_tr_loss = 0

            for step, mini_batch in enumerate(train_loader):
                input_ids, labels = mini_batch['input_ids'], mini_batch['labels']
                input_ids, labels = input_ids.to(device), labels.to(device)
                attention_mask = mini_batch['attention_mask']
                attention_mask = attention_mask.to(device)

                entity_ids = mini_batch['entity_ids']
                entity_ids = entity_ids.to(device)

                # You have to reset the gradients of all model parameters
                # before to take another step in gradient descent.
                optimizer.zero_grad()

                # Take feed-forward
                outputs = model(
                    input_ids,
                    attention_mask=attention_mask,
                    entity_ids=entity_ids,
                    labels=labels
                )
                loss, logits = outputs[0], outputs[1]

                # Perform a backward pass to calculate the gradients.
                #loss.backward()
                loss.backward()
                # track train loss
                total_tr_loss += loss.item()
                # update parameters
                optimizer.step()
                # Update the learning rate.
                scheduler.step()

            # Calculate the average loss over the training data.
            avg_tr_loss = total_tr_loss / len(train_loader)
            print('Epoch {} - loss={:.4e}'.format(
                epoch+1,
                avg_tr_loss
            ))

            # Put the model into evaluation mode
            model.eval()
            # Reset the validation loss and accuracy for this epoch.
            total_val_loss, total_val_accuracy = 0, 0
            preds, true_labels = [], []
            for step, mini_batch in enumerate(valid_loader):
                input_ids, labels = mini_batch['input_ids'], mini_batch['labels']
                input_ids, labels = input_ids.to(device), labels.to(device)
                attention_mask = mini_batch['attention_mask']
                attention_mask = attention_mask.to(device)

                entity_ids = mini_batch['entity_ids']
                entity_ids = entity_ids.to(device)

                # Telling the model not to compute or store gradients,
                # saving memory and speeding up validation
                with torch.no_grad():
                    outputs = model(
                        input_ids,
                        attention_mask=attention_mask,
                        entity_ids=entity_ids,
                        labels=labels
                    )
                    loss, logits = outputs[0], outputs[1]

                    # Calculate the accuracy for this batch of test sentences.
                    total_val_loss += loss.mean().item()

                    # Calculate accuracy only if 'y' is LongTensor,
                    # which means that 'y' is one-hot representation.
                    if isinstance(labels, torch.LongTensor) or isinstance(labels, torch.cuda.LongTensor):
                        accuracy = (torch.argmax(logits, dim=-1) == labels).sum() / float(labels.size(0))
                    else:
                        accuracy = 0

                    total_val_accuracy += float(accuracy)

                    # Move logits and labels to CPU
                    logits = logits.detach().cpu().numpy()
                    labels = labels.to('cpu').numpy()
                    # 2d array
                    for pred in np.argmax(logits, axis=-1):
                        preds.append([pred])
                    for label in labels:
                        true_labels.append([label])

            pred_class, true_class = [], []
            for pred, true_label in zip(preds, true_labels):
                pred_class.append(
                    [index_to_label.get(pred[0])]
                )
                true_class.append(
                    [index_to_label.get(true_label[0])]
                )

            avg_val_loss = total_val_loss / len(valid_loader)
            avg_val_acc = total_val_accuracy / len(valid_loader)
            avg_val_report = classification_report(pred_class, true_class)
            avg_micro_f1_score = f1_score(pred_class, true_class, average='micro')
            avg_macro_f1_score = f1_score(pred_class, true_class, average='macro')

            self.check_best(model, avg_val_loss)

            print('Validation - loss={:.4e} accuracy={:.4f} best_loss={:.4f} micro_f1={:.4f} macro_f1={:.4f}'.format(
                avg_val_loss,
                avg_val_acc,
                self.best_loss,
                avg_micro_f1_score,
                avg_macro_f1_score
            ))
            print(avg_val_report)
            print()

        model.load_state_dict(self.best_model)

        return model


    def test(
            self,
            model,
            test_loader,
            index_to_label,
            device,
    ):

        # Put the model into evaluation mode
        model.eval()
        # Reset the validation loss and accuracy for this epoch.
        total_test_loss, total_test_accuracy = 0, 0
        preds, true_labels = [], []
        for step, mini_batch in enumerate(test_loader):
            input_ids, labels = mini_batch['input_ids'], mini_batch['labels']
            input_ids, labels = input_ids.to(device), labels.to(device)
            attention_mask = mini_batch['attention_mask']
            attention_mask = attention_mask.to(device)

            entity_ids = mini_batch['entity_ids']
            entity_ids = entity_ids.to(device)

            # Telling the model not to compute or store gradients,
            # saving memory and speeding up validation
            with torch.no_grad():
                outputs = model(
                    input_ids,
                    attention_mask=attention_mask,
                    entity_ids=entity_ids,
                    labels=labels
                )
                loss, logits = outputs[0], outputs[1]

                # Calculate the accuracy for this batch of test sentences.
                total_test_loss += loss.mean().item()

                # Calculate accuracy only if 'y' is LongTensor,
                # which means that 'y' is one-hot representation.
                if isinstance(labels, torch.LongTensor) or isinstance(labels, torch.cuda.LongTensor):
                    accuracy = (torch.argmax(logits, dim=-1) == labels).sum() / float(labels.size(0))
                else:
                    accuracy = 0

                total_test_accuracy += float(accuracy)

                #pred_class += torch.argmax(logits, dim=-1).cpu().numpy()
                #true_label += labels.cpu().numpy()

                # Move logits and labels to CPU
                logits = logits.detach().cpu().numpy()
                labels = labels.to('cpu').numpy()
                # 2d array
                for pred in np.argmax(logits, axis=-1):
                    preds.append([pred])
                for label in labels:
                    true_labels.append([label])

        pred_class, true_class = [], []
        for pred, true_label in zip(preds, true_labels):
            pred_class.append(
                [index_to_label.get(pred[0])]
            )
            true_class.append(
                [index_to_label.get(true_label[0])]
            )

        avg_test_loss = total_test_loss / len(test_loader)
        avg_test_acc = total_test_accuracy / len(test_loader)
        avg_micro_f1_score = f1_score(pred_class, true_class, average='micro')
        avg_macro_f1_score = f1_score(pred_class, true_class, average='macro')
        avg_test_report = classification_report(pred_class, true_class)

        print('Test - loss={:.4e} accuracy={:.4f} micro_f1={:.4f} macro_f1={:.4f}'.format(
            avg_test_loss,
            avg_test_acc,
            avg_micro_f1_score,
            avg_macro_f1_score
        ))
        print()
        print(avg_test_report)

