import torch
from training.torch.set_up import set_up_training_components
from tqdm import tqdm


def train_model(
    model,
    train_dataloader,
    val_dataloader,
    num_epochs=10,
    lr=0.001,
    pos_class_weight=None,
    class_weights=None,
    save_dir=None,
    use_lengths=False,
    patience=5,
    min_earlystop_delta=0.1,
):
    # Initialise model

    best_val_loss = float("inf")

    # Define loss function and optimizer
    if class_weights and not pos_class_weight:
        pos_class_weight = class_weights[0]

    criterion, optimizer, scheduler, early_stopper = set_up_training_components(
        model, lr, pos_class_weight, patience, min_earlystop_delta
    )

    training_losses = []
    val_losses = []

    # train model
    for epoch in tqdm(range(num_epochs)):
        epoch_train_loss = run_1_epoch(
            model, train_dataloader, criterion, optimizer, train=True
        )
        epoch_val_loss = run_1_epoch(model, val_dataloader, criterion, train=False)

        training_losses.append(epoch_train_loss)
        val_losses.append(epoch_val_loss)
        scheduler.step(epoch_val_loss)

        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            if save_dir:
                torch.save(model.state_dict(), save_dir)
        if early_stopper.early_stop(epoch_val_loss):
            print(f"Stopping at epoch {epoch+1}")
            break

        tqdm.write(
            f"Epoch {epoch+1}/{num_epochs} |"
            f"Train loss: {epoch_train_loss:.4f} | Early stopping loss: {epoch_val_loss:.4f} | "
            f"Learning rate: {scheduler.get_last_lr()[0]}"
        )

    return training_losses, val_losses, best_val_loss


def run_1_epoch(
    model,
    dataloader,
    criterion,
    optimizer=None,
    train: bool = True,
    additional_metric=None,
):
    """
    Run one epoch over a dataloader in either training or validation mode.
    Shows progress bar and returns average loss.
    Optionally returns an additional metric which takes binary true labels and binary prediction.
    """
    model.train(mode=train)

    running_loss = 0.0
    all_preds = []
    all_labels = []

    progress_bar = tqdm(dataloader, desc="Train" if train else "Val", leave=False)

    for batch in progress_bar:
        inputs = batch[0]
        labels = batch[1]
        static_data = batch[-1]
        if labels.dim() == 1:
            labels = labels.unsqueeze(1)
        with torch.set_grad_enabled(train):
            outputs = model(inputs, static_data)
            if outputs.dim() == 1:
                outputs = outputs.unsqueeze(1)
            loss = criterion(outputs, labels)

            # Calculate predicted outcomes for additional metric
            probabilities = torch.sigmoid(outputs)
            predicted = (probabilities > 0.5).int().numpy()
            all_preds.extend(predicted)
            all_labels.extend(labels.numpy())

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        running_loss += loss.item() * inputs.size(0)

        # update progress bar with batch loss
        progress_bar.set_postfix(loss=loss.item())
    epoch_loss = running_loss / len(dataloader.dataset)

    if additional_metric:
        return epoch_loss, additional_metric(all_labels, all_preds)

    return epoch_loss
