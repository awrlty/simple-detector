import os
import numpy as np
from datetime import datetime

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from model import CornerDetectionNet
from loss import CornerLoss
import config
from dataset import PointDataset
# from utils import mean_average_precision


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Frequency to print/log the results.
    print_freq = 5
    tb_log_freq = 5

    # Training hyper parameters.
    momentum = 0.9
    weight_decay = 5.0e-4

    # Load model.
    model = CornerDetectionNet().to(device)

    # Setup loss and optimizer.
    criterion = CornerLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=config.LEARNING_RATE, momentum=momentum, weight_decay=weight_decay)

    # Load Pascal-VOC dataset.
    train_dataset = PointDataset(set_type='train', augment=True)
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, drop_last=True, num_workers=8)

    val_dataset = PointDataset(set_type='valid')
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=True, drop_last=True, num_workers=8)

    print('Number of training images: ', len(train_dataset))

    # Open TensorBoardX summary writer
    log_dir = datetime.now().strftime('%b%d_%H-%M-%S')
    log_dir = os.path.join('results/corner_detector', log_dir)
    writer = SummaryWriter(log_dir=log_dir)

    # Training loop.
    logfile = open(os.path.join(log_dir, 'log.txt'), 'a+')
    best_val_loss = np.inf

    # torch.autograd.set_detect_anomaly(True)
    for epoch in range(config.EPOCHS):
        print('\n')
        print('Starting epoch {} / {}'.format(epoch, config.EPOCHS))

        # Training.
        model.train()
        total_loss = 0.0
        total_batch = 0

        for i, (imgs, targets) in enumerate(train_loader):
            # Update learning rate.
            lr = config.LEARNING_RATE

            # Load data as a batch.
            batch_size_this_iter = imgs.size(0)
            imgs, targets = imgs.to(device), targets.to(device)

            # Forward to compute loss.
            preds = model(imgs)

            loss = criterion(preds, targets)
            loss_this_iter = loss.item()
            total_loss += loss_this_iter
            total_batch += batch_size_this_iter

            # Backward to update model weight.
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Print current loss.
            if i % print_freq == 0:
                print('Epoch [%d/%d], Iter [%d/%d], LR: %.6f, Loss: %.4f, Average Loss: %.4f'
                % (epoch, config.EPOCHS, i, len(train_loader), lr, loss_this_iter / batch_size_this_iter, total_loss / float(total_batch)))

            # TensorBoard.
            n_iter = epoch * len(train_loader) + i
            if n_iter % tb_log_freq == 0:
                writer.add_scalar('train/loss', loss_this_iter, n_iter)
                writer.add_scalar('lr', lr, n_iter)

        # Validation.
        model.eval()
        val_loss = 0.0
        total_batch = 0
        batch_map = 0.0

        for imgs, targets in val_loader:
            # Load data as a batch.
            batch_size_this_iter = imgs.size(0)
            imgs, targets = imgs.to(device), targets.to(device)

            # Forward to compute validation loss.
            with torch.no_grad():
                preds = model(imgs)

            loss = criterion(preds, targets)
            loss_this_iter = loss.item()
            val_loss += loss_this_iter
            total_batch += batch_size_this_iter
            # batch_map += mean_average_precision(preds, targets, iou_thresh=0.3)

        val_loss /= float(total_batch)
        batch_map /= float(total_batch)

        # Save results.
        logfile.writelines(str(epoch) + '\t' + str(val_loss) + '\n')
        logfile.flush()
        torch.save(model.state_dict(), os.path.join(log_dir, 'model_latest.pth'))

        if best_val_loss > val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(log_dir, 'model_best.pth'))

        # Print.
        print(f'Epoch [{epoch}/{config.EPOCHS}], Val Loss: {val_loss: .4f}, Best Val Loss: {best_val_loss: .4f}')
        # print(f'Epoch [{epoch}/{config.EPOCHS}], Val Loss: {val_loss: .4f}, Best Val Loss: {best_val_loss: .4f}, mAP: {batch_map: .4f}')

        # TensorBoard.
        writer.add_scalar('test/loss', val_loss, epoch)

    writer.close()
    logfile.close()
