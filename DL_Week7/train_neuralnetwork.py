import torch.optim
from torchvision.transforms import ToTensor
from cifar_dataloader import CIFARDataset,AnimalDataset
from model import SimpleCNN
from torch.utils.data import DataLoader
import torch.nn as nn
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from sklearn.metrics import classification_report, accuracy_score
from argparse import ArgumentParser
from tqdm.autonotebook import tqdm
from torch.utils.tensorboard import SummaryWriter
import os
import shutil
def get_args():
    parser = ArgumentParser(description="CNN training")
    parser.add_argument("--root","-r",type=str,default="./data", help="root")
    parser.add_argument("--epochs","-e",type=int,default=100, help="Number of epochs")
    parser.add_argument("--batch-size","-b",type=int,default=100, help="Batch Size") #batch-size or batch_size is acceptable
    parser.add_argument("--image-size", "-i", type=int, default=224, help="Image_Size")
    parser.add_argument("--logging", "-l", type=str, default="tensorboard", help="logging")
    parser.add_argument("--trained-models", "-t", type=str, default="trained_models")
    args = parser.parse_args()
    return args
if __name__ == '__main__':
    args = get_args()
    #print(args.epochs)
    #print(args.batch_size)# Must be batch_size
    transform = Compose([
        Resize((args.image_size, args.image_size)),
        ToTensor(),
    ])
    train_dataset = AnimalDataset(root=args.root, train=True, transform = transform)
    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size= args.batch_size,
        shuffle=True,
        num_workers=4,
        drop_last=True
    )
    test_dataset = AnimalDataset(root=args.root, train=False, transform = transform)
    test_dataloader = DataLoader(
        dataset=test_dataset,
        batch_size= args.batch_size,
        shuffle=False,
        num_workers=4,
        drop_last=False
    )
    if os.path.isdir(args.logging):
        shutil.rmtree(args.logging) # If folder exists tensorboard then it will be deleted using shutil library
                                    # For the non-existing then use os.rmdir
    if not os.path.isdir(args.trained_models):
        os.mkdir(args.trained_models)
    writer = SummaryWriter(args.logging)
    model = SimpleCNN(num_classes=10)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)
    num_iters = len(train_dataloader)
    if torch.cuda.is_available():
        model.cuda()

    best_acc = 0
    for epoch in range(args.epochs):
        model.train()
        progress_bar = tqdm(train_dataloader, colour="cyan")
        for iter, (images, labels) in enumerate(progress_bar):
            if torch.cuda.is_available():
                images = images.cuda()
                labels = labels.cuda()
            # forward
            outputs = model(images)
            loss_value = criterion(outputs, labels)
            if (iter + 1) % 10:
               progress_bar.set_description("Epoch {}/{}. Iteration {}/{}. Loss {:.3f}".format(epoch+1, args.epochs, iter+1, num_iters, loss_value))
            writer.add_scalar("Train/Loss",loss_value, num_iters*epoch+iter)
            # backward
            optimizer.zero_grad()
            loss_value.backward()
            optimizer.step()

        model.eval()
        all_predictions = []
        all_labels = []
        for iter, (images, labels) in enumerate(test_dataloader):
            all_labels.extend(labels)
            if torch.cuda.is_available():
                images = images.cuda()
                labels = labels.cuda()

            with torch.no_grad():
                predictions = model(images)   # predictions shape 64x10
                indices = torch.argmax(predictions.cpu(), dim=1)
                all_predictions.extend(indices)
                loss_value = criterion(predictions, labels)
        all_labels = [label.item() for label in all_labels]
        all_predictions = [prediction.item() for prediction in all_predictions]
        accuracy = accuracy_score(all_labels,all_predictions)
        print("Epoch: {} : Accuracy: {}".format(epoch+1, accuracy_score(all_labels, all_predictions)) )
        writer.add_scalar("Val/Acc",accuracy,epoch)
        torch.save(model.state_dict(),"{}/last_cnn.pt".format(args.trained_models))
        if (accuracy > best_acc):
            torch.save(model.state_dict(), "{}/best_cnn.pt".format(args.trained_models))
            best_acc = accuracy
        #print(classification_report(all_labels, all_predictions))
        #tensorboard --logdir tensorboard
        


