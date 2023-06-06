import pandas as pd
import pathlib
import time

import torch
from sklearn.metrics import accuracy_score
from dataset.poison import build_poisoned_training_set, build_testset
from models.badnet import BadNet
from utils import dotdict

def train_one_epoch(data_loader, model, criterion, optimizer, device):
    running_loss = 0
    model.train()

    for _, (images, labels) in enumerate(data_loader):

        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        output = model(images)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    
    return {
        'loss': running_loss / len(data_loader)
    }


def evaluate_badnets(clean_val_loader, poisoned_val_loader, model, device):
    clean_acc = eval(clean_val_loader, model, device)
    asr = eval(poisoned_val_loader, model, device)

    return {
        'clean_acc': clean_acc['acc'], 'clean_loss': clean_acc['loss'],
        'asr': asr['acc'], 'asr_loss': asr['loss'],
    }

def eval(data_loader, model, device):
    criterion = torch.nn.CrossEntropyLoss()
    model.eval()
    y_true, y_pred = [], []
    loss_sum = []

    for _, (images, labels) in enumerate(data_loader):
        images, labels = images.to(device), labels.to(device)
        output = model(images)
        loss = criterion(output, labels)
        loss_sum.append(loss.item())
        y_true.append(labels.cpu())
        y_pred.append(output)
    
    y_true = torch.cat(y_true, dim=0).cpu().numpy()
    y_pred = torch.cat(y_pred, dim=0).cpu().detach().numpy()
    loss = sum(loss_sum) / len(loss_sum)

    return {
        'acc': accuracy_score(y_true, y_pred.argmax(axis=1)),
        'loss': loss
    }


def main():


    args = {
        'dataset': 'cifar10',
        'data_path': './dataset/cifar10',
        'nb_classes': 10,
        'epochs': 100,
        'input_channels': 3,
        'output_channels': 10,
        'batch_size': 32,
        'num_workers': 8,
        'lr': 0.001,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'poison_rate': 0.1,
        'trigger_label': 0,
        'trigger_path': './triggers/trigger_white.png',
        'trigger_size': 5,
        'model_path': './checkpoints/badnet',
    }
    args = dotdict(args)


    # create related path
    pathlib.Path('./checkpoints/').mkdir(parents=True, exist_ok=True)
    pathlib.Path('./logs/').mkdir(parents=True, exist_ok=True)

    clean_train_set, poisoned_train_set = build_poisoned_training_set(args)
    clean_val_set, poisoned_val_set = build_testset(args)

    clean_train_loader = torch.utils.data.DataLoader(clean_train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    poisoned_train_loader = torch.utils.data.DataLoader(poisoned_train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    clean_val_loader = torch.utils.data.DataLoader(clean_val_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    poisoned_val_loader = torch.utils.data.DataLoader(poisoned_val_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    clean_model = BadNet(args.input_channels, args.output_channels).to(args.device)
    poisoned_model = BadNet(args.input_channels, args.output_channels).to(args.device)

    criterion = torch.nn.CrossEntropyLoss()
    clean_optimizer = torch.optim.Adam(clean_model.parameters(), lr=args.lr)
    poisoned_optimizer = torch.optim.Adam(poisoned_model.parameters(), lr=args.lr)
    
    train_model(clean_model, clean_train_loader, criterion, clean_optimizer, clean_val_loader, poisoned_val_loader, args, False)
    train_model(poisoned_model, poisoned_train_loader, criterion, poisoned_optimizer, clean_val_loader, poisoned_val_loader, args, True)
    
    

def train_model(model, train_loader, criterion, optimizer, clean_val_loader, poisoned_val_loader, args, isPoisoned):
    start = time.time()

    stats = []  # Move the creation of 'stats' outside the loop

    model_path = args.model_path  # Store the initial model path

    if isPoisoned:
        model_path += "_poisoned.pth"  # Update the model path for poisoned model
    else:
        model_path += "_clean.pth"  # Update the model path for clean model

    # Train a clean model
    for epoch in range(args.epochs):
        train_stats = train_one_epoch(train_loader, model, criterion, optimizer, args.device)

        if isPoisoned:
            test_stats = evaluate_badnets(clean_val_loader, poisoned_val_loader, model, args.device)
            print(f"# EPOCH {epoch}   loss: {train_stats['loss']:.4f} Test Acc: {test_stats['clean_acc']:.4f}, ASR: {test_stats['asr']:.4f}\n")
        else:
            test_stats = eval(clean_val_loader, model, args.device)
            print(f"# EPOCH {epoch}   loss: {train_stats['loss']:.4f} Test Acc: {test_stats['acc']:.4f}\n")

        torch.save(model.state_dict(), model_path)

        log_stats = {
            **{f'train_{k}': v for k, v in train_stats.items()},
            **{f'test_{k}': v for k, v in test_stats.items()},
            'epoch': epoch,
        }
        
        # save the training stats
        stats.append(log_stats)

    df = pd.DataFrame(stats)
    if isPoisoned:  
        df.to_csv("./logs/trigger%d.csv" % (args.trigger_label), index=False, encoding='utf-8')
    else:
        df.to_csv("./logs/clean.csv", index=False, encoding='utf-8')

    end = time.time()
    print(f"Training time: {end - start}s")

if __name__ == "__main__":
    main()