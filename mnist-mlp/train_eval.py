from torch import nn
import tqdm
import torch.optim as optim

def train(config,model,train_loader,test_loader):
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(),lr = config.lr, momentum=config.momentum)
    
    losses = []
    acces = []
    eval_losses = []
    eval_acces = []

    for epoch in range(config.num_epoches):
        train_loss = 0
        train_acc = 0
        model.train()
        if epoch%5==0:
            optimizer.param_groups[0]['lr']*=0.1
        for img,label in tqdm.tqdm(train_loader):
            img=img.to(config.device)
            label = label.to(config.device)
            img = img.view(img.size(0),-1)
            out = model(img)
            loss = criterion(out,label)
            #有问题
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _,pred = out.max(1)
            num_correct = (pred == label).sum().item()
            acc = num_correct / img.shape[0]
            train_acc += acc

        losses.append(train_loss / len(train_loader))
        acces.append(train_acc / len(train_loader))

        eval_loss = 0
        eval_acc = 0
        model.eval()
        for img,label in tqdm.tqdm(test_loader):
            img=img.to(config.device)
            label=label.to(config.device)
            img = img.view(img.size(0),-1)
            out = model(img)
            loss = criterion(out,label)

            eval_loss += loss.item()
            _,pred = out.max(1)
            num_correct = (pred == label).sum().item()
            acc = num_correct / img.shape[0]
            eval_acc += acc

        eval_losses.append(eval_loss / len(test_loader))
        eval_acces.append(eval_loss / len(test_loader))
        print('epoch: {}, train loss: {:.4f}, train acc: {:.4f}, test loss: {:.4f}, test acc: {:.4f}'
            .format(epoch, train_loss /len(train_loader),train_acc /len(train_loader),eval_loss / len(test_loader),eval_acc / len(test_loader)))
