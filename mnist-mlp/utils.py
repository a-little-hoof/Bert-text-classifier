import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import mnist 

def build_dataset(train_batch_size,test_batch_size):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5],[0.5])])
    #download
    train_dataset = mnist.MNIST('./data',train=True,transform=transform)
    test_dataset = mnist.MNIST('./data',train=False,transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False)
    return train_loader,test_loader