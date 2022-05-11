# Courtesy of https://pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html
# Courtesy of https://colab.research.google.com/drive/1jrKpcF6AVCh1M6_2aW9j-QpWnzOZh_mh?usp=sharing#scrollTo=vDjSqSeQyJYk
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from PIL import Image
import torchvision.transforms as transform



# Define model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.Tanh(),
            nn.Linear(512, 64),
            nn.Tanh(),
            nn.Linear(64, 10),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.linear_relu_stack(x)


class MNIST:
    """
     # Initialise global variable of the class MNIST
    """

    def __int__(self):

        # Get training data from  directory.
        self.training_data = datasets.MNIST(
            root="./",
            train=True,
            download=False,
            transform=ToTensor(),
        )

        # Get test_model data from directory.
        self.test_data = datasets.MNIST(
            root="./",
            train=False,
            download=False,
            transform=ToTensor(),
        )

        self.batch_size = 512
        # Get cpu or gpu device for training.
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # print("Using {} device".format(self.device))
        self.model = NeuralNetwork().to(self.device)

        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=1e-2)

    """
    # print(X.shape)
    # Compute prediction error
    # Training pass
    # Backpropagation
    # And optimizes its weights here
    @:param dataloader - input data loader containing  training data and batch size 
    @:param model  - model to be trained 
    @:param loss_fn  - loss function we use in a model
    @:param optimizer - model optimizer
    @:return list of epoch loss
    """

    def train(self, dataloader, model, loss_fn, optimizer):
        # size = len(dataloader.dataset)
        epoch_loss = []
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.reshape(-1, 28 * 28).to(self.device), y.to(self.device)
            pred = model(X)
            loss = loss_fn(pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss.append(loss.detach())
        return epoch_loss

    """
     -Validate model using test data in dataloader 
     @:param dataloader  - dataloader we use to validate model
     @:param model  - model we validate 
     @:return accuracy of the model
    """

    def validate(self, dataloader, model):
        model.eval()
        accuracy = None
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.reshape(-1, 28 * 28).to(self.device), y.to(self.device)
            logits = self.model(X)
            accuracy = self.compute_accuracy(logits, y).detach()
            return accuracy
        return accuracy

    """
        @:param predicted: probabilities of of output
        @:param expected: expected value 
        @:return accuracy of the model 
    """

    def compute_accuracy(self, predicted, expected):
        pred = predicted.argmax(dim=1)
        return (pred == expected).type(torch.float).mean()

    """
       - Early stopping is that every certain number of learning epochs,
       - we compute an evaluation measure (e.g., classification accuracy) on the validation set. 
       - If there is an improvement over the previous time, we store the current values of the neural network weights, 
       - And if there is no improvement for several times, we stop learning and restore the weights of the last best model.
    """

    def evaluation(self):
        train_loss = []
        validation_acc = []
        best_model = None
        best_acc = None
        best_epoch = None
        max_epoch = 10000
        no_improvement = 5
        epochList = []

        for n_epoch in range(max_epoch):
            train_dataloader = DataLoader(self.training_data, batch_size=self.batch_size)  # train dataloader
            epoch_loss = self.train(train_dataloader, self.model, self.loss_fn, self.optimizer)

            train_loss.append(torch.tensor(epoch_loss).mean())
            epochList.append(n_epoch)
            test_dataloader = DataLoader(self.test_data, batch_size=self.batch_size)
            acc = self.validate(test_dataloader, self.model)
            validation_acc.append(acc)
            if best_acc is None or acc > best_acc:
                # print("New best epoch ", n_epoch, "acc", acc)
                best_acc = acc
                best_model = self.model.state_dict()
                best_epoch = n_epoch
            if best_epoch + no_improvement <= n_epoch:
                # print("No improvement for", no_improvement, "epochs")
                break

        self.model.load_state_dict(best_model)
        return train_loss, epochList, validation_acc

    """ 
    - Model save on director Models
    :param model: is the model we save
   """

    def save_model(self, path: str = "Models/classifier_1.pt"):
        torch.save(self.model.state_dict(), path)

    """
    * load image
    * transform to tensor
    * Add batch parameter
    @:param path - path to image 
    @:return tensor data of the image"""

    def read_userImage(self, path: str):
        img = Image.open(path)
        to_tensor = transform.ToTensor()
        tensor = to_tensor(img).unsqueeze(0)

        return tensor

    """
    - Load model saved model
    :param path : is the path of model we load
    :return model loaded 
    """

    def get_model(self, path: str = "Models/classifier_1.pt"):
        model = NeuralNetwork()
        model.load_state_dict(torch.load(path))
        model.eval()
        return model

    """   -Predict image
        @:param img - image we  predict 
        @:param model - model predict given image
        @:return  predicted number """

    def predict(self, img, model) -> int:
        X = img.reshape(-1, 28 * 28).to(self.device)
        pred = model(X)
        str1 = str(pred.argmax())
        return int(str1[str1.find('(') + 1:len(str1) - 1].strip(' '))

    def main(self):
        print("Pytorch Output. . .\n. . .")
        self.evaluation()
        self.save_model()
        print("Done!")
        model = self.get_model()
        input_object = None
        while input_object != "exit":
            input_object = input("Please enter a filepath:\n")
            if input_object == "exit": break
            try:
                input_object = self.read_userImage(input_object)
                print("Classifier: ", self.predict(input_object, model))
            except:
                print("Enter correct path or image.")


run = MNIST()
run.__int__()
run.main()
