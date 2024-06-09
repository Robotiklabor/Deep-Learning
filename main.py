import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from model import Net
from dataloader import load_csv
import itertools
import matplotlib.pyplot as plt


class TrainTestLoop:
    def __init__(self):
        # Load dataset
        x_data_scaled, y_data_scaled = load_csv()
        # Splitting the data
        split_ratio = 0.7
        self.x_train, self.x_test, self.y_train, self.y_test = self.train_test_split(x_data_scaled, y_data_scaled,
                                                                             split_ratio)
        # Load model
        self.model = Net()
        # Convert them into Tensors
        self.x_train = torch.Tensor(self.x_train)
        self.y_train = torch.Tensor(self.y_train)
        self.x_test = torch.Tensor(self.x_test)
        self.y_test = torch.Tensor(self.y_test)

    @staticmethod
    def train_test_split(X, y, split_ratio):
        indices = np.arange(X.shape[0])
        np.random.seed(41)
        np.random.shuffle(indices)
        split_index = int(split_ratio * len(indices))
        train_indices = indices[:split_index]
        test_indices = indices[split_index:]
        return X[train_indices], X[test_indices], y[train_indices], y[test_indices]

    def train(self):

        learning_rates = [0.1, 0.01, 0.001]
        optimizers = ['Adam', 'SGD with momentum', 'RMSprop']
        criteria = ['CrossEntropyLoss', 'L1Loss', 'MSELoss']
        best_hyperparameters = None
        best_test_accuracy = 0.0

        for lr, optimizer_name, criterion_name in itertools.product(learning_rates, optimizers, criteria):
            model = Net()

            if optimizer_name == 'Adam':
                optimizer = optim.Adam(model.parameters(), lr=lr)
            elif optimizer_name == 'SGD with momentum':
                optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
            elif optimizer_name == 'RMSprop':
                optimizer = optim.RMSprop(model.parameters(), lr=lr)

            if criterion_name == 'CrossEntropyLoss':
                criterion = nn.CrossEntropyLoss()
                criterion_type = 'classification'
            elif criterion_name == 'L1Loss':
                criterion = nn.L1Loss()
                criterion_type = 'regression'
            elif criterion_name == 'MSELoss':
                criterion = nn.MSELoss()
                criterion_type = 'regression'

            num_epochs = 100
            train_losses = []
            test_losses = []
            test_accuracies = []

            for epoch in range(num_epochs):

                ######### Training Loop ##############

                optimizer.zero_grad()
                outputs = model(self.x_train)
                loss = criterion(outputs, self.y_train)
                loss.backward()
                optimizer.step()
                train_losses.append(loss.item())

                ######### Testing Loop ################

                with torch.no_grad():
                    model.eval()
                    test_outputs = model(self.x_test)
                    test_loss = criterion(test_outputs, self.y_test)
                    test_losses.append(test_loss.item())

                    # Calculate test accuracy (based on criterion used)

                    if criterion_type == 'classification':
                        _, predicted = torch.max(test_outputs, 1)
                        accuracy = (predicted == self.y_test.argmax(dim=1)).sum().item() / len(self.y_test)
                    else:
                        accuracy = ((test_outputs - self.y_test) ** 2).mean().sqrt().item()

                    test_accuracies.append(accuracy)

            # Calculate the best test accuracy

            if criterion_type == 'classification':
                max_test_accuracy = max(test_accuracies)
            else:
                max_test_accuracy = min(test_accuracies)

            if max_test_accuracy > best_test_accuracy:
                best_test_accuracy = max_test_accuracy
                best_hyperparameters = (lr, optimizer_name, criterion_name)

            ### Training and Test loss curves for all the combination of Hyperparameters ###
            fig, ax = plt.subplots()
            ax.set_xlabel('Epoch', fontsize=10)
            ax.set_ylabel('Loss', fontsize=10)
            ax.set_title(
                f'Training and Test Loss Curve\nLR: {lr}, Optimizer: {optimizer_name}, Criterion: {criterion_name}',
                fontsize=10, color='blue')
            ax.plot(range(num_epochs), train_losses, color='blue', label='Training Loss')
            ax.plot(range(num_epochs), test_losses, color='red', label='Test Loss')
            ax.legend()
            ax.tick_params(axis='x', colors='green', labelsize=8)
            ax.tick_params(axis='y', colors='green', labelsize=8)
            plt.pause(1)
            plt.close()

            # Saving the model weights using torch save
        filename = f"model_lr{best_hyperparameters[0]}_optimizer{best_hyperparameters[1]}_criterion{best_hyperparameters[2]}.pt"
        torch.save(self.model.state_dict(), filename)

        # Load the model weights
        loaded_model = Net()
        loaded_model.load_state_dict(torch.load(filename))
        state_dict = loaded_model.state_dict()

        # Print the model weights
        print("Model weights: ")
        for param_tensor in state_dict:
            print(param_tensor, "\n", state_dict[param_tensor])

        print("\n")

        # Best hyperparameters
        print("Best Hyperparameters:")
        print(f"Learning Rate: {best_hyperparameters[0]}")
        print(f"Optimizer: {best_hyperparameters[1]}")
        print(f"Criterion: {best_hyperparameters[2]}")


if __name__ == '__main__':
    print("Training and Testing the Model")
    print("\n")
    train_test_loop = TrainTestLoop()
    train_test_loop.train()
