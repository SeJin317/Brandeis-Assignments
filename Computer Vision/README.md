# Computer Vision HW1_ Sejin Kim

### This project uses MNIST dataset to make a classification model. I used multi-layer perceptron model, and RESNET model. The multi-layer perceptron model had an accuracy of 97%, and RESNET model had an accuracy of 99%. 

## Setup and Installation

### Prerequisites
- Python
- pip

### Creating a Virtual Environment

```bash
# Navigate to your project directory
cd /Users/Guest/Downloads

# Create a virtual environment named 'env'
python -m venv env

# Activate the virtual environment
# On Windows
.\env\Scripts\activate
# On macOS and Linux
source env/bin/activate

pip install torch torchvision

python CV_HW1_part1_Sejin Kim.py
```

## Task 1
```python
#initial setting
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
device = 'cuda' if torch.cuda.is_available() else 'cpu'
```

```python
# Timer Decorator
import time
def timer(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        computation_time = end_time - start_time
        print(f"Execution time of {func.__name__}: {computation_time} seconds")
        return result
    return wrapper
```

```python
# Hyperparameters
CONFIG = {
    'lr': 0.1,
    'epochs': 50,
    'min_batch': 512
}
```

```python
# Training code
def make_train_step(model, loss_fn, optimizer):
    def train_step_fn(x, y):

        model.train()
        y_hat = model(x)
        loss = loss_fn(y_hat, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        return loss.item()
    return train_step_fn

def evaluate_step(x, y):
    y_hat = model(x)
    result = torch.sum(torch.argmax(y_hat, axis=1) == y)
    return result, len(y)
```

```python
# get train, test data
from torch.utils.data.dataloader import DataLoader

train_loader = DataLoader(dataset=train_dataset, batch_size=CONFIG['min_batch'], shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=CONFIG['min_batch'], shuffle=False)
```

```python
class LogisticClassifier(nn.Module):
    def __init__(self, hidden_variables=None, input_output_dim=(28*28, 10)):
        super().__init__()
        # Initial Setting
        self.input_variable_dim = input_output_dim[0]
        self.output_variable_dim = input_output_dim[1]
        variable_dim = self.input_variable_dim
        self.layer = nn.Sequential()
        # Construct Multi-Layers when hidden_variables is not None
        if hidden_variables is not None:
            self.list_hidden_variable = hidden_variables
            for i, hidden_variable in enumerate(self.list_hidden_variable):
                self.layer.add_module('layer_' + str(i), nn.Linear(variable_dim, hidden_variable))
                self.layer.add_module('custom_activation_' + str(i), nn.ReLU())
                variable_dim = hidden_variable
        self.layer.add_module('classifier_layer', nn.Linear(variable_dim, self.output_variable_dim))

    def forward(self, x):
    # Computes the outputs / predictions
        x = x.view(-1, self.input_variable_dim)
        y_hat = self.layer(x)
        return y_hat
```

```python
model = LogisticClassifier().to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=CONFIG['lr'])
train_step = make_train_step(model, loss_fn, optimizer)
```

```python
def train_model(epochs=1000, eval_test_accuracy=False):

    for epoch in tqdm(range(epochs), desc='train'):
        mini_batch_losses = []
        for x_minibatch, y_minibatch in train_loader:
            x_minibatch = x_minibatch.to(device)
            y_minibatch = y_minibatch.to(device)
            mini_batch_loss = train_step(x_minibatch, y_minibatch)
            mini_batch_losses.append(mini_batch_loss)

        # Evaluate train loss
        if (epoch + 1) % 10 == 0:
            loss = np.mean(mini_batch_losses)
            print("train loss at {} epoch:{}".format(epoch + 1, loss))

        # Evaluate test accuracy
    if eval_test_accuracy:
        with torch.no_grad():
            test_accuracy = 0
            test_result = 0
            test_cnt = 0
            for x_minibatch_test, y_minibatch_test in test_loader:
                x_minibatch_test = x_minibatch_test.to(device)
                y_minibatch_test = y_minibatch_test.to(device)
                result, cnt = evaluate_step(x_minibatch_test, y_minibatch_test)
                test_result += result
                test_cnt += cnt
            test_accuracy = 100 * test_result / test_cnt
            print("test accuracy: {}%".format(test_accuracy))
```

```python
# Train with a Multi-Layer Perceptron
model = LogisticClassifier(hidden_variables=[256]).to(device)
optimizer = optim.SGD(model.parameters(), lr=CONFIG['lr'])
train_step = make_train_step(model, loss_fn, optimizer)
train_model(CONFIG['epochs'], eval_test_accuracy=True)
```

![Result of Logistic Regression model]()


### The result of the model using Multi-layer perceptron with logistic regression model with 'lr': 0.1, 'epochs': 50, 'min_batch': 512 was 97.34% accuracy. 
### I tried different epochs with epoch 100, batch size with 256 but the above model showed the best results. 


## Task 2

```python
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
device = 'cuda' if torch.cuda.is_available() else 'cpu'
```

```python
# Timer Decorator
import time
def timer(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        computation_time = end_time - start_time
        print(f"Execution time of {func.__name__}: {computation_time} seconds")
        return result
    return wrapper
```

```python
# Hyperparameters
CONFIG = {
    'lr': 0.1,
    'epochs': 10,
    'min_batch': 32,
    'dropout': 0.0,
    'weight_decay': 0,
}
```

```python
# Training code
def make_train_step(model, loss_fn, optimizer):
    def train_step_fn(x, y):

        model.train()
        y_hat = model(x)
        loss = loss_fn(y_hat, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        return loss.item()
    return train_step_fn

def evaluate_step(x, y):
    y_hat = model(x)
    result = torch.sum(torch.argmax(y_hat, axis=1) == y)
    return result, len(y)
```

```python
@timer
def train_model(epochs=1000, eval_test_accuracy=False):

    for epoch in tqdm(range(epochs), desc='train'):
        model.train()
        mini_batch_losses = []
        for x_minibatch, y_minibatch in train_loader:
            x_minibatch = x_minibatch.to(device)
            y_minibatch = y_minibatch.to(device)
            mini_batch_loss = train_step(x_minibatch, y_minibatch)
            mini_batch_losses.append(mini_batch_loss)

        # Evaluate train loss
        if (epoch + 1) % 10 == 0:
            loss = np.mean(mini_batch_losses)
            print("train loss at {} epoch:{}".format(epoch + 1, loss))

        # Evaluate test accuracy
    if eval_test_accuracy:
        model.eval()
        with torch.no_grad():
            test_accuracy = 0
            test_result = 0
            test_cnt = 0
            for x_minibatch_test, y_minibatch_test in test_loader:
                x_minibatch_test = x_minibatch_test.to(device)
                y_minibatch_test = y_minibatch_test.to(device)
                result, cnt = evaluate_step(x_minibatch_test, y_minibatch_test)
                test_result += result
                test_cnt += cnt
            test_accuracy = 100 * test_result / test_cnt
            print("test accuracy: {}%".format(test_accuracy))
```


```python
from torch.utils.data.dataloader import DataLoader

train_loader = DataLoader(dataset=train_dataset, batch_size=CONFIG['min_batch'], shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=CONFIG['min_batch'], shuffle=False)
```

```python
import torchvision.models as models

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models

class ResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet = models.resnet18(pretrained=False)

        # Modify the first convolutional layer to accept 1-channel input
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, 10)

    def forward(self, x):
        return self.resnet(x)


model = ResNet().to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=CONFIG['lr'], weight_decay=CONFIG['weight_decay'])
train_step = make_train_step(model, loss_fn, optimizer)

# Train Convolutional Nets
train_model(CONFIG['epochs'], eval_test_accuracy=True)
```

### The result of the model using ResNet-18 was 99.28% accuracy.
