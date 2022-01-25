import torch
import random
import numpy
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt
import pandas as pd

from SGD_IKSA import SGD_IKSA


# Running an experiment


def experiment(device, model, trainset, testset, batch_size, no_epochs, lr, momentum,  optimizer_type, LM_f, LM_c, LM_c_run_min):

    """[summary]

    Args:
        device ([]):
        model ([type]): [description]
        trainset ([type]): [description]
        testset ([type]): [description]
        no_epochs ([type]): [description]
        lr ([type]): [description]
        momentum ([type]): [description]
        optimizer_type ([type]): [description]
        LM_f ([type]): [description]
        LM_c ([type]): [description]
        LM_c_run_min ([type]): [description]

    Returns:
        [type]: [description]
    """
    # We want to save the hyperparameters used for this experiment in a dictionary.
    #param_dict = {"lr": lr, "LM_f": LM_f, "optim_type": optimizer_type, \
    #                "no_epochs": no_epochs, "LM_c": LM_c, "momentum": momentum}
    param_dict = {"no_epochs": no_epochs,
                    "lr": lr,
                    "momentum": momentum,
                    "optimizer_type": optimizer_type,
                    "LM_f": LM_f.__name__,
                    "LM_c": LM_c,
                    "LM_c_run_min": LM_c_run_min}
    # We save the loss values.
    loss_list = []
    # We save the c values.
    c_list = []
    # We save a shorter loss list, of average values (every 2000 minibatch)
    average_loss_list = []

    # We set the seed at the beginning of each experiment to ensure reproducibility
    torch.manual_seed(0)
    #random.seed(0)
    #numpy.random.seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)

    # We load the data
    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        numpy.random.seed(worker_seed)
        random.seed(worker_seed)

    g = torch.Generator()
    g.manual_seed(0)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                        shuffle=True, num_workers=2, worker_init_fn = seed_worker, generator = g)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                        shuffle=False, num_workers=2, worker_init_fn = seed_worker, generator = g)

    # Define model and move to device
    net = model
    net.to(device)

    # Define loss function
    criterion = nn.CrossEntropyLoss()

    # Define optimizer
    if optimizer_type == "Original":
        optimizer = optim.SGD(net.parameters(), lr=lr, momentum=momentum)
    elif optimizer_type == "LM":
        optimizer = SGD_IKSA(net.parameters(), LM_f, lr=lr, momentum=momentum)

    for epoch in range(no_epochs):

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # input data
            inputs, labels = data[0].to(device), data[1].to(device)

            optimizer.zero_grad()
            outputs = net(inputs)

            loss = criterion(outputs, labels)
            loss_list.append(loss)
            loss.backward()

            #if LM_c_run_min:
            #     if loss < LM_c:
            #        LM_c = loss
            if optimizer_type == "LM":
              if LM_c_run_min:
                if loss < LM_c:
                  LM_c = loss.item()
              optimizer.step(LM_c, loss)
            else:
                optimizer.step()

            c_list.append(LM_c)
            running_loss += loss.item()
            if i % 2000 == 1999:    # every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, i + 1, running_loss / 2000))
                average_loss_list.append(running_loss / 2000)
                running_loss = 0.0

    print('Finished Training')

    # TESTING ACCURACY

    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            # calculate outputs by running images through the network
            outputs = net(images)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))

    results_dict = {"param_dict": param_dict,
                    "loss_list": loss_list,
                    "c_list": c_list,
                    "average_loss_list": average_loss_list,
                    "accuracy": accuracy}
    return results_dict


# list of dictionaries
def create_report(list_of_results_dicts):

    columns = ["Optimizer", "Epochs", "Learning rate", "Momentum", "LM_f", "LM_C", "C_running_min",\
                "Min Loss", "Last Loss", "Last Average Loss", "Test Accuracy"]

    rows = []

    for d in list_of_results_dicts:

        optimizer = d["param_dict"]["optimizer_type"]
        epochs = d["param_dict"]["no_epochs"]
        lr = d["param_dict"]["lr"]
        momentum = d["param_dict"]["momentum"]
        f = d["param_dict"]["LM_f"] # will need to be formatted to string
        c = d["param_dict"]["LM_c"]
        c_running_loss = str(d["param_dict"]["LM_c_run_min"])
        min_loss = min(d["loss_list"]).item()
        last_loss = d["loss_list"][-1].item()
        last_average_loss = d["average_loss_list"][-1]
        test_accuracy = d["accuracy"]
        rows.append([optimizer, epochs, lr, momentum, \
            f, c, c_running_loss, min_loss, last_loss, last_average_loss, test_accuracy])


    report_df = pd.DataFrame(columns = columns, data = rows)

    return report_df

def create_loss_sheet(list_of_results_dicts):

    loss_dict = {}
    for d in list_of_results_dicts:

        optimizer = d["param_dict"]["optimizer_type"]
        epochs = d["param_dict"]["no_epochs"]
        lr = d["param_dict"]["lr"]
        momentum = d["param_dict"]["momentum"]
        f = d["param_dict"]["LM_f"] # will need to be formatted to string
        c = d["param_dict"]["LM_c"]
        c_running_loss = str(d["param_dict"]["LM_c_run_min"])

        key = f"{optimizer}_{epochs}_{lr}_{momentum}_{f}_{c}_{c_running_loss}"
        value = d["loss_list"]

        # turn into list of floats
        map_obj = map(torch.Tensor.item, value)
        value = list(map_obj)

        loss_dict[key] = value

    loss_df = pd.DataFrame(loss_dict)

    return loss_df

def create_c_sheet(list_of_results_dicts):

    c_dict = {}
    for d in list_of_results_dicts:

        optimizer = d["param_dict"]["optimizer_type"]
        epochs = d["param_dict"]["no_epochs"]
        lr = d["param_dict"]["lr"]
        momentum = d["param_dict"]["momentum"]
        f = d["param_dict"]["LM_f"] # will need to be formatted to string
        c = d["param_dict"]["LM_c"]
        c_running_loss = str(d["param_dict"]["LM_c_run_min"])

        key = f"{optimizer}_{epochs}_{lr}_{momentum}_{f}_{c}_{c_running_loss}"
        value = d["c_list"]

        c_dict[key] = value

    c_df = pd.DataFrame(c_dict)

    return c_df


def create_average_loss_sheet(list_of_results_dicts):

    average_loss_dict = {}
    for d in list_of_results_dicts:

        optimizer = d["param_dict"]["optimizer_type"]
        epochs = d["param_dict"]["no_epochs"]
        lr = d["param_dict"]["lr"]
        momentum = d["param_dict"]["momentum"]
        f = d["param_dict"]["LM_f"] # will need to be formatted to string
        c = d["param_dict"]["LM_c"]
        c_running_loss = str(d["param_dict"]["LM_c_run_min"])

        key = f"{optimizer}_{epochs}_{lr}_{momentum}_{f}_{c}_{c_running_loss}"
        value = d["average_loss_list"]

        average_loss_dict[key] = value

    average_loss_df = pd.DataFrame(average_loss_dict)

    return average_loss_df


# TO DO:

# find a way to store the loss and c values for each experiment
# plots as well
# test these function in colab
# import them as a git repository
# deal with f as a string somehow

# Tuesday
# do the same for adam