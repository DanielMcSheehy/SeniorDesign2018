import torch

def train(net, batch, batch_size, n_epochs, truth_vector, learning_rate):
    criterion = torch.nn.CrossEntropyLoss()
    #Todo: Switch from SGD
    optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate)
    
    for epoch in range(n_epochs): 
        # Forward pass: Compute predicted y by passing x to the model
        y_pred = net(batch)
        # Compute and print loss
        loss = criterion(y_pred, truth_vector)
        print("Epoch: ", epoch, " Loss: ", loss.item())

        # Zero gradients, perform a backward pass, and update the weights.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

def test(net, testing_batch_array, truth_vector):
    batch_size = 1
    criterion = torch.nn.CrossEntropyLoss()
    n_correct, n_total = 0, 0
    for i, batch in enumerate(testing_batch_array):
        # Forward pass: Compute predicted y by passing x to the model
        answer = net(batch)
        n_correct += (torch.max(answer, 1)[1].view(batch_size) == truth_vector[i]).sum().item()
        n_total += batch_size
        train_acc = 100. * n_correct/n_total
        print("Traning accuracy: ", train_acc)
