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

#Todo: Find accuracy equation:
def test(net, testing_batch, truth_vector):
    criterion = torch.nn.CrossEntropyLoss()
    # Forward pass: Compute predicted y by passing x to the model
    y_pred = net(testing_batch)
    # Compute and print loss
    loss = criterion(y_pred, truth_vector)
    print("Loss of model: ", loss.item())
