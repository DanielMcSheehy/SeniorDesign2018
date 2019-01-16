import torch

def train(net, batch, batch_size, n_epochs, truth_vector, learning_rate):
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate)
    for epoch in range(n_epochs):
        # Forward pass: Compute predicted y by passing x to the model
        y_pred = net(batch)
        # Compute and print loss
        loss = criterion(y_pred, truth_vector)
        print(epoch, loss.item())

        # Zero gradients, perform a backward pass, and update the weights.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()