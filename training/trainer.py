import time


def calculate_execution_time(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"{func.__name__} took {execution_time:.6f} seconds to execute.")
        return result
    return wrapper


@calculate_execution_time
def train(model, optimizer, criterion, dataloader, device, epochs):
    loss_train = []
    print("Training just started")
    for epoch in range(epochs):
        model.train()
        n_training_batches, epoch_loss = 0, 0
        for idx, (imgs, captions) in enumerate(dataloader):
            optimizer.zero_grad()
            imgs = imgs.to(device)
            captions = captions.to(device)
            output = model(imgs, captions)
            loss = criterion(
                output.reshape(-1, output.shape[2]), captions.reshape(-1)
            )
            epoch_loss += loss
            loss.backward()
            optimizer.step()

            n_training_batches += 1
        avg_epoch_loss = epoch_loss/n_training_batches  # better check this
        loss_train.append(avg_epoch_loss.item())
        if epoch % 1 == 0:
            print(f"epoch: {epoch + 1} -> Loss: {avg_epoch_loss:.8f}", end=" ---------------- ")
    return loss_train
