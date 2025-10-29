import matplotlib.pyplot as plt


def visualize_dataset_examples(dataloader, num_examples=10, figsize=(15,5), cmap='gray'):
    """
    Visualize examples from a dataloader in a grid.
    
    Args:
        dataloader: PyTorch dataloader containing images and labels
        num_examples (int): Number of examples to display
        figsize (tuple): Figure size (width, height)
        cmap (str): Color map for displaying images
    """
    # Get a batch of images and labels
    dataiter = iter(dataloader)
    images, labels = next(dataiter)

    # Create a grid of images
    plt.figure(figsize=figsize)
    for i in range(num_examples):
        plt.subplot(2, 5, i+1)
        plt.imshow(images[i][0], cmap=cmap)  # [0] to get the first channel since MNIST is grayscale
        plt.title(f'Label: {labels[i]}')
        plt.axis('off')
    plt.tight_layout()
    plt.show()
