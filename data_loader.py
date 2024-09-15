import torchvision.transforms as transforms

def preprocess_data(data):
    # Enhanced preprocessing with data augmentation
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    augmented_data = [transform(image) for image in data]
    return augmented_data