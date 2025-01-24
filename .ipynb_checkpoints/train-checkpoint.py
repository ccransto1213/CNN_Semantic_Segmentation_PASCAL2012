from basic_fcn import *
import time
from torch.utils.data import DataLoader
import torch
import gc
import voc
import torchvision.transforms.v2 as standard_transforms
import util
import numpy as np
import multiprocessing
import matplotlib.pyplot as plt

# Base Model Parameters
MODEL_PATH = "best_model.pt"
USE_WORKERS = True
EARLY_STOPPING_PATIENCE = 5 # Set to 30 to disable early stopping

# Step 4 Parameters
USE_SCHEDULER = False
USE_AUGMENTATION = False
USE_WEIGHTED_LOSS = False

# Step 5 Parameters
USE_ALTERNATIVE_ARCHITECTURE = False
USE_RESNET50 = False

#model name
model_name = 'baseline'

class MaskToTensor(object):
    def __call__(self, img):
        return torch.from_numpy(np.array(img, dtype=np.int32)).long()


def init_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        torch.nn.init.xavier_uniform_(m.weight.data)
        torch.nn.init.normal_(m.bias.data) #xavier not applicable for biases



#TODO Get class weights
def getClassWeights():
    # TODO for Q4.c || Caculate the weights for the classes
    num_classes=21
    class_inst=torch.zeros(num_classes)
    total_pixels=0
    for _, mask in train_loader: #loop through every mask in train set

        for id in range(num_classes):
            class_inst+=torch.sum(mask==id) #count number of class instances in each mask
    
        total_pixels+=mask.numel() #sum total number of pixels in each mask
    class_freq=class_inst/total_pixels #get class frequencies 
    eps=1e-8
    class_weights=1.0/(class_freq+eps) #Invert frequencies for class weights, lower frequency higher weight
    class_weights=class_weights/torch.sum(class_weights) #normalize so they sum to 1
    return class_weights

num_workers=0
if USE_WORKERS:
    num_workers=2

if USE_SCHEDULER:
    model_name = 'learning rate scheduler model'

augment_transform = None
if USE_AUGMENTATION:
    model_name = 'data augmentation model'
    augment_transform = standard_transforms.Compose([
        standard_transforms.RandomResizedCrop(size=(224, 224), scale=(0.8, 1.0), ratio=(0.75, 1.33)),
        standard_transforms.RandomHorizontalFlip(),
        standard_transforms.RandomRotation(degrees=10),
    ])

if USE_WEIGHTED_LOSS:
    model_name = 'weighted loss model'

# normalize using imagenet averages
mean_std = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) 
# toImage() and ToDtype() together do the same as ToTensor in transform.v2
input_transform = standard_transforms.Compose([
    standard_transforms.ToImage(),
    standard_transforms.ToDtype(torch.float32, scale=True),
    standard_transforms.Normalize(*mean_std)
])

# input_transform = standard_transforms.Compose([
#     standard_transforms.ToTensor(),
#     standard_transforms.Normalize(*mean_std)
# ])

target_transform = MaskToTensor()

train_dataset = voc.VOC('train', transform=input_transform, target_transform=target_transform, both_transform=augment_transform)
val_set = voc.VOC('val', transform=input_transform, target_transform=target_transform)
# test_dataset = voc.VOC('test', transform=input_transform, target_transform=target_transform)
split_size=int(0.5*len(val_set))
val_dataset, test_dataset =torch.utils.data.random_split(val_set, [split_size, len(val_set)-split_size]) #splitting validation set and train set in half, per piazza instructions

train_loader = DataLoader(dataset=train_dataset, batch_size= 16, shuffle=True, num_workers=num_workers)
val_loader = DataLoader(dataset=val_dataset, batch_size= 16, shuffle=False, num_workers=num_workers)
test_loader = DataLoader(dataset=test_dataset, batch_size= 16, shuffle=False, num_workers=num_workers)


epochs = 30

n_class = 21

if USE_ALTERNATIVE_ARCHITECTURE:
    model_name = 'alternative model'
    fcn_model = Alternative(n_class=n_class) 
    fcn_model.apply(init_weights)
elif USE_RESNET50:
    model_name = 'resnet50 model'
    fcn_model = Resnet(n_class=n_class)
else:
    fcn_model = FCN(n_class=n_class)
    fcn_model.apply(init_weights)


device = "cuda" if torch.cuda.is_available() else "cpu" # TODO determine which device to use (cuda or cpu)

optimizer = torch.optim.Adam(fcn_model.parameters(), lr=0.001) # TODO choose an optimizer
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=0.0001)
class_weights = getClassWeights().to(device) if USE_WEIGHTED_LOSS else None
criterion = nn.CrossEntropyLoss(weight=class_weights)

fcn_model = fcn_model.to(device) # TODO transfer the model to the device
training_loss = []
validation_loss = []

# TODO
def train():
    """
    Train a deep learning model using mini-batches.

    - Perform forward propagation in each epoch.
    - Compute loss and conduct backpropagation.
    - Update model weights.
    - Evaluate model on validation set for mIoU score.
    - Save model state if mIoU score improves.
    - Implement early stopping if necessary.

    Returns:
        None.
    """

    best_iou_score = 0.0

    wait_epochs = 0 # how many epochs without encountering a better model
    for epoch in range(epochs):
        ts = time.time()
        for iter, (inputs, labels) in enumerate(train_loader):
            # TODO  reset optimizer gradients
            optimizer.zero_grad()


            # both inputs and labels have to reside in the same device as the model's
            inputs =  inputs.to(device) # TODO transfer the input to the same device as the model's
            labels =  labels.to(device) # TODO transfer the labels to the same device as the model's

            outputs = fcn_model(inputs) # TODO  Compute outputs. we will not need to transfer the output, it will be automatically in the same device as the model's!

            loss = criterion(outputs, labels)  #TODO  calculate loss

            # TODO  backpropagate
            loss.backward()

            # TODO  update the weights
            optimizer.step()


            if iter % 20 == 0:
                print("epoch{}, iter{}, loss: {}".format(epoch, iter, loss.item()))
        print("Finish epoch {}, time elapsed {}".format(epoch, time.time() - ts))
        
        current_miou_score, current_loss_score = val(epoch)
        wait_epochs += 1

        training_loss.append(loss.item())
        validation_loss.append(current_loss_score)

        if current_miou_score > best_iou_score:
            best_iou_score = current_miou_score
            # save the best model
            torch.save(fcn_model.state_dict(), MODEL_PATH)
            wait_epochs = 0
        elif wait_epochs >= EARLY_STOPPING_PATIENCE:
            print("Early stopping at epoch: ", epoch)
            break

        if USE_SCHEDULER:
            scheduler.step()
    
 #TODO
def val(epoch):
    """
    Validate the deep learning model on a validation dataset.

    - Set model to evaluation mode.
    - Disable gradient calculations.
    - Iterate over validation data loader:
        - Perform forward pass to get outputs.
        - Compute loss and accumulate it.
        - Calculate and accumulate mean Intersection over Union (IoU) scores and pixel accuracy.
    - Print average loss, IoU, and pixel accuracy for the epoch.
    - Switch model back to training mode.

    Args:
        epoch (int): The current epoch number.

    Returns:
        tuple: Mean IoU score and mean loss for this validation epoch.
    """
    fcn_model.eval() # Put in eval mode (disables batchnorm/dropout) !
    
    losses = []
    mean_iou_scores = []
    accuracy = []

    with torch.no_grad(): # we don't need to calculate the gradient in the validation/testing

        for iter, (inputs, labels) in enumerate(val_loader):
            # First part is same as train()
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = fcn_model(inputs)

            loss = criterion(outputs, labels)
            losses.append(loss.item())

            # Now we need to compute iou and accuracy
            pred = torch.argmax(outputs, dim=1) # Predicted class and dim=1 sums over the classes in NCHW

            iou = util.iou(pred, labels)
            mean_iou_scores.append(iou)

            acc = util.pixel_acc(pred, labels)
            accuracy.append(acc)

    print(f"Loss at epoch: {epoch} is {torch.mean(torch.tensor(losses))}")
    print(f"IoU at epoch: {epoch} is {torch.mean(torch.tensor(mean_iou_scores))}")
    print(f"Pixel acc at epoch: {epoch} is {torch.mean(torch.tensor(accuracy))}")

    fcn_model.train() #TURNING THE TRAIN MODE BACK ON TO ENABLE BATCHNORM/DROPOUT!!
    
    return torch.mean(torch.tensor(mean_iou_scores)), torch.mean(torch.tensor(losses))

 #TODO
def modelTest():
    """
    Test the deep learning model using a test dataset.

    - Load the model with the best weights.
    - Set the model to evaluation mode.
    - Iterate over the test data loader:
        - Perform forward pass and compute loss.
        - Accumulate loss, IoU scores, and pixel accuracy.
    - Print average loss, IoU, and pixel accuracy for the test data.
    - Switch model back to training mode.

    Returns:
        None. Outputs average test metrics to the console.
    """

    fcn_model.eval()  # Put in eval mode (disables batchnorm/dropout) !

    losses = []
    mean_iou_scores = []
    accuracy = []

    with torch.no_grad():  # we don't need to calculate the gradient in the validation/testing

        for iter, (inputs, labels) in enumerate(test_loader):

            # Same as val()
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = fcn_model(inputs)

            loss = criterion(outputs, labels)
            losses.append(loss.item())

            pred = torch.argmax(outputs, dim=1) 

            iou = util.iou(pred, labels)
            mean_iou_scores.append(iou)

            acc = util.pixel_acc(pred, labels)
            accuracy.append(acc)

    print(f"Loss is {torch.mean(torch.tensor(losses))}")
    print(f"IoU is {torch.mean(torch.tensor(mean_iou_scores))}")
    print(f"Pixel is {torch.mean(torch.tensor(accuracy))}")

    fcn_model.train()  #TURNING THE TRAIN MODE BACK ON TO ENABLE BATCHNORM/DROPOUT!!


def plotModel():
    plt.plot(training_loss, label='Training Loss')
    plt.plot(validation_loss, label='Validation Loss')
    plt.ylabel('Value')
    plt.xlabel('Epoch')
    plt.legend()  
    plt.title(f"loss graph of {model_name}")
    plt.savefig('./images/%s.png' % model_name)
    
def exportModel(inputs):    
    """
    Export the output of the model for given inputs.

    - Set the model to evaluation mode.
    - Load the model with the best saved weights.
    - Perform a forward pass with the model to get output.
    - Switch model back to training mode.

    Args:
        inputs: Input data to the model.

    Returns:
        Output from the model for the given inputs.
    """

    fcn_model.eval() # Put in eval mode (disables batchnorm/dropout) !
    
    
    # TODO Then Load your best model using saved_model_path
    fcn_model.load_state_dict(torch.load(MODEL_PATH))

    
    inputs = inputs.to(device)
    
    output_image = fcn_model(inputs)
    
    fcn_model.train()  #TURNING THE TRAIN MODE BACK ON TO ENABLE BATCHNORM/DROPOUT!!
    
    return output_image

if __name__ == "__main__":
    print(f"Device is {device}")
    val(0)  # show the accuracy before training
    train()
    modelTest()
    plotModel()

    # housekeeping
    gc.collect()
    torch.cuda.empty_cache()
