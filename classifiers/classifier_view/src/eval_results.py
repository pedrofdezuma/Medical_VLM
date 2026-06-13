import torch
from utils import plot_learning_curves

# Paths
MODEL_NAME = "mobilenet_v2" # Change this to match the trained model (resnet18, vgg16, mobilenet_v2)
NUM_TRAIN="4k"
PRETRAINED=True
NON_ROTATED=True
NO_AUGMENTATION=True

HISTORY_PATH = f'D:/classifiers/classifier_view/checkpoints/{MODEL_NAME}-{NUM_TRAIN}'

if PRETRAINED:
    HISTORY_PATH+='-PT'
if NON_ROTATED:
    HISTORY_PATH+='-NR'
if NO_AUGMENTATION:
    HISTORY_PATH+="-NA"
    
HISTORY_PATH+='/history.pth' 

def main():
    # Load Loss and Accuracy history
    if torch.cuda.is_available():
        history = torch.load(HISTORY_PATH, weights_only=True)
   

    image_save_path=plot_learning_curves(history, MODEL_NAME,NUM_TRAIN,pretrained=PRETRAINED, non_rotated=NON_ROTATED, no_augmentation=NO_AUGMENTATION)
    
    print(f"Figure saved in {image_save_path}")
    
        

if __name__ == '__main__':
    main()
