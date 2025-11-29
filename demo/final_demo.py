import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Sampler
from torchvision import transforms, models
from PIL import Image
import cv2
from sklearn.metrics import average_precision_score, roc_auc_score
from scipy.ndimage import gaussian_filter
import streamlit as st
import matplotlib.pyplot as plt
from io import BytesIO

# Set default device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# We will use the necessory components like CNN class and the loss class
# from our cnn.ipynb, as it is in this interface
class DeviationLoss(nn.Module):
    # loss function inherited from nn module
    def __init__(self):
        super().__init__()

    def forward(self, y_pred, y_true):
        confidence_margin = 5.0
        ref = torch.normal(mean=0.0, std=torch.ones(5000)).to(device)
        dev = (y_pred - torch.mean(ref)) / torch.std(ref)
        inlier_loss = torch.abs(dev)
        outlier_loss = torch.abs((confidence_margin - dev).clamp_(min=0.))
        return torch.mean((1 - y_true) * inlier_loss + y_true * outlier_loss)

# Feature extraction networks
class feature_resnet18(nn.Module):
    def __init__(self):
        super(feature_resnet18, self).__init__()
        self.net = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        
    def forward(self, x):
        x = self.net.conv1(x)
        x = self.net.bn1(x)
        x = self.net.relu(x)
        x = self.net.maxpool(x)
        x = self.net.layer1(x)
        x = self.net.layer2(x)
        x = self.net.layer3(x)
        x = self.net.layer4(x)
        # we are returning not the classified output, but the features extracted 
        # by first few layers of resnet18
        return x

# Define output dimensions for different backbones
# We have used only one backbone resnet18 here
# Other like resnet50, alexnet can also be used for feature extraction
NET_OUT_DIM = {'resnet18': 512}

def build_feature_extractor(backbone):
    if backbone == "resnet18":
        print("Feature extractor: ResNet-18")
        return feature_resnet18()
    else:
        raise NotImplementedError(f"Backbone {backbone} not implemented")

# Main network model
class SemiADNet(nn.Module):
    def __init__(self, args):
        super(SemiADNet, self).__init__()
        self.args = args
        self.feature_extractor = build_feature_extractor(self.args.backbone)
        self.conv = nn.Conv2d(in_channels=NET_OUT_DIM[self.args.backbone], out_channels=1, kernel_size=1, padding=0)

    def forward(self, image):
        if self.args.n_scales == 0:
            raise ValueError("n_scales must be greater than 0")

        image_pyramid = list()
        # Scoring the image at different scales, to capture the features at different levels
        for s in range(self.args.n_scales):
            image_scaled = F.interpolate(image, size=self.args.img_size // (2 ** s)) if s > 0 else image
            feature = self.feature_extractor(image_scaled)

            scores = self.conv(feature)
            if self.args.topk > 0:
                scores = scores.view(int(scores.size(0)), -1)
                topk = max(int(scores.size(1) * self.args.topk), 1)
                scores = torch.topk(torch.abs(scores), topk, dim=1)[0]
                scores = torch.mean(scores, dim=1).view(-1, 1)
            else:
                scores = scores.view(int(scores.size(0)), -1)
                scores = torch.mean(scores, dim=1).view(-1, 1)

            image_pyramid.append(scores)
        scores = torch.cat(image_pyramid, dim=1)
        score = torch.mean(scores, dim=1)
        return score.view(-1, 1)

# Utility functions
def aucPerformance(mse, labels, prt=True):
    roc_auc = roc_auc_score(labels, mse)
    ap = average_precision_score(labels, mse)
    if prt:
        print(f"AUC-ROC: {roc_auc:.4f}, AUC-PR: {ap:.4f}")
    return roc_auc, ap

def convert_to_grayscale(im_as_arr):
    grayscale_im = np.sum(np.abs(im_as_arr), axis=0)
    im_max = np.percentile(grayscale_im, 99)
    im_min = np.min(grayscale_im)
    if im_max > im_min:
        grayscale_im = np.clip((grayscale_im - im_min) / (im_max - im_min), 0, 1)
    grayscale_im = np.expand_dims(grayscale_im, axis=0)
    return grayscale_im

def show_cam_on_image(img, mask, label=None, title=None, save_path=None):
    """
    Display the CAM visualization
    """
    fig = plt.figure(figsize=(15, 5))
    
    # Original image
    plt.subplot(1, 3 if label is not None else 2, 1)
    plt.imshow(img)
    plt.title("Original Image")
    plt.axis('off')
    
    # Heatmap
    heatmap = cv2.applyColorMap(np.uint8(255*mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    
    plt.subplot(1, 3 if label is not None else 2, 2)
    plt.imshow(cam)
    plt.title("Anomaly Heatmap")
    plt.axis('off')
    
    # Ground truth (if available)
    if label is not None:
        plt.subplot(1, 3, 3)
        plt.imshow(label)
        plt.title("Ground Truth")
        plt.axis('off')
    
    if title:
        plt.suptitle(title)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    
    return fig

# Functions which we use to plot the heatmap fo gradient appearing
# in the model training for that image in the form of heatmap
# This helps us to visualize the anomaly part in the image
def get_anomaly_heatmap(model, img_tensor):
    model.eval()
    model.zero_grad()
    
    # Get anomaly score as output, then 
    # we do backprop to do gradient calculation
    img_tensor.requires_grad = True
    output = model(img_tensor)
    output.backward()
    
    # Process gradient for heatmap 
    grad = img_tensor.grad
    grad_temp = convert_to_grayscale(grad.cpu().numpy().squeeze(0))
    grad_temp = grad_temp.squeeze(0)
    grad_temp = gaussian_filter(grad_temp, sigma=4)
    
    # Return both the heatmap and the anomaly score
    return grad_temp, output.item()

def load_pretrained_model(model_path, args):
    """Load a pre-trained DevNet model"""
    model = SemiADNet(args).to(device)
    
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Loaded model from {model_path}")
        return model, True
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return model, False

class Args:
    """Simple class to hold model arguments"""
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

# Streamlit app
def main():
    st.set_page_config(
        page_title="DevNet Anomaly Detection",
        page_icon="🔍",
        layout="wide"
    )
    
    st.title("Deviation Network (DevNet) for Anomaly Detection")
    
    st.markdown("""
    ### Deep Few-shot Anomaly Detection with Deviation Networks
    
    This demo showcases anomaly detection and localization (finding the part of image causing anomaly) using the Deviation Network (DevNet) on image datasets.
    approach from the paper [Deep Anomaly Detection with Deviation Networks](https://arxiv.org/abs/1911.08623).
    
    Upload an image to detect and localize potential anomalies.
    """)
    
    # Sidebar configuration
    st.sidebar.title("Model Configuration")
    
    # Model parameters
    backbone = st.sidebar.selectbox(
        "Feature Extractor Backbone",
        ["resnet18"],
        index=0
    )
    
    category = st.sidebar.selectbox(
        "Product Category",
        ["carpet", "grid", "leather", "tile", "wood", 
         "bottle", "cable", "capsule", "hazelnut", "metal_nut",
         "pill", "screw", "toothbrush", "transistor", "zipper"],
        index=0
    )
    
    # Basically this decides the resolution of the image
    img_size = st.sidebar.select_slider(
        "Image Size",
        options=[128, 224, 320, 448],
        value=224
    )
    
    # This is the number of scales we want to use for the image
    # This is used to capture the features at different levels of the image
    n_scales = st.sidebar.slider(
        "Number of Scales",
        min_value=1,
        max_value=3,
        value=2
    )
    
    # This threshold is used to classify the image
    # as anomaly or not based on the output score
    anomaly_threshold = st.sidebar.slider(
        "Anomaly Score Threshold",
        min_value=0.0,
        max_value=10.0, 
        value=1.0,
        step=0.1
    )
    
    # Create model arguments
    args = Args(
        backbone=backbone,
        img_size=img_size,
        n_scales=n_scales,
        topk=0.1,
        classname=category
    )
    
    # Check for model directory and available models
    model_dir = "./models"
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
        
    # Check for pre-trained model
    model_path = os.path.join(model_dir, f"{category}_model.pkl")
    model_available = os.path.exists(model_path)
    
    # Upload model section
    st.sidebar.markdown("---")
    st.sidebar.subheader("Model Management")
    
    uploaded_model = st.sidebar.file_uploader("Upload trained model (.pkl)", type=["pkl"])
    if uploaded_model is not None:
        # Save the uploaded model
        with open(model_path, "wb") as f:
            f.write(uploaded_model.getbuffer())
        st.sidebar.success(f"Model saved to {model_path}")
        model_available = True
    
    # We have split the main content area into two columns
    # One for showing the uploaded image and the other for showing the results
    # That is, the anomaly score and the heatmap
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Upload Image")
        uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png", "bmp"])
        
        if uploaded_file is not None:
            # Display the uploaded image
            image = Image.open(uploaded_file).convert('RGB')
            st.image(image, caption="Uploaded Image", use_container_width=True)
            
            # Create a button to run detection
            run_detection = st.button("Detect Anomalies")
            
            if run_detection:
                if not model_available:
                    st.error("No model available for selected category. Please upload a trained model first.")
                else:
                    with st.spinner("Processing image..."):
                        try:
                            # Load model
                            model, success = load_pretrained_model(model_path, args)
                            
                            if not success:
                                st.error("Failed to load model. Please check the model file and try again.")
                            else:
                                # Preprocess image
                                transform = transforms.Compose([
                                    transforms.Resize((img_size, img_size)),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                ])
                                
                                input_tensor = transform(image).unsqueeze(0).to(device)
                                
                                # Get anomaly heatmap
                                heatmap, score = get_anomaly_heatmap(model, input_tensor)
                                
                                # Create visualization using the exact notebook method
                                img_np = np.array(image.resize((img_size, img_size))) / 255.0
                                
                                # Status based on score compared to threshold
                                status = "Anomalous" if score > anomaly_threshold else "Normal"
                                
                                # Show results in the second column
                                with col2:
                                    st.subheader("Detection Results")
                                    
                                    # Display score and status
                                    st.metric("Anomaly Score", f"{score:.4f}", delta=status)
                                    
                                    # Use the notebook's visualization function
                                    fig = show_cam_on_image(img_np, heatmap, title=f"Anomaly Score: {score:.4f} - {status}")
                                    st.pyplot(fig)
                                    
                                    # Add download button for the visualization
                                    buf = BytesIO()
                                    fig.savefig(buf, format="png")
                                    buf.seek(0)
                                    st.download_button(
                                        label="Download Visualization",
                                        data=buf,
                                        file_name=f"anomaly_detection_{category}.png",
                                        mime="image/png"
                                    )
                        
                        except Exception as e:
                            st.error(f"Error processing image: {str(e)}")
                            st.exception(e)
    
    # If no image is uploaded yet
    if uploaded_file is None:
        with col2:
            st.subheader("Detection Results")
            st.info("Upload an image and click 'Detect Anomalies' to see results.")
    
    # We have added a section to show the information about the model and the demo
    st.markdown("---")
    st.subheader("About DevNet")
    st.markdown("""
    **Deviation Network (DevNet)** is an approach for few-shot anomaly detection that can work with minimal 
    anomaly examples. It uses a reference distribution and trains the network to push anomaly scores 
    away from this distribution.
    
    **Key features:**
    * Few-shot learning: Works with just a few labeled anomaly examples
    * Explainable visualization: Generates heatmaps showing anomalous regions
    * Multi-scale analysis: Examines features at different image scales
    
    For more information, see the [original paper](https://arxiv.org/abs/1911.08623).
    """)
    
    # We have also added the instructions for using the demo
    st.sidebar.markdown("---")
    st.sidebar.subheader("Notes")
    st.sidebar.markdown("""
    **Using this demo:**
    
    1. Select model configuration
    2. Upload a trained model if available
    3. Upload an image to analyze
    4. Click "Detect Anomalies"
    
    **Pre-trained models:**  
    This demo requires pre-trained models from the DevNet notebook.
    """)

if __name__ == "__main__":
    main()