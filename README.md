# AINaim: "Reviving Heritage, One Page at a Time"

## Executive Summary

AINaim, standing for **Artificial Intelligence for Navigating Ancient Imagery**, represents the pinnacle of blending traditional cultural heritage with cutting-edge AI technology. Aimed at transforming scanned Hebrew documents into pristine, digital formats, AINaim leverages sophisticated machine learning models to enhance, correct, and digitize ancient texts. Our mission is to ensure that precious historical documents are not only preserved but made more accessible and editable for future generations, encapsulated by our slogan, "Reviving Heritage, One Page at a Time."

## Project Components

AINaim integrates several advanced components into a seamless pipeline:

1. **Data Preparation**: Utilizing a rich dataset of scanned Hebrew pages, augmented to simulate real-world degradation.
2. **Document Enhancement (DE-GAN)**: A custom Generative Adversarial Network fine-tuned for restoring the intricacies of Hebrew script.
3. **Auto-Rotation**: Intelligent correction of document orientation to ensure proper alignment for reading and analysis.
4. **Optical Character Recognition (OCR)**: State-of-the-art OCR technology tailored for Hebrew text extraction.
5. **Editable PDF Generation**: Conversion of OCR text into formatted, editable PDF documents, preserving the original layout and style.

### 1. Data Preparation

A robust dataset forms the foundation of AINaim's training process, crafted through meticulous augmentation to mirror various scanning conditions.

#### Data Augmentation Code

```python
import augraphy
import fitz  # PyMuPDF for PDF processing
from PIL import Image
import numpy as np
import os

def pdf_to_images(pdf_path, output_folder):
    """Converts each page of a PDF to an individual image."""
    doc = fitz.open(pdf_path)
    for page_number in range(len(doc)):
        page = doc.load_page(page_number)
        pix = page.get_pixmap()
        output_path = os.path.join(output_folder, f"page_{page_number}.png")
        pix.save(output_path)

def augment_images(input_folder, output_folder):
    """Applies augmentation to simulate scanning issues on images."""
    aug_pipeline = augraphy.create_paper_augraphy_pipeline()
    
    for file_name in os.listdir(input_folder):
        if file_name.endswith('.png'):
            input_path = os.path.join(input_folder, file_name)
            image = Image.open(input_path)
            image_np = np.array(image)
            
            augmented_image_np = aug_pipeline.augment(image_np)
            augmented_image = Image.fromarray(augmented_image_np)
            
            output_path = os.path.join(output_folder, file_name)
            augmented_image.save(output_path)
```

### 2. DE-GAN for Document Enhancement

DE-GAN intricately enhances scanned documents, meticulously restoring even the most minute details of Hebrew script, lost through degradation. More information about DE-GAN model [here](https://arxiv.org/pdf/2010.08764.pdf)

#### Example DE_GAN code
```python
import tensorflow as tf
from tensorflow.keras import layers, Model

def make_generator_model():
    model = tf.keras.Sequential([
        # Start with a dense layer that takes a noise vector as input
        layers.Dense(7*7*256, use_bias=False, input_shape=(100,)),
        layers.BatchNormalization(),
        layers.LeakyReLU(),

        layers.Reshape((7, 7, 256)),
        # Upsample to the desired image size
        layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False),
        layers.BatchNormalization(),
        layers.LeakyReLU(),

        layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False),
        layers.BatchNormalization(),
        layers.LeakyReLU(),

        layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh')
    ])
    return model

def make_discriminator_model():
    model = tf.keras.Sequential([
        # Input shape is the size of the document image
        layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same', input_shape=[28, 28, 1]),
        layers.LeakyReLU(),
        layers.Dropout(0.3),

        layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'),
        layers.LeakyReLU(),
        layers.Dropout(0.3),

        layers.Flatten(),
        layers.Dense(1)
    ])
    return model

```

### 3. Auto-Rotation

Our auto-rotation model ensures every document is perfectly aligned, paving the way for flawless OCR processing. Tesseract OCR has the capability to detect the orientation and script of the text in an image and can be used to auto-rotate an image for better OCR results.

#### Here is a basic example of how to use Tesseract for this purpose:

```python
import pytesseract
from PIL import Image

def correct_orientation(image_path):
    # Load the image
    image = Image.open(image_path)
    
    # Use Tesseract to detect orientation
    osd = pytesseract.image_to_osd(image)
    rotation_angle = int(re.search('Rotate: (\d+)', osd).group(1))
    
    # Correct the orientation
    if rotation_angle != 0:
        rotated_image = image.rotate(-rotation_angle, expand=True)
        return rotated_image
    else:
        return image

```

### 4. OCR Application

AINaim will incorporate advanced OCR techniques, fine-tuned for the Hebrew language, ensuring precise text extraction.

### 5. PDF Generation

The culmination of AINaim's pipeline produces editable PDFs, crafted with an unwavering commitment to maintaining the original document's integrity and layout.

#### Here is a first example code:

```python
import fitz
from PIL import Image
import pytesseract
import io

# Step 1: OCR with Tesseract to extract text from an image
def ocr_image_to_text(image_path):
    # Open the image
    img = Image.open(image_path)
    # Use Tesseract to do OCR on the image
    text = pytesseract.image_to_string(img)
    return text

# Step 2: Create a PDF with the recognized text
def create_pdf_with_text(text, output_pdf_path):
    # Create a PDF document
    doc = fitz.open()
    # Add a page
    page = doc.new_page()
    # Add text to the page
    page.insert_text((72, 72), text)  # Starting at x=72, y=72
    # Save the document
    doc.save(output_pdf_path)
    doc.close()

# Example usage
image_path = 'your_image_here.png'  # Replace with your image file path
output_pdf_path = 'output.pdf'

# Convert image to text
recognized_text = ocr_image_to_text(image_path)

# Create PDF with recognized text
create_pdf_with_text(recognized_text, output_pdf_path)

```

## The Future of AINaim

While AINaim is still under development and has not been deployed, its vision is clear: to ensure that ancient texts are not merely preserved as artifacts but are brought to life for future generations to explore and learn from. "Reviving Heritage, One Page at a Time" is not just our mission—it's our promise to the world, ensuring that the wisdom of the past continues to enlighten the future.

As development progresses, AINaim will continue to refine its methodologies, enhance its algorithms, and expand its capabilities, all while maintaining the highest standards of accuracy, respect, and dedication to cultural heritage preservation.


