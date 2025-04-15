from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
import numpy as np
import os
import tempfile
import pydicom
import nibabel as nib
import cv2
import time
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Configure CORS for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins - restrict this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Model is loaded globally at startup
model = None

@app.on_event("startup")
async def load_model():
    global model
    try:
        model_path = "/home/antriksh/Desktop/DAV PROJECT/backend/saved_model.h5"
        if os.path.exists(model_path):
            model = tf.keras.models.load_model(model_path)
            logger.info("Model loaded successfully")
        else:
            logger.error(f"Model file not found at {model_path}")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")

@app.get("/")
async def root():
    return {"message": "Medical Imaging Analysis API is running"}

def load_and_process_nifti(file_path):
    """Load and process a NIfTI file"""
    try:
        img = nib.load(file_path)
        volume = img.get_fdata()
        logger.info(f"Loaded NIfTI file with shape: {volume.shape}")
        return volume
    except Exception as e:
        logger.error(f"Error loading NIfTI file: {e}")
        raise HTTPException(status_code=400, detail=f"Invalid NIfTI file: {str(e)}")

def load_and_process_dicom(file_path):
    """Load and process a DICOM file"""
    try:
        dicom_data = pydicom.dcmread(file_path)
        pixel_array = dicom_data.pixel_array
        logger.info(f"Loaded DICOM file with shape: {pixel_array.shape}")
        return pixel_array
    except Exception as e:
        logger.error(f"Error loading DICOM file: {e}")
        raise HTTPException(status_code=400, detail=f"Invalid DICOM file: {str(e)}")

def format_prediction(prediction):
    """
    Format the prediction result for the 3-class classification model (PD, SWEDD, Control)
    """
    # Map indexes to class names
    class_names = {0: "PD", 1: "SWEDD", 2: "Control"}
    
    # Get the prediction result
    prediction_list = prediction.tolist()
    
    # Get the class with highest probability
    class_index = np.argmax(prediction, axis=1)[0]
    class_name = class_names[class_index]
    confidence = float(prediction[0][class_index])
    
    # Return formatted result
    return {
        "class_name": class_name,
        "class_index": int(class_index),
        "confidence": confidence,
        "probabilities": {
            "PD": float(prediction[0][0]),
            "SWEDD": float(prediction[0][1]),
            "Control": float(prediction[0][2])
        }
    }

def preprocess_volume(volume, target_size=(64, 64, 64)):
    """
    Normalizes and resizes the 3D volume based on the training code.
    """
    try:
        logger.info(f"Preprocessing volume with initial shape: {volume.shape}")
        
        # Normalize (Scale values between 0 and 1)
        volume = volume.astype(np.float32) / np.max(volume) if np.max(volume) > 0 else volume.astype(np.float32)
        
        # Handle different dimension possibilities
        if volume.ndim == 4 and volume.shape[-1] == 1:
            volume = np.squeeze(volume, axis=-1)  # Remove extra channel dimension
        elif volume.ndim == 2:
            # Single slice - expand to 3D
            volume = np.expand_dims(volume, axis=2)
            
        num_slices = volume.shape[2] if volume.ndim >= 3 else 1
        
        # Handle 2D vs 3D volumes
        if volume.ndim < 3:
            logger.warning(f"Volume has unexpected shape {volume.shape}, expanding dimensions")
            return np.zeros(target_size + (1,))  # Return zero volume of correct shape
            
        # Resize each 2D slice
        resized_slices = []
        for i in range(min(num_slices, target_size[2])):
            slice_2d = volume[:, :, i] if i < num_slices else np.zeros(volume.shape[:2])
            resized_slice = cv2.resize(slice_2d, target_size[:2])
            resized_slices.append(resized_slice)
            
        # If we don't have enough slices, pad with zeros
        while len(resized_slices) < target_size[2]:
            resized_slices.append(np.zeros(target_size[:2]))
            
        # Stack resized slices
        resized_volume = np.stack(resized_slices, axis=2)  # Using axis=2 for z-dimension
        
        # Add channel dimension
        final_volume = np.expand_dims(resized_volume, axis=3)
        
        logger.info(f"Preprocessed volume shape: {final_volume.shape}")
        return final_volume
        
    except Exception as e:
        logger.error(f"Error in preprocessing: {str(e)}")
        logger.error(f"Volume shape: {volume.shape}, dtype: {volume.dtype}")
        # Return a zero volume of the correct shape
        return np.zeros(target_size + (1,))

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    if not model:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        # Read the uploaded file
        contents = await file.read()
        
        # Determine file type from file extension
        file_ext = Path(file.filename).suffix.lower()
        
        # Save to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as temp_file:
            temp_file.write(contents)
            temp_path = temp_file.name
        
        try:
            # Process different file types
            if file_ext in ['.nii', '.gz']:
                pixel_array = load_and_process_nifti(temp_path)
            elif file_ext == '.dcm':
                pixel_array = load_and_process_dicom(temp_path)
            else:
                raise HTTPException(
                    status_code=400,
                    detail=f"Unsupported file format: {file_ext}. Please upload a .nii, .nii.gz, or .dcm file."
                )
                
        except Exception as e:
            logger.error(f"Error processing file: {str(e)}")
            raise HTTPException(status_code=400, detail=f"Error processing file: {str(e)}")
        finally:
            # Clean up the temporary file
            if os.path.exists(temp_path):
                os.unlink(temp_path)
        
        # Preprocess the volume
        processed_volume = preprocess_volume(pixel_array)
        logger.info(f"After preprocessing shape: {processed_volume.shape}")
        
        # Ensure proper shape (batch, x, y, z, channels) for the model
        # Should be (1, 64, 64, 64, 1)
        if processed_volume.shape != (64, 64, 64, 1):
            logger.warning(f"Unexpected shape after preprocessing: {processed_volume.shape}, reshaping...")
            # Try to reshape correctly based on what we have
            processed_volume = np.reshape(processed_volume, (64, 64, 64, 1))
        
        # Add batch dimension - VERY IMPORTANT
        model_input = np.expand_dims(processed_volume, axis=0)
        
        logger.info(f"Final model input shape: {model_input.shape}")
        
        # Make prediction
        start_time = time.time()
        prediction = model.predict(model_input)
        processing_time = (time.time() - start_time) * 1000
        
        # Format the prediction result
        result = format_prediction(prediction)
        
        return {
            "prediction": result,
            "processing_time_ms": processing_time
        }
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")
    
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)