import streamlit as st
from fastai.vision.all import *

# Define the Streamlit app
def main():
    st.title("Mongolian Food Image Classifier")
    st.header("Upload an image of Mongolian food to classify")

    # Upload image
    uploaded_file = st.file_uploader("Choose an image...", type="jpg")

    # Display uploaded image and make predictions
    if uploaded_file is not None:
        image = PILImage.create(uploaded_file)
        st.image(image, caption='Uploaded Image.', use_column_width=True)

        # Load data
        path = Path('food')
        dls = ImageDataLoaders.from_name_func(
            path, get_image_files(path), valid_pct=0.2,
            label_func=lambda x: x.parent.name, item_tfms=Resize(224))

        # Train model
        learn = cnn_learner(dls, resnet18, metrics=accuracy)
        learn.fine_tune(1)

        # Classify image
        pred, pred_idx, probs = learn.predict(image)
        st.write(f"Prediction: {pred}; Probability: {probs[pred_idx]:.4f}")

# Run the Streamlit app
if __name__ == '__main__':
    main()
