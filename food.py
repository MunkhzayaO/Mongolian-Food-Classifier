import streamlit as st
from fastai.vision.all import *
from duckduckgo_search import DDGS

# Define functions for image search and downloading
def search_images(term, max_images=30):
    print(f"Searching for '{term}'")
    with DDGS() as ddgs:
        search_results = ddgs.images(keywords=term)
        image_urls = [next(search_results).get("image") for _ in range(max_images)]
        return L(image_urls)

def download_images(path, urls):
    path.mkdir(exist_ok=True)
    for i, url in enumerate(urls):
        download_url(url, path/f"image_{i}.jpg")

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

        # Create DataBlock for image classification
        foods = DataBlock(blocks=(ImageBlock, CategoryBlock),
                          get_items=get_image_files,
                          splitter=RandomSplitter(valid_pct=0.2, seed=42),
                          get_y=parent_label,
                          item_tfms=Resize(224),
                          batch_tfms=aug_transforms())

        # Create DataLoader
        path = Path('food')
        dls = foods.dataloaders(path)

        # Load trained model
        learn = cnn_learner(dls, resnet18)

        # Classify image
        pred, pred_idx, probs = learn.predict(image)
        st.write(f"Prediction: {pred}; Probability: {probs[pred_idx]:.4f}")

# Run the Streamlit app
if __name__ == '__main__':
    main()
