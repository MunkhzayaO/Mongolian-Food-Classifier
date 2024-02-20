import streamlit as st
from fastai.vision.all import *
import gdown

st.markdown("""# Mongolian Five Animal Classifier

Mongolia has five traditional herd animals: Horses, Cattle, Sheep, Goats, and Camels. This app allows you to upload an image of one of these five animals and the connected model will classify it for you. Upload an image and try it out!

This app was created as a demo for the Deep Learning course at LETU Mongolia American University.""")

st.markdown("""### Upload your image here""")

image_file = st.file_uploader("Image Uploader", type=["png","jpg","jpeg"])

## Model Loading Section
model_path = Path("export.pkl")

if not model_path.exists():
    with st.spinner("Downloading model... this may take awhile! \n Don't stop it!"):
        url = 'https://colab.research.google.com/uc?id=1ZGPfy9RvyzDyf_U9bnTbreNXEx0pvFCf'
        output = 'export.pkl'
        gdown.download(url, output, quiet=False)
    learn_inf = load_learner('export.pkl')
else:
    learn_inf = load_learner('export.pkl')

col1, col2 = st.columns(2)
if image_file is not None:
    img = PILImage.create(image_file)
    pred, pred_idx, probs = learn_inf.predict(img)

    with col1:
        st.markdown(f"""### Predicted animal: {pred.capitalize()}""")
        st.markdown(f"""### Probability: {round(max(probs.tolist()), 3) * 100}%""")
    with col2:
        st.image(img, width=300)
