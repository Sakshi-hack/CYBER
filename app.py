
import io
import streamlit as st
import torch
import torch.nn.functional as F
import torchvision.models as models
from torchvision.models import ResNet18_Weights
from torchvision.transforms.functional import to_tensor
from PIL import Image

st.set_page_config(page_title="CyberShield AI", page_icon="🛡️", layout="wide")

st.title("🛡️ CyberShield AI")
st.subheader("Robustness Against Adversarial Attacks")
st.write("Demonstration of how small image perturbations can affect AI predictions.")

st.info("Prototype using a pretrained ResNet-18 model. It may differ from the model described in your project report.")

@st.cache_resource(show_spinner="Loading AI model...")
def load_model():
    weights = ResNet18_Weights.DEFAULT
    model = models.resnet18(weights=weights)
    model.eval()

    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

    return model, weights.meta["categories"], mean, std

def predict(model, categories, image_tensor, mean, std):
    normalized = (image_tensor - mean) / std
    with torch.no_grad():
        probabilities = torch.softmax(model(normalized), dim=1)
        confidence, index = probabilities.max(dim=1)
    return categories[index.item()], confidence.item() * 100

def create_fgsm(model, image_tensor, mean, std, epsilon):
    normalized = (image_tensor - mean) / std

    with torch.no_grad():
        original_class = model(normalized).argmax(dim=1)

    attack_input = image_tensor.clone().detach().requires_grad_(True)
    logits = model((attack_input - mean) / std)
    loss = F.cross_entropy(logits, original_class)

    model.zero_grad(set_to_none=True)
    loss.backward()

    adversarial = attack_input + epsilon * attack_input.grad.sign()
    return adversarial.detach().clamp(0, 1)

def tensor_to_pil(tensor):
    array = tensor.squeeze(0).detach().cpu().permute(1, 2, 0).numpy()
    array = (array * 255).round().clip(0, 255).astype("uint8")
    return Image.fromarray(array)

try:
    model, categories, mean, std = load_model()
except Exception as error:
    st.error("The AI model could not be loaded. Check the deployment logs.")
    st.code(str(error))
    st.stop()

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
epsilon = st.slider("Attack strength (epsilon)", 0.0, 0.08, 0.02, 0.005)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB").resize((224, 224))
    original = to_tensor(image).unsqueeze(0)

    original_label, original_conf = predict(
        model, categories, original, mean, std
    )

    col1, col2 = st.columns(2)

    with col1:
        st.image(image, caption="Original Image", use_container_width=True)

    with col2:
        st.metric("Original prediction", original_label)
        st.metric("Confidence", f"{original_conf:.2f}%")

    if st.button("⚡ Apply FGSM Attack", type="primary"):
        with st.spinner("Generating adversarial example..."):
            adversarial = create_fgsm(
                model, original, mean, std, epsilon
            )
            attacked_label, attacked_conf = predict(
                model, categories, adversarial, mean, std
            )

            st.session_state["result"] = {
                "filename": uploaded_file.name,
                "image": tensor_to_pil(adversarial),
                "label": attacked_label,
                "confidence": attacked_conf,
                "epsilon": epsilon,
            }

    result = st.session_state.get("result")

    if result and result["filename"] == uploaded_file.name:
        st.markdown("### Results Comparison")
        left, right = st.columns(2)

        with left:
            st.image(image, caption="Original Image", use_container_width=True)
            st.write("Prediction:", original_label)
            st.write(f"Confidence: {original_conf:.2f}%")

        with right:
            st.image(
                result["image"],
                caption="Adversarial Image",
                use_container_width=True,
            )
            st.write("Prediction:", result["label"])
            st.write(f"Confidence: {result['confidence']:.2f}%")

        if original_label != result["label"]:
            st.warning("The predicted class changed after the attack.")
        else:
            st.info("The predicted class stayed the same. The confidence may have changed.")

        buffer = io.BytesIO()
        result["image"].save(buffer, format="PNG")
        st.download_button(
            "Download adversarial image",
            data=buffer.getvalue(),
            file_name="adversarial_image.png",
            mime="image/png",
        )

with st.expander("About FGSM"):
    st.write(
        "Fast Gradient Sign Method (FGSM) uses the gradient of the model's "
        "loss with respect to the input image to generate a small perturbation."
    )

with st.expander("Project limitations"):
    st.write(
        "This prototype uses a pretrained ImageNet ResNet-18 model. "
        "A changed prediction is not guaranteed for every image. "
        "This demonstration does not prove that an adversarial defense is effective."
    )
