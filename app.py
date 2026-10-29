
import io
import hashlib
import pandas as pd
import streamlit as st
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.models import resnet18, ResNet18_Weights
from torchvision.transforms.functional import to_tensor


# ---------------- PAGE CONFIGURATION ----------------
st.set_page_config(
    page_title="CyberShield AI | Robustness Lab",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------- ACCESSIBLE CYBERSECURITY THEME ----------------
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #071426 0%, #0b1d33 55%, #102943 100%);
    color: #f1f7ff;
}
.stApp p, .stApp li, .stApp label,
.stApp [data-testid="stCaptionContainer"],
.stApp [data-testid="stMarkdownContainer"] {
    font-size: 19px !important;
    line-height: 1.6 !important;
    color: #e6efff;
}
.stApp h1 {
    font-size: 38px !important;
    line-height: 1.3 !important;
    color: #ffffff !important;
}
.stApp h2 {
    font-size: 30px !important;
    color: #ffffff !important;
}
.stApp h3 {
    font-size: 25px !important;
    color: #ffffff !important;
}
section[data-testid="stSidebar"] {
    background-color: #0b1b2d;
}
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] li,
section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] {
    font-size: 19px !important;
    line-height: 1.65 !important;
    color: #f1f7ff !important;
}
[data-testid="stMetric"] {
    background: #112b43;
    border: 1px solid #1c6480;
    border-radius: 12px;
    padding: 16px;
}
[data-testid="stMetricLabel"] {
    font-size: 18px !important;
    color: #c5eaff !important;
}
[data-testid="stMetricValue"] {
    font-size: 32px !important;
    color: #55d8f5 !important;
}
[data-testid="stMetricDelta"] {
    font-size: 17px !important;
}
.stButton button, .stDownloadButton button {
    font-size: 19px !important;
    min-height: 48px !important;
    border-radius: 9px;
    font-weight: 700;
}
.stApp input, .stApp textarea,
.stApp [data-baseweb="select"] {
    font-size: 18px !important;
}
div[data-testid="stAlert"] {
    font-size: 18px !important;
}
hr {
    border-color: #24516c;
}
</style>
""", unsafe_allow_html=True)


# ---------------- MODEL AND PREPROCESSING ----------------
@st.cache_resource(show_spinner="Loading pretrained ResNet-18 model...")
def load_model():
    weights = ResNet18_Weights.DEFAULT
    model = resnet18(weights=weights)
    model.eval()
    return model, weights.meta["categories"]


def normalize_image(x):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    return (x - mean) / std


def predict(model, x, categories):
    with torch.no_grad():
        logits = model(normalize_image(x))
        probabilities = torch.softmax(logits, dim=1)
        confidence, index = probabilities.max(dim=1)

    return categories[index.item()], confidence.item()


def apply_fgsm(model, x, epsilon):
    """
    FGSM: x_adv = clip(x + epsilon * sign(gradient), 0, 1)

    The original prediction is used as a pseudo-label.
    The attack increases the loss to test input sensitivity.
    """
    attack_input = x.clone().detach().requires_grad_(True)

    logits = model(normalize_image(attack_input))
    pseudo_label = logits.detach().argmax(dim=1)
    loss = F.cross_entropy(logits, pseudo_label)

    model.zero_grad(set_to_none=True)
    loss.backward()

    adversarial = (
        attack_input + epsilon * attack_input.grad.sign()
    ).clamp(0, 1).detach()

    return adversarial


def tensor_to_image(x):
    array = (x.detach().squeeze(0).cpu().clamp(0, 1).permute(1, 2, 0).numpy() * 255)
    return Image.fromarray(array.astype("uint8"))


# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.title("🛡️ CyberShield AI")
    st.subheader("Robustness Lab")
    st.write("Evaluate how a pretrained image classifier responds to adversarial perturbations.")

    epsilon = st.slider(
        "Attack strength (epsilon)",
        min_value=0.0,
        max_value=0.08,
        value=0.02,
        step=0.005,
        help="Higher epsilon allows a larger pixel perturbation.",
    )

    st.markdown("---")
    st.subheader("System Information")
    st.write("**Model:** ResNet-18")
    st.write("**Weights:** ImageNet pretrained")
    st.write("**Attack:** FGSM")
    st.write("**Input:** Image")
    st.markdown("---")
    st.caption("Educational demonstration. Results depend on the input image and attack strength.")


# ---------------- MAIN HEADER ----------------
st.title("🛡️ CyberShield AI")
st.subheader("Adversarial Robustness Testing Dashboard")
st.write(
    "Upload an image, obtain the model's original prediction, "
    "apply an FGSM perturbation, and compare the resulting predictions."
)

st.markdown("---")


# ---------------- LOAD MODEL ----------------
try:
    model, categories = load_model()
except Exception as error:
    st.error(
        "The model could not be loaded. This can happen if model weights "
        "cannot be downloaded or a dependency fails."
    )
    with st.expander("Technical error details"):
        st.code(str(error))
    st.stop()


# ---------------- IMAGE UPLOAD ----------------
uploaded_file = st.file_uploader(
    "Upload an image for testing",
    type=["jpg", "jpeg", "png"],
    help="For best results, use a clear image of an object or animal.",
)

if uploaded_file is None:
    st.info("Upload an image to begin the robustness experiment.")
    st.markdown("### Experiment workflow")
    st.write("1. Upload an image.")
    st.write("2. Get the original model prediction.")
    st.write("3. Apply FGSM with the selected epsilon.")
    st.write("4. Compare predictions and confidence values.")
    st.write("5. Download the experiment summary.")
    st.stop()


# ---------------- PREPARE IMAGE ----------------
try:
    image_bytes = uploaded_file.getvalue()
    image_hash = hashlib.sha256(image_bytes).hexdigest()
    original_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    display_image = original_image.resize((224, 224))
    x = to_tensor(display_image).unsqueeze(0)
except Exception as error:
    st.error("The uploaded file could not be processed as an image.")
    st.code(str(error))
    st.stop()


# ---------------- ORIGINAL PREDICTION ----------------
original_label, original_confidence = predict(model, x, categories)

left, right = st.columns([1, 1], gap="large")

with left:
    st.subheader("Original Image")
    st.image(display_image, use_container_width=True)
    st.metric("Predicted class", original_label)
    st.metric("Top-1 confidence", f"{original_confidence * 100:.2f}%")


with right:
    st.subheader("Attack Configuration")
    st.write("**Attack:** Fast Gradient Sign Method (FGSM)")
    st.write(f"**Epsilon:** `{epsilon:.3f}`")
    st.write("**Pixel range:** 0 to 1")
    st.write("**Model:** ResNet-18 / ImageNet")
    st.write(
        "The attack computes input gradients and adds a bounded "
        "perturbation in the gradient-sign direction."
    )

    run_attack = st.button(
        "⚡ Apply FGSM Attack",
        type="primary",
        use_container_width=True,
    )


# ---------------- RUN ATTACK ----------------
if run_attack:
    with st.spinner("Generating adversarial image and evaluating the model..."):
        adversarial_tensor = apply_fgsm(model, x, epsilon)
        adversarial_label, adversarial_confidence = predict(
            model, adversarial_tensor, categories
        )

        st.session_state["cyber_result"] = {
            "image_hash": image_hash,
            "epsilon": epsilon,
            "adversarial_tensor": adversarial_tensor.cpu(),
            "adversarial_label": adversarial_label,
            "adversarial_confidence": adversarial_confidence,
            "original_label": original_label,
            "original_confidence": original_confidence,
        }


# ---------------- DISPLAY RESULTS ----------------
result = st.session_state.get("cyber_result")

if (
    result is not None
    and result["image_hash"] == image_hash
    and abs(result["epsilon"] - epsilon) < 1e-9
):
    adversarial_tensor = result["adversarial_tensor"]
    adversarial_label = result["adversarial_label"]
    adversarial_confidence = result["adversarial_confidence"]

    confidence_change = (
        adversarial_confidence - original_confidence
    ) * 100
    class_changed = original_label != adversarial_label

    st.markdown("---")
    st.header("Experiment Results")

    m1, m2, m3 = st.columns(3)
    m1.metric("Original prediction", original_label)
    m2.metric("Adversarial prediction", adversarial_label)
    m3.metric("Confidence change", f"{confidence_change:+.2f} pp")

    if class_changed:
        st.warning(
            "The predicted class changed after the perturbation. "
            "This example demonstrates sensitivity to the selected attack."
        )
    else:
        st.info(
            "The predicted class stayed the same for this example. "
            "This does not prove that the model is fully robust."
        )

    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.subheader("Original Input")
        st.image(display_image, use_container_width=True)
        st.write(f"**Class:** {original_label}")
        st.write(f"**Confidence:** {original_confidence * 100:.2f}%")

    with col2:
        st.subheader("Adversarial Input")
        st.image(tensor_to_image(adversarial_tensor), use_container_width=True)
        st.write(f"**Class:** {adversarial_label}")
        st.write(f"**Confidence:** {adversarial_confidence * 100:.2f}%")

    st.subheader("Confidence Comparison")
    chart_data = pd.DataFrame(
        {
            "Confidence (%)": [
                original_confidence * 100,
                adversarial_confidence * 100,
            ]
        },
        index=["Original image", "Adversarial image"],
    )
    st.bar_chart(chart_data, y_label="Confidence (%)")

    st.subheader("Automatic Experiment Analysis")
    st.write(
        f"The pretrained ResNet-18 model initially classified the image as "
        f"**{original_label}**, with a top-1 confidence of "
        f"**{original_confidence * 100:.2f}%**. An FGSM perturbation was then "
        f"generated using epsilon **{epsilon:.3f}**. After the perturbation, "
        f"the model predicted **{adversarial_label}**, with a top-1 confidence "
        f"of **{adversarial_confidence * 100:.2f}%**."
    )

    if class_changed:
        st.write(
            "Interpretation: the predicted class changed in this experiment, "
            "indicating that this input was sensitive to the tested perturbation. "
            "This single example does not measure overall model robustness."
        )
    else:
        st.write(
            "Interpretation: the predicted class remained unchanged in this "
            "experiment. The confidence may still have changed, and one image "
            "is not enough to establish overall robustness."
        )

    st.subheader("FGSM Formula")
    st.latex(
        r"x_{\mathrm{adv}} = "
        r"\operatorname{clip}_{[0,1]}"
        r"\left(x + \epsilon \cdot "
        r"\operatorname{sign}(\nabla_x J(\theta,x,y))\right)"
    )
    st.write(
        "**Explanation:** x is the original input, epsilon controls the "
        "perturbation size, J is the loss function, and the gradient indicates "
        "how the loss changes with respect to the input pixels. This demo uses "
        "the original predicted class as a pseudo-label."
    )

    report = f"""CYBERSHIELD AI — EXPERIMENT SUMMARY

Model: ResNet-18 pretrained on ImageNet
Attack: Fast Gradient Sign Method (FGSM)
Epsilon: {epsilon:.3f}

Original predicted class: {original_label}
Original top-1 confidence: {original_confidence * 100:.2f}%

Adversarial predicted class: {adversarial_label}
Adversarial top-1 confidence: {adversarial_confidence * 100:.2f}%

Confidence change: {confidence_change:+.2f} percentage points
Predicted class changed: {"Yes" if class_changed else "No"}

Interpretation:
{"The predicted class changed for this input, indicating sensitivity to this tested perturbation." if class_changed else "The predicted class did not change for this input under the tested perturbation."}

Limitations:
This is a single-image demonstration using a pretrained classifier and a pseudo-label.
It is not a comprehensive benchmark of robustness. Results may vary by image and epsilon.
"""

    st.download_button(
        "⬇️ Download Experiment Summary",
        data=report,
        file_name="cybershield_experiment_summary.txt",
        mime="text/plain",
        use_container_width=True,
    )

else:
    st.caption(
        "Choose an epsilon value and click Apply FGSM Attack to generate results."
    )


# ---------------- FOOTER ----------------
st.markdown("---")
st.caption(
    "CyberShield AI | Educational adversarial robustness demonstration | "
    "ResNet-18 + FGSM"
)
