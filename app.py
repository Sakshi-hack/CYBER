
import io
import hashlib

import pandas as pd
import streamlit as st
import torch
import torch.nn.functional as F
import torchvision.models as models
from torchvision.models import ResNet18_Weights
from torchvision.transforms.functional import to_tensor
from PIL import Image


# -------------------- PAGE CONFIGURATION --------------------

st.set_page_config(
    page_title="CyberShield AI | Robustness Lab",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# -------------------- CUSTOM UI THEME --------------------

st.markdown("""
<style>
.stApp {
    background: linear-gradient(145deg, #07111f 0%, #0b1730 55%, #101b35 100%);
    color: #e6f1ff;
}
.block-container {
    max-width: 1250px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}
h1, h2, h3 {
    color: #eaf5ff !important;
}
p, label, li {
    color: #c6d7ec;
}
[data-testid="stMetric"] {
    background: linear-gradient(135deg, #10233f, #132d4c);
    border: 1px solid #244766;
    border-radius: 14px;
    padding: 18px;
}
[data-testid="stMetricLabel"] {
    color: #91b8d9 !important;
}
[data-testid="stMetricValue"] {
    color: #56e0e8 !important;
}
div.stButton > button {
    background: linear-gradient(90deg, #087fbd, #08b6c9);
    color: white;
    border: 0;
    border-radius: 10px;
    font-weight: 700;
    min-height: 46px;
}
div.stButton > button:hover {
    border: 1px solid #71f4ff;
    color: white;
}
div[data-testid="stExpander"] {
    border: 1px solid #244766;
    border-radius: 12px;
}
div[data-testid="stFileUploader"] {
    background: #10213a;
    border: 1px dashed #287b9b;
    border-radius: 12px;
    padding: 12px;
}
.cyber-banner {
    background: linear-gradient(110deg, #102743, #123d56);
    border: 1px solid #287b9b;
    border-radius: 18px;
    padding: 24px;
    margin-bottom: 20px;
}
.cyber-kicker {
    color: #57e6ee;
    font-size: 0.82rem;
    font-weight: 700;
    letter-spacing: 2px;
}
.cyber-subtitle {
    color: #b8d0e8;
    font-size: 1.05rem;
}
.small-note {
    color: #8eabc7;
    font-size: 0.85rem;
}
</style>
""", unsafe_allow_html=True)


# -------------------- HEADER --------------------

st.markdown("""
<div class="cyber-banner">
    <div class="cyber-kicker">CYBERSECURITY • AI RESEARCH LAB</div>
    <h1>🛡️ CyberShield AI</h1>
    <div class="cyber-subtitle">
        Adversarial Robustness Analysis & FGSM Demonstration
    </div>
</div>
""", unsafe_allow_html=True)

st.write(
    "Investigate how a small, gradient-based input perturbation "
    "can affect the predictions of an image classification model."
)

st.caption(
    "PROJECT PROTOTYPE  |  IMAGE CLASSIFICATION  |  FGSM ATTACK"
)


# -------------------- SIDEBAR --------------------

with st.sidebar:
    st.markdown("## 🛡️ Control Center")
    st.write("**Model:** ResNet-18")
    st.write("**Dataset classes:** ImageNet")
    st.write("**Attack:** Fast Gradient Sign Method")
    st.write("**Execution:** CPU")

    st.divider()

    epsilon = st.slider(
        "Attack strength (epsilon)",
        min_value=0.0,
        max_value=0.08,
        value=0.02,
        step=0.005,
        help="Controls the maximum pixel-space perturbation.",
    )

    st.caption(
        "Higher epsilon permits a larger perturbation. "
        "It does not guarantee a changed prediction."
    )

    st.divider()
    st.markdown("### Experiment workflow")
    st.write("1. Upload an image")
    st.write("2. Inspect the original prediction")
    st.write("3. Apply the FGSM attack")
    st.write("4. Compare predictions and confidence")


# -------------------- MODEL --------------------

@st.cache_resource(show_spinner="Loading pretrained AI model...")
def load_model():
    weights = ResNet18_Weights.DEFAULT
    model = models.resnet18(weights=weights)
    model.eval()

    mean = torch.tensor(
        [0.485, 0.456, 0.406]
    ).view(1, 3, 1, 1)

    std = torch.tensor(
        [0.229, 0.224, 0.225]
    ).view(1, 3, 1, 1)

    return model, weights.meta["categories"], mean, std


def predict(model, categories, image_tensor, mean, std):
    normalized = (image_tensor - mean) / std

    with torch.no_grad():
        logits = model(normalized)
        probabilities = torch.softmax(logits, dim=1)
        confidence, index = probabilities.max(dim=1)

    return (
        categories[index.item()],
        float(confidence.item() * 100),
    )


def create_fgsm(model, image_tensor, mean, std, epsilon):
    # Use the model's original predicted class as the reference label.
    normalized = (image_tensor - mean) / std

    with torch.no_grad():
        original_class = model(normalized).argmax(dim=1)

    attack_input = (
        image_tensor.clone().detach().requires_grad_(True)
    )

    logits = model((attack_input - mean) / std)
    loss = F.cross_entropy(logits, original_class)

    model.zero_grad(set_to_none=True)
    loss.backward()

    perturbation = epsilon * attack_input.grad.sign()

    adversarial = (
        attack_input + perturbation
    ).detach().clamp(0, 1)

    return adversarial


def tensor_to_pil(tensor):
    array = (
        tensor.squeeze(0)
        .detach()
        .cpu()
        .permute(1, 2, 0)
        .numpy()
    )

    array = (
        (array * 255)
        .round()
        .clip(0, 255)
        .astype("uint8")
    )

    return Image.fromarray(array)


# -------------------- LOAD MODEL SAFELY --------------------

try:
    model, categories, mean, std = load_model()
except Exception as error:
    st.error(
        "The model could not be loaded. Check the deployment logs "
        "and the availability of the pretrained model weights."
    )
    with st.expander("Technical error details"):
        st.code(str(error))
    st.stop()


# -------------------- IMAGE UPLOAD --------------------

st.markdown("## 01 / Input Image")

uploaded_file = st.file_uploader(
    "Upload an image for robustness testing",
    type=["jpg", "jpeg", "png"],
    help="Choose a clear image of an object or animal.",
)

if uploaded_file is None:
    st.info(
        "Upload an image to begin the experiment. "
        "The original prediction will appear here."
    )

    st.markdown("## 02 / Analysis Preview")

    col1, col2, col3 = st.columns(3)
    col1.metric("Classification model", "ResNet-18")
    col2.metric("Attack method", "FGSM")
    col3.metric("Current epsilon", f"{epsilon:.3f}")

    st.markdown(
        '<p class="small-note">The model uses pretrained ImageNet '
        'weights. Its prediction may differ from the exact model '
        'described in your project report.</p>',
        unsafe_allow_html=True,
    )

    st.stop()


# -------------------- ORIGINAL PREDICTION --------------------

try:
    uploaded_bytes = uploaded_file.getvalue()
    image_id = hashlib.sha256(uploaded_bytes).hexdigest()

    image = Image.open(io.BytesIO(uploaded_bytes)).convert("RGB")
    image = image.resize((224, 224))

    original = to_tensor(image).unsqueeze(0)

    original_label, original_conf = predict(
        model, categories, original, mean, std
    )

except Exception as error:
    st.error("The uploaded image could not be processed.")
    st.code(str(error))
    st.stop()


st.markdown("## 02 / Baseline Analysis")

left, right = st.columns([1.15, 1])

with left:
    st.image(
        image,
        caption="Original input image — resized to 224 × 224",
        use_container_width=True,
    )

with right:
    st.markdown("### Original model prediction")
    st.metric("Predicted class", original_label)
    st.metric("Top-class confidence", f"{original_conf:.2f}%")

    st.caption(
        "Confidence is the model's softmax score, "
        "not a guarantee that the prediction is correct."
    )


# -------------------- RUN FGSM --------------------

st.markdown("## 03 / Adversarial Experiment")

st.write(
    f"Current perturbation budget: **epsilon = {epsilon:.3f}**"
)

if st.button("⚡ Apply FGSM Attack", type="primary", use_container_width=True):
    with st.spinner("Calculating the adversarial example..."):
        try:
            adversarial = create_fgsm(
                model, original, mean, std, epsilon
            )

            attacked_label, attacked_conf = predict(
                model, categories, adversarial, mean, std
            )

            adversarial_image = tensor_to_pil(adversarial)

            st.session_state["attack_result"] = {
                "image_id": image_id,
                "epsilon": epsilon,
                "original_label": original_label,
                "original_conf": original_conf,
                "attacked_label": attacked_label,
                "attacked_conf": attacked_conf,
                "original_image": image.copy(),
                "adversarial_image": adversarial_image,
            }

        except Exception as error:
            st.error("The FGSM experiment could not be completed.")
            st.code(str(error))


# -------------------- RESULTS DASHBOARD --------------------

result = st.session_state.get("attack_result")

if result and result["image_id"] == image_id:
    st.markdown("---")
    st.markdown("## 04 / Results Dashboard")

    original_label = result["original_label"]
    original_conf = result["original_conf"]
    attacked_label = result["attacked_label"]
    attacked_conf = result["attacked_conf"]
    used_epsilon = result["epsilon"]

    prediction_changed = original_label != attacked_label
    confidence_delta = attacked_conf - original_conf

    metric1, metric2, metric3 = st.columns(3)

    metric1.metric(
        "Original prediction",
        original_label,
        help="Top predicted ImageNet class before the attack.",
    )

    metric2.metric(
        "Adversarial prediction",
        attacked_label,
        help="Top predicted ImageNet class after the attack.",
    )

    metric3.metric(
        "Top-class confidence change",
        f"{confidence_delta:+.2f} pp",
        help=(
            "Difference between the two top-class confidence scores, "
            "measured in percentage points."
        ),
    )

    if prediction_changed:
        st.warning(
            "Prediction changed: the top predicted class differs "
            "between the original and adversarial inputs."
        )
    else:
        st.info(
            "Prediction unchanged: the top predicted class stayed the "
            "same. This does not mean the attack had no effect."
        )

    st.markdown("### Visual comparison")

    col_original, col_adversarial = st.columns(2)

    with col_original:
        st.image(
            result["original_image"],
            caption="Original image",
            use_container_width=True,
        )
        st.write(f"**Prediction:** {original_label}")
        st.write(f"**Top-class confidence:** {original_conf:.2f}%")

    with col_adversarial:
        st.image(
            result["adversarial_image"],
            caption="FGSM adversarial image",
            use_container_width=True,
        )
        st.write(f"**Prediction:** {attacked_label}")
        st.write(f"**Top-class confidence:** {attacked_conf:.2f}%")

    st.markdown("### Confidence comparison")

    chart_data = pd.DataFrame(
        {
            "Top-class confidence (%)": [
                original_conf,
                attacked_conf,
            ]
        },
        index=[
            f"Original: {original_label}",
            f"Adversarial: {attacked_label}",
        ],
    )

    st.bar_chart(
        chart_data,
        y_label="Confidence (%)",
        color="#29d9e8",
    )

    st.caption(
        "These bars show each input's top-class confidence. "
        "When the predicted classes differ, the scores belong to "
        "different classes and should not be interpreted as the "
        "confidence of one fixed class before and after the attack."
    )

    # -------------------- AUTOMATIC ENGLISH ANALYSIS --------------------

    st.markdown("### Automatic experiment analysis")

    if prediction_changed:
        conclusion = (
            f"The top predicted class changed from '{original_label}' "
            f"to '{attacked_label}' after applying FGSM with "
            f"epsilon = {used_epsilon:.3f}. This result demonstrates "
            "that this particular image classification was sensitive "
            "to the tested perturbation."
        )
    else:
        conclusion = (
            f"The top predicted class remained '{original_label}' "
            f"after applying FGSM with epsilon = {used_epsilon:.3f}. "
            "The attack did not change the top predicted class for "
            "this input, although the confidence score may have changed."
        )

    st.write(conclusion)

    st.write(
        f"The original top-class confidence was {original_conf:.2f}%, "
        f"and the adversarial input's top-class confidence was "
        f"{attacked_conf:.2f}%. The difference between these scores "
        f"was {confidence_delta:+.2f} percentage points."
    )

    st.info(
        "Scientific limitation: one image is not enough to establish "
        "overall model robustness. A systematic evaluation requires "
        "multiple inputs, a defined success criterion, and suitable "
        "evaluation metrics."
    )

    # -------------------- DOWNLOADABLE SUMMARY --------------------

    report = f"""CYBERSHIELD AI — FGSM EXPERIMENT SUMMARY

Project: Robustness Against Adversarial Attacks
Model: Pretrained ResNet-18
Class labels: ImageNet
Attack method: Fast Gradient Sign Method (FGSM)
Epsilon: {used_epsilon:.3f}

Original predicted class: {original_label}
Original top-class confidence: {original_conf:.2f}%

Adversarial predicted class: {attacked_label}
Adversarial top-class confidence: {attacked_conf:.2f}%

Prediction changed: {"Yes" if prediction_changed else "No"}
Top-class confidence difference: {confidence_delta:+.2f} percentage points

Analysis:
{conclusion}

Method:
FGSM adds a perturbation based on the sign of the gradient of the
classification loss with respect to the input image.

Limitations:
This is a single-image demonstration using a pretrained ImageNet
ResNet-18 model. It does not prove overall robustness or demonstrate
that a defense is effective.
"""

    st.download_button(
        "⬇️ Download experiment summary",
        data=report,
        file_name="cybershield_experiment_summary.txt",
        mime="text/plain",
        use_container_width=True,
    )


# -------------------- VIVA & THEORY --------------------

st.markdown("---")
st.markdown("## 05 / Research Notes & Viva Support")

with st.expander("📘 FGSM formula and explanation"):
    st.markdown("**Fast Gradient Sign Method (FGSM)**")

    st.latex(
        r"x_{\mathrm{adv}} = "
        r"\operatorname{Clip}_{[0,1]}"
        r"\left(x + \epsilon \cdot "
        r"\operatorname{sign}(\nabla_x L(f(x), y))\right)"
    )

    st.write(
        "**In simple terms:** FGSM calculates how the input image "
        "affects the model's loss, then uses the sign of that gradient "
        "to create a small perturbation intended to increase the loss."
    )

    st.write("**Key terms:**")
    st.markdown("""
    - **x:** Original image represented as pixel values.
    - **x_adv:** Adversarial image after applying the perturbation.
    - **epsilon:** Maximum perturbation budget in pixel space.
    - **L:** Classification loss function.
    - **Gradient:** Indicates how the loss changes with respect to the input.
    - **Clip:** Keeps pixel values within the valid range from 0 to 1.
    """)

    st.caption(
        "Implementation note: this prototype uses the model's original "
        "predicted class as the reference label for the FGSM loss."
    )

with st.expander("🎓 Viva-ready explanation"):
    st.markdown("""
    **Q. What is adversarial robustness?**

    Adversarial robustness is a model's ability to maintain reliable
    predictions when its input is subjected to small, deliberately
    designed perturbations.

    **Q. What is FGSM?**

    FGSM stands for Fast Gradient Sign Method. It uses the sign of the
    gradient of the loss with respect to the input to generate a
    perturbation in a single gradient-based step.

    **Q. What does epsilon control?**

    Epsilon controls the maximum allowed perturbation in pixel space.
    A larger value permits a larger change, but does not guarantee
    that the predicted class will change.

    **Q. What does this prototype demonstrate?**

    It compares the top prediction and confidence of a pretrained
    ResNet-18 model before and after applying FGSM to one uploaded image.

    **Q. What are its limitations?**

    It uses a pretrained ImageNet model and tests one image at a time.
    It is not a complete robustness benchmark and does not implement
    or validate a defense mechanism.
    """)

with st.expander("⚠️ Model details and limitations"):
    st.write(
        "This prototype uses pretrained ResNet-18 weights trained on "
        "ImageNet. Its predictions may differ from a model trained "
        "specifically for the dataset in a project report."
    )
    st.write(
        "The app demonstrates an adversarial attack, not a defense. "
        "A changed prediction on one image is evidence for that example "
        "only; it is not proof of general vulnerability or robustness."
    )

st.markdown("---")
st.markdown(
    '<p class="small-note">CyberShield AI • Academic demonstration '
    'of adversarial machine learning</p>',
    unsafe_allow_html=True,
)
