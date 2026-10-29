
import io
import hashlib

import pandas as pd
import streamlit as st
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.models import resnet18, ResNet18_Weights
from torchvision.transforms.functional import to_tensor


# =========================================================
# PAGE CONFIGURATION
# =========================================================
st.set_page_config(
    page_title="CyberShield AI | Robustness Lab",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =========================================================
# BLACK & WHITE THEME + LARGE ACCESSIBLE TEXT
# =========================================================
st.markdown("""
<style>
/* Main black background */
.stApp {
    background: #080808 !important;
    color: #ffffff !important;
}

/* Main body text */
.stApp p,
.stApp li,
.stApp label,
.stApp [data-testid="stMarkdownContainer"],
.stApp [data-testid="stCaptionContainer"] {
    color: #ffffff !important;
    font-size: 21px !important;
    line-height: 1.65 !important;
}

/* Headings */
.stApp h1 {
    color: #ffffff !important;
    font-size: 42px !important;
    font-weight: 800 !important;
    line-height: 1.3 !important;
}

.stApp h2 {
    color: #ffffff !important;
    font-size: 34px !important;
    font-weight: 750 !important;
}

.stApp h3 {
    color: #ffffff !important;
    font-size: 28px !important;
    font-weight: 700 !important;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #151515 !important;
    border-right: 2px solid #555555 !important;
}

section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] li,
section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] {
    color: #ffffff !important;
    font-size: 21px !important;
    line-height: 1.65 !important;
}

/* Metric cards */
[data-testid="stMetric"] {
    background: #171717 !important;
    border: 2px solid #777777 !important;
    border-radius: 12px !important;
    padding: 18px !important;
}

[data-testid="stMetricLabel"] {
    color: #eeeeee !important;
    font-size: 20px !important;
}

[data-testid="stMetricValue"] {
    color: #ffffff !important;
    font-size: 32px !important;
    font-weight: 800 !important;
}

[data-testid="stMetricDelta"] {
    color: #ffffff !important;
    font-size: 18px !important;
}

/* Buttons */
.stButton button,
.stDownloadButton button {
    background: #ffffff !important;
    color: #000000 !important;
    border: 2px solid #ffffff !important;
    border-radius: 8px !important;
    font-size: 21px !important;
    font-weight: 800 !important;
    min-height: 54px !important;
    padding: 12px 18px !important;
}

.stButton button:hover,
.stDownloadButton button:hover {
    background: #dddddd !important;
    color: #000000 !important;
}

/* Slider and form labels */
.stSlider [data-testid="stMarkdownContainer"] p {
    color: #ffffff !important;
    font-size: 20px !important;
}

/* Inputs */
.stApp input,
.stApp textarea {
    font-size: 20px !important;
}

/* File uploader */
[data-testid="stFileUploader"] {
    border: 2px dashed #aaaaaa !important;
    border-radius: 12px !important;
    padding: 12px !important;
}

/* Separators */
hr {
    border-color: #777777 !important;
}

/* Alerts */
div[data-testid="stAlert"] {
    border: 1px solid #aaaaaa !important;
    font-size: 20px !important;
}

/* Code blocks */
.stApp code {
    font-size: 18px !important;
}

/* Download and button text stays readable */
.stButton button p,
.stDownloadButton button p {
    color: #000000 !important;
    font-size: 21px !important;
}

/* Footer */
.stApp [data-testid="stCaptionContainer"] p {
    color: #dddddd !important;
    font-size: 17px !important;
}
</style>
""", unsafe_allow_html=True)


# =========================================================
# MODEL LOADING
# =========================================================
@st.cache_resource(show_spinner="Loading ResNet-18 model...")
def load_model():
    weights = ResNet18_Weights.DEFAULT
    model = resnet18(weights=weights)
    model.eval()
    categories = weights.meta["categories"]
    return model, categories


# =========================================================
# IMAGE PREPROCESSING
# =========================================================
def normalize_image(image_tensor):
    mean = torch.tensor(
        [0.485, 0.456, 0.406]
    ).view(1, 3, 1, 1)

    std = torch.tensor(
        [0.229, 0.224, 0.225]
    ).view(1, 3, 1, 1)

    return (image_tensor - mean) / std


# =========================================================
# MODEL PREDICTION
# =========================================================
def predict(model, image_tensor, categories):
    with torch.no_grad():
        logits = model(normalize_image(image_tensor))
        probabilities = torch.softmax(logits, dim=1)
        confidence, class_index = probabilities.max(dim=1)

    label = categories[class_index.item()]
    confidence_value = confidence.item()

    return label, confidence_value


# =========================================================
# FGSM ADVERSARIAL ATTACK
# =========================================================
def apply_fgsm(model, image_tensor, epsilon):
    """
    FGSM formula:
    x_adv = clip(x + epsilon * sign(gradient), 0, 1)

    The model's original predicted class is used as a pseudo-label.
    """

    attack_input = (
        image_tensor.clone().detach().requires_grad_(True)
    )

    logits = model(normalize_image(attack_input))

    pseudo_label = logits.detach().argmax(dim=1)

    loss = F.cross_entropy(logits, pseudo_label)

    model.zero_grad(set_to_none=True)
    loss.backward()

    perturbation = epsilon * attack_input.grad.sign()

    adversarial_image = (
        attack_input + perturbation
    ).clamp(0, 1).detach()

    return adversarial_image


# =========================================================
# CONVERT TENSOR TO PIL IMAGE
# =========================================================
def tensor_to_image(image_tensor):
    image_array = (
        image_tensor.detach()
        .squeeze(0)
        .cpu()
        .clamp(0, 1)
        .permute(1, 2, 0)
        .numpy()
        * 255
    )

    return Image.fromarray(image_array.astype("uint8"))


# =========================================================
# SIDEBAR
# =========================================================
with st.sidebar:
    st.title("🛡️ CyberShield AI")
    st.subheader("Robustness Testing Lab")

    st.write(
        "An adversarial machine learning demonstration "
        "for evaluating image classifier sensitivity."
    )

    st.markdown("---")

    epsilon = st.slider(
        "FGSM Attack Strength (Epsilon)",
        min_value=0.0,
        max_value=0.08,
        value=0.02,
        step=0.005,
        help="Higher epsilon allows a larger pixel perturbation.",
    )

    st.markdown("---")
    st.subheader("System Information")

    st.write("**Model:** ResNet-18")
    st.write("**Dataset:** ImageNet pretrained weights")
    st.write("**Attack:** FGSM")
    st.write("**Task:** Image classification")

    st.markdown("---")

    st.write("**Experiment workflow**")
    st.write("1. Upload an image")
    st.write("2. Get original prediction")
    st.write("3. Generate adversarial input")
    st.write("4. Compare model outputs")

    st.markdown("---")
    st.caption("Educational cybersecurity project")


# =========================================================
# MAIN DASHBOARD
# =========================================================
st.title("🛡️ CYBERSHIELD AI")
st.header("Adversarial Robustness Testing Dashboard")

st.write(
    "This application demonstrates how a small, calculated "
    "input perturbation can affect the predictions of a "
    "pretrained image classification model."
)

st.markdown("---")


# =========================================================
# LOAD MODEL SAFELY
# =========================================================
try:
    model, categories = load_model()

except Exception as error:
    st.error(
        "The model could not be loaded. Check the deployment "
        "logs and confirm that model weights can be downloaded."
    )

    with st.expander("Technical error details"):
        st.code(str(error))

    st.stop()


# =========================================================
# IMAGE UPLOAD
# =========================================================
st.header("1. Upload Test Image")

uploaded_file = st.file_uploader(
    "Choose an image (JPG, JPEG or PNG)",
    type=["jpg", "jpeg", "png"],
)

if uploaded_file is None:
    st.info(
        "Upload an image to start the experiment. "
        "A clear animal or object image is recommended."
    )

    st.subheader("How the system works")

    st.write(
        "**Step 1:** The uploaded image is processed and "
        "passed to the pretrained ResNet-18 model."
    )

    st.write(
        "**Step 2:** The model produces an original class "
        "prediction and confidence score."
    )

    st.write(
        "**Step 3:** FGSM calculates an input gradient and "
        "generates a perturbed image."
    )

    st.write(
        "**Step 4:** The model evaluates the adversarial image, "
        "and the application compares both predictions."
    )

    st.stop()


# =========================================================
# PROCESS IMAGE
# =========================================================
try:
    image_bytes = uploaded_file.getvalue()

    image_hash = hashlib.sha256(
        image_bytes
    ).hexdigest()

    original_image = Image.open(
        io.BytesIO(image_bytes)
    ).convert("RGB")

    display_image = original_image.resize((224, 224))

    image_tensor = to_tensor(
        display_image
    ).unsqueeze(0)

except Exception as error:
    st.error("Unable to process this uploaded image.")
    st.code(str(error))
    st.stop()


# =========================================================
# ORIGINAL PREDICTION
# =========================================================
original_label, original_confidence = predict(
    model,
    image_tensor,
    categories,
)

st.header("2. Original Model Prediction")

original_col1, original_col2 = st.columns(
    [1, 1],
    gap="large",
)

with original_col1:
    st.image(
        display_image,
        caption="Original input image",
        use_container_width=True,
    )

with original_col2:
    st.metric(
        "Predicted Class",
        original_label,
    )

    st.metric(
        "Top-1 Confidence",
        f"{original_confidence * 100:.2f}%",
    )

    st.write(
        "The confidence value represents the model's "
        "softmax probability for its top predicted class."
    )


# =========================================================
# ATTACK CONFIGURATION
# =========================================================
st.markdown("---")
st.header("3. Generate Adversarial Example")

st.write(
    f"Selected epsilon: **{epsilon:.3f}**. "
    "Click the button to apply FGSM to the uploaded image."
)

run_attack = st.button(
    "⚡ APPLY FGSM ATTACK",
    type="primary",
    use_container_width=True,
)


# =========================================================
# RUN FGSM ATTACK
# =========================================================
if run_attack:
    try:
        with st.spinner(
            "Generating adversarial input and evaluating model..."
        ):
            adversarial_tensor = apply_fgsm(
                model,
                image_tensor,
                epsilon,
            )

            adversarial_label, adversarial_confidence = predict(
                model,
                adversarial_tensor,
                categories,
            )

            st.session_state["cyber_result"] = {
                "image_hash": image_hash,
                "epsilon": epsilon,
                "adversarial_tensor": adversarial_tensor.cpu(),
                "adversarial_label": adversarial_label,
                "adversarial_confidence": adversarial_confidence,
            }

    except Exception as error:
        st.error("The FGSM attack could not be completed.")

        with st.expander("Technical error details"):
            st.code(str(error))


# =========================================================
# DISPLAY ATTACK RESULTS
# =========================================================
result = st.session_state.get("cyber_result")

if (
    result is not None
    and result["image_hash"] == image_hash
    and abs(result["epsilon"] - epsilon) < 1e-9
):
    adversarial_tensor = result["adversarial_tensor"]

    adversarial_label = result["adversarial_label"]

    adversarial_confidence = result[
        "adversarial_confidence"
    ]

    class_changed = (
        original_label != adversarial_label
    )

    confidence_change = (
        adversarial_confidence - original_confidence
    ) * 100

    st.markdown("---")
    st.header("4. Adversarial Experiment Results")

    metric1, metric2, metric3 = st.columns(3)

    with metric1:
        st.metric(
            "Original Class",
            original_label,
        )

    with metric2:
        st.metric(
            "Adversarial Class",
            adversarial_label,
        )

    with metric3:
        st.metric(
            "Confidence Change",
            f"{confidence_change:+.2f} pp",
        )

    if class_changed:
        st.warning(
            "The predicted class changed after the perturbation. "
            "This example demonstrates sensitivity to the tested attack."
        )
    else:
        st.info(
            "The predicted class remained unchanged for this example. "
            "This does not establish that the model is fully robust."
        )

    result_col1, result_col2 = st.columns(
        2,
        gap="large",
    )

    with result_col1:
        st.subheader("Original Image")

        st.image(
            display_image,
            use_container_width=True,
        )

        st.write(f"**Class:** {original_label}")

        st.write(
            f"**Confidence:** {original_confidence * 100:.2f}%"
        )

    with result_col2:
        st.subheader("Adversarial Image")

        adversarial_image = tensor_to_image(
            adversarial_tensor
        )

        st.image(
            adversarial_image,
            use_container_width=True,
        )

        st.write(f"**Class:** {adversarial_label}")

        st.write(
            f"**Confidence:** {adversarial_confidence * 100:.2f}%"
        )

    # -----------------------------------------------------
    # CONFIDENCE CHART
    # -----------------------------------------------------
    st.markdown("---")
    st.header("5. Confidence Comparison")

    chart_data = pd.DataFrame(
        {
            "Confidence (%)": [
                original_confidence * 100,
                adversarial_confidence * 100,
            ]
        },
        index=[
            "Original Image",
            "Adversarial Image",
        ],
    )

    st.bar_chart(
        chart_data,
        y_label="Confidence (%)",
    )

    # -----------------------------------------------------
    # AUTOMATIC EXPERIMENT ANALYSIS
    # -----------------------------------------------------
    st.markdown("---")
    st.header("6. Automatic Experiment Analysis")

    st.write(
        f"The pretrained ResNet-18 model classified the "
        f"original image as **{original_label}**, with a "
        f"top-1 confidence of "
        f"**{original_confidence * 100:.2f}%**."
    )

    st.write(
        f"FGSM was then applied using epsilon "
        f"**{epsilon:.3f}**. The adversarial image was "
        f"classified as **{adversarial_label}**, with a "
        f"top-1 confidence of "
        f"**{adversarial_confidence * 100:.2f}%**."
    )

    if class_changed:
        st.write(
            "**Interpretation:** The predicted class changed "
            "for this input. This indicates sensitivity to the "
            "tested perturbation, but does not measure overall "
            "model robustness."
        )
    else:
        st.write(
            "**Interpretation:** The predicted class remained "
            "the same for this input. Confidence may still have "
            "changed. One image is insufficient to establish "
            "overall robustness."
        )

    st.write(
        "**Confidence change** is reported in percentage points, "
        "calculated as adversarial confidence minus original confidence."
    )

    # -----------------------------------------------------
    # FGSM FORMULA
    # -----------------------------------------------------
    st.markdown("---")
    st.header("7. FGSM Technical Explanation")

    st.latex(
        r"x_{\mathrm{adv}} = "
        r"\operatorname{clip}_{[0,1]}"
        r"\left(x + \epsilon \cdot "
        r"\operatorname{sign}(\nabla_x J(\theta,x,y))\right)"
    )

    st.write(
        "**x:** Original image input."
    )

    st.write(
        "**x_adv:** Generated adversarial image."
    )

    st.write(
        "**Epsilon (ε):** Controls the maximum perturbation "
        "added to each normalized input pixel."
    )

    st.write(
        "**Gradient:** Indicates how the loss changes with "
        "respect to the input."
    )

    st.write(
        "**Sign function:** Uses the sign of each gradient value "
        "to determine the perturbation direction."
    )

    st.write(
        "**Clipping:** Keeps image pixel values within the "
        "valid range from 0 to 1."
    )

    st.caption(
        "Implementation note: this demonstration uses the model's "
        "original predicted class as a pseudo-label when calculating "
        "the FGSM gradient."
    )

    # -----------------------------------------------------
    # DOWNLOADABLE REPORT
    # -----------------------------------------------------
    st.markdown("---")
    st.header("8. Download Experiment Report")

    report = f"""
CYBERSHIELD AI
ADVERSARIAL ROBUSTNESS EXPERIMENT SUMMARY

Model: ResNet-18 pretrained on ImageNet
Attack: Fast Gradient Sign Method (FGSM)
Epsilon: {epsilon:.3f}

Original predicted class: {original_label}
Original top-1 confidence: {original_confidence * 100:.2f}%

Adversarial predicted class: {adversarial_label}
Adversarial top-1 confidence: {adversarial_confidence * 100:.2f}%

Confidence change: {confidence_change:+.2f} percentage points
Predicted class changed: {"Yes" if class_changed else "No"}

INTERPRETATION
{"The predicted class changed for this image under the tested perturbation." if class_changed else "The predicted class did not change for this image under the tested perturbation."}

LIMITATIONS
This is a single-image educational demonstration using a pretrained
classifier and a pseudo-label. It is not a comprehensive robustness
benchmark. Results depend on the input image and epsilon value.
"""

    st.download_button(
        "⬇️ DOWNLOAD EXPERIMENT SUMMARY",
        data=report,
        file_name="cybershield_experiment_summary.txt",
        mime="text/plain",
        use_container_width=True,
    )

else:
    st.caption(
        "Select an epsilon value and click APPLY FGSM ATTACK "
        "to generate the experiment results."
    )


# =========================================================
# FOOTER
# =========================================================
st.markdown("---")

st.caption(
    "CYBERSHIELD AI | Adversarial Robustness Demonstration | "
    "ResNet-18 + FGSM"
)
