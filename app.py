
import io
from datetime import datetime

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import models, transforms


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------
st.set_page_config(
    page_title="CyberShield AI",
    page_icon="🛡️",
    layout="wide"
)


# --------------------------------------------------
# DARK THEME AND STYLING
# --------------------------------------------------
st.markdown("""
<style>
.stApp {
    background-color: #080808;
    color: #ffffff;
}
[data-testid="stSidebar"] {
    background-color: #151515;
}
h1 {
    color: #ffffff !important;
    font-size: 42px !important;
    font-weight: 800 !important;
}
h2 {
    color: #ffffff !important;
    font-size: 34px !important;
}
h3 {
    color: #ffffff !important;
    font-size: 28px !important;
}
p, label, li {
    color: #eeeeee;
    font-size: 19px;
}
div[data-testid="stMetricValue"] {
    color: #22c55e;
    font-size: 30px;
}
div[data-testid="stMetricLabel"] {
    color: #ffffff;
    font-size: 17px;
}
.stButton > button {
    background-color: #16a34a;
    color: white;
    font-size: 19px;
    font-weight: bold;
    border-radius: 10px;
    border: none;
    padding: 10px 18px;
}
.stButton > button:hover {
    background-color: #15803d;
    color: white;
}
div[data-testid="stAlert"] {
    font-size: 17px;
}
</style>
""", unsafe_allow_html=True)


# --------------------------------------------------
# LOAD PRETRAINED RESNET-18
# --------------------------------------------------
@st.cache_resource
def load_model():
    weights = models.ResNet18_Weights.DEFAULT
    model = models.resnet18(weights=weights)
    model.eval()

    return model, weights.meta["categories"]


model, class_names = load_model()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# ImageNet normalization
mean = torch.tensor(
    [0.485, 0.456, 0.406],
    device=device
).view(1, 3, 1, 1)

std = torch.tensor(
    [0.229, 0.224, 0.225],
    device=device
).view(1, 3, 1, 1)

# Resize and crop without normalizing yet.
# This lets FGSM perturb pixel values in the [0, 1] range.
image_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor()
])


# --------------------------------------------------
# PREDICTION FUNCTION
# --------------------------------------------------
def predict_image(pixel_tensor):
    normalized = (pixel_tensor - mean) / std

    with torch.no_grad():
        logits = model(normalized)
        probabilities = torch.softmax(logits, dim=1)

    confidence, predicted_index = probabilities.max(dim=1)

    return (
        int(predicted_index.item()),
        float(confidence.item()) * 100,
        probabilities[0].detach().cpu()
    )


# --------------------------------------------------
# FGSM ATTACK
# --------------------------------------------------
def fgsm_attack(pixel_tensor, epsilon, original_class):
    image = pixel_tensor.clone().detach().to(device)
    image.requires_grad_(True)

    model.zero_grad(set_to_none=True)

    normalized = (image - mean) / std
    logits = model(normalized)

    # Use the original prediction as the attack reference.
    target = torch.tensor(
        [original_class],
        dtype=torch.long,
        device=device
    )

    loss = F.cross_entropy(logits, target)
    loss.backward()

    # Untargeted FGSM: increase the loss of the original class.
    perturbation = epsilon * image.grad.sign()

    adversarial_image = (
        image.detach() + perturbation
    ).clamp(0, 1)

    return adversarial_image.detach()


# --------------------------------------------------
# SEMICIRCLE CONFIDENCE GAUGES
# --------------------------------------------------
def create_confidence_gauges(original_conf, adversarial_conf):
    fig = go.Figure()

    gauge_settings = [
        {
            "value": original_conf,
            "title": "Original Image",
            "color": "#22c55e",
            "domain": {"x": [0.00, 0.48], "y": [0.02, 0.98]}
        },
        {
            "value": adversarial_conf,
            "title": "Adversarial Image",
            "color": "#f59e0b",
            "domain": {"x": [0.52, 1.00], "y": [0.02, 0.98]}
        }
    ]

    for item in gauge_settings:
        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=item["value"],
                title={
                    "text": item["title"],
                    "font": {"size": 21, "color": "white"}
                },
                number={
                    "suffix": "%",
                    "font": {"size": 34, "color": "white"},
                    "valueformat": ".2f"
                },
                gauge={
                    "shape": "angular",
                    "axis": {
                        "range": [0, 100],
                        "tickwidth": 1,
                        "tickcolor": "#cccccc",
                        "tickfont": {"color": "white", "size": 11}
                    },
                    "bar": {
                        "color": item["color"],
                        "thickness": 0.27
                    },
                    "bgcolor": "#171717",
                    "borderwidth": 1,
                    "bordercolor": "#444444",
                    "steps": [
                        {"range": [0, 50], "color": "#252525"},
                        {"range": [50, 80], "color": "#303030"},
                        {"range": [80, 100], "color": "#3a3a3a"}
                    ]
                },
                domain=item["domain"]
            )
        )

    fig.update_layout(
        height=340,
        margin=dict(l=20, r=20, t=65, b=15),
        paper_bgcolor="#080808",
        plot_bgcolor="#080808",
        font={"color": "white"},
        showlegend=False
    )

    return fig


# --------------------------------------------------
# MAIN DASHBOARD
# --------------------------------------------------
st.title("🛡️ CyberShield AI")
st.subheader("Robustness Against Adversarial Attacks")

st.markdown("""
An interactive image-classification experiment using a pretrained
ResNet-18 model and the Fast Gradient Sign Method (FGSM).
""")

st.info(
    "Upload an image, observe its original prediction, "
    "and test how an FGSM perturbation affects the model."
)


# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------
st.sidebar.title("⚙️ Experiment Settings")

epsilon = st.sidebar.slider(
    "FGSM Epsilon (Perturbation Strength)",
    min_value=0.00,
    max_value=0.10,
    value=0.02,
    step=0.01,
    help="Higher epsilon allows a larger pixel perturbation."
)

st.sidebar.markdown("---")
st.sidebar.markdown("### About the Model")
st.sidebar.write("Architecture: ResNet-18")
st.sidebar.write("Weights: ImageNet pretrained")
st.sidebar.write("Attack: Untargeted FGSM")
st.sidebar.write(f"Device: {device.type.upper()}")

st.sidebar.markdown("---")
st.sidebar.markdown("### FGSM Formula")
st.latex(r"x_{adv}=clip(x+\epsilon \cdot sign(\nabla_x J),0,1)")


# --------------------------------------------------
# IMAGE UPLOAD
# --------------------------------------------------
uploaded_file = st.file_uploader(
    "📤 Upload an image",
    type=["jpg", "jpeg", "png", "webp"]
)

if uploaded_file is not None:
    try:
        pil_image = Image.open(uploaded_file).convert("RGB")
    except Exception:
        st.error("The uploaded file could not be read as an image.")
        st.stop()

    pixel_tensor = image_transform(pil_image).unsqueeze(0).to(device)

    st.markdown("---")
    st.header("1. Original Image")

    image_col, info_col = st.columns([1, 1])

    with image_col:
        st.image(
            pil_image,
            caption="Uploaded Image",
            use_container_width=True
        )

    original_index, original_conf, original_probs = predict_image(
        pixel_tensor
    )
    original_label = class_names[original_index]

    with info_col:
        st.markdown("### Original Prediction")
        st.success(f"Predicted Label: **{original_label}**")
        st.metric(
            "Original Confidence",
            f"{original_conf:.2f}%"
        )
        st.caption(
            "Confidence is the model's predicted probability for "
            "its top class, not a guarantee that the prediction is correct."
        )

    # --------------------------------------------------
    # RUN FGSM
    # --------------------------------------------------
    st.markdown("---")
    st.header("2. Adversarial Attack")

    st.write(f"Selected epsilon: **{epsilon:.2f}**")

    if st.button("🚀 Apply FGSM Attack", use_container_width=True):
        with st.spinner("Generating adversarial image and evaluating model..."):
            adversarial_tensor = fgsm_attack(
                pixel_tensor,
                epsilon,
                original_index
            )

            adversarial_index, adversarial_conf, adversarial_probs = (
                predict_image(adversarial_tensor)
            )
            adversarial_label = class_names[adversarial_index]

            adversarial_array = (
                adversarial_tensor[0]
                .detach()
                .cpu()
                .permute(1, 2, 0)
                .numpy()
            )
            adversarial_pil = Image.fromarray(
                (adversarial_array * 255).round().clip(0, 255).astype("uint8")
            )

            confidence_change = adversarial_conf - original_conf
            label_changed = original_index != adversarial_index

            # --------------------------------------------------
            # IMAGE COMPARISON
            # --------------------------------------------------
            st.markdown("---")
            st.header("3. Original vs. Adversarial Image")

            original_col, adversarial_col = st.columns(2)

            with original_col:
                st.image(
                    pixel_tensor[0].detach().cpu().permute(1, 2, 0).numpy(),
                    caption="Original (preprocessed)",
                    use_container_width=True
                )
                st.write(f"**Prediction:** {original_label}")
                st.write(f"**Confidence:** {original_conf:.2f}%")

            with adversarial_col:
                st.image(
                    adversarial_pil,
                    caption="FGSM Adversarial Image",
                    use_container_width=True
                )
                st.write(f"**Prediction:** {adversarial_label}")
                st.write(f"**Confidence:** {adversarial_conf:.2f}%")

            # --------------------------------------------------
            # SEMICIRCLE GAUGES
            # --------------------------------------------------
            st.markdown("---")
            st.header("4. Confidence Comparison")

            gauge_figure = create_confidence_gauges(
                original_conf,
                adversarial_conf
            )
            st.plotly_chart(
                gauge_figure,
                use_container_width=True
            )

            metric_col1, metric_col2, metric_col3 = st.columns(3)

            with metric_col1:
                st.metric(
                    "Original Confidence",
                    f"{original_conf:.2f}%"
                )

            with metric_col2:
                st.metric(
                    "Adversarial Confidence",
                    f"{adversarial_conf:.2f}%",
                    delta=f"{confidence_change:+.2f} percentage points"
                )

            with metric_col3:
                st.metric(
                    "Prediction Changed",
                    "YES" if label_changed else "NO"
                )

            # --------------------------------------------------
            # AUTOMATIC ANALYSIS
            # --------------------------------------------------
            st.markdown("---")
            st.header("5. Automatic Experiment Analysis")

            if label_changed:
                st.warning(
                    f"The predicted label changed from **{original_label}** "
                    f"to **{adversarial_label}** after applying FGSM. "
                    "This example demonstrates a change in the model's "
                    "prediction under the selected perturbation."
                )
            else:
                st.success(
                    f"The predicted label remained **{original_label}**. "
                    "For this image and epsilon, the attack did not change "
                    "the top-1 predicted class."
                )

            if confidence_change > 0:
                st.write(
                    f"The adversarial image's top-class confidence is "
                    f"{confidence_change:.2f} percentage points higher."
                )
            elif confidence_change < 0:
                st.write(
                    f"The adversarial image's top-class confidence is "
                    f"{abs(confidence_change):.2f} percentage points lower."
                )
            else:
                st.write("The top-class confidence did not change.")

            st.caption(
                "A single image is not sufficient to establish overall "
                "model robustness. Test multiple images and epsilon values. "
                "The model's confidence is not the same as accuracy."
            )

            # --------------------------------------------------
            # TOP-5 PREDICTIONS
            # --------------------------------------------------
            st.markdown("---")
            st.header("6. Top-5 Prediction Comparison")

            original_top5 = torch.topk(original_probs, 5)
            adversarial_top5 = torch.topk(adversarial_probs, 5)

            top5_df = pd.DataFrame({
                "Rank": range(1, 6),
                "Original Label": [
                    class_names[i] for i in original_top5.indices.tolist()
                ],
                "Original Confidence (%)": [
                    round(float(v) * 100, 2)
                    for v in original_top5.values.tolist()
                ],
                "Adversarial Label": [
                    class_names[i] for i in adversarial_top5.indices.tolist()
                ],
                "Adversarial Confidence (%)": [
                    round(float(v) * 100, 2)
                    for v in adversarial_top5.values.tolist()
                ]
            })

            st.dataframe(top5_df, use_container_width=True, hide_index=True)

            # --------------------------------------------------
            # DOWNLOAD REPORT
            # --------------------------------------------------
            st.markdown("---")
            st.header("7. Download Experiment Report")

            report = f"""
CYBERSHIELD AI
Robustness Against Adversarial Attacks
--------------------------------------

Experiment date and time: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

MODEL DETAILS
Architecture: ResNet-18
Pretrained weights: ImageNet
Attack: Untargeted Fast Gradient Sign Method (FGSM)
Epsilon: {epsilon:.2f}
Device: {device.type.upper()}

PREDICTION RESULTS
Original predicted label: {original_label}
Original top-class confidence: {original_conf:.2f}%

Adversarial predicted label: {adversarial_label}
Adversarial top-class confidence: {adversarial_conf:.2f}%

Prediction changed: {"Yes" if label_changed else "No"}
Confidence change (adversarial minus original):
{confidence_change:+.2f} percentage points

INTERPRETATION
{"The top-1 predicted label changed after the attack." if label_changed else "The top-1 predicted label remained the same after the attack."}

This report describes one experiment. It does not establish overall
model robustness or accuracy. Multiple images and perturbation levels
are needed for a broader evaluation.

FGSM FORMULA
x_adv = clip(x + epsilon * sign(gradient), 0, 1)

Here, x is the input image, epsilon controls perturbation strength,
and the gradient is calculated from the loss for the original
predicted class.
"""

            st.download_button(
                label="📥 Download Experiment Report (.txt)",
                data=report,
                file_name="CyberShield_AI_Experiment_Report.txt",
                mime="text/plain",
                use_container_width=True
            )

            csv_buffer = io.StringIO()
            top5_df.to_csv(csv_buffer, index=False)

            st.download_button(
                label="📊 Download Top-5 Results (.csv)",
                data=csv_buffer.getvalue(),
                file_name="CyberShield_AI_Top5_Results.csv",
                mime="text/csv",
                use_container_width=True
            )

else:
    st.info("👆 Upload an image above to begin the experiment.")

    st.markdown("---")
    st.header("How the Experiment Works")

    st.markdown("""
    1. **Input:** Upload an image.
    2. **Baseline:** ResNet-18 predicts the top class and confidence.
    3. **Attack:** FGSM adds a small gradient-based perturbation.
    4. **Evaluation:** The model predicts the perturbed image again.
    5. **Comparison:** Review the labels, confidence gauges, and results.
    6. **Report:** Download the experiment summary and top-5 predictions.
    """)
