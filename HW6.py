"""Homework #6 — Streamlit object counting (HW05-style workflow).

Run locally: py -3.13 -m streamlit run HW6.py
Dependencies: streamlit, numpy, Pillow, scipy
"""

from io import BytesIO

import numpy as np
import streamlit as st
from PIL import Image, ImageFilter, ImageOps
from scipy import ndimage as ndi


st.set_page_config(page_title="HW6 Object Counting", layout="wide")
st.title("Homework #6 — Object Counting")
st.write("이미지를 업로드해 이진화, 잡음 제거, Connected Component 분석으로 물체 개수를 계산합니다.")

with st.sidebar:
    st.header("설정")
    uploaded = st.file_uploader("분석할 이미지 업로드", type=["jpg", "jpeg", "png"])
    threshold = st.slider("Threshold 값", 0, 255, 128, 1)
    target = st.radio("물체의 밝기", ["어두운 물체", "밝은 물체"], index=0)
    median_size = st.select_slider("Median Filter 크기", options=[1, 3, 5, 7], value=3,
                                   help="1은 필터 적용 안 함. 크기가 너무 크면 작은 물체가 사라질 수 있어요.")
    connectivity = st.radio("Connected Component 연결 기준", [4, 8], horizontal=True, index=0)
    remove_border = st.checkbox("테두리에 닿은 영역 제거", value=True,
                                help="오른쪽 위와 같이 영상 경계에 붙은 잡음 제거에 유용합니다. 실제 물체도 테두리에 닿으면 없어집니다.")
    min_pixels = st.number_input("최소 물체 크기 (픽셀)", min_value=1, max_value=1000000,
                                 value=60, step=10,
                                 help="이 값보다 작은 Connected Component는 점 잡음으로 간주합니다.")


def analyze(image: Image.Image, threshold: int, target: str, median_size: int,
            connectivity: int, remove_border: bool, min_pixels: int):
    """Return grayscale, raw mask, cleaned mask, labels and areas."""
    gray_image = ImageOps.grayscale(image)
    if median_size > 1:
        gray_image = gray_image.filter(ImageFilter.MedianFilter(size=median_size))
    gray = np.asarray(gray_image, dtype=np.uint8)

    raw_mask = (gray < threshold) if target == "어두운 물체" else (gray > threshold)
    structure = ndi.generate_binary_structure(2, 1 if connectivity == 4 else 2)
    initial_labels, count = ndi.label(raw_mask, structure=structure)
    keep = np.ones(count + 1, dtype=bool)
    keep[0] = False

    if remove_border and count:
        touching = np.unique(np.concatenate([
            initial_labels[0, :], initial_labels[-1, :],
            initial_labels[:, 0], initial_labels[:, -1]
        ]))
        keep[touching] = False

    sizes = np.bincount(initial_labels.ravel(), minlength=count + 1)
    keep[sizes < min_pixels] = False
    keep[0] = False
    clean_mask = keep[initial_labels]

    # Re-label after filtering; all subsequent measurements use the CLEAN mask.
    labels, n_objects = ndi.label(clean_mask, structure=structure)
    areas = np.bincount(labels.ravel(), minlength=n_objects + 1)[1:]
    return gray, raw_mask, clean_mask, labels, areas


def labels_to_rgb(labels: np.ndarray, num: int):
    rgb = np.zeros((*labels.shape, 3), dtype=np.uint8)
    if num == 0:
        return rgb
    # Deterministic distinct colors; background remains black.
    from colorsys import hsv_to_rgb
    for lab in range(1, num + 1):
        hue = (lab * 0.61803398875) % 1
        rgb[labels == lab] = np.array(hsv_to_rgb(hue, 0.75, 1.0)) * 255
    return rgb


if uploaded is None:
    st.info("왼쪽에서 JPG/PNG 이미지를 업로드하면 분석 결과가 나타납니다.")
    st.stop()

try:
    image = ImageOps.exif_transpose(Image.open(uploaded)).convert("RGB")
    gray, raw_mask, clean_mask, labels, areas = analyze(
        image, threshold, target, median_size, connectivity, remove_border, int(min_pixels)
    )
except Exception as error:
    st.error(f"이미지를 처리하지 못했습니다: {error}")
    st.stop()

st.metric("검출된 물체 개수", len(areas))
col1, col2 = st.columns(2)
with col1:
    st.subheader("① 원본")
    st.image(image, use_container_width=True)
with col2:
    st.subheader("② Grayscale / Median")
    st.image(gray, clamp=True, use_container_width=True)
col3, col4 = st.columns(2)
with col3:
    st.subheader("③ Threshold 결과")
    st.image((raw_mask.astype(np.uint8) * 255), clamp=True, use_container_width=True)
with col4:
    st.subheader("④ 잡음 제거 후 마스크")
    st.image((clean_mask.astype(np.uint8) * 255), clamp=True, use_container_width=True)

st.subheader("⑤ Connected Components")
st.image(labels_to_rgb(labels, len(areas)), use_container_width=True)

if len(areas):
    st.write(f"가장 큰 물체: **{int(areas.max()):,}픽셀**, "
             f"가장 작은 물체: **{int(areas.min()):,}픽셀**")
    with st.expander("물체별 픽셀 개수 보기", expanded=False):
        st.dataframe([{"Object": i, "Pixels": int(area)}
                      for i, area in enumerate(areas, 1)],
                     hide_index=True, use_container_width=True)
else:
    st.warning("검출된 물체가 없습니다. Threshold나 최소 크기, 테두리 제거 설정을 조정해 보세요.")

buf = BytesIO()
Image.fromarray((clean_mask.astype(np.uint8) * 255)).save(buf, format="PNG")
st.download_button("결과 마스크 PNG 다운로드", data=buf.getvalue(),
                   file_name="HW6_mask.png", mime="image/png")
st.caption("참고: 이 알고리즘은 밝기 기준으로 분리되는 물체에 적합합니다. 배경과 겹치거나 닿아 있는 물체는 하나로 셀 수 있습니다.")
