import streamlit as st
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from rembg import remove
from streamlit_drawable_canvas import st_canvas
import mediapipe as mp
from io import BytesIO

st.set_page_config(page_title="Bike Analysis", layout="wide")
st.title("🚴‍♂️ Cyclist Position Analysis")

# ==========================
# HELPERS
# ==========================
def euclid_len(p1, p2):
    return float(np.hypot(p2[0] - p1[0], p2[1] - p1[1]))

def angle(a, b, c):
    """Angle ABC (degrees) – vertex at point B."""
    ang = np.degrees(
        np.arctan2(c[1] - b[1], c[0] - b[0]) -
        np.arctan2(a[1] - b[1], a[0] - b[0])
    )
    ang = ang if ang >= 0 else ang + 360
    if ang > 180:
        ang = 360 - ang
    return float(np.round(ang, 1))

def draw_line(img, p1, p2, color=(0, 255, 0), thickness=2):
    cv2.line(img, p1, p2, color, thickness)

def draw_point(img, p, name=None, color=(0, 0, 255)):
    cv2.circle(img, p, 5, color, -1)
    if name:
        cv2.putText(img, name, (p[0] + 6, p[1] - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

def draw_angle_text(img, b, text, color=(255, 0, 0)):
    cv2.putText(img, text, (b[0] + 10, b[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

# --- tabs ---
tab1, tab2 = st.tabs(["📸 Frontal Analysis", "📐 Bike fitting (side view)"])

# ======================================================
# TAB 1 – FRONTAL AREA
# ======================================================
with tab1:
    st.header("Frontal area analysis + metrics")

    wheel_diameter_m = st.number_input("Wheel diameter [m]", min_value=0.5, max_value=0.8, value=0.622, step=0.001)
    uploaded_files = st.file_uploader("Upload frontal photos", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

    results = []

    for file in uploaded_files:
        image = Image.open(file).convert("RGB")
        img_array = np.array(image)

        segmented_img = remove(image)
        segmented_img = Image.open(segmented_img) if not isinstance(segmented_img, Image.Image) else segmented_img
        seg_array = np.array(segmented_img)

        if seg_array.shape[2] == 4:
            alpha = seg_array[:, :, 3]
            binary_mask = alpha > 0
            pink_bg = np.full(seg_array.shape, [255, 0, 255, 255], dtype=np.uint8)
            a = alpha / 255.0
            for c in range(3):
                pink_bg[:, :, c] = (seg_array[:, :, c] * a + pink_bg[:, :, c] * (1 - a)).astype(np.uint8)
            seg_array = pink_bg[:, :, :3]
        else:
            binary_mask = np.sum(seg_array, axis=2) > 0

        frontal_area_px = int(np.sum(binary_mask))

        st.subheader(f"📸 {file.name}")
        st.markdown("""
        ✏️ Draw **three lines** in this order:  
        1) 🔵 **Wheel** (vertical wheel diameter)  
        2) 🟢 **Elbow–elbow** (elbow width)  
        3) 🔴 **Head–handlebar** (vertical difference)  
        """)

        canvas_result = st_canvas(
            fill_color="rgba(255, 0, 0, 0.3)",
            stroke_width=3,
            stroke_color="blue",
            background_image=image,
            update_streamlit=True,
            height=img_array.shape[0],
            width=img_array.shape[1],
            drawing_mode="line",
            key=f"front_{file.name}"
        )

        scale = None
        elbow_cm, head_cm = None, None

        if canvas_result.json_data is not None and len(canvas_result.json_data["objects"]) >= 3:
            objects = canvas_result.json_data["objects"]

            # line 1: wheel
            x1, y1 = objects[0]["x1"], objects[0]["y1"]
            x2, y2 = objects[0]["x2"], objects[0]["y2"]
            pixel_length_wheel = float(np.hypot(x2 - x1, y2 - y1))
            if pixel_length_wheel > 0:
                scale = wheel_diameter_m / pixel_length_wheel

            # line 2: elbows
            x1, y1 = objects[1]["x1"], objects[1]["y1"]
            x2, y2 = objects[1]["x2"], objects[1]["y2"]
            pixel_length_elbows = float(np.hypot(x2 - x1, y2 - y1))
            if scale:
                elbow_cm = round(pixel_length_elbows * scale * 100, 1)

            # line 3: head–handlebar
            x1, y1 = objects[2]["x1"], objects[2]["y1"]
            x2, y2 = objects[2]["x2"], objects[2]["y2"]
            pixel_length_head = float(np.hypot(x2 - x1, y2 - y1))
            if scale:
                head_cm = round(pixel_length_head * scale * 100, 1)

            frontal_area_m2 = round(frontal_area_px * (scale ** 2), 4) if scale else None

            results.append({
                "File": file.name,
                "Area [px²]": frontal_area_px,
                "Area [m²]": frontal_area_m2,
                "Elbows [cm]": elbow_cm,
                "Head–Handlebar [cm]": head_cm
            })

            st.image(seg_array, caption="Segmentation (U²-Net – cyclist + bike)")

    if results:
        df = pd.DataFrame(results).sort_values("Area [m²]", na_position="last")
        st.subheader("📊 Position ranking")
        st.dataframe(df, use_container_width=True)
        df_valid = df.dropna(subset=["Area [m²]"])
        if not df_valid.empty:
            best = df_valid.iloc[0]
            st.success(
                f"🏆 Best: {best['File']} → {best['Area [m²]']} m², "
                f"Elbows: {best['Elbows [cm]']} cm, Head–Handlebar: {best['Head–Handlebar [cm]']} cm"
            )

# ======================================================
# TAB 2 – BIKE FITTING (SIDE VIEW)
# ======================================================
with tab2:
    st.header("Bike fitting – side view (drag & drop points)")

    # --- thresholds settings ---
    with st.expander("⚙️ Angle thresholds settings"):
        knee_min = st.number_input("Knee angle – min", 100, 180, 140)
        knee_max = st.number_input("Knee angle – max", 100, 180, 150)

        elbow_min = st.number_input("Elbow angle – min", 100, 180, 140)
        elbow_max = st.number_input("Elbow angle – max", 100, 180, 160)

        hip_min = st.number_input("Hip angle – min", 30, 120, 40)
        hip_max = st.number_input("Hip angle – max", 30, 120, 55)

        shoulder_min = st.number_input("Shoulder angle – min", 70, 160, 90)
        shoulder_max = st.number_input("Shoulder angle – max", 70, 160, 110)

        neck_min = st.number_input("Torso/neck angle – min", 60, 120, 85)
        neck_max = st.number_input("Torso/neck angle – max", 60, 120, 95)

    uploaded_side = st.file_uploader("Upload side view photo", type=["jpg", "jpeg", "png"], accept_multiple_files=False)

    if uploaded_side:
        image = Image.open(uploaded_side).convert("RGB")
        img_array = np.array(image)
        h, w, _ = img_array.shape

        mp_pose = mp.solutions.pose
        with mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5) as pose:
            results_pose = pose.process(cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR))

        required_points = ["Shoulder", "Elbow", "Wrist", "Hip", "Knee", "Ankle", "Ear"]

        if results_pose.pose_landmarks:
            lm = results_pose.pose_landmarks.landmark
            def gp(idx): return int(lm[idx].x * w), int(lm[idx].y * h)
            auto_points = {
                "Shoulder": gp(mp_pose.PoseLandmark.RIGHT_SHOULDER.value),
                "Elbow": gp(mp_pose.PoseLandmark.RIGHT_ELBOW.value),
                "Wrist": gp(mp_pose.PoseLandmark.RIGHT_WRIST.value),
                "Hip": gp(mp_pose.PoseLandmark.RIGHT_HIP.value),
                "Knee": gp(mp_pose.PoseLandmark.RIGHT_KNEE.value),
                "Ankle": gp(mp_pose.PoseLandmark.RIGHT_ANKLE.value),
                "Ear": gp(mp_pose.PoseLandmark.RIGHT_EAR.value),
            }
        else:
            st.info("ℹ️ No automatic detection – inserted initial points, move them manually on the photo.")
            auto_points = {
                "Shoulder": (int(w * 0.35), int(h * 0.35)),
                "Elbow":   (int(w * 0.50), int(h * 0.48)),
                "Wrist":   (int(w * 0.60), int(h * 0.62)),
                "Hip":     (int(w * 0.40), int(h * 0.60)),
                "Knee":    (int(w * 0.45), int(h * 0.78)),
                "Ankle":   (int(w * 0.48), int(h * 0.94)),
                "Ear":     (int(w * 0.45), int(h * 0.22)),
            }

        radius = 6
        init_objects = []
        for name in required_points:
            x, y = auto_points[name]
            init_objects.append({
                "type": "circle",
                "left": x - radius,
                "top": y - radius,
                "radius": radius,
                "fill": "red",
                "stroke": "red",
                "strokeWidth": 1,
                "name": name
            })

        st.markdown("✏️ **Drag** the red points to correct places. Angles and lines recalculate automatically.")

        canvas_result = st_canvas(
            background_image=image,
            initial_drawing={"objects": init_objects},
            update_streamlit=True,
            height=h,
            width=w,
            drawing_mode="transform",
            key="pose_canvas_drag"
        )

        final_points = {}
        if canvas_result.json_data is not None and "objects" in canvas_result.json_data:
            objs = [o for o in canvas_result.json_data["objects"] if o.get("type") == "circle"]
            for idx, name in enumerate(required_points):
                if idx < len(objs):
                    o = objs[idx]
                    r = int(o.get("radius", radius))
                    cx = int(o.get("left", 0) + r)
                    cy = int(o.get("top", 0) + r)
                    final_points[name] = (cx, cy)
                else:
                    final_points[name] = auto_points[name]
        else:
            final_points = auto_points

        img_draw = img_array.copy()
        seg_pairs = [
            ("Shoulder", "Elbow"),
            ("Elbow", "Wrist"),
            ("Shoulder", "Hip"),
            ("Hip", "Knee"),
            ("Knee", "Ankle"),
            ("Shoulder", "Ear"),
        ]
        for a, b in seg_pairs:
            draw_line(img_draw, final_points[a], final_points[b], (0, 255, 0), 2)
        for name, pt in final_points.items():
            draw_point(img_draw, pt, name, (0, 0, 255))

        knee_angle = angle(final_points["Hip"], final_points["Knee"], final_points["Ankle"])
        elbow_angle = angle(final_points["Shoulder"], final_points["Elbow"], final_points["Wrist"])
        back_angle  = angle(final_points["Shoulder"], final_points["Hip"], final_points["Knee"])
        neck_back   = angle(final_points["Ear"], final_points["Shoulder"], final_points["Hip"])
        shoulder_angle = angle(final_points["Hip"], final_points["Shoulder"], final_points["Elbow"])

        draw_angle_text(img_draw, final_points["Knee"], f"{knee_angle}")
        draw_angle_text(img_draw, final_points["Elbow"], f"{elbow_angle}")
        draw_angle_text(img_draw, final_points["Hip"], f"{back_angle}")
        draw_angle_text(img_draw, final_points["Ear"], f"{neck_back}")
        draw_angle_text(img_draw, final_points["Shoulder"], f"{shoulder_angle}")

        st.image(img_draw, caption="Points, lines and angles (live)")

        pts_df = pd.DataFrame(
            [{"Point": name, "x": pt[0], "y": pt[1]} for name, pt in final_points.items()]
        )
        ang_df = pd.DataFrame([{
            "Knee angle (Hip–Knee–Ankle)": knee_angle,
            "Elbow angle (Shoulder–Elbow–Wrist)": elbow_angle,
            "Hip angle / back (Shoulder–Hip–Knee)": back_angle,
            "Torso/neck (Ear–Shoulder–Hip)": neck_back,
            "Shoulder angle (Hip–Shoulder–Elbow)": shoulder_angle,
        }])

        st.subheader("📍 Point positions")
        st.dataframe(pts_df, use_container_width=True)

        st.subheader("📐 Position angles")
        st.dataframe(ang_df, use_container_width=True)

        # export CSV
        export_df = pts_df.copy()
        for col, val in ang_df.iloc[0].items():
            export_df[col] = val
        csv_bytes = export_df.to_csv(index=False).encode("utf-8")
        st.download_button("💾 Download data (CSV)", data=csv_bytes, file_name="bike_fitting.csv", mime="text/csv")

        # export PNG
        img_pil = Image.fromarray(cv2.cvtColor(img_draw, cv2.COLOR_BGR2RGB))
        buf = BytesIO()
        img_pil.save(buf, format="PNG")
        st.download_button("🖼️ Download image (PNG)", data=buf.getvalue(), file_name="bike_fitting.png", mime="image/png")

        # angle ratings
        def rate_angle(val, min_ok, max_ok):
            if min_ok <= val <= max_ok:
                return "✅ Optimal"
            elif (val < min_ok - 5) or (val > max_ok + 5):
                return "❌ Poor"
            else:
                return "⚠️ Needs improvement"

        ratings = {
            "Knee angle": rate_angle(knee_angle, knee_min, knee_max),
            "Elbow angle": rate_angle(elbow_angle, elbow_min, elbow_max),
            "Hip angle": rate_angle(back_angle, hip_min, hip_max),
            "Torso/neck angle": rate_angle(neck_back, neck_min, neck_max),
            "Shoulder angle": rate_angle(shoulder_angle, shoulder_min, shoulder_max),
        }

        st.subheader("📊 Angle assessment")
        ratings_df = pd.DataFrame([ratings])
        st.dataframe(ratings_df, use_container_width=True)

    else:
        st.info("Upload a side view photo to start fitting.")
