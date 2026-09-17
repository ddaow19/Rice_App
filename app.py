# ════
# FF69_Rice | Smart Agriculture — Streamlit Prototype Application
# ════

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import folium
from streamlit_folium import st_folium

# ────
# PAGE CONFIG
# ────
st.set_page_config(
    page_title="FF69_Rice | Smart Agriculture",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ────
# LOAD MODEL & ARTIFACTS (Cache เพื่อไม่ต้องโหลดซ้ำทุกครั้ง)
# ────
@st.cache_resource
def load_artifacts():
    base_dir = os.path.dirname(__file__)

    # ชื่อไฟล์โมเดลที่รองรับ (ลองตามลำดับ เพื่อรองรับทั้งไฟล์ DT และ SVM)
    candidate_model_files = ["best_model_DT.pkl", "best_model_SVM.pkl", "model.pkl"]
    model_path = None
    for fname in candidate_model_files:
        full_path = os.path.join(base_dir, fname)
        if os.path.exists(full_path):
            model_path = full_path
            break
    if model_path is None:
        raise FileNotFoundError(
            f"ไม่พบไฟล์โมเดล (ลองหาแล้ว: {', '.join(candidate_model_files)}) "
            f"ในโฟลเดอร์ {base_dir}"
        )

    with open(model_path, "rb") as f:
        model = pickle.load(f)
    with open(os.path.join(base_dir, "scaler.pkl"), "rb") as f:
        scaler = pickle.load(f)
    with open(os.path.join(base_dir, "label_encoder.pkl"), "rb") as f:
        le = pickle.load(f)

    feature_names = None
    feature_path = os.path.join(base_dir, "feature_names.pkl")
    if os.path.exists(feature_path):
        with open(feature_path, "rb") as f:
            feature_names = pickle.load(f)

    return model, scaler, le, feature_names


LOAD_ERROR = None
try:
    model, scaler, le, feature_names = load_artifacts()
    MODEL_LOADED = True
except Exception as e:
    MODEL_LOADED = False
    LOAD_ERROR = str(e)
    # ต้องกำหนดตัวแปรว่างไว้ เพื่อไม่ให้ NameError เวลาโค้ดส่วนอื่นอ้างถึง
    model, scaler, le, feature_names = None, None, None, None

# ────
# FERTILIZER RECOMMENDATION LOGIC
# ────
CLASS_COLORS = {"S1": "#15803D", "S2": "#EAB308", "S3": "#F97316", "N": "#DC2626"}
CLASS_LABELS = {
    "S1": "เหมาะสมสูง",
    "S2": "เหมาะสมปานกลาง",
    "S3": "เหมาะสมน้อย",
    "N": "ไม่เหมาะสม"
}

def fertilizer_recommendation(suitability_class):
    rec_map = {
        "S1": ["สภาพดินดีมาก รักษาระดับปุ๋ยตามค่าวิเคราะห์ดินเดิม"],
        "S2": ["ควรเพิ่มปุ๋ยไนโตรเจน (N) หรือโพแทสเซียม (K) ตามค่าวิเคราะห์"],
        "S3": ["ควรปรับปรุงคุณสมบัติดินก่อนเพาะปลูก เช่น ปรับ pH หรือเพิ่มอินทรียวัตถุ"],
        "N":  ["ไม่แนะนำให้ปลูกในสภาพดินปัจจุบัน ควรตรวจวิเคราะห์ดินเพิ่มเติม"]
    }
    return rec_map.get(suitability_class, ["ไม่มีคำแนะนำ"])


def calculate_custom_fertilizer(soil_texture, n_level, p_ppm, k_ppm):
    n_req_map = {"ต่ำมาก": 12.0, "ต่ำ": 9.0, "ปานกลาง": 6.0, "สูง": 3.0}
    n_req = n_req_map.get(n_level, 6.0)

    if p_ppm < 5:
        p_req = 6.0
    elif p_ppm <= 15:
        p_req = 4.0
    else:
        p_req = 0.0

    if k_ppm < 40:
        k_req = 6.0 if "เหนียว" in soil_texture else 8.0
    elif k_ppm <= 80:
        k_req = 3.0 if "เหนียว" in soil_texture else 4.0
    else:
        k_req = 0.0

    dap_kg = (p_req / 0.46) if p_req > 0 else 0.0
    n_from_dap = dap_kg * 0.18
    mop_kg = (k_req / 0.60) if k_req > 0 else 0.0
    n_remain = max(0.0, n_req - n_from_dap)
    urea_kg = (n_remain / 0.46) if n_remain > 0 else 0.0

    return {
        "46-0-0 (ยูเรีย)": round(urea_kg, 1),
        "18-46-0 (DAP)": round(dap_kg, 1),
        "0-0-60 (MOP)": round(mop_kg, 1)
    }

# ────
# PREDICTION FUNCTION (Robust ต่อจำนวน feature)
# ────
def predict_suitability(input_df):
    if feature_names is not None:
        X = input_df.copy()
        for col in feature_names:
            if col not in X.columns:
                X[col] = np.nan
        X = X[feature_names]
    else:
        X = input_df.copy()

    n_target = getattr(model, "n_features_in_", X.shape[1])

    if X.shape[1] < n_target and hasattr(scaler, "mean_"):
        means = scaler.mean_
        for i in range(X.shape[1], n_target):
            X[f"_pad_{i}"] = means[i] if i < len(means) else 0.0

    if X.shape[1] > n_target:
        X = X.iloc[:, :n_target]

    X_scaled = scaler.transform(X.values.astype(float))
    y_pred_enc = model.predict(X_scaled)
    return le.inverse_transform(y_pred_enc)

# ────
# SIDEBAR NAVIGATION
# ────
st.sidebar.title("🌾 FF69_Rice")
st.sidebar.caption("Smart Agriculture System")

menu = st.sidebar.radio(
    "เมนู",
    ["Dashboard", "Map View", "Suitability Prediction", "ปุ๋ยสั่งตัด (SSNM)"]
)

if not MODEL_LOADED:
    st.sidebar.error(f"โหลดโมเดลไม่สำเร็จ: {LOAD_ERROR}")

# ────
# PAGE: DASHBOARD
# ────
if menu == "Dashboard":
    st.title("Dashboard ภาพรวมแปลงเกษตร")
    st.caption("สรุปผลวิเคราะห์จากข้อมูลจุดสำรวจล่าสุดในเมนู Map View")

    # ตรวจสอบว่ามีข้อมูลที่สร้างจากหน้า Map View แล้วหรือไม่
    map_data = st.session_state.get("map_data", None)

    if map_data is None or map_data.empty:
        st.info(
            "ยังไม่มีข้อมูลสำหรับแสดงผล กรุณาไปที่เมนู **Map View** "
            "กำหนดจำนวนจุดข้อมูล และกดปุ่ม **สร้างข้อมูลและทำนายผล** ก่อน"
        )

    else:
        data = map_data.copy()

        # ป้องกันกรณีไม่มีคอลัมน์ Suitability
        if "Suitability" not in data.columns:
            data["Suitability"] = "N"

        # ────
        # SUMMARY METRICS
        # ────
        total_points = len(data)
        avg_ph = data["pH"].mean()
        avg_moisture = data["Soil Moisture (%)"].mean()
        avg_temperature = data["Temperature (°C)"].mean()

        st.subheader("สรุปข้อมูลล่าสุด")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                label="จุดสำรวจทั้งหมด",
                value=f"{total_points} จุด"
            )

        with col2:
            st.metric(
                label="ค่า pH เฉลี่ย",
                value=f"{avg_ph:.2f}"
            )

        with col3:
            st.metric(
                label="ความชื้นดินเฉลี่ย",
                value=f"{avg_moisture:.1f}%"
            )

        with col4:
            st.metric(
                label="อุณหภูมิเฉลี่ย",
                value=f"{avg_temperature:.1f} °C"
            )

        st.markdown("")

        # ────
        # SUITABILITY SUMMARY
        # ────
        suitability_order = ["S1", "S2", "S3", "N"]

        suitability_summary = (
            data["Suitability"]
            .astype(str)
            .value_counts()
            .reindex(suitability_order, fill_value=0)
            .rename_axis("ระดับความเหมาะสม")
            .reset_index(name="จำนวนจุด")
        )

        suitability_summary["คำอธิบาย"] = suitability_summary[
            "ระดับความเหมาะสม"
        ].map(CLASS_LABELS)

        # หาจำนวนจุดที่เหมาะสมสูงและปานกลาง
        suitable_points = int(
            suitability_summary.loc[
                suitability_summary["ระดับความเหมาะสม"].isin(["S1", "S2"]),
                "จำนวนจุด"
            ].sum()
        )

        suitable_percent = (
            (suitable_points / total_points) * 100
            if total_points > 0 else 0
        )

        left_col, right_col = st.columns([1.15, 1])

        with left_col:
            st.subheader("การกระจายระดับความเหมาะสม")

            st.bar_chart(
                suitability_summary,
                x="ระดับความเหมาะสม",
                y="จำนวนจุด",
                color="#15803D",
                use_container_width=True
            )

            st.caption(
                "S1 = เหมาะสมสูง | S2 = เหมาะสมปานกลาง | "
                "S3 = เหมาะสมน้อย | N = ไม่เหมาะสม"
            )

        with right_col:
            st.subheader("ภาพรวมการเพาะปลูก")

            st.metric(
                label="พื้นที่ที่อยู่ในระดับเหมาะสม (S1 + S2)",
                value=f"{suitable_percent:.1f}%",
                delta=f"{suitable_points} จาก {total_points} จุด"
            )

            st.markdown("#### คำแนะนำเบื้องต้น")

            dominant_class = suitability_summary.loc[
                suitability_summary["จำนวนจุด"].idxmax(),
                "ระดับความเหมาะสม"
            ]

            dominant_label = CLASS_LABELS.get(dominant_class, "ไม่ระบุ")

            st.success(
                f"ผลการประเมินส่วนใหญ่อยู่ในระดับ **{dominant_class} "
                f"({dominant_label})**"
            )

            recommendations = fertilizer_recommendation(dominant_class)

            for recommendation in recommendations:
                st.write(f"• {recommendation}")

        st.markdown("")

        # ────
        # SOIL NUTRIENT CHART
        # ────
        st.subheader("เปรียบเทียบธาตุอาหารในดินรายจุดสำรวจ")

        nutrient_chart = data[
            ["Nitrogen", "Phosphorus", "Potassium"]
        ].copy()

        nutrient_chart.index = [
            f"จุดที่ {index + 1}"
            for index in range(len(nutrient_chart))
        ]

        st.bar_chart(
            nutrient_chart,
            color=["#2563EB", "#EAB308", "#F97316"],
            use_container_width=True
        )

        st.caption(
            "กราฟแสดงค่าธาตุอาหารหลัก: "
            "Nitrogen (N), Phosphorus (P) และ Potassium (K)"
        )

        # ────
        # SOIL / ENVIRONMENT TREND
        # ────
        st.subheader("แนวโน้มสภาพดินและสภาพแวดล้อมรายจุด")

        environment_chart = data[
            ["pH", "Soil Moisture (%)", "Temperature (°C)"]
        ].copy()

        environment_chart.index = [
            f"จุดที่ {index + 1}"
            for index in range(len(environment_chart))
        ]

        st.line_chart(
            environment_chart,
            color=["#7C3AED", "#15803D", "#DC2626"],
            use_container_width=True
        )

        # ────
        # DATA TABLE
        # ────
        with st.expander("ดูตารางข้อมูลจุดสำรวจทั้งหมด"):
            display_columns = [
                "Latitude",
                "Longitude",
                "pH",
                "Nitrogen",
                "Phosphorus",
                "Potassium",
                "Soil Moisture (%)",
                "Temperature (°C)",
                "Suitability"
            ]

            st.dataframe(
                data[display_columns],
                use_container_width=True,
                hide_index=True
            )

# ────
# PAGE: MAP VIEW
# ────
# ────
# PAGE: MAP VIEW
# ────
elif menu == "Map View":
    st.title("แผนที่ระดับความเหมาะสมรายแปลง")

    num_points = st.slider("จำนวนจุดข้อมูลจำลอง (IoT)", 3, 50, 10)

    # สร้างตัวแปรเก็บข้อมูลแผนที่ไว้ใน session
    if "map_data" not in st.session_state:
        st.session_state.map_data = None

    if st.button("สร้างข้อมูลและทำนายผล", type="primary"):
        lat_c, lon_c = 14.99455672384764, 103.1963208119938

        data = pd.DataFrame({
            "Latitude": lat_c + np.random.uniform(-0.01, 0.01, num_points),
            "Longitude": lon_c + np.random.uniform(-0.01, 0.01, num_points),
            "pH": np.random.uniform(5.5, 7.5, num_points),
            "Nitrogen": np.random.uniform(0.1, 0.5, num_points),
            "Phosphorus": np.random.uniform(0.05, 0.3, num_points),
            "Potassium": np.random.uniform(0.1, 0.4, num_points),
            "Soil Moisture (%)": np.random.uniform(20, 60, num_points),
            "Temperature (°C)": np.random.uniform(25, 35, num_points),
        })

        if MODEL_LOADED:
            try:
                input_data = data.drop(columns=["Latitude", "Longitude"])
                data["Suitability"] = predict_suitability(input_data)
            except Exception as e:
                st.warning(
                    f"โมเดลทำนายผลไม่ได้ จะแสดงจุดบนแผนที่เป็นสถานะ N แทน: {e}"
                )
                data["Suitability"] = "N"
        else:
            st.warning("ไม่พบโมเดล จะแสดงจุดบนแผนที่เป็นสถานะ N")
            data["Suitability"] = "N"

        # เก็บข้อมูลไว้ เพื่อให้แผนที่ไม่หายหลัง Streamlit rerun
        st.session_state.map_data = data

    # แสดงผลหลังมีการกดปุ่มแล้ว
    if st.session_state.map_data is not None:
        data = st.session_state.map_data

        lat_c = data["Latitude"].mean()
        lon_c = data["Longitude"].mean()

        m = folium.Map(
            location=[lat_c, lon_c],
            zoom_start=14,
            tiles="OpenStreetMap"
        )

        for _, row in data.iterrows():
            suitability = str(row["Suitability"])
            color = CLASS_COLORS.get(suitability, "#6B7280")

            popup_html = f"""
                <b>ผลประเมิน: {suitability}</b><br>
                pH: {row["pH"]:.2f}<br>
                Nitrogen: {row["Nitrogen"]:.3f}<br>
                Phosphorus: {row["Phosphorus"]:.3f}<br>
                Potassium: {row["Potassium"]:.3f}<br>
                ความชื้นดิน: {row["Soil Moisture (%)"]:.1f}%<br>
                อุณหภูมิ: {row["Temperature (°C)"]:.1f} °C
            """

            folium.CircleMarker(
                location=[row["Latitude"], row["Longitude"]],
                radius=8,
                color=color,
                fill=True,
                fill_color=color,
                fill_opacity=0.80,
                tooltip=f"ระดับความเหมาะสม: {suitability}",
                popup=folium.Popup(popup_html, max_width=280)
            ).add_to(m)

        st.subheader("ผลลัพธ์บนแผนที่")
        st_folium(
            m,
            use_container_width=True,
            height=550,
            key="rice_suitability_map"
        )

        st.subheader("ตารางข้อมูลจุดสำรวจ")
        st.dataframe(data, use_container_width=True)

    else:
        st.info("กำหนดจำนวนจุดข้อมูล แล้วกดปุ่ม “สร้างข้อมูลและทำนายผล” เพื่อแสดงแผนที่")

# ────
# PAGE: SUITABILITY PREDICTION (Manual Input)
# ────
elif menu == "Suitability Prediction":
    st.title("ทำนายระดับความเหมาะสมของพื้นที่ปลูกข้าว")

    with st.form("prediction_form"):
        c1, c2, c3 = st.columns(3)
        ph = c1.number_input("pH", 3.0, 9.0, 6.4, 0.1)
        n_val = c1.number_input("Nitrogen", 0.0, 1.0, 0.25, 0.01)
        p_val = c2.number_input("Phosphorus", 0.0, 1.0, 0.15, 0.01)
        k_val = c2.number_input("Potassium", 0.0, 1.0, 0.20, 0.01)
        moisture = c3.number_input("Soil Moisture (%)", 0.0, 100.0, 40.0, 1.0)
        temp = c3.number_input("Temperature (°C)", 0.0, 50.0, 30.0, 0.5)

        submitted = st.form_submit_button("ทำนายผล")

    if submitted:
        input_df = pd.DataFrame({
            "pH": [ph], "Nitrogen": [n_val], "Phosphorus": [p_val],
            "Potassium": [k_val], "Soil Moisture (%)": [moisture],
            "Temperature (°C)": [temp]
        })

        if MODEL_LOADED:
            try:
                pred = predict_suitability(input_df)[0]
                color = CLASS_COLORS.get(pred, "#6B7280")
                label = CLASS_LABELS.get(pred, pred)

                st.markdown(
                    f"""<div style="padding:16px;border:2px solid {color};
                    background-color:{color}22;border-radius:6px;">
                    <h3 style="color:{color};margin:0;">ผลการทำนาย: {pred} — {label}</h3>
                    </div>""",
                    unsafe_allow_html=True
                )

                st.subheader("คำแนะนำ")
                for rec in fertilizer_recommendation(pred):
                    st.write(f"- {rec}")

            except Exception as e:
                st.error(f"เกิดข้อผิดพลาดในการทำนาย: {e}")
        else:
            st.error("ไม่สามารถทำนายได้ เนื่องจากไม่พบไฟล์โมเดล")

# ────
# PAGE: SSNM FERTILIZER CALCULATOR
# ────
elif menu == "ปุ๋ยสั่งตัด (SSNM)":
    st.title("คำนวณปุ๋ยสั่งตัดตามค่าวิเคราะห์ดิน")
    st.caption("Site-Specific Nutrient Management (SSNM)")

    c1, c2 = st.columns(2)
    with c1:
        soil_texture = st.selectbox("ลักษณะเนื้อดิน", ["ดินเหนียว", "ดินทราย/ดินร่วน"])
        n_level = st.selectbox("ระดับไนโตรเจน (จาก OM)", ["ต่ำมาก", "ต่ำ", "ปานกลาง", "สูง"])
    with c2:
        p_ppm = st.number_input("ค่าฟอสฟอรัส P (ppm)", 0.0, 100.0, 6.5, 0.5)
        k_ppm = st.number_input("ค่าโพแทสเซียม K (ppm)", 0.0, 200.0, 50.0, 1.0)

    if st.button("คำนวณสูตรปุ๋ย"):
        result = calculate_custom_fertilizer(soil_texture, n_level, p_ppm, k_ppm)

        st.subheader("ปริมาณแม่ปุ๋ยที่ต้องใช้ (กก./ไร่)")
        cols = st.columns(3)
        for col, (name, val) in zip(cols, result.items()):
            col.metric(name, f"{val} กก.")

        st.subheader("แผนการแบ่งใส่ปุ๋ย 2 รอบ")
        split_df = pd.DataFrame({
            "ระยะ": ["รอบที่ 1 (รองพื้น/แตกกอ)", "รอบที่ 2 (กำเนิดช่อดอก)"],
            "46-0-0 (กก.)": [round(result["46-0-0 (ยูเรีย)"] * 0.5, 1)] * 2,
            "18-46-0 (กก.)": [result["18-46-0 (DAP)"], 0.0],
            "0-0-60 (กก.)": [round(result["0-0-60 (MOP)"] * 0.5, 1)] * 2
        })
        st.table(split_df)

# ────
st.sidebar.divider()
st.sidebar.caption("FF69_Rice Research Prototype © 2026")