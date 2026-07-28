import os
import textwrap
import streamlit as st
from ultralytics import YOLO
from PIL import Image
import pandas as pd
import base64


# =========================================================
# KONFIGURASI HALAMAN
# =========================================================

st.set_page_config(
    page_title="Waste Detection",
    page_icon="♻️",
    layout="wide",
    initial_sidebar_state="expanded"
)

with open("assets/waste-illustration.png", "rb") as f:
    waste_image = base64.b64encode(f.read()).decode()


# =========================================================
# FUNGSI HTML
# =========================================================

def html(content):
    content = "\n".join(
        line.strip()
        for line in content.splitlines()
    )

    st.markdown(
        content,
        unsafe_allow_html=True
    )


# =========================================================
# SESSION STATE
# =========================================================

if "history" not in st.session_state:
    st.session_state.history = []


# =========================================================
# CSS
# =========================================================

html("""
<style>

.stApp {
    background-color: #F7FAF5;
}

#MainMenu {
    visibility: hidden;
}

footer {
    visibility: hidden;
}

header {
    visibility: hidden;
}


/* =====================================
   SIDEBAR
===================================== */

[data-testid="stSidebar"] {
    background-color: #FFFFFF;
    border-right: 1px solid #E1EBDD;
}

[data-testid="stSidebar"] > div:first-child {
    padding-top: 20px;
}


/* =====================================
   HERO
===================================== */

.hero {
    background: linear-gradient(
        135deg,
        #EEF8E9,
        #FFFFFF
    );

    border: 1px solid #DCEBD5;
    border-radius: 22px;

    padding: 32px;

    margin-bottom: 22px;

    box-shadow:
        0 5px 20px rgba(50, 100, 40, 0.05);
}

.hero-title {
    color: #3F8526;

    font-size: 42px;

    font-weight: 900;

    line-height: 1;

    letter-spacing: -1px;
}

.hero-slogan {
    color: #39702C;

    font-size: 17px;

    font-weight: 500;

    margin-top: 12px;
}

.hero-description {
    color: #52604F;

    font-size: 14px;

    line-height: 1.7;

    margin-top: 16px;

    max-width: 650px;
}


/* =====================================
   CARD
===================================== */

.card {
    background: #FFFFFF;

    border: 1px solid #E0E9DC;

    border-radius: 18px;

    padding: 24px;

    margin-bottom: 18px;

    box-shadow:
        0 4px 15px rgba(50, 90, 40, 0.04);
}

.card-title {
    color: #1F2D20;

    font-size: 18px;

    font-weight: 750;
}

.card-description {
    color: #788477;

    font-size: 13px;

    margin-top: 5px;
}


/* =====================================
   METRIC CARD
===================================== */

.metric-box {
    background: #FFFFFF;

    border: 1px solid #DDE8D8;

    border-radius: 15px;

    padding: 20px 10px;

    text-align: center;
}

.metric-number {
    color: #3F8526;

    font-size: 28px;

    font-weight: 800;
}

.metric-label {
    color: #71806E;

    font-size: 12px;

    margin-top: 5px;
}


/* =====================================
   INFO BOX
===================================== */

.info-box {
    background: #F2F8EF;

    border: 1px solid #DCEBD5;

    border-radius: 15px;

    padding: 18px;

    color: #465443;
}


/* =====================================
   SIDEBAR TITLE
===================================== */

.menu-label {
    color: #3F8526;

    font-size: 11px;

    font-weight: 800;

    text-transform: uppercase;

    margin-top: 18px;

    margin-bottom: 8px;
}


/* =====================================
   BUTTON
===================================== */

.stButton > button {

    background-color: #3F8526;

    color: white;

    border: none;

    border-radius: 10px;

    font-weight: 600;

    padding: 10px 18px;

}

.stButton > button:hover {

    background-color: #326C1E;

    color: white;

}


/* =====================================
   FOOTER
===================================== */

.footer {

    text-align: center;

    color: #7A8776;

    font-size: 11px;

    padding: 25px 0;

}

</style>
""")

# =====================================
# HERO DASHBOARD DENGAN ILUSTRASI
# =====================================

st.markdown("""
<style>

.hero-main {
    width: 100%;
    min-height: 300px;

    background: linear-gradient(
        135deg,
        #EEF8E9,
        #FFFFFF
    );

    border: 1px solid #DCEBD5;
    border-radius: 22px;

    padding: 35px 45px;
    margin-bottom: 5px;

    display: flex;
    align-items: center;
    justify-content: space-between;

    box-sizing: border-box;
    overflow: hidden;

    box-shadow:
        0 5px 20px rgba(50, 100, 40, 0.05);
}

.hero-content {
    width: 58%;
}

.hero-image {
    width: 35%;
    height: 240px;

    display: flex;
    justify-content: center;
    align-items: center;
}

.hero-image img {
    width: 230px;
    height: 210px;

    object-fit: contain;
}

</style>
""", unsafe_allow_html=True)

# =========================================================
# MODEL
# =========================================================

model_path = {

    "Skenario Baseline":
        "models/Baselinebest.pt",

    "Skenario Combined":
        "models/Combinedbest.pt",

    "Skenario Flip":
        "models/Flipbest.pt",

    "Skenario Lighting":
        "models/Lightingbest.pt",

    "Skenario Noise":
        "models/Noisebest.pt"

}


@st.cache_resource
def load_model(path):

    return YOLO(path)


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    # =========================================
    # LOGO
    # =========================================

    if os.path.exists("assets/logo.png"):

        st.image(
            "assets/logo.png",
            use_container_width=True
        )

    else:

        html("""
        <div style="
            text-align:center;
            padding:10px 0 20px 0;
        ">

            <div style="
                font-size:48px;
            ">
                ♻️
            </div>

            <div style="
                color:#3F8526;
                font-size:25px;
                font-weight:900;
                line-height:1;
            ">
                WASTE<br>
                DETECTION
            </div>

            <div style="
                color:#52724A;
                font-size:12px;
                margin-top:8px;
            ">
                See Waste. Make a Difference.
            </div>

        </div>
        """)


    st.divider()


    # =========================================
    # MENU
    # =========================================

    html("""
    <div class="menu-label">
        Menu Utama
    </div>
    """)


    halaman = st.radio(

        "Navigasi",

        [
            "📊  Dashboard",
            "🎯  Deteksi Sampah",
            "🕘  Riwayat Deteksi",
            "📈  Statistik",
            "ⓘ  Tentang Aplikasi"
        ],

        label_visibility="collapsed"
    )


    st.divider()


    # =========================================
    # INFO MODEL
    # =========================================

    html("""
    <div class="info-box">

        <b style="color:#3F8526;">
            ⚙️ Informasi Sistem
        </b>

        <p>

        <b>Model</b><br>
        YOLO11n

        <p>

        <b>Task</b><br>
        Waste Detection

        <p>

        <b>Framework</b><br>
        Ultralytics

        <p>

        <b>Platform</b><br>
        Streamlit

    </div>
    """)


    # =========================================
    # FOOTER
    # =========================================

    html("""
    <div class="footer">

        🌿 Waste Detection<br>
        Version 1.0<br><br>

        See Waste. Make a Difference.

    </div>
    """)


# =========================================================
# DASHBOARD
# =========================================================

if halaman == "📊  Dashboard":

    html(f"""
    <div class="hero-main">

        <div class="hero-content">

            <div class="hero-title">
                WASTE<br>
                DETECTION
            </div>

            <div class="hero-slogan">
                See Waste. Make a Difference.
            </div>

            <div class="hero-description">
                Sistem deteksi dan klasifikasi sampah
                menggunakan YOLO11n dengan berbagai
                skenario pengujian untuk menghasilkan
                deteksi yang lebih akurat dan andal.
            </div>

        </div>

        <div class="hero-image">
            <img src="data:image/png;base64,{waste_image}">
        </div>

    </div>
    """)

    st.markdown(
        '<div style="height: 5px;"></div>',
        unsafe_allow_html=True
    )



    # =========================================
    # WELCOME
    # =========================================

    html("""
    <div class="card">

        <div class="card-title">
            👋 Selamat Datang di Waste Detection
        </div>

        <div class="card-description">

            Gunakan menu di sebelah kiri untuk
            melakukan deteksi sampah, melihat
            riwayat hasil deteksi, dan melihat
            statistik model.

        </div>

    </div>
    """)


    # =========================================
    # DASHBOARD STATISTIC
    # =========================================

    total_images = len(
        st.session_state.history
    )

    total_objects = sum(

        item["Jumlah Objek"]

        for item
        in st.session_state.history

    )


    col1, col2, col3 = st.columns(3)


    with col1:

        html(f"""
        <div class="metric-box">

            <div class="metric-number">
                {total_images}
            </div>

            <div class="metric-label">
                Total Gambar Dideteksi
            </div>

        </div>
        """)


    with col2:

        html(f"""
        <div class="metric-box">

            <div class="metric-number">
                {total_objects}
            </div>

            <div class="metric-label">
                Total Objek Terdeteksi
            </div>

        </div>
        """)


    with col3:

        html("""
        <div class="metric-box">

            <div class="metric-number">
                5
            </div>

            <div class="metric-label">
                Skenario Model
            </div>

        </div>
        """)


# =========================================================
# DETEKSI SAMPAH
# =========================================================

elif halaman == "🎯  Deteksi Sampah":

    html("""
    <div class="hero">

        <div class="hero-title">
            DETEKSI SAMPAH
        </div>

        <div class="hero-slogan">
            Detect Smarter, Sort Better.
        </div>

        <div class="hero-description">

            Upload satu atau lebih gambar sampah
            untuk dianalisis menggunakan model YOLO11n.

        </div>

    </div>
    """)

    # =========================================
    # PENGATURAN MODEL
    # =========================================

    st.markdown("### ⚙️ Model Deteksi")

    st.caption(
        "Pilih skenario YOLO11n yang ingin digunakan "
        "sebelum melakukan deteksi."
    )

    col1, col2 = st.columns(2)

    with col1:

        skenario = st.selectbox(
            "Pilih Skenario Model",
            [
                "Skenario Baseline",
                "Skenario Combined",
                "Skenario Flip",
                "Skenario Lighting",
                "Skenario Noise"
            ]
        )

    with col2:

        confidence = st.slider(
            "Confidence Threshold",
            min_value=0.10,
            max_value=0.90,
            value=0.50,
            step=0.05
        )

    st.divider()


    # =========================================
    # UPLOAD
    # =========================================

    html("""
    <div class="card">

        <div class="card-title">
            📤 Upload Gambar Sampah
        </div>

        <div class="card-description">

            Pilih satu atau beberapa gambar
            sekaligus untuk diproses.

        </div>

    </div>
    """)


    uploaded_files = st.file_uploader(

        "Upload gambar",

        type=[
            "jpg",
            "jpeg",
            "png"
        ],

        accept_multiple_files=True,

        label_visibility="collapsed"

    )


    if uploaded_files:

        st.success(
            f"📷 {len(uploaded_files)} gambar berhasil dipilih."
        )


        if st.button(
            "🔍 Mulai Deteksi — {skenario.replace('Skenario ', '')}",
            use_container_width=True
        ):

            selected_model = model_path[
                skenario
            ]


            # =====================================
            # CHECK MODEL
            # =====================================

            if not os.path.exists(
                selected_model
            ):

                st.error(
                    f"Model tidak ditemukan: "
                    f"{selected_model}"
                )

                st.stop()


            # =====================================
            # LOAD MODEL
            # =====================================

            model = load_model(
                selected_model
            )


            all_results = []


            progress = st.progress(0)


            st.markdown(
                "## 🎯 Hasil Deteksi"
            )


            # =====================================
            # LOOP GAMBAR
            # =====================================

            for i, uploaded_file in enumerate(
                uploaded_files
            ):

                image = Image.open(
                    uploaded_file
                ).convert("RGB")


                with st.spinner(
                    f"Mendeteksi gambar "
                    f"{i + 1} dari "
                    f"{len(uploaded_files)}..."
                ):

                    results = model.predict(

                        source=image,

                        conf=confidence,

                        verbose=False

                    )


                result = results[0]


                # =================================
                # HASIL GAMBAR
                # =================================

                plotted = result.plot()

                plotted_rgb = plotted[
                    :, :, ::-1
                ]


                col1, col2 = st.columns(2)


                with col1:

                    st.image(

                        image,

                        caption="Gambar Input",

                        use_container_width=True

                    )


                with col2:

                    st.image(

                        plotted_rgb,

                        caption="Hasil Deteksi YOLO11",

                        use_container_width=True

                    )


                # =================================
                # DETEKSI OBJECT
                # =================================

                object_count = 0


                if (

                    result.boxes is not None

                    and len(result.boxes) > 0

                ):

                    class_ids = (

                        result.boxes.cls
                        .cpu()
                        .numpy()
                        .astype(int)

                    )


                    confs = (

                        result.boxes.conf
                        .cpu()
                        .numpy()

                    )


                    for class_id, conf in zip(
                        class_ids,
                        confs
                    ):

                        class_name = (
                            model.names[
                                int(class_id)
                            ]
                        )


                        object_count += 1


                        all_results.append({

                            "Gambar":
                                uploaded_file.name,

                            "Objek":
                                class_name,

                            "Confidence":
                                round(
                                    float(conf) * 100,
                                    2
                                ),

                            "Skenario":
                                skenario

                        })


                else:

                    all_results.append({

                        "Gambar":
                            uploaded_file.name,

                        "Objek":
                            "Tidak terdeteksi",

                        "Confidence":
                            0,

                        "Skenario":
                            skenario

                    })


                # =================================
                # SIMPAN HISTORY
                # =================================

                st.session_state.history.append({

                    "Gambar":
                        uploaded_file.name,

                    "Jumlah Objek":
                        object_count,

                    "Skenario":
                        skenario

                })


                progress.progress(

                    (i + 1)
                    /
                    len(uploaded_files)

                )


            # =====================================
            # RINGKASAN
            # =====================================

            df = pd.DataFrame(
                all_results
            )


            detected_df = df[
                df["Objek"]
                !=
                "Tidak terdeteksi"
            ]


            total_images = len(
                uploaded_files
            )


            total_objects = len(
                detected_df
            )


            if total_objects > 0:

                average_confidence = (

                    detected_df[
                        "Confidence"
                    ].mean()

                )

            else:

                average_confidence = 0


            st.markdown(
                "## 📊 Ringkasan Hasil"
            )


            col1, col2, col3, col4 = st.columns(4)


            with col1:

                html(f"""
                <div class="metric-box">

                    <div class="metric-number">
                        {total_images}
                    </div>

                    <div class="metric-label">
                        Total Gambar
                    </div>

                </div>
                """)


            with col2:

                html(f"""
                <div class="metric-box">

                    <div class="metric-number">
                        {total_objects}
                    </div>

                    <div class="metric-label">
                        Objek Terdeteksi
                    </div>

                </div>
                """)


            with col3:

                html(f"""
                <div class="metric-box">

                    <div class="metric-number">
                        {average_confidence:.2f}%
                    </div>

                    <div class="metric-label">
                        Rata-rata Confidence
                    </div>

                </div>
                """)


            with col4:

                html(f"""
                <div class="metric-box">

                    <div class="metric-number"
                         style="font-size:16px;">

                        {skenario.replace(
                            "Skenario ",
                            ""
                        )}

                    </div>

                    <div class="metric-label">
                        Model Aktif
                    </div>

                </div>
                """)


            # =====================================
            # TABEL
            # =====================================

            st.markdown(
                "## 📋 Detail Hasil Deteksi"
            )


            table_df = df.copy()


            table_df.insert(

                0,

                "No",

                range(
                    1,
                    len(table_df) + 1
                )

            )


            table_df[
                "Confidence"
            ] = table_df[
                "Confidence"
            ].apply(

                lambda x:
                f"{x:.2f}%"
                if x > 0
                else "-"

            )


            st.dataframe(

                table_df,

                use_container_width=True,

                hide_index=True

            )


            # =====================================
            # DOWNLOAD CSV
            # =====================================

            csv = df.to_csv(
                index=False
            ).encode("utf-8")


            st.download_button(

                "⬇️ Download Hasil CSV",

                data=csv,

                file_name=
                "hasil_deteksi.csv",

                mime="text/csv",

                use_container_width=True

            )


    else:

        st.info(
            "📤 Silakan upload minimal satu gambar."
        )


# =========================================================
# RIWAYAT
# =========================================================

elif halaman == "🕘  Riwayat Deteksi":

    html("""
    <div class="hero">

        <div class="hero-title">
            RIWAYAT DETEKSI
        </div>

        <div class="hero-slogan">
            Track Your Detection History.
        </div>

        <div class="hero-description">
            Menampilkan riwayat gambar yang
            telah diproses selama aplikasi berjalan.
        </div>

    </div>
    """)

    if len(st.session_state.history) == 0:

        st.info("Belum ada riwayat deteksi.")

    else:

        history_df = pd.DataFrame(
            st.session_state.history
        )

        st.dataframe(
            history_df,
            use_container_width=True,
            hide_index=True
        )


# =========================================================
# STATISTIK
# =========================================================

elif halaman == "📈  Statistik":

    html("""
    <div class="hero">

        <div class="hero-title">
            STATISTIK
        </div>

        <div class="hero-slogan">
            Understand Your Detection Results.
        </div>

    </div>
    """)


    if len(
        st.session_state.history
    ) == 0:

        st.info(
            "Belum ada data statistik."
        )

    else:

        df_history = pd.DataFrame(
            st.session_state.history
        )


        total_images = len(
            df_history
        )


        total_objects = int(
            df_history[
                "Jumlah Objek"
            ].sum()
        )


        total_scenarios = (
            df_history[
                "Skenario"
            ].nunique()
        )


        col1, col2, col3 = st.columns(3)


        with col1:

            st.metric(
                "Total Gambar",
                total_images
            )


        with col2:

            st.metric(
                "Total Objek",
                total_objects
            )


        with col3:

            st.metric(
                "Skenario Digunakan",
                total_scenarios
            )


        st.markdown(
            "### 📊 Jumlah Objek per Skenario"
        )


        chart_data = (

            df_history

            .groupby(
                "Skenario"
            )["Jumlah Objek"]

            .sum()

        )


        st.bar_chart(
            chart_data
        )


# =========================================================
# TENTANG APLIKASI
# =========================================================

elif halaman == "ⓘ  Tentang Aplikasi":

    html("""
    <div class="hero">

        <div class="hero-title">
            TENTANG APLIKASI
        </div>

        <div class="hero-slogan">
            Smart Detection for a Cleaner Future.
        </div>

        <div class="hero-description">

            Kenali sampah dengan lebih cepat
            menggunakan teknologi computer vision.

        </div>

    </div>
    """)


    col1, col2 = st.columns(2)


    # =========================================
    # ABOUT
    # =========================================

    with col1:

        html("""
        <div class="card">

            <div class="card-title">
                ♻️ Waste Detection
            </div>

            <br>

            <div style="
                color:#52604F;
                line-height:1.8;
                font-size:14px;
            ">

                Waste Detection merupakan aplikasi
                berbasis kecerdasan buatan yang digunakan
                untuk mendeteksi dan mengklasifikasikan
                objek sampah berdasarkan gambar.

                <br><br><br>

                Sistem ini menggunakan model
                <b>YOLO11n</b> untuk melakukan proses
                object detection secara otomatis.

            </div>

        </div>
        """)


    # =========================================
    # TECHNOLOGY
    # =========================================

    with col2:

        html("""
        <div class="card">

            <div class="card-title">
                🤖 Teknologi
            </div>

            <br>

            <div style="
                color:#52604F;
                line-height:1.8;
                font-size:14px;
            ">

                <b>Model</b><br>
                YOLO11n

                <p>

                <b>Framework</b><br>
                Ultralytics

                <p>

                <b>Interface</b><br>
                Streamlit

                <p>

                <b>Skenario Model</b><br>
                Baseline, Combined, Flip,
                Lighting, dan Noise.

            </div>

        </div>
        """)


    # =========================================
    # TUJUAN
    # =========================================

    html("""
    <div class="info-box">

        <b style="color:#3F8526;">
            🎯 Tujuan Aplikasi
        </b>

        <br><br>

        Aplikasi ini dikembangkan untuk membantu
        proses identifikasi sampah secara otomatis
        menggunakan teknologi computer vision,
        sekaligus membandingkan hasil dari beberapa
        skenario model YOLO11n.

    </div>
    """)


# =========================================================
# SELESAI
# =========================================================
