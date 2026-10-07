import io
import numpy as np
import tensorflow as tf
from PIL import Image
from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# 1. Path to your SavedModel file
MODEL_PATH = "/Users/ich/Documents/Project/skin_disease_tf_model.keras"
model = tf.keras.models.load_model(MODEL_PATH)

# 2. Define Class Names
CLASS_NAMES = [
    "Eczema",
    "Warts Molluscum and other Viral Infections",
    "Melanoma",
    "Atopic Dermatitis",
    "Basal Cell Carcinoma (BCC)",
    "Melanocytic Nevi (NV)",
    "Benign Keratosis-like Lesions (BKL)",
    "Psoriasis pictures Lichen Planus and related diseases",
    "Seborrheic Keratoses and other Benign Tumors",
    "Tinea Ringworm Candidiasis and other Fungal Infections"
]

# 3. Symptoms & Care Methods mapping (English & Thai)
DISEASE_INFO = {
    "Eczema": {
        "symptoms": [
            "Dry, sensitive, and inflamed skin",
            "Intense itching that may worsen at night",
            "Red to brownish-gray patches on hands, feet, neck, or elbows"
        ],
        "symptoms_th": [
            "ผิวแห้ง บอบบาง และอักเสบ",
            "มีอาการคันอย่างรุนแรง ซึ่งอาจแย่ลงในตอนกลางคืน",
            "มีผื่นสีแดงถึงสีน้ำตาลเทาบริเวณมือ เทา คอ หรือข้อพับ"
        ],
        "care": [
            "Moisturize skin at least twice a day with fragrance-free creams",
            "Avoid harsh soaps, detergents, and sudden temperature changes",
            "Apply mild over-the-counter hydrocortisone creams as needed"
        ],
        "care_th": [
            "ทาครีมบำรุงผิวที่ไม่มีน้ำหอมอย่างน้อยวันละ 2 ครั้ง",
            "หลีกเลี่ยงสบู่ น้ำยาซักผ้าที่มีสารเคมีรุนแรง และการเปลี่ยนแปลงอุณหภูมิฉับพลัน",
            "ใช้ยาทาทาไฮโดรคอร์ติโซนชนิดอ่อนตามคำแนะนำเพื่อบรรเทาอาการคัน"
        ]
    },
    "Warts Molluscum and other Viral Infections": {
        "symptoms": [
            "Small, firm, dome-shaped bumps with a dimple in the middle (Molluscum)",
            "Rough, grainy skin growths on fingers or feet (Warts)",
            "Cluster of tiny, painless or slightly itchy skin lesions"
        ],
        "symptoms_th": [
            "ตุ่มนุ่มเล็กๆ รูปร่างคล้ายโดมและมีรอยบุ๋มตรงกลาง (หูดข้าวสุก)",
            "ตุ่มเนื้อผิวขรุขระบริเวณนิ้วมือหรือฝ่าเท้า (หูด)",
            "กลุ่มตุ่มบนผิวหนังขนาดเล็ก ไม่เจ็บหรือมีอาการคันเล็กน้อย"
        ],
        "care": [
            "Avoid picking, scratching, or popping the lesions to prevent spreading",
            "Keep the affected area clean, dry, and covered with a bandage",
            "Do not share personal items like towels, razors, or clothing"
        ],
        "care_th": [
            "แกะ เกา หรือบีบตุ่มเพื่อป้องกันการแพร่กระจายของเชื้อ",
            "รักษาบริเวณที่เป็นให้สะอาด แห้ง และปิดด้วยผ้าพลาสเตอร์",
            "ไม่ใช้ของใช้ส่วนตัวร่วมกับผู้อื่น เช่น ผ้าเช็ดตัว มีดโกน หรือเสื้อผ้า"
        ]
    },
    "Melanoma": {
        "symptoms": [
            "Asymmetrical mole with irregular, scalloped, or poorly defined borders",
            "Color variation within the mole (brown, black, pink, red, or blue)",
            "Mole that is changing in size, shape, color, or bleeding"
        ],
        "symptoms_th": [
            "ไฝที่มีรูปร่างไม่สมมาตร ขอบเขตไม่สม่ำเสมอ หรือขอบเขตไม่ชัดเจน",
            "ไฝที่มีหลายสีในเม็ดเดียว (น้ำตาล, ดำ, ชมพู, แดง หรือน้ำเงิน)",
            "ไฝที่มีการเปลี่ยนแปลงขนาด รูปร่าง สี หรือมีเลือดออก"
        ],
        "care": [
            "Schedule an urgent evaluation with a dermatologist for a biopsy",
            "Protect skin from direct sunlight using SPF 50+ sunscreen and protective gear",
            "Do not attempt to scrape or treat the spot with topical creams"
        ],
        "care_th": [
            "พบแพทย์ผิวหนังโดยด่วนเพื่อตรวจตัดชิ้นเนื้อพิสูจน์",
            "ปกป้องผิวจากแสงแดดโดยตรงด้วยครีมกันแดด SPF 50+ และสวมเสื้อผิวมิดชิด",
            "ห้ามพยายามขูด ตัด หรือใช้ยาทารักษาเองโดยเด็ดขาด"
        ]
    },
    "Atopic Dermatitis": {
        "symptoms": [
            "Severe itching leading to cracked, raw, or swollen skin",
            "Crusty or oozing patches during flare-ups",
            "Thickened or leathery skin from chronic rubbing"
        ],
        "symptoms_th": [
            "อาการคันอย่างรุนแรงจนทำให้ผิวหนังแตก แสบ หรือพอง",
            "ผื่นมีสะเก็ดหรือมีน้ำเหลืองซึมในช่วงที่มีอาการกำเริบ",
            "ผิวหนังหนาขึ้นหรือแข็งเป็นแผ่นจากการถูหรือเกาเป็นเวลานาน"
        ],
        "care": [
            "Take short, lukewarm showers and apply thick moisturizer immediately",
            "Wear soft, breathable cotton fabrics and avoid wool or scratchy materials",
            "Identify and avoid individual environmental triggers (pollen, pet dander)"
        ],
        "care_th": [
            "อาบน้ำอุ่นในเวลาสั้นๆ และทาครีมบำรุงผิวเนื้อเข้มข้นทันทีหลังอาบน้ำ",
            "สวมเสื้อผ้าผ้าฝ้ายที่นุ่ม ระบายอากาศได้ดี หลีกเลี่ยงผ้าขนสัตว์",
            "สังเกตและหลีกเลี่ยงสิ่งกระตุ้น เช่น เกสรดอกไม้ หรือขนสัตว์"
        ]
    },
    "Basal Cell Carcinoma (BCC)": {
        "symptoms": [
            "Pearly or waxy bump with visible tiny blood vessels",
            "Flat, flesh-colored, or brown scar-like lesion",
            "A bleeding or scabbing sore that heals and returns repeatedly"
        ],
        "symptoms_th": [
            "ตุ่มนูนสีมุกหรือคล้ายขี้ผึ้ง มองเห็นเส้นเลือดฝอยเล็กๆ ได้",
            "รอยแผลราบสีเนื้อหรือสีน้ำตาลคล้ายแผลเป็น",
            "แผลตกสะเก็ดหรือมีเลือดออกที่หายแล้วกลับมาเป็นซ้ำเรื่อยๆ"
        ],
        "care": [
            "Consult a dermatologist for surgical evaluation or specialized removal",
            "Avoid sun exposure during peak UV hours (10 AM - 4 PM)",
            "Regularly monitor your skin for new or changing spots"
        ],
        "care_th": [
            "ปรึกษาแพทย์ผิวหนังเพื่อประเมินการผ่าตัดหรือการรักษาเฉพาะทาง",
            "หลีกเลี่ยงการสัมผัสแสงแดดในช่วงเวลาที่มี UV สูง (10.00 น. - 16.00 น.)",
            "สังเกตและตรวจเช็กผิวหนังเป็นประจำหากมีจุดใหม่หรือการเปลี่ยนแปลง"
        ]
    },
    "Melanocytic Nevi (NV)": {
        "symptoms": [
            "Small, well-defined round or oval spots (Common Moles)",
            "Uniform color ranging from pink, tan, to dark brown",
            "Flat or slightly raised smooth surface"
        ],
        "symptoms_th": [
            "จุดขนาดเล็ก รูปร่างกลมหรือรีที่มีขอบชัดเจน (ไฝธรรมดา)",
            "สีสม่ำเสมอกัน ตั้งแต่ชมพู น้ำตาลอ่อน ถึงน้ำตาลเข้ม",
            "ผิวเรียบหรือนูนขึ้นมาจากผิวหนังเล็กน้อย"
        ],
        "care": [
            "Generally harmless; monitor for ABCDE changes (Asymmetry, Border, Color, Diameter, Evolving)",
            "Apply daily sunscreen to prevent darkening or structural changes",
            "Have routine full-body skin checks with a physician"
        ],
        "care_th": [
            "โดยทั่วไปไม่อันตราย ควรเฝ้าระวังการเปลี่ยนแปลงตามหลัก ABCDE",
            "ทาครีมกันแดดเป็นประจำเพื่อป้องกันไม่ให้ไฝคล้ำลงหรือเปลี่ยนสภาพ",
            "ตรวจเช็กผิวหนังทั่วร่างกายกับแพทย์เป็นประจำ"
        ]
    },
    "Benign Keratosis-like Lesions (BKL)": {
        "symptoms": [
            "Waxy, 'stuck-on' appearance on the skin surface",
            "Light tan to black elevated spots",
            "Rough, dry, or slightly scaly texture"
        ],
        "symptoms_th": [
            "ลักษณะคล้ายขี้ผึ้งหรือแปะติดอยู่บนพื้นผิวผิวหนัง",
            "ตุ่มนูนมีสีตั้งแต่สีน้ำตาลอ่อนไปจนถึงสีดำ",
            "ผิวสัมผัสขรุขระ แห้ง หรือเป็นสะเก็ดเล็กน้อย"
        ],
        "care": [
            "Harmless non-cancerous growths; no medical treatment is required unless irritated",
            "Avoid scratching or picking to prevent secondary skin infections",
            "Consult a dermatologist if the lesion becomes painful or bleeds"
        ],
        "care_th": [
            "เนื้องอกที่ไม่ใช่มะเร็งและไม่อันตราย ไม่จำเป็นต้องรักษาเว้นแต่จะระคายเคือง",
            "หลีกเลี่ยงการแกะหรือเกาเพื่อป้องกันการติดเชื้อแบคทีเรียแทรกซ้อน",
            "พบแพทย์หากรอยโรคมีอาการเจ็บหรือมีเลือดออก"
        ]
    },
    "Psoriasis pictures Lichen Planus and related diseases": {
        "symptoms": [
            "Thick red patches covered with silvery scales (Psoriasis)",
            "Flat-topped, itchy, purplish bumps (Lichen Planus)",
            "Joint pain or nail pitting in some cases"
        ],
        "symptoms_th": [
            "ผื่นแดงหนาปกคลุมด้วยสะเก็ดสีเงิน (โรคสะเก็ดเงิน)",
            "ตุ่มคันสีม่วงยอดราบ (โรคไลเคน พลานัส)",
            "อาจมีอาการปวดข้อหรือเล็บเป็นหลุมร่วมด้วยในบางราย"
        ],
        "care": [
            "Keep skin moisturized to reduce scaling and itching",
            "Use prescribed topical corticosteroids or light therapy as advised by a doctor",
            "Manage stress levels and avoid skin injuries, which can trigger flare-ups"
        ],
        "care_th": [
            "รักษาความชุ่มชื้นของผิวหนังเสมอเพื่อลดการลอกและอาการคัน",
            "ใช้ยาทาสเตียรอยด์ตามแพทย์สั่ง หรือรับการบำบัดด้วยแสงตามคำแนะนำ",
            "จัดการความเครียดและหลีกเลี่ยงการเกิดบาดแผลที่ผิวหนัง ซึ่งเป็นตัวกระตุ้นผื่น"
        ]
    },
    "Seborrheic Keratoses and other Benign Tumors": {
        "symptoms": [
            "Harmless, elevated skin growths that look like drop-wax",
            "Colors ranging from light yellow-brown to dark black",
            "Commonly found on chest, back, shoulders, or face"
        ],
        "symptoms_th": [
            "ตุ่มเนื้อนูนไม่อันตราย ลักษณะคล้ายหยดขี้ผึ้งเกาะผิว",
            "มีสีตั้งแต่สีเหลืองน้ำตาลอ่อนไปจนถึงสีดำเข้ม",
            "มักพบตามหน้าอก หลัง ไหล่ หรือใบหน้า"
        ],
        "care": [
            "No medical treatment required unless rubbed by clothing or cosmetically undesired",
            "Avoid trying to cut or peel off the growths at home",
            "A physician can easily remove them via cryotherapy or curettage if needed"
        ],
        "care_th": [
            "ไม่จำเป็นต้องรักษา ยกเว้นจะถูกเสียดสีกับเสื้อผ้าหรือต้องการเอาออกเพื่อความสวยงาม",
            "ห้ามพยายามตัดหรือแกะออกด้วยตนเองที่บ้าน",
            "แพทย์สามารถเอาออกได้ง่ายด้วยการจี้เย็น (Cryotherapy) หรือขูดออก"
        ]
    },
    "Tinea Ringworm Candidiasis and other Fungal Infections": {
        "symptoms": [
            "Red, circular, ring-shaped rash with raised edges (Ringworm)",
            "Itchy, red, moist skin in skin folds (Candidiasis)",
            "Scaling, cracking, or peeling skin between toes or fingers"
        ],
        "symptoms_th": [
            "ผื่นแดงรูปวงกลมคล้ายวงแหวน ขอบนูนขอบชัด (กลาก)",
            "ผื่นแดงเปียกชื้นและคันบริเวณข้อพับหรืออับชื้น (เชื้อราแคนดิดา)",
            "ผิวหนังลอก แตก หรือเป็นสะเก็ดบริเวณซอกนิ้วมือนิ้วเท้า (ฮ่องกงฟุต)"
        ],
        "care": [
            "Apply over-the-counter topical antifungal creams (e.g., Clotrimazole or Miconazole)",
            "Keep the affected skin area completely clean and dry",
            "Wear loose-fitting, breathable cotton clothing and change socks daily"
        ],
        "care_th": [
            "ใช้ยาทาฆ่าเชื้อรา (เช่น Clotrimazole หรือ Miconazole) ทาบริเวณที่เป็น",
            "ดูแลบริเวณที่เป็นผื่นให้สะอาดและแห้งอยู่เสมอ",
            "สวมเสื้อผ้าผ้าฝ้ายที่ระบายอากาศได้ดี ไม่อับชื้น และเปลี่ยนถุงเท้าทุกวัน"
        ]
    }
}

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Skin Condition Analyzer</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
</head>
<body class="bg-slate-50 min-h-screen flex items-center justify-center p-4">

    <div class="bg-white w-full max-w-xl rounded-2xl shadow-xl overflow-hidden border border-slate-100 my-8">
        <!-- Header -->
        <div class="bg-indigo-600 px-6 py-6 text-white text-center">
            <i class="fa-solid font-bold text-3xl mb-2 fa-microscope"></i>
            <h1 class="text-2xl font-bold">Skin Lesion Classifier</h1>
            <p class="text-indigo-100 text-sm mt-1">Upload or capture an image to analyze skin conditions</p>
        </div>

        <div class="p-6 space-y-6">
            <!-- Mode Selection Tabs -->
            <div class="flex rounded-xl bg-slate-100 p-1 border border-slate-200">
                <button type="button" id="tab-upload" class="flex-1 py-2 text-sm font-semibold rounded-lg bg-white text-indigo-600 shadow-sm transition-all flex items-center justify-center space-x-2">
                    <i class="fa-solid fa-cloud-arrow-up"></i>
                    <span>Upload Image</span>
                </button>
                <button type="button" id="tab-camera" class="flex-1 py-2 text-sm font-semibold rounded-lg text-slate-600 hover:text-indigo-600 transition-all flex items-center justify-center space-x-2">
                    <i class="fa-solid fa-camera"></i>
                    <span>Take Picture</span>
                </button>
            </div>

            <!-- Upload / Camera Input Form -->
            <form id="upload-form" class="space-y-4">
                
                <!-- Option 1: File Dropzone -->
                <div id="upload-mode-container">
                    <div id="dropzone" class="border-2 border-dashed border-indigo-200 hover:border-indigo-500 rounded-xl p-6 text-center cursor-pointer transition-colors bg-indigo-50/30 hover:bg-indigo-50/50 relative">
                        <input type="file" id="image-input" accept="image/*" class="hidden">
                        
                        <div id="upload-prompt" class="space-y-2">
                            <i class="fa-solid fa-file-image text-4xl text-indigo-400"></i>
                            <p class="text-slate-600 font-medium text-sm">Click to select or drag & drop</p>
                            <p class="text-slate-400 text-xs">PNG, JPG, JPEG or WEBP</p>
                        </div>

                        <div id="file-preview-container" class="hidden relative">
                            <img id="file-preview" src="#" alt="Preview" class="max-h-56 mx-auto rounded-lg shadow-sm border border-slate-200">
                            <p id="file-name" class="text-xs text-slate-500 mt-2 font-medium truncate"></p>
                        </div>
                    </div>
                </div>

                <!-- Option 2: Live Camera Capture -->
                <div id="camera-mode-container" class="hidden space-y-3">
                    <div class="relative bg-slate-900 rounded-xl overflow-hidden aspect-video flex items-center justify-center border border-slate-200 shadow-inner">
                        <video id="webcam" autoplay playsinline class="w-full h-full object-cover"></video>
                        <canvas id="canvas" class="hidden"></canvas>
                        <img id="camera-preview" src="#" alt="Captured Preview" class="hidden w-full h-full object-cover">
                    </div>

                    <div class="flex space-x-2">
                        <button type="button" id="snap-btn" class="flex-1 bg-indigo-600 hover:bg-indigo-700 text-white font-medium py-2 px-3 rounded-lg text-sm transition flex items-center justify-center space-x-2">
                            <i class="fa-solid fa-circle-dot"></i>
                            <span>Capture Photo</span>
                        </button>
                        <button type="button" id="retake-btn" class="hidden flex-1 bg-slate-200 hover:bg-slate-300 text-slate-700 font-medium py-2 px-3 rounded-lg text-sm transition flex items-center justify-center space-x-2">
                            <i class="fa-solid fa-rotate-left"></i>
                            <span>Retake</span>
                        </button>
                    </div>
                </div>

                <!-- Submit Button -->
                <button type="submit" id="submit-btn" class="w-full bg-indigo-600 hover:bg-indigo-700 text-white font-semibold py-3 px-4 rounded-xl shadow-md transition-all duration-200 flex items-center justify-center space-x-2">
                    <i class="fa-solid fa-wand-magic-sparkles"></i>
                    <span>Analyze Image</span>
                </button>
            </form>

            <!-- Results Section -->
            <div id="result-container" class="hidden space-y-4">
                <div class="bg-indigo-50 border border-indigo-100 rounded-xl p-4">
                    <span class="text-xs font-semibold uppercase tracking-wider text-indigo-600">Predicted Diagnosis</span>
                    <h3 id="result-class" class="text-xl font-bold text-slate-800 mt-1">--</h3>

                    <div class="mt-3">
                        <div class="flex justify-between text-xs font-medium text-slate-600 mb-1">
                            <span>Confidence</span>
                            <span id="result-confidence-text">0%</span>
                        </div>
                        <div class="w-full bg-indigo-100 rounded-full h-2.5 overflow-hidden">
                            <div id="result-bar" class="bg-indigo-600 h-2.5 rounded-full transition-all duration-500 ease-out" style="width: 0%"></div>
                        </div>
                    </div>
                </div>

                <div class="bg-amber-50/60 border border-amber-200/80 rounded-xl p-4">
                    <h4 class="font-bold text-amber-800 flex items-center space-x-2 mb-2 text-sm">
                        <i class="fa-solid fa-triangle-exclamation"></i>
                        <span>Common Symptoms (EN)</span>
                    </h4>
                    <ul id="symptoms-list" class="list-disc list-inside space-y-1 text-slate-700 text-sm"></ul>
                </div>

                <div class="bg-amber-50/60 border border-amber-200/80 rounded-xl p-4">
                    <h4 class="font-bold text-amber-800 flex items-center space-x-2 mb-2 text-sm">
                        <i class="fa-solid fa-triangle-exclamation"></i>
                        <span>อาการโดยทั่วไป (TH)</span>
                    </h4>
                    <ul id="symptomsTH-list" class="list-disc list-inside space-y-1 text-slate-700 text-sm"></ul>
                </div>

                <div class="bg-emerald-50/60 border border-emerald-200/80 rounded-xl p-4">
                    <h4 class="font-bold text-emerald-800 flex items-center space-x-2 mb-2 text-sm">
                        <i class="fa-solid fa-heart-pulse"></i>
                        <span>Recommended Care & Next Steps (EN)</span>
                    </h4>
                    <ul id="care-list" class="list-disc list-inside space-y-1 text-slate-700 text-sm"></ul>
                </div>

                <div class="bg-emerald-50/60 border border-emerald-200/80 rounded-xl p-4">
                    <h4 class="font-bold text-emerald-800 flex items-center space-x-2 mb-2 text-sm">
                        <i class="fa-solid fa-heart-pulse"></i>
                        <span>การดูแลรักษาและข้อแนะนำ (TH)</span>
                    </h4>
                    <ul id="careTH-list" class="list-disc list-inside space-y-1 text-slate-700 text-sm"></ul>
                </div>
            </div>

            <!-- Error Banner -->
            <div id="error-banner" class="hidden p-3 bg-red-50 border border-red-100 text-red-600 rounded-xl text-sm flex items-center space-x-2">
                <i class="fa-solid fa-circle-exclamation"></i>
                <span id="error-text">An error occurred.</span>
            </div>
        </div>
    </div>

    <script>
        // Mode Toggling
        const tabUpload = document.getElementById('tab-upload');
        const tabCamera = document.getElementById('tab-camera');
        const uploadContainer = document.getElementById('upload-mode-container');
        const cameraContainer = document.getElementById('camera-mode-container');

        // Webcam Elements
        const webcam = document.getElementById('webcam');
        const canvas = document.getElementById('canvas');
        const snapBtn = document.getElementById('snap-btn');
        const retakeBtn = document.getElementById('retake-btn');
        const cameraPreview = document.getElementById('camera-preview');
        let mediaStream = null;
        let capturedBlob = null;
        let activeMode = 'upload'; // 'upload' or 'camera'

        tabUpload.addEventListener('click', () => {
            activeMode = 'upload';
            tabUpload.className = "flex-1 py-2 text-sm font-semibold rounded-lg bg-white text-indigo-600 shadow-sm transition-all flex items-center justify-center space-x-2";
            tabCamera.className = "flex-1 py-2 text-sm font-semibold rounded-lg text-slate-600 hover:text-indigo-600 transition-all flex items-center justify-center space-x-2";
            uploadContainer.classList.remove('hidden');
            cameraContainer.classList.add('hidden');
            stopCamera();
        });

        tabCamera.addEventListener('click', () => {
            activeMode = 'camera';
            tabCamera.className = "flex-1 py-2 text-sm font-semibold rounded-lg bg-white text-indigo-600 shadow-sm transition-all flex items-center justify-center space-x-2";
            tabUpload.className = "flex-1 py-2 text-sm font-semibold rounded-lg text-slate-600 hover:text-indigo-600 transition-all flex items-center justify-center space-x-2";
            cameraContainer.classList.remove('hidden');
            uploadContainer.classList.add('hidden');
            startCamera();
        });

        // Camera Logic
        async function startCamera() {
            try {
                mediaStream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'environment' }, audio: false });
                webcam.srcObject = mediaStream;
                webcam.classList.remove('hidden');
                cameraPreview.classList.add('hidden');
                snapBtn.classList.remove('hidden');
                retakeBtn.classList.add('hidden');
                capturedBlob = null;
            } catch (err) {
                showError("Unable to access camera. Please allow camera permissions.");
            }
        }

        function stopCamera() {
            if (mediaStream) {
                mediaStream.getTracks().forEach(track => track.stop());
                mediaStream = null;
            }
        }

        snapBtn.addEventListener('click', () => {
            canvas.width = webcam.videoWidth;
            canvas.height = webcam.videoHeight;
            const ctx = canvas.getContext('2d');
            ctx.drawImage(webcam, 0, 0, canvas.width, canvas.height);

            canvas.toBlob((blob) => {
                capturedBlob = blob;
                cameraPreview.src = URL.createObjectURL(blob);
                cameraPreview.classList.remove('hidden');
                webcam.classList.add('hidden');
                snapBtn.classList.add('hidden');
                retakeBtn.classList.remove('hidden');
            }, 'image/jpeg');
        });

        retakeBtn.addEventListener('click', () => {
            capturedBlob = null;
            cameraPreview.classList.add('hidden');
            webcam.classList.remove('hidden');
            snapBtn.classList.remove('hidden');
            retakeBtn.classList.add('hidden');
        });

        // File Upload Dropzone Logic
        const dropzone = document.getElementById('dropzone');
        const fileInput = document.getElementById('image-input');
        const uploadPrompt = document.getElementById('upload-prompt');
        const filePreviewContainer = document.getElementById('file-preview-container');
        const filePreview = document.getElementById('file-preview');
        const fileName = document.getElementById('file-name');

        dropzone.addEventListener('click', () => fileInput.click());

        fileInput.addEventListener('change', (e) => {
            if (e.target.files.length > 0) showFilePreview(e.target.files[0]);
        });

        ['dragenter', 'dragover'].forEach(name => {
            dropzone.addEventListener(name, (e) => {
                e.preventDefault();
                dropzone.classList.add('border-indigo-500', 'bg-indigo-50');
            }, false);
        });

        ['dragleave', 'drop'].forEach(name => {
            dropzone.addEventListener(name, (e) => {
                e.preventDefault();
                dropzone.classList.remove('border-indigo-500', 'bg-indigo-50');
            }, false);
        });

        dropzone.addEventListener('drop', (e) => {
            const files = e.dataTransfer.files;
            if (files.length > 0) {
                fileInput.files = files;
                showFilePreview(files[0]);
            }
        });

        function showFilePreview(file) {
            const reader = new FileReader();
            reader.onload = (e) => {
                filePreview.src = e.target.result;
                fileName.innerText = file.name;
                uploadPrompt.classList.add('hidden');
                filePreviewContainer.classList.remove('hidden');
                hideError();
            };
            reader.readAsDataURL(file);
        }

        // Form Submission
        const form = document.getElementById('upload-form');
        const submitBtn = document.getElementById('submit-btn');
        const resultContainer = document.getElementById('result-container');
        const resultClass = document.getElementById('result-class');
        const resultConfidenceText = document.getElementById('result-confidence-text');
        const resultBar = document.getElementById('result-bar');
        const symptomsList = document.getElementById('symptoms-list');
        const symptomsTHList = document.getElementById('symptomsTH-list');
        const careList = document.getElementById('care-list');
        const careTHList = document.getElementById('careTH-list');
        const errorBanner = document.getElementById('error-banner');
        const errorText = document.getElementById('error-text');

        function showError(msg) {
            errorText.innerText = msg;
            errorBanner.classList.remove('hidden');
        }

        function hideError() {
            errorBanner.classList.add('hidden');
        }

        form.addEventListener('submit', async (e) => {
            e.preventDefault();
            hideError();

            const formData = new FormData();

            if (activeMode === 'upload') {
                if (fileInput.files.length === 0) {
                    showError("Please select or drop an image file.");
                    return;
                }
                formData.append('file', fileInput.files[0]);
            } else if (activeMode === 'camera') {
                if (!capturedBlob) {
                    showError("Please capture a photo first.");
                    return;
                }
                formData.append('file', capturedBlob, 'camera_capture.jpg');
            }

            resultContainer.classList.add('hidden');
            submitBtn.disabled = true;
            submitBtn.innerHTML = `<i class="fa-solid fa-spinner animate-spin"></i><span>Analyzing Image...</span>`;

            try {
                const response = await fetch('/predict', {
                    method: 'POST',
                    body: formData
                });

                const data = await response.json();

                if (response.ok) {
                    resultClass.innerText = data.class;
                    resultConfidenceText.innerText = `${data.confidence}%`;
                    resultBar.style.width = `${data.confidence}%`;

                    symptomsList.innerHTML = (data.info.symptoms || []).map(item => `<li>${item}</li>`).join('');
                    symptomsTHList.innerHTML = (data.info.symptoms_th || []).map(item => `<li>${item}</li>`).join('');
                    careList.innerHTML = (data.info.care || []).map(item => `<li>${item}</li>`).join('');
                    careTHList.innerHTML = (data.info.care_th || []).map(item => `<li>${item}</li>`).join('');

                    resultContainer.classList.remove('hidden');
                } else {
                    showError(data.error || "Failed to process image.");
                }
            } catch (err) {
                showError("Unable to connect to the backend server.");
            } finally {
                submitBtn.disabled = false;
                submitBtn.innerHTML = `<i class="fa-solid fa-wand-magic-sparkles"></i><span>Analyze Image</span>`;
            }
        });
    </script>
</body>
</html>
"""

@app.route("/", methods=["GET"])
def home():
    return render_template_string(HTML_TEMPLATE)

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    try:
        image = Image.open(io.BytesIO(file.read())).convert("RGB")
        image = image.resize((224, 224))
        
        img_array = np.array(image, dtype=np.float32)
        img_batch = np.expand_dims(img_array, axis=0)

        predictions = model.predict(img_batch)

        idx = int(np.argmax(predictions[0]))
        predicted_class = CLASS_NAMES[idx]
        confidence = float(predictions[0][idx])

        info = DISEASE_INFO.get(predicted_class, {
            "symptoms": ["No specific symptoms recorded."],
            "symptoms_th": ["ไม่มีข้อมูลอาการในระบบ"],
            "care": ["Consult a medical professional."],
            "care_th": ["แนะนำให้ปรึกษาแพทย์เพื่อรับคำแนะนำ"]
        })

        return jsonify({
            "class": predicted_class,
            "confidence": round(confidence * 100, 2),
            "info": info
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)