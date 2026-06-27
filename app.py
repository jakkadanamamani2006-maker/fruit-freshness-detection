import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import re
import io
import gdown
from gtts import gTTS

# ---------------- PAGE CONFIG ----------------

st.set_page_config(
    page_title="FreshSense AI",
    page_icon="🍎",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ================================================================
# LANGUAGE DICTIONARY
# ================================================================

LANGUAGES = {
    "English 🇬🇧": {
        "app_title": "FreshSense AI",
        "hero_title": "🍎 FreshSense AI",
        "hero_sub": "AI-Powered Fruit Freshness Detection · Upload any fruit image to begin",
        "sidebar_about": "About Project",
        "sidebar_desc": "AI-powered fruit freshness detection.\n\n**Fresh fruits** → fruits float upward ✅\n\n**Rotten fruits** → fruits fall down ❌\n\n**Supported Fruits:**\n- Apple 🍎\n- Banana 🍌\n- Orange 🍊",
        "sidebar_tech": "Technologies Used",
        "model_ready": "✅ Model Ready",
        "step_upload": "STEP 1 — UPLOAD",
        "upload_title": "📤 Upload a Fruit Image",
        "upload_hint": "Supports JPG, JPEG, PNG",
        "uploaded_caption": "Your uploaded image",
        "how_works_label": "HOW IT WORKS",
        "how_works_title": "🔬 Three Simple Steps",
        "step1": "1️⃣  Upload a clear photo of a fruit",
        "step2": "2️⃣  Our CNN model analyzes it instantly",
        "step3": "3️⃣  See fresh / rotten result with confidence score",
        "how_works_note": "Fresh fruits make the background come alive 🍎<br>Rotten fruits make them fall away 🍂",
        "analyzing": "🔍 Analyzing fruit...",
        "result_label": "PREDICTION RESULT",
        "status_fresh": "Fresh",
        "status_rotten": "Rotten",
        "confidence": "Confidence Score",
        "safe": "✅ Safe to Eat",
        "discard": "❌ Do Not Consume",
        "msg_fresh": "🌟 Great news! Your **{fruit}** looks fresh and ready to eat.",
        "msg_rotten": "⚠️ This **{fruit}** appears to be rotten. Please discard it.",
        "footer": "FreshSense AI · Built with TensorFlow + Streamlit + CNN Deep Learning",
        "footer_sub": "🍎 Fresh fruits float · 🍂 Rotten fruits fall",
        "downloading": "⬇️ Downloading AI model...",
        "select_lang": "🌐 Language / భాష / मराठी",
        "voice_section": "🔊 Voice Output",
        "voice_select_label": "Select voice language",
        "voice_enable_label": "Enable voice result",
        "autoplay_label": "Auto-play voice on result",
        "listen_label": "LISTEN TO RESULT",
        "voice_error": "⚠️ Voice output unavailable. Please check your internet connection.",
    },
    "తెలుగు 🇮🇳": {
        "app_title": "ఫ్రెష్‌సెన్స్ AI",
        "hero_title": "🍎 ఫ్రెష్‌సెన్స్ AI",
        "hero_sub": "AI ఆధారిత పండ్ల తాజాదనం గుర్తింపు · ప్రారంభించడానికి పండు చిత్రాన్ని అప్‌లోడ్ చేయండి",
        "sidebar_about": "ప్రాజెక్ట్ గురించి",
        "sidebar_desc": "AI ఆధారిత పండ్ల తాజాదనం గుర్తింపు.\n\n**తాజా పండ్లు** → పైకి తేలుతాయి ✅\n\n**పాడైన పండ్లు** → కిందకు పడతాయి ❌\n\n**మద్దతు ఉన్న పండ్లు:**\n- యాపిల్ 🍎\n- అరటి 🍌\n- నారంగి 🍊",
        "sidebar_tech": "వాడిన సాంకేతికతలు",
        "model_ready": "✅ మోడల్ సిద్ధంగా ఉంది",
        "step_upload": "దశ 1 — అప్‌లోడ్",
        "upload_title": "📤 పండు చిత్రాన్ని అప్‌లోడ్ చేయండి",
        "upload_hint": "JPG, JPEG, PNG మద్దతు ఉంది",
        "uploaded_caption": "మీరు అప్‌లోడ్ చేసిన చిత్రం",
        "how_works_label": "ఇది ఎలా పని చేస్తుంది",
        "how_works_title": "🔬 మూడు సరళమైన దశలు",
        "step1": "1️⃣  పండు యొక్క స్పష్టమైన ఫోటో అప్‌లోడ్ చేయండి",
        "step2": "2️⃣  మా CNN మోడల్ తక్షణమే విశ్లేషిస్తుంది",
        "step3": "3️⃣  తాజా / పాడైన ఫలితాన్ని చూడండి",
        "how_works_note": "తాజా పండ్లు నేపథ్యాన్ని చేతన్యవంతం చేస్తాయి 🍎<br>పాడైన పండ్లు కిందకు పడతాయి 🍂",
        "analyzing": "🔍 పండును విశ్లేషిస్తోంది...",
        "result_label": "అంచనా ఫలితం",
        "status_fresh": "తాజా",
        "status_rotten": "పాడైన",
        "confidence": "నమ్మకం స్కోర్",
        "safe": "✅ తినడానికి సురక్షితం",
        "discard": "❌ వినియోగించకండి",
        "msg_fresh": "🌟 అద్భుతం! మీ **{fruit}** తాజాగా ఉంది మరియు తినడానికి సిద్ధంగా ఉంది.",
        "msg_rotten": "⚠️ ఈ **{fruit}** పాడైనట్లు కనిపిస్తోంది. దయచేసి పారవేయండి.",
        "footer": "ఫ్రెష్‌సెన్స్ AI · TensorFlow + Streamlit + CNN తో నిర్మించబడింది",
        "footer_sub": "🍎 తాజా పండ్లు తేలతాయి · 🍂 పాడైన పండ్లు పడతాయి",
        "downloading": "⬇️ AI మోడల్ డౌన్‌లోడ్ అవుతోంది...",
        "select_lang": "🌐 భాష ఎంచుకోండి",
        "voice_section": "🔊 వాయిస్ అవుట్‌పుట్",
        "voice_select_label": "వాయిస్ భాషను ఎంచుకోండి",
        "voice_enable_label": "వాయిస్ ఫలితాన్ని ప్రారంభించండి",
        "autoplay_label": "ఫలితం వచ్చినప్పుడు స్వయంగా వినిపించు",
        "listen_label": "ఫలితాన్ని వినండి",
        "voice_error": "⚠️ వాయిస్ అందుబాటులో లేదు. దయచేసి ఇంటర్నెట్ కనెక్షన్‌ని తనిఖీ చేయండి.",
    },
    "हिन्दी 🇮🇳": {
        "app_title": "फ्रेशसेंस AI",
        "hero_title": "🍎 फ्रेशसेंस AI",
        "hero_sub": "AI आधारित फल ताजगी पहचान प्रणाली · शुरू करने के लिए फल की छवि अपलोड करें",
        "sidebar_about": "प्रोजेक्ट के बारे में",
        "sidebar_desc": "AI आधारित फल ताजगी पहचान।\n\n**ताजे फल** → ऊपर तैरते हैं ✅\n\n**सड़े फल** → नीचे गिरते हैं ❌\n\n**समर्थित फल:**\n- सेब 🍎\n- केला 🍌\n- संतरा 🍊",
        "sidebar_tech": "उपयोग की गई तकनीकें",
        "model_ready": "✅ मॉडल तैयार है",
        "step_upload": "चरण 1 — अपलोड",
        "upload_title": "📤 फल की छवि अपलोड करें",
        "upload_hint": "JPG, JPEG, PNG समर्थित",
        "uploaded_caption": "आपकी अपलोड की गई छवि",
        "how_works_label": "यह कैसे काम करता है",
        "how_works_title": "🔬 तीन सरल चरण",
        "step1": "1️⃣  फल की स्पष्ट फोटो अपलोड करें",
        "step2": "2️⃣  हमारा CNN मॉडल तुरंत विश्लेषण करता है",
        "step3": "3️⃣  ताजा / सड़ा परिणाम देखें",
        "how_works_note": "ताजे फल बैकग्राउंड को जीवंत बनाते हैं 🍎<br>सड़े फल नीचे गिरते हैं 🍂",
        "analyzing": "🔍 फल का विश्लेषण हो रहा है...",
        "result_label": "पूर्वानुमान परिणाम",
        "status_fresh": "ताजा",
        "status_rotten": "सड़ा हुआ",
        "confidence": "विश्वास स्कोर",
        "safe": "✅ खाने के लिए सुरक्षित",
        "discard": "❌ उपभोग न करें",
        "msg_fresh": "🌟 बहुत अच्छा! आपका **{fruit}** ताजा है और खाने के लिए तैयार है।",
        "msg_rotten": "⚠️ यह **{fruit}** सड़ा हुआ लगता है। कृपया इसे फेंक दें।",
        "footer": "फ्रेशसेंस AI · TensorFlow + Streamlit + CNN के साथ बनाया गया",
        "footer_sub": "🍎 ताजे फल तैरते हैं · 🍂 सड़े फल गिरते हैं",
        "downloading": "⬇️ AI मॉडल डाउनलोड हो रहा है...",
        "select_lang": "🌐 भाषा चुनें",
        "voice_section": "🔊 वॉइस आउटपुट",
        "voice_select_label": "वॉइस भाषा चुनें",
        "voice_enable_label": "वॉइस परिणाम सक्षम करें",
        "autoplay_label": "परिणाम आने पर स्वतः सुनाएं",
        "listen_label": "परिणाम सुनें",
        "voice_error": "⚠️ वॉइस उपलब्ध नहीं है। कृपया अपना इंटरनेट कनेक्शन जांचें।",
    },
    "Tamil தமிழ் 🇮🇳": {
        "app_title": "ஃப்ரெஷ்சென்ஸ் AI",
        "hero_title": "🍎 ஃப்ரெஷ்சென்ஸ் AI",
        "hero_sub": "AI சக்தி கொண்ட பழம் புத்துணர்வு கண்டறிதல் · தொடங்க பழத்தின் படத்தை பதிவேற்றவும்",
        "sidebar_about": "திட்டம் பற்றி",
        "sidebar_desc": "AI சக்தி கொண்ட பழம் புத்துணர்வு கண்டறிதல்.\n\n**புத்தம் புதிய பழங்கள்** → மேல் மிதக்கும் ✅\n\n**கெட்டுப்போன பழங்கள்** → கீழே விழும் ❌\n\n**ஆதரிக்கப்படும் பழங்கள்:**\n- ஆப்பிள் 🍎\n- வாழைப்பழம் 🍌\n- ஆரஞ்சு 🍊",
        "sidebar_tech": "பயன்படுத்திய தொழில்நுட்பங்கள்",
        "model_ready": "✅ மாதிரி தயார்",
        "step_upload": "படி 1 — பதிவேற்று",
        "upload_title": "📤 பழத்தின் படத்தை பதிவேற்றவும்",
        "upload_hint": "JPG, JPEG, PNG ஆதரிக்கப்படுகிறது",
        "uploaded_caption": "உங்கள் பதிவேற்றிய படம்",
        "how_works_label": "இது எப்படி வேலை செய்கிறது",
        "how_works_title": "🔬 மூன்று எளிய படிகள்",
        "step1": "1️⃣  பழத்தின் தெளிவான புகைப்படம் பதிவேற்றவும்",
        "step2": "2️⃣  எங்கள் CNN மாதிரி உடனடியாக பகுப்பாய்வு செய்யும்",
        "step3": "3️⃣  புதிய / கெட்ட முடிவை பாருங்கள்",
        "how_works_note": "புத்தம் புதிய பழங்கள் பின்னணியை உயிர்ப்பிக்கும் 🍎<br>கெட்டுப்போன பழங்கள் கீழே விழும் 🍂",
        "analyzing": "🔍 பழத்தை பகுப்பாய்வு செய்கிறது...",
        "result_label": "கணிப்பு முடிவு",
        "status_fresh": "புதியது",
        "status_rotten": "கெட்டுப்போனது",
        "confidence": "நம்பிக்கை மதிப்பெண்",
        "safe": "✅ சாப்பிட பாதுகாப்பானது",
        "discard": "❌ உட்கொள்ள வேண்டாம்",
        "msg_fresh": "🌟 அருமை! உங்கள் **{fruit}** புத்தம் புதியதாக உள்ளது.",
        "msg_rotten": "⚠️ இந்த **{fruit}** கெட்டுப்போனதாக தெரிகிறது. தயவுசெய்து அதை தூக்கி எறியுங்கள்.",
        "footer": "ஃப்ரெஷ்சென்ஸ் AI · TensorFlow + Streamlit + CNN உடன் கட்டப்பட்டது",
        "footer_sub": "🍎 புதிய பழங்கள் மிதக்கும் · 🍂 கெட்ட பழங்கள் விழும்",
        "downloading": "⬇️ AI மாதிரி பதிவிறக்கம் ஆகிறது...",
        "select_lang": "🌐 மொழியைத் தேர்ந்தெடுக்கவும்",
        "voice_section": "🔊 குரல் வெளியீடு",
        "voice_select_label": "குரல் மொழியைத் தேர்ந்தெடுக்கவும்",
        "voice_enable_label": "குரல் முடிவை இயக்கு",
        "autoplay_label": "முடிவு வந்தவுடன் தானாக கேட்கவும்",
        "listen_label": "முடிவைக் கேளுங்கள்",
        "voice_error": "⚠️ குரல் கிடைக்கவில்லை. உங்கள் இணைய இணைப்பைச் சரிபார்க்கவும்.",
    },
    "Kannada ಕನ್ನಡ 🇮🇳": {
        "app_title": "ಫ್ರೆಶ್‌ಸೆನ್ಸ್ AI",
        "hero_title": "🍎 ಫ್ರೆಶ್‌ಸೆನ್ಸ್ AI",
        "hero_sub": "AI ಆಧಾರಿತ ಹಣ್ಣಿನ ತಾಜಾತನ ಪತ್ತೆ · ಪ್ರಾರಂಭಿಸಲು ಹಣ್ಣಿನ ಚಿತ್ರ ಅಪ್‌ಲೋಡ್ ಮಾಡಿ",
        "sidebar_about": "ಯೋಜನೆ ಬಗ್ಗೆ",
        "sidebar_desc": "AI ಆಧಾರಿತ ಹಣ್ಣಿನ ತಾಜಾತನ ಪತ್ತೆ.\n\n**ತಾಜಾ ಹಣ್ಣುಗಳು** → ಮೇಲಕ್ಕೆ ತೇಲುತ್ತವೆ ✅\n\n**ಕೊಳೆತ ಹಣ್ಣುಗಳು** → ಕೆಳಕ್ಕೆ ಬೀಳುತ್ತವೆ ❌\n\n**ಬೆಂಬಲಿತ ಹಣ್ಣುಗಳು:**\n- ಸೇಬು 🍎\n- ಬಾಳೆಹಣ್ಣು 🍌\n- ಕಿತ್ತಳೆ 🍊",
        "sidebar_tech": "ಬಳಸಿದ ತಂತ್ರಜ್ಞಾನಗಳು",
        "model_ready": "✅ ಮಾದರಿ ಸಿದ್ಧ",
        "step_upload": "ಹಂತ 1 — ಅಪ್‌ಲೋಡ್",
        "upload_title": "📤 ಹಣ್ಣಿನ ಚಿತ್ರ ಅಪ್‌ಲೋಡ್ ಮಾಡಿ",
        "upload_hint": "JPG, JPEG, PNG ಬೆಂಬಲಿಸುತ್ತದೆ",
        "uploaded_caption": "ನಿಮ್ಮ ಅಪ್‌ಲೋಡ್ ಮಾಡಿದ ಚಿತ್ರ",
        "how_works_label": "ಇದು ಹೇಗೆ ಕೆಲಸ ಮಾಡುತ್ತದೆ",
        "how_works_title": "🔬 ಮೂರು ಸರಳ ಹಂತಗಳು",
        "step1": "1️⃣  ಹಣ್ಣಿನ ಸ್ಪಷ್ಟ ಫೋಟೋ ಅಪ್‌ಲೋಡ್ ಮಾಡಿ",
        "step2": "2️⃣  ನಮ್ಮ CNN ಮಾದರಿ ತಕ್ಷಣ ವಿಶ್ಲೇಷಿಸುತ್ತದೆ",
        "step3": "3️⃣  ತಾಜಾ / ಕೊಳೆತ ಫಲಿತಾಂಶ ನೋಡಿ",
        "how_works_note": "ತಾಜಾ ಹಣ್ಣುಗಳು ಹಿನ್ನೆಲೆಯನ್ನು ಜೀವಂತಗೊಳಿಸುತ್ತವೆ 🍎<br>ಕೊಳೆತ ಹಣ್ಣುಗಳು ಕೆಳಕ್ಕೆ ಬೀಳುತ್ತವೆ 🍂",
        "analyzing": "🔍 ಹಣ್ಣನ್ನು ವಿಶ್ಲೇಷಿಸಲಾಗುತ್ತಿದೆ...",
        "result_label": "ಮುನ್ಸೂಚನೆ ಫಲಿತಾಂಶ",
        "status_fresh": "ತಾಜಾ",
        "status_rotten": "ಕೊಳೆತ",
        "confidence": "ವಿಶ್ವಾಸ ಅಂಕ",
        "safe": "✅ ತಿನ್ನಲು ಸುರಕ್ಷಿತ",
        "discard": "❌ ಸೇವಿಸಬೇಡಿ",
        "msg_fresh": "🌟 ಅದ್ಭುತ! ನಿಮ್ಮ **{fruit}** ತಾಜಾವಾಗಿದೆ ಮತ್ತು ತಿನ್ನಲು ಸಿದ್ಧ.",
        "msg_rotten": "⚠️ ಈ **{fruit}** ಕೊಳೆತಂತೆ ಕಾಣುತ್ತದೆ. ದಯವಿಟ್ಟು ಅದನ್ನು ಎಸೆಯಿರಿ.",
        "footer": "ಫ್ರೆಶ್‌ಸೆನ್ಸ್ AI · TensorFlow + Streamlit + CNN ನೊಂದಿಗೆ ನಿರ್ಮಿಸಲಾಗಿದೆ",
        "footer_sub": "🍎 ತಾಜಾ ಹಣ್ಣುಗಳು ತೇಲುತ್ತವೆ · 🍂 ಕೊಳೆತ ಹಣ್ಣುಗಳು ಬೀಳುತ್ತವೆ",
        "downloading": "⬇️ AI ಮಾದರಿ ಡೌನ್‌ಲೋಡ್ ಆಗುತ್ತಿದೆ...",
        "select_lang": "🌐 ಭಾಷೆ ಆಯ್ಕೆ ಮಾಡಿ",
        "voice_section": "🔊 ಧ್ವನಿ ಔಟ್‌ಪುಟ್",
        "voice_select_label": "ಧ್ವನಿ ಭಾಷೆಯನ್ನು ಆಯ್ಕೆಮಾಡಿ",
        "voice_enable_label": "ಧ್ವನಿ ಫಲಿತಾಂಶವನ್ನು ಸಕ್ರಿಯಗೊಳಿಸಿ",
        "autoplay_label": "ಫಲಿತಾಂಶ ಬಂದಾಗ ಸ್ವಯಂಚಾಲಿತವಾಗಿ ಪ್ಲೇ ಮಾಡಿ",
        "listen_label": "ಫಲಿತಾಂಶವನ್ನು ಕೇಳಿ",
        "voice_error": "⚠️ ಧ್ವನಿ ಲಭ್ಯವಿಲ್ಲ. ದಯವಿಟ್ಟು ನಿಮ್ಮ ಇಂಟರ್ನೆಟ್ ಸಂಪರ್ಕವನ್ನು ಪರಿಶೀಲಿಸಿ.",
    },
    "Malayalam മലയാളം 🇮🇳": {
        "app_title": "ഫ്രെഷ്‌സെൻസ് AI",
        "hero_title": "🍎 ഫ്രെഷ്‌സെൻസ് AI",
        "hero_sub": "AI-ശക്തിയുള്ള പഴം പുതുമ കണ്ടെത്തൽ · ആരംഭിക്കാൻ ഒരു പഴത്തിന്റെ ചിത്രം അപ്‌ലോഡ് ചെയ്യുക",
        "sidebar_about": "പ്രോജക്ടിനെക്കുറിച്ച്",
        "sidebar_desc": "AI-ശക്തിയുള്ള പഴം പുതുമ കണ്ടെത്തൽ.\n\n**പുതിയ പഴങ്ങൾ** → മുകളിലേക്ക് പൊങ്ങുന്നു ✅\n\n**ചീഞ്ഞ പഴങ്ങൾ** → താഴേക്ക് വീഴുന്നു ❌\n\n**പിന്തുണയ്ക്കുന്ന പഴങ്ങൾ:**\n- ആപ്പിൾ 🍎\n- വാഴപ്പഴം 🍌\n- ഓറഞ്ച് 🍊",
        "sidebar_tech": "ഉപയോഗിച്ച സാങ്കേതികവിദ്യകൾ",
        "model_ready": "✅ മോഡൽ തയ്യാർ",
        "step_upload": "ഘട്ടം 1 — അപ്‌ലോഡ്",
        "upload_title": "📤 പഴത്തിന്റെ ചിത്രം അപ്‌ലോഡ് ചെയ്യുക",
        "upload_hint": "JPG, JPEG, PNG പിന്തുണയ്ക്കുന്നു",
        "uploaded_caption": "നിങ്ങൾ അപ്‌ലോഡ് ചെയ്ത ചിത്രം",
        "how_works_label": "ഇത് എങ്ങനെ പ്രവർത്തിക്കുന്നു",
        "how_works_title": "🔬 മൂന്ന് ലളിതമായ ഘട്ടങ്ങൾ",
        "step1": "1️⃣  പഴത്തിന്റെ വ്യക്തമായ ഫോട്ടോ അപ്‌ലോഡ് ചെയ്യുക",
        "step2": "2️⃣  ഞങ്ങളുടെ CNN മോഡൽ ഉടനടി വിശകലനം ചെയ്യുന്നു",
        "step3": "3️⃣  പുതിയ / ചീഞ്ഞ ഫലം കാണുക",
        "how_works_note": "പുതിയ പഴങ്ങൾ പശ്ചാത്തലം ഉജ്ജ്വലമാക്കുന്നു 🍎<br>ചീഞ്ഞ പഴങ്ങൾ താഴേക്ക് വീഴുന്നു 🍂",
        "analyzing": "🔍 പഴം വിശകലനം ചെയ്യുന്നു...",
        "result_label": "പ്രവചന ഫലം",
        "status_fresh": "പുതിയത്",
        "status_rotten": "ചീഞ്ഞത്",
        "confidence": "ആത്മവിശ്വാസ സ്കോർ",
        "safe": "✅ കഴിക്കാൻ സുരക്ഷിതം",
        "discard": "❌ ഉപഭോഗിക്കരുത്",
        "msg_fresh": "🌟 മികച്ചത്! നിങ്ങളുടെ **{fruit}** പുതിയതും കഴിക്കാൻ തയ്യാറുമാണ്.",
        "msg_rotten": "⚠️ ഈ **{fruit}** ചീഞ്ഞതായി കാണുന്നു. ദയവായി ഇത് ഉപേക്ഷിക്കുക.",
        "footer": "ഫ്രെഷ്‌സെൻസ് AI · TensorFlow + Streamlit + CNN ഉപയോഗിച്ച് നിർമ്മിച്ചത്",
        "footer_sub": "🍎 പുതിയ പഴങ്ങൾ പൊങ്ങുന്നു · 🍂 ചീഞ്ഞ പഴങ്ങൾ വീഴുന്നു",
        "downloading": "⬇️ AI മോഡൽ ഡൗൺലോഡ് ചെയ്യുന്നു...",
        "select_lang": "🌐 ഭാഷ തിരഞ്ഞെടുക്കുക",
        "voice_section": "🔊 വോയ്സ് ഔട്ട്പുട്ട്",
        "voice_select_label": "വോയ്സ് ഭാഷ തിരഞ്ഞെടുക്കുക",
        "voice_enable_label": "വോയ്സ് ഫലം പ്രവർത്തനക്ഷമമാക്കുക",
        "autoplay_label": "ഫലം വരുമ്പോൾ സ്വയമേവ പ്ലേ ചെയ്യുക",
        "listen_label": "ഫലം കേൾക്കുക",
        "voice_error": "⚠️ വോയ്സ് ലഭ്യമല്ല. നിങ്ങളുടെ ഇന്റർനെറ്റ് കണക്ഷൻ പരിശോധിക്കുക.",
    },
}

# ================================================================
# VOICE / TTS CONFIGURATION
# ================================================================

# gTTS language codes for each supported language
VOICE_LANG_CODES = {
    "English 🇬🇧": "en",
    "తెలుగు 🇮🇳": "te",
    "हिन्दी 🇮🇳": "hi",
    "Tamil தமிழ் 🇮🇳": "ta",
    "Kannada ಕನ್ನಡ 🇮🇳": "kn",
    "Malayalam മലయாళം 🇮🇳": "ml",
}

# Fruit names translated per language, so the spoken (and displayed)
# fruit name matches the chosen language instead of staying in English
FRUIT_NAMES = {
    "English 🇬🇧": {"apple": "Apple", "banana": "Banana", "orange": "Orange"},
    "తెలుగు 🇮🇳": {"apple": "యాపిల్", "banana": "అరటి", "orange": "నారంగి"},
    "हिन्दी 🇮🇳": {"apple": "सेब", "banana": "केला", "orange": "संतरा"},
    "Tamil தமிழ் 🇮🇳": {"apple": "ஆப்பிள்", "banana": "வாழைப்பழம்", "orange": "ஆரஞ்சு"},
    "Kannada ಕನ್ನಡ 🇮🇳": {"apple": "ಸೇಬು", "banana": "ಬಾಳೆಹಣ್ಣು", "orange": "ಕಿತ್ತಳೆ"},
    "Malayalam മലയാളം 🇮🇳": {"apple": "ആപ്പിൾ", "banana": "വാഴപ്പഴം", "orange": "ഓറഞ്ച്"},
}

# Strip markdown bold + emoji so the TTS engine reads clean, natural sentences
_EMOJI_PATTERN = re.compile(
    "["
    "\U0001F300-\U0001FAFF"
    "\U00002600-\U000027BF"
    "\U0001F1E0-\U0001F1FF"
    "\uFE0F"
    "\u200D"
    "]+",
    flags=re.UNICODE,
)


def clean_for_speech(text: str) -> str:
    text = text.replace("**", "")
    text = _EMOJI_PATTERN.sub("", text)
    return re.sub(r"\s+", " ", text).strip()


@st.cache_data(show_spinner=False)
def generate_speech_bytes(text: str, lang_code: str):
    """Generate MP3 audio bytes for the given text using gTTS. Returns None on failure."""
    try:
        tts = gTTS(text=text, lang=lang_code)
        mp3_fp = io.BytesIO()
        tts.write_to_fp(mp3_fp)
        mp3_fp.seek(0)
        return mp3_fp.read()
    except Exception:
        return None


# ================================================================
# CSS
# ================================================================

def get_animation_css(state="idle"):
    if state == "fresh":
        bg = "linear-gradient(135deg, #d4f7dc 0%, #a8edba 40%, #f0fff4 100%)"
        anim_name = "floatUp"
        anim_css = """
        @keyframes floatUp {
            0%   { transform: translateY(0px) rotate(0deg);    opacity: 1; }
            50%  { transform: translateY(-60px) rotate(12deg);  opacity: 0.9; }
            100% { transform: translateY(-130px) rotate(-8deg); opacity: 0; }
        }"""
        fruit_filter = "drop-shadow(0 0 12px rgba(34,197,94,0.7))"
    elif state == "rotten":
        bg = "linear-gradient(135deg, #fff1e6 0%, #ffd6b0 40%, #fff8f0 100%)"
        anim_name = "fallDown"
        anim_css = """
        @keyframes fallDown {
            0%   { transform: translateY(0px) rotate(0deg);   opacity: 1; }
            40%  { transform: translateY(40px) rotate(-15deg); opacity: 0.85; }
            100% { transform: translateY(110px) rotate(20deg); opacity: 0; }
        }"""
        fruit_filter = "drop-shadow(0 0 12px rgba(234,88,12,0.7)) grayscale(40%)"
    else:
        bg = "linear-gradient(135deg, #f0fff4 0%, #e6f7ff 50%, #fff8f0 100%)"
        anim_name = "floatGentle"
        anim_css = """
        @keyframes floatGentle {
            0%   { transform: translateY(0px) rotate(0deg); }
            50%  { transform: translateY(-25px) rotate(5deg); }
            100% { transform: translateY(0px) rotate(0deg); }
        }"""
        fruit_filter = "drop-shadow(0 2px 8px rgba(0,0,0,0.15))"

    return f"""
    <style>
    .stApp {{
        background: {bg};
        transition: background 1.2s ease;
        font-family: 'Segoe UI', system-ui, sans-serif;
    }}
    /* ── REMOVE TOP WHITESPACE & CENTER MAIN CONTENT ── */
    [data-testid="stAppViewContainer"] {{
        display: flex !important;
    }}
    section.main,
    [data-testid="stMain"] {{
        display: flex !important;
        flex-direction: column !important;
        align-items: center !important;
        width: 100% !important;
    }}
    .main .block-container,
    [data-testid="stMainBlockContainer"],
    [data-testid="block-container"] {{
        padding-top: 0.5rem !important;
        padding-bottom: 1rem !important;
        padding-left: 2rem !important;
        padding-right: 2rem !important;
        max-width: 1000px !important;
        width: 100% !important;
        margin-left: auto !important;
        margin-right: auto !important;
    }}
    header[data-testid="stHeader"] {{
        height: 0 !important;
        min-height: 0 !important;
        display: none !important;
    }}
    div[data-testid="stToolbar"] {{
        display: none !important;
    }}
    {anim_css}
    .fruit-bg {{
        position: fixed; inset: 0;
        pointer-events: none; z-index: 0; overflow: hidden;
    }}
    .fruit-bg span {{
        position: absolute; font-size: 2.2rem;
        filter: {fruit_filter};
        animation: {anim_name} 3.5s ease-in-out infinite;
        opacity: 0.55; user-select: none;
    }}
    .fruit-bg span:nth-child(1)  {{ left:5%;  top:10%; animation-delay:0s;   animation-duration:3.2s; font-size:2.8rem; }}
    .fruit-bg span:nth-child(2)  {{ left:15%; top:70%; animation-delay:0.6s; animation-duration:4.0s; font-size:1.8rem; }}
    .fruit-bg span:nth-child(3)  {{ left:25%; top:30%; animation-delay:1.1s; animation-duration:3.6s; font-size:2.5rem; }}
    .fruit-bg span:nth-child(4)  {{ left:40%; top:80%; animation-delay:0.3s; animation-duration:4.2s; font-size:2.0rem; }}
    .fruit-bg span:nth-child(5)  {{ left:55%; top:15%; animation-delay:1.8s; animation-duration:3.0s; font-size:3.0rem; }}
    .fruit-bg span:nth-child(6)  {{ left:65%; top:60%; animation-delay:0.9s; animation-duration:3.8s; font-size:1.7rem; }}
    .fruit-bg span:nth-child(7)  {{ left:75%; top:40%; animation-delay:1.4s; animation-duration:3.4s; font-size:2.3rem; }}
    .fruit-bg span:nth-child(8)  {{ left:85%; top:85%; animation-delay:0.5s; animation-duration:4.5s; font-size:1.9rem; }}
    .fruit-bg span:nth-child(9)  {{ left:90%; top:20%; animation-delay:2.0s; animation-duration:3.1s; font-size:2.6rem; }}
    .fruit-bg span:nth-child(10) {{ left:50%; top:50%; animation-delay:1.5s; animation-duration:3.9s; font-size:2.1rem; }}
    .fruit-bg span:nth-child(11) {{ left:10%; top:50%; animation-delay:0.8s; animation-duration:4.1s; font-size:1.6rem; }}
    .fruit-bg span:nth-child(12) {{ left:33%; top:5%;  animation-delay:2.3s; animation-duration:3.3s; font-size:2.4rem; }}
    .hero-title {{
        font-size: 3.0rem; font-weight: 800; text-align: center;
        letter-spacing: -1px; color: #14532d; margin-bottom: 0.1rem;
        text-shadow: 0 2px 12px rgba(34,197,94,0.15);
    }}
    .hero-sub {{
        text-align: center; font-size: 1.05rem; color: #6b7280;
        margin-bottom: 1.2rem; letter-spacing: 0.5px;
    }}
    .card {{
        background: rgba(255,255,255,0.82);
        backdrop-filter: blur(10px); -webkit-backdrop-filter: blur(10px);
        border-radius: 20px; padding: 22px 26px;
        box-shadow: 0 4px 24px rgba(0,0,0,0.08);
        border: 1px solid rgba(255,255,255,0.6);
        margin-bottom: 14px; position: relative; z-index: 1;
    }}
    .result-fresh {{ border-left: 5px solid #22c55e; background: rgba(240,255,244,0.88); }}
    .result-rotten {{ border-left: 5px solid #ef4444; background: rgba(255,241,235,0.88); }}
    .result-title  {{ font-size: 1.9rem; font-weight: 700; margin-bottom: 6px; }}
    .result-status {{ font-size: 1.2rem;  font-weight: 600; margin-bottom: 4px; }}
    .confidence-label {{ font-size: 0.9rem; color: #4b5563; margin-bottom: 2px; }}
    .section-label {{
        font-size: 0.72rem; font-weight: 700; letter-spacing: 2px;
        text-transform: uppercase; color: #9ca3af; margin-bottom: 4px;
    }}
    .badge-fresh  {{ display:inline-block; background:#dcfce7; color:#15803d; border-radius:999px; padding:4px 16px; font-size:0.85rem; font-weight:700; margin-top:8px; }}
    .badge-rotten {{ display:inline-block; background:#fee2e2; color:#b91c1c; border-radius:999px; padding:4px 16px; font-size:0.85rem; font-weight:700; margin-top:8px; }}
    [data-testid="stSidebar"] {{
        background: rgba(255,255,255,0.82) !important;
        backdrop-filter: blur(12px);
        border-right: 1px solid rgba(0,0,0,0.06);
    }}
    .footer {{
        text-align:center; color:#9ca3af; font-size:0.82rem;
        padding:16px 0 6px; position:relative; z-index:1;
    }}
    [data-testid="stFileUploader"] {{ position:relative; z-index:1; }}
    .stProgress > div > div {{ border-radius:999px; }}
    /* language pill */
    .lang-pill {{
        display:inline-block; background:rgba(255,255,255,0.7);
        border:1px solid #d1fae5; border-radius:999px;
        padding:2px 12px; font-size:0.82rem; color:#14532d;
        font-weight:600; margin-left:6px;
    }}
    </style>
    """


def get_fruit_bg_html(state="idle", predicted_class=None):
    if state == "fresh":
        if predicted_class and "banana" in predicted_class:
            fruits = ["🍌","🍎","🍊","🍌","🍎","🍊","🍌","🍎","🍊","🍌","🍎","🍊"]
        elif predicted_class and "orange" in predicted_class:
            fruits = ["🍊","🍎","🍌","🍊","🍊","🍎","🍌","🍊","🍊","🍎","🍌","🍊"]
        else:
            fruits = ["🍎","🍌","🍊","🍎","🍌","🍊","🍎","🍌","🍊","🍎","🍌","🍊"]
    elif state == "rotten":
        fruits = ["🍂","🤢","🍂","🤢","🍂","🤢","🍂","🤢","🍂","🤢","🍂","🤢"]
    else:
        fruits = ["🍎","🍌","🍊","🍋","🍇","🍓","🍎","🍌","🍊","🍋","🍇","🍓"]
    spans = "".join(f"<span>{f}</span>" for f in fruits)
    return f'<div class="fruit-bg">{spans}</div>'


# ================================================================
# SESSION STATE
# ================================================================

if "anim_state" not in st.session_state:
    st.session_state.anim_state = "idle"
if "predicted_class" not in st.session_state:
    st.session_state.predicted_class = None

# ================================================================
# SIDEBAR — language selector + voice settings
# ================================================================

with st.sidebar:
    lang_choice = st.selectbox(
        "🌐 Language / భాష / भाषा",
        list(LANGUAGES.keys()),
        index=0
    )
    T = LANGUAGES[lang_choice]  # active translation dict

    st.markdown(f"# 🍎 {T['app_title']}")
    st.write("---")
    st.markdown(f"## {T['sidebar_about']}")
    st.write(T["sidebar_desc"])
    st.write("---")
    st.markdown(f"### {T['sidebar_tech']}")
    st.write("""
    - TensorFlow / CNN
    - Streamlit
    - Python
    - gTTS (Voice Output)
    """)
    st.write("---")
    st.success(T["model_ready"])

    # ---------------- VOICE SETTINGS ----------------
    st.write("---")
    st.markdown(f"### {T['voice_section']}")
    voice_lang_choice = st.selectbox(
        T["voice_select_label"],
        list(LANGUAGES.keys()),
        index=list(LANGUAGES.keys()).index(lang_choice),
        key="voice_lang_select",
    )
    voice_enabled = st.checkbox(T["voice_enable_label"], value=True, key="voice_enable_checkbox")
    autoplay_enabled = st.checkbox(T["autoplay_label"], value=True, key="autoplay_checkbox")
    st.caption("🌐 Voice output needs an internet connection (Google TTS).")

# ================================================================
# CSS + BACKGROUND
# ================================================================

st.markdown(get_animation_css(st.session_state.anim_state), unsafe_allow_html=True)
st.markdown(get_fruit_bg_html(st.session_state.anim_state, st.session_state.predicted_class), unsafe_allow_html=True)

# ================================================================
# HEADER
# ================================================================

st.markdown(f'<div class="hero-title">{T["hero_title"]}</div>', unsafe_allow_html=True)
st.markdown(f'<div class="hero-sub">{T["hero_sub"]}</div>', unsafe_allow_html=True)

# ================================================================
# MODEL LOAD
# ================================================================

MODEL_PATH = "fruit_freshness_model.keras"

if not os.path.exists(MODEL_PATH):
    with st.spinner(T["downloading"]):
        gdown.download(
            "https://drive.google.com/uc?id=1JHax46671U2VmEyEfqIP51B5bySHLEzb",
            MODEL_PATH,
            quiet=False
        )

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()

class_names = [
    "apple", "banana", "orange",
    "rottenapples", "rottenbanana", "rottenoranges"
]

# ================================================================
# LAYOUT
# ================================================================

col1, col2 = st.columns([1.2, 1], gap="medium")

with col1:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown(f'<div class="section-label">{T["step_upload"]}</div>', unsafe_allow_html=True)
    st.markdown(f"### {T['upload_title']}")
    uploaded_file = st.file_uploader(
        T["upload_hint"],
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed"
    )
    st.markdown('</div>', unsafe_allow_html=True)

    if uploaded_file:
        image = Image.open(uploaded_file)
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.image(image, caption=T["uploaded_caption"], use_column_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

with col2:
    if not uploaded_file:
        st.markdown(f"""
        <div class="card">
          <div class="section-label">{T["how_works_label"]}</div>
          <h3>{T["how_works_title"]}</h3>
          <p>{T["step1"]}</p>
          <p>{T["step2"]}</p>
          <p>{T["step3"]}</p>
          <hr style="border-color:#e5e7eb; margin:14px 0;">
          <p style="color:#6b7280; font-size:0.9rem;">{T["how_works_note"]}</p>
        </div>
        """, unsafe_allow_html=True)

# ================================================================
# PREDICTION
# ================================================================

if uploaded_file:
    img = image.resize((224, 224))
    img_array = np.array(img)
    if len(img_array.shape) == 2:
        img_array = np.stack([img_array] * 3, axis=-1)
    if img_array.shape[-1] == 4:
        img_array = img_array[:, :, :3]
    img_array = np.expand_dims(img_array, axis=0).astype("float32")

    with st.spinner(T["analyzing"]):
        prediction = model.predict(img_array)

    predicted_class = class_names[np.argmax(prediction)]
    confidence = float(np.max(prediction)) * 100
    is_rotten = "rotten" in predicted_class

    st.session_state.anim_state = "rotten" if is_rotten else "fresh"
    st.session_state.predicted_class = predicted_class

    # Extract a clean fruit key (apple / banana / orange) regardless of fresh/rotten class
    fruit_key = predicted_class.replace("rotten", "")
    if fruit_key.endswith("s"):
        fruit_key = fruit_key[:-1]

    if is_rotten:
        condition_label = T["status_rotten"]
        emoji = "❌"
        card_class = "result-rotten"
        badge_class = "badge-rotten"
        badge_text = T["discard"]
        state_icon = "🍂"
        conf_color = "#b91c1c"
    else:
        condition_label = T["status_fresh"]
        emoji = "✅"
        card_class = "result-fresh"
        badge_class = "badge-fresh"
        badge_text = T["safe"]
        state_icon = "🌿"
        conf_color = "#15803d"

    # Fruit name translated into the UI display language
    fruit_name = FRUIT_NAMES.get(lang_choice, FRUIT_NAMES["English 🇬🇧"]).get(
        fruit_key, fruit_key.capitalize()
    )
    # Fruit name translated into the chosen VOICE language (may differ from UI language)
    voice_fruit_name = FRUIT_NAMES.get(voice_lang_choice, FRUIT_NAMES["English 🇬🇧"]).get(
        fruit_key, fruit_key.capitalize()
    )

    # Re-inject updated animation
    st.markdown(get_animation_css(st.session_state.anim_state), unsafe_allow_html=True)
    st.markdown(get_fruit_bg_html(st.session_state.anim_state, predicted_class), unsafe_allow_html=True)

    st.markdown(f"""
    <div class="card {card_class}">
      <div class="section-label">{T["result_label"]}</div>
      <div class="result-title">{emoji} {fruit_name}</div>
      <div class="result-status">Status: <strong>{condition_label}</strong> &nbsp;{state_icon}</div>
      <div class="confidence-label">{T["confidence"]}</div>
      <div style="font-size:2rem; font-weight:800; color:{conf_color};">{confidence:.1f}%</div>
      <span class="{badge_class}">{badge_text}</span>
    </div>
    """, unsafe_allow_html=True)

    st.progress(int(confidence))
    st.markdown('<div style="height:6px;"></div>', unsafe_allow_html=True)

    if not is_rotten:
        st.success(T["msg_fresh"].format(fruit=fruit_name))
    else:
        st.error(T["msg_rotten"].format(fruit=fruit_name))

    # ---------------- VOICE OUTPUT ----------------
    if voice_enabled:
        T_voice = LANGUAGES[voice_lang_choice]
        voice_template = T_voice["msg_rotten"] if is_rotten else T_voice["msg_fresh"]
        voice_text = clean_for_speech(voice_template.format(fruit=voice_fruit_name))

        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown(f'<div class="section-label">{T_voice["listen_label"]}</div>', unsafe_allow_html=True)

        audio_bytes = generate_speech_bytes(
            voice_text, VOICE_LANG_CODES.get(voice_lang_choice, "en")
        )
        if audio_bytes:
            try:
                st.audio(audio_bytes, format="audio/mp3", autoplay=autoplay_enabled)
            except TypeError:
                # Older Streamlit versions that don't support the 'autoplay' kwarg
                st.audio(audio_bytes, format="audio/mp3")
        else:
            st.warning(T["voice_error"])

        st.markdown('</div>', unsafe_allow_html=True)

# ================================================================
# FOOTER
# ================================================================

st.markdown(f"""
<div class="footer">
    {T["footer"]}<br>
    <span style="font-size:0.75rem;">{T["footer_sub"]}</span>
</div>
""", unsafe_allow_html=True)