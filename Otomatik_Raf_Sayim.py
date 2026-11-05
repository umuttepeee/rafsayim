import streamlit as st
import json
import base64
import requests
import pandas as pd
from PIL import Image
import io
import time

# 1. API Anahtarını Alma
try:
    API_KEY = st.secrets["GEMINI_API_KEY"]
except Exception:
    API_KEY = "umutapiapplication"

st.title("📸 Otomatik Raf Sayım Aracı")
st.write("Markaları ve ön yüz sayılarını yapay zeka ile anında tespit edin.")

# 2. Görüntü Yükleme Seçenekleri (Sekmeler)
tab1, tab2 = st.tabs(["📷 Kamera ile Çek", "📂 Fotoğraf Yükle"])

resim = None

with tab1:
    kamera_resmi = st.camera_input("Rafın fotoğrafını çekin")
    if kamera_resmi:
        resim = kamera_resmi

with tab2:
    yuklenen_resim = st.file_uploader("Galeriden veya bilgisayardan fotoğraf seçin", type=["jpg", "jpeg", "png"])
    if yuklenen_resim:
        resim = yuklenen_resim

# 3. Analiz Kısmı
if resim is not None:
    img = Image.open(resim)
    st.image(img, caption="İşlenecek Görsel", use_container_width=True)
    
    if st.button("Analiz Et"):
        with st.spinner("Modeller doğrulanıyor ve ürünler sayılıyor..."):
            tampon = io.BytesIO()
            img.convert("RGB").save(tampon, format="JPEG")
            img_b64 = base64.b64encode(tampon.getvalue()).decode("utf-8")
            
            prompt = """
            Bu görseldeki ürün markalarını ve doğrudan öne (bize) dönük olan (ön yüz) sayılarını say. 
            Çıktıyı SADECE aşağıdaki JSON formatında ver. Başka hiçbir açıklama, selamlama veya markdown işareti ekleme: 
            [{"marka": "MarkaAdı", "on_yuz_sayisi": 0}]
            """
            
            headers = {"Content-Type": "application/json"}
            payload = {
                "contents": [
                    {
                        "parts": [
                            {"text": prompt},
                            {
                                "inline_data": {
                                    "mime_type": "image/jpeg",
                                    "data": img_b64
                                }
                            }
                        ]
                    }
                ]
            }
            
            try:
                # 1. Adım: API Anahtarınızın yetkili olduğu modelleri Google'dan doğrudan çek
                list_url = f"https://generativelanguage.googleapis.com/v1beta/models?key={API_KEY}"
                list_res = requests.get(list_url, timeout=30)
                
                aktif_modeller = []
                if list_res.status_code == 200:
                    modeller_verisi = list_res.json().get("models", [])
                    # generateContent destekleyen ve gemini içeren modelleri filtrele
                    for m in modeller_verisi:
                        methods = m.get("supportedGenerationMethods", [])
                        m_name = m.get("name", "")
                        if "generateContent" in methods and "gemini" in m_name:
                            aktif_modeller.append(m_name)
                
                # Eğer liste çekilemezse varsayılan model havuzunu kullan
                if not aktif_modeller:
                    aktif_modeller = ["models/gemini-2.5-flash", "models/gemini-flash-latest"]

                cevap_json = {}
                basarili = False
                son_hata = ""

                # 2. Adım: Yetkili modelleri sırayla dene
                for model_tam_adi in aktif_modeller:
                    url = f"https://generativelanguage.googleapis.com/v1beta/{model_tam_adi}:generateContent?key={API_KEY}"
                    
                    for deneme in range(2):
                        response = requests.post(url, headers=headers, json=payload, timeout=60)
                        
                        try:
                            cevap_json = response.json()
                        except Exception:
                            cevap_json = {"error": {"message": response.text}}
                            
                        if response.status_code == 200:
                            basarili = True
                            break
                        elif response.status_code == 503:
                            time.sleep(2)
                            continue
                        else:
                            son_hata = cevap_json.get("error", {}).get("message", f"HTTP {response.status_code}")
                            break
                            
                    if basarili:
                        break
                
                if not basarili:
                    st.error(f"API Hatası: {son_hata}")
                else:
                    sonuc_metni = cevap_json["candidates"][0]["content"]["parts"][0]["text"]
                    sonuc_metni = sonuc_metni.replace("```json", "").replace("```", "").strip()
                    veri = json.loads(sonuc_metni)
                    
                    st.success("✅ Sayım başarıyla tamamlandı!")
                    df = pd.DataFrame(veri)
                    st.dataframe(df, use_container_width=True)
                    
                    st.download_button(
                        label="Veriyi CSV Olarak İndir",
                        data=df.to_csv(index=False).encode("utf-8"),
                        file_name="raf_sayimi.csv",
                        mime="text/csv"
                    )

            except Exception as e:
                st.error("Bir hata oluştu. Lütfen görseli ve bağlantıyı kontrol edin.")
                st.write("Hata Detayı:", e)
