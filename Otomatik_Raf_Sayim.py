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
        with st.spinner("Yapay zeka ürünleri sayıyor... Lütfen bekleyin."):
            tampon = io.BytesIO()
            img.convert("RGB").save(tampon, format="JPEG")
            img_b64 = base64.b64encode(tampon.getvalue()).decode("utf-8")
            
            prompt = """
            Bu görseldeki ürün markalarını ve doğrudan öne (bize) dönük olan (ön yüz) sayılarını say. 
            Çıktıyı SADECE aşağıdaki JSON formatında ver. Başka hiçbir açıklama, selamlama veya markdown işareti ekleme: 
            [{"marka": "MarkaAdı", "on_yuz_sayisi": 0}]
            """
            
            url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-3.8-flash:generateContent?key={API_KEY}"
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
                # 503 yoğunluk hatalarına karşı 3 kez otomatik tekrar deneme mekanizması
                max_deneme = 3
                bekleme = 2
                cevap_json = {}
                
                for deneme in range(max_deneme):
                    response = requests.post(url, headers=headers, json=payload, timeout=60)
                    cevap_json = response.json()
                    
                    if response.status_code == 503 and deneme < max_deneme - 1:
                        time.sleep(bekleme)
                        bekleme *= 2
                        continue
                    break
                
                if response.status_code != 200:
                    hata_mesaji = cevap_json.get("error", {}).get("message", "Bilinmeyen hata")
                    st.error(f"API Hatası ({response.status_code}): {hata_mesaji}")
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
