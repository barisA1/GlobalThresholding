
# 📊 Global Thresholding with Pixel Intensity Histogram

Bu Python programı, kullanıcıların piksel yoğunluk değerlerine dayanan bir eşik değerini hesaplamasına yardımcı olur. Program, matplotlib kullanarak bir histogram oluşturur ve hesaplanan eşik değerini grafikte gösterir.

## 📄 İçindekiler
- [Giriş](#-giriş)
- [Nasıl Çalışır?](#-nasıl-çalışır)
- [Kurulum](#-kurulum)
- [Kullanım](#-kullanım)
- [Sonuçlar](#-sonuçlar)


## 📘 Giriş
Bu proje, piksel yoğunluklarının histogramını analiz ederek optimal bir eşik değeri hesaplar. Eşik değeri, piksel yoğunluklarını iki gruba ayırmak için kullanılır: eşik değerinden büyük olanlar ve küçük veya eşit olanlar.

## 🔍 Nasıl Çalışır?
1. **Veri Girişi:**
   - Piksel yoğunluk değerleri ve sayıları manuel olarak girilir.
   
2. **Eşik Değeri Hesaplama:**
   - Başlangıç eşik değeri, tüm piksel değerlerinin ortalaması olarak hesaplanır.
   - Piksel yoğunlukları, eşik değerine göre iki gruba ayrılır.
   - Her grubun ortalama yoğunluk değeri hesaplanır ve yeni eşik değeri olarak kullanılır.
   - Eşik değeri, belirli bir yakınsama değerine ulaşana kadar güncellenir.
   
3. **Histogram ve Eşik Değeri Grafiği:**
   - Piksel yoğunluklarının histogramı çizilir ve hesaplanan eşik değeri grafikte gösterilir.

## 🛠️ Kurulum
Bu projeyi çalıştırmak için aşağıdaki kütüphanelerin kurulu olması gerekmektedir:
- pandas
- numpy
- matplotlib

Gerekli kütüphaneleri şu komutla yükleyebilirsiniz:
```bash
pip install pandas numpy matplotlib
```

## 💻 Kullanım
Aşağıdaki kodu çalıştırarak projeyi kullanabilirsiniz:

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Manually input the pixel intensity and count data
data = {
    'Intensity': [100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150],
    'Count': [12, 18, 32, 48, 52, 65, 55, 42, 32, 16, 10, 5, 18, 25, 32, 40, 65, 43, 32, 20, 10, 4]
}

df = pd.DataFrame(data)

# Convert the dataframe to a full list of pixel values
pixels = []
for index, row in df.iterrows():
    pixels.extend([row['Intensity']] * row['Count'])

pixels = np.array(pixels)

# Initialize the threshold
T0 = np.mean(pixels)

# Convergence threshold
threshold = 0.5

while True:
    # Divide into two groups
    G1 = pixels[pixels > T0]
    G2 = pixels[pixels <= T0]

    # Calculate mean values
    m1 = np.mean(G1) if len(G1) > 0 else 0
    m2 = np.mean(G2) if len(G2) > 0 else 0

    # New threshold
    T1 = (m1 + m2) / 2

    # Check for convergence
    if abs(T1 - T0) < threshold:
        break

    T0 = T1

# Print optimum threshold value
print(f"Optimum Threshold Value: {T0}")

# Calculate number of pixels above and below the threshold
G1_count = len(pixels[pixels > T0])
G2_count = len(pixels[pixels <= T0])

# Print results
print(f"Number of pixels above threshold: {G1_count}")
print(f"Number of pixels below or equal to threshold: {G2_count}")

# Plot the histogram and threshold
plt.hist(pixels, bins=range(100, 151), edgecolor='black')
plt.axvline(T0, color='red', linestyle='dashed', linewidth=1)
plt.title('Pixel Intensity Histogram with Threshold')
plt.xlabel('Intensity')
plt.ylabel('Frequency')

# Save the histogram as an image file
plt.savefig('histogram.png')

# Show the histogram
plt.show()
```

## 📊 Sonuçlar
Aşağıdaki grafikte, piksel yoğunluklarının histogramını ve hesaplanan eşik değerini görebilirsiniz. Kırmızı kesikli çizgi, optimal eşik değerini temsil eder.

![Histogram](histogram.png)

