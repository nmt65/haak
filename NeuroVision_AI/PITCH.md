# NeuroVision AI — pitch de hackathon

## Mesajul central

**Transformăm ce vede camera în conturul fiecărui lucru din fața ta, desenat cu câteva sute de puncte de lumină, plus direcția lui prin sunet.** NeuroVision AI este un prototip de cercetare în substituție senzorială și compresie perceptuală. Nu redă vederea, nu stimulează creierul și nu este un instrument validat pentru navigație.

## Pitch oral (aproximativ 3 minute)

**0:00–0:30, problema.** O cameră surprinde milioane de pixeli. Orice proteză vizuală sau sistem de substituție senzorială are însă doar zeci sau sute de canale. Întrebarea nu este cum vedem mai mult, ci ce păstrăm atunci când avem foarte puține canale.

**0:30–1:10, soluția.** Răspunsul nostru: păstrăm forma. În browser, un singur model (YOLOE-26n-seg) găsește lucrurile din imagine, din câteva mii de tipuri posibile: persoane, scaune, căni, genți, plante. Pentru fiecare dă masca exactă, pixel cu pixel. Din mască extragem conturul real și îl transformăm într-o matrice de fosfene simulate. Interiorul rămâne întunecat, iar forma nu e niciodată o cutie sau un șablon. Același contur apare și pe imaginea camerei, ca să se vadă că fosfenele spun adevărul.

**1:10–1:40, ce se aude și ce se mai vede.** Fiecare sector (stânga, centru, dreapta) are sunetul lui. Butonul „Adâncime” adaugă ordinea aproape–departe (Depth Anything V2), niciodată metri. Panoul Wi-Fi CSI arată, printr-un backend opțional, cum se combină camera cu un senzor radio de prezență.

**1:40–2:15, de ce contează.** Literatura despre vederea protetică simulată arată că, la rezoluție mică, conturul unui obiect se recunoaște mai ușor decât o pată plină. Noi facem asta live, pentru aproape orice obiect, local pe laptop, fără să trimitem imagini nicăieri.

**2:15–2:45, onestitate.** Fosfenele sunt o ilustrație software, nu o predicție clinică. O zonă fără detecții înseamnă „necunoscut”, nu „liber”. Modelele CSI din demo sunt antrenate pe date sintetice și sunt marcate ca atare. Nu prezentăm rezultate pe oameni pe care nu le avem.

**2:45–3:00, încheiere.** NeuroVision AI este un laborator deschis pentru întrebarea „ce informație spațială merită păstrată când canalele sunt puține?”. Căutăm colaborare cu cercetători în accesibilitate și cu persoane nevăzătoare.

## Demonstrația pentru juriu (3 minute)

1. **Pornește** (30 s): un clic pe „Pornește” deschide camera și sunetul. Modelul e inclus în proiect, deci merge și fără internet.
2. **Orice obiect** (60 s): pune în fața camerei, pe rând, o persoană, o cană, un rucsac, o sticlă. Arată conturul de pe cameră și aceeași siluetă în fosfene.
3. **Direcții și sunet** (30 s): mută obiectul stânga–dreapta; harta și sunetul se schimbă. Tasta **M** oprește sunetul.
4. **Adâncime** (30 s): pornește „Adâncime”; obiectele mai apropiate devin mai luminoase.
5. **Wi-Fi CSI** (30 s, doar cu backend-ul pornit): „Conectează”, apoi arată acordul cameră–radio și eticheta „date sintetice”.

**Dacă ceva nu merge:** fără WebGPU, modelul rulează pe WebAssembly, mai încet (~2 actualizări pe secundă). Backend-ul este opțional; ecranul principal funcționează fără el.

## Structura slide-urilor

| Slide | Titlu | Mesaj |
| --- | --- | --- |
| 1 | NeuroVision AI | Puține canale, forma păstrată. |
| 2 | Problema | Milioane de pixeli → zeci–sute de canale. Ce păstrăm? |
| 3 | Pipeline | Cameră → mască pentru fiecare obiect → contur → fosfene + sunet. |
| 4 | Demo live | Orice obiect, conturul pe cameră și în fosfene. |
| 5 | Incertitudine | Adâncime relativă, „necunoscut” ≠ „liber”, date sintetice marcate. |
| 6 | Pașii următori | Date WiMANS reale, studii așezate cu utilizatori, purtabil, haptic. |

## Întrebări probabile

**„E un implant sau un dispozitiv medical?”** Nu. E software experimental; matricea conține valori 0–1, nu parametri de stimulare.

**„De unde știe ce e un obiect?”** YOLOE-26n-seg (Ultralytics, licență AGPL-3.0) este antrenat pe mai multe seturi publice de date și cunoaște 4585 de tipuri de lucruri. Numele rare pot fi greșite, dar conturul vine oricum din masca reală a obiectului.

**„Cât de repede merge?”** În testele noastre, ~180 ms pe cadru cu WebGPU și ~450 ms cu WebAssembly. Imaginea camerei rămâne fluentă; conturul se actualizează de 2–5 ori pe secundă.

**„Poate ghida pe cineva pe stradă?”** Nu. Detectorul poate rata trepte sau gropi. Orice testare cu oameni se face așezat.

**„Ce aduce Wi-Fi CSI?”** Un indiciu de prezență, fără poziție. Rezultatele reale apar doar după antrenarea pe WiMANS; până atunci modelele sunt sintetice și marcate ca atare.
