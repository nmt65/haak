> **Incepi pe Windows?** Deschide [GITHUB_SI_BACKEND_LOCAL.md](GITHUB_SI_BACKEND_LOCAL.md). Are pasii pentru upload GitHub si pornirea backend-ului local prin 3 fisiere `.bat`, fara baza de date. Logo-ul original si favicon-ul sunt deja incluse.

# NeuroVision AI

Prototip experimental de cercetare și educațional, realizat pentru un hackathon Life Science (12 ore).

> NeuroVision AI **nu** stimulează creierul, **nu** redă vederea, **nu** este dispozitiv medical și **nu** înlocuiește bastonul alb, câinele-ghid sau alte mijloace validate de mobilitate. Toate datele „neurale” sunt simulate computațional.

---

## 1. Problema investigată

O persoană nevăzătoare poate avea dificultăți în a identifica obstacolele și configurația mediului din fața sa. Bastonul alb oferă informații esențiale despre mediul apropiat, dar nu o hartă a întregii scene.

Întrebarea de inginerie: **putem transforma o scenă vizuală complexă într-o reprezentare spațială compactă, care să poată fi transmisă printr-un alt canal senzorial (sunet) sau studiată ca intrare ipotetică pentru o viitoare neuroproteză vizuală?**

### Unde se încadrează proiectul

Orice sistem de „vedere artificială” trebuie să comprime o imagine de milioane de pixeli în câteva zeci sau sute de canale: aproximativ 60 de electrozi la implantul cortical Orion, 96–100 la matricele intracorticale studiate la om, 378 de pixeli la implantul retinian PRIMA, 394 de electrozi la BrainPort (pe limbă). Problema grea nu este camera, ci **ce informație păstrăm**.

NeuroVision AI este un **codec perceptual non-invaziv**: alege informația relevantă (obiecte, direcție, adâncime relativă), o reduce la o matrice mică și o transmite prin sunet. Fosfenele simulate arată cum ar arăta aceeași matrice ca intrare ipotetică pentru o proteză. Ideea, susținută de literatura de specialitate, este că o astfel de codificare trebuie validată mai întâi pe un canal sigur (audio, haptic), cu utilizatori nevăzători, înainte de orice tentativă invazivă.

Cifrele de mai sus provin dintr-un raport de cercetare din noiembrie 2026 furnizat de echipă; verificați sursele originale înainte de a le cita.

## 2. Ce face produsul

Un singur ecran live, fără mod demo și fără pagini de prezentare:

```
Camera → YOLOE-26n-seg (detecție + mască pe pixel pentru fiecare obiect, 4585 de tipuri)
       → conturul real al fiecărui obiect / persoane
       → același contur: desenat pe imaginea camerei + fosfene (exact 256 de canale virtuale, 16 × 16)
       ↘ direcții (stânga / centru / dreapta) → hartă + sunet spațial
       ↘ (opțional) adâncime relativă Depth Anything V2
       ↘ (opțional) backend Python: Wi-Fi CSI + fuziune cu camera
```

- **Camera**: imagine live; fiecare obiect și fiecare persoană primește conturul real, desenat din masca de segmentare, plus o etichetă. Nu se desenează cutii. Modelul rulează local în browser (ONNX Runtime, WebGPU sau WebAssembly), inclusiv fără internet. Cadrele nu părăsesc calculatorul, cu excepția modului YOLO26n, care le trimite doar backend-ului local.
- **Orice obiect**: YOLOE „prompt-free” cunoaște 4585 de tipuri de lucruri (persoane, mobilă, căni, pantofi, plante, genți, indicatoare...). Fiecare are masca lui, deci conturul urmează forma reală. Nu există niciun contur prestabilit: un obiect fără mască nu primește nimic. Suprafețele (pământ, cer) și etichetele de scenă care acoperă tot cadrul sunt ignorate, pentru că ar fi doar o linie de orizont.
- **Fosfene pe contur**: fosfenele se aprind exact pe conturul afișat pe cameră; interiorul rămâne întunecat. Matricea are 16 × 16 poziții fixe și este adaptată proporțiilor ferestrei camerei (celulele pot fi dreptunghiulare). Intensitățile 0–1 sunt doar valori simulate. Detalii în `src/services/objectOutline.ts` și `src/services/contourEncoder.ts`.
- **Direcții**: hartă de sus, centrată pe utilizator, și ocuparea pe cele trei sectoare. Fără adâncime, distanța rămâne necunoscută.
- **Sunet**: ton grav în stânga, două impulsuri în centru, ton înalt în dreapta. Tasta **M** oprește sunetul.
- **Adâncime** (buton în antet): ordine aproape–departe, niciodată metri. Modelul se descarcă la prima pornire.
- **Wi-Fi CSI** (panou, cu backend-ul pornit): sursă simulată, înregistrată sau ESP32, semnalul CSI, clasificatoarele CSI / cameră / fuziune și acordul dintre cameră și radio. Punctul violet de lângă fosfene arată prezența radio: radioul nu dă poziție, deci nu intră în matrice.

## 3. Ce este simulat

| Element | Statut |
|---|---|
| Fosfenele | Ilustrație software, **nu** predicție validată a vederii prin proteză |
| Matricea de activare | Valori 0–1, **nu** curenți sau parametri de stimulare |
| Harta pe direcții | Câmp vizual presupus de 60°; fără adâncime, distanța **nu** este estimată |
| Modelele CSI implicite | Antrenate pe date **sintetice** până la antrenarea pe WiMANS (marcat în interfață) |

## 4. Rulare

Cerințe: Node.js 20+, Chrome sau Edge recent.

```bash
npm install
npm run dev        # http://localhost:5173, apoi „Pornește”
npm test           # teste unitare (Vitest)
npm run build      # verificare TypeScript + build de producție
```

Camera funcționează doar pe `localhost` sau HTTPS.

### Backend-ul Wi-Fi CSI (opțional)

Instalare și antrenare: [neurovision_benchmark/README.md](neurovision_benchmark/README.md). Din folderul `neurovision_benchmark`:

```bash
python -m uvicorn server.app:app --port 8000
```

Instalează dependențele Python înainte de rulare (`pip install -r requirements.txt -r requirements-server.txt` în folderul `neurovision_benchmark`). Testează `http://localhost:8000/api/health` — trebuie să primești `{"status":"ok", ...}`. La prima pornire fără modele existente, backend-ul creează o rulare de test pe date **sintetice**; aceasta poate întârzia pornirea. În aplicație: **Wi-Fi CSI → Adresa backend-ului: http://localhost:8000 → Salvează URL → Conectează**.

**Pe Vercel:** Vercel publică numai frontend-ul Vite din acest proiect. Backend-ul FastAPI trebuie găzduit separat pe o platformă Python (server/VPS, Render etc.) cu WebSockets și HTTPS. În frontend setează variabila de mediu `VITE_BACKEND_URL=https://backend.example.com` (sau introdu adresa în panoul CSI). În backend setează `NV_CORS_ORIGINS=https://numele-proiectului.vercel.app` și repornește serverul. Fără această configurare apar erori de CORS sau WebSocket. Adresa `localhost` nu poate conecta backend-ul de pe alt calculator. Consultă `PAS_CU_PAS.md`.

**256 canale:** matrice virtuală fixă 16 × 16; fiecare canal are intensitate software 0–1 (nu impulsuri pentru electrozi). Butonul **Export JSON** salvează ordinea canalelor 0–255 și intensitățile la momentul capturii. Simularea fosfenelor funcționează și fără backend.

## 5. Modele folosite

- **YOLOE-26n-seg prompt-free** (Ultralytics, licență AGPL-3.0), exportat în ONNX la 384 × 384: `public/models/yoloe-26n-seg-pf-384.onnx` (15,6 MB, inclus în proiect). Regenerare: `python neurovision_benchmark/export_browser_model.py`. Viteză măsurată în testele noastre: ~180 ms pe cadru cu WebGPU, ~450 ms cu WebAssembly (deci 2–5 actualizări pe secundă).
- **Depth Anything V2 Small** (prin `@huggingface/transformers`): doar când pornești „Adâncime”.
- Backend: clasificatoare CSI / cameră / fuziune (PyTorch), **YOLO26n** opțional pentru detecție pe server.

## 6. Structura proiectului

```
src/
  app/App.tsx                    ecranul live
  context/PipelineContext.tsx    bucla cameră → detecție + măști → contur → fosfene / hartă / sunet
  context/BackendContext.tsx     conexiunea WebSocket la backend-ul Wi-Fi CSI
  services/objectDetectionService.ts  YOLOE în browser (ONNX Runtime)
  services/yoloSegmentation.ts   decodarea ieșirii YOLOE: cutii, duplicate, măști
  services/objectOutline.ts      mască → contur
  services/contourEncoder.ts     contur → matricea de fosfene
  services/                      cameră, adâncime, sectoare, audio
  components/                    antet, cameră, fosfene, direcții, panoul Wi-Fi CSI
public/models/                   modelul YOLOE în format ONNX
tests/                           teste unitare (inclusiv conturul)
neurovision_benchmark/           backend FastAPI + benchmark PyTorch
```

## 7. Hardware necesar pentru cercetări viitoare (nu face parte din prototip)

- Ochelari cu cameră (ideal cameră stereo sau de adâncime) și procesare locală purtabilă.
- Căști cu conducție osoasă, ca să nu acopere sunetele din jur; opțional 4–8 motoare de vibrație.
- Pentru cercetare neuroprotetică: interfețe neurale validate clinic, dezvoltate de echipe specializate, în studii aprobate etic. Software-ul de față ar putea servi doar ca strat de percepție și codificare.

## 8. Limitări științifice

- O cameră și un detector de obiecte **nu pot garanta** un drum liber; o regiune fără detecții este „necunoscută”, nu „sigură”.
- Detectorul poate rata trepte, gropi sau obstacole la nivelul capului, iar etichetele rare pot fi greșite (de exemplu un autobuz numit „recreational vehicle”). Suprafețele mari (pământ, cer) sunt ignorate intenționat.
- Unghiurile sunt aproximative (câmp vizual presupus, fără calibrare).
- Adâncimea monoculară este relativă la fiecare cadru și poate greși pe suprafețe lucioase sau fără textură.
- Percepția fosfenelor la purtătorii de implant variază mult; un electrod nu este un pixel perfect.

Pentru evaluare, folosiți doar sarcini în poziție așezată. Nu cereți nimănui să meargă cu ochii închiși pe baza acestui prototip.

### Logo propriu

Pune PNG-ul original la `public/branding/neurovision-logo.png`. Logo-ul din header și favicon-ul browserului se vor actualiza automat după încărcare. Fără fișier, este afișată marca implicită.
