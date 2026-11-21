# NeuroVision AI — GitHub + backend LOCAL (fara baza de date)

Acesta este **codul sursa complet**, cu logo-ul incarcat, favicon, simularea a 256 de canale (16 × 16), frontend React/Vite si backend Python/FastAPI.

## 1. Pune pe GitHub

1. Dezarhiveaza fisierul ZIP primit.
2. Intra in GitHub -> New repository -> nume, de exemplu `NeuroVision-AI` -> Create repository.
3. In repository apasa **Add file -> Upload files**. Selecteaza **CONTINUTUL** folderului dezarhivat (nu arhiva ZIP si nu folderul parinte). Asigura-te ca `package.json` si `index.html` ajung direct in radacina repository-ului. Daca interfata web nu permite incarcarea completa a directoarelor, foloseste GitHub Desktop / `git add .`, `git commit`, `git push`.
4. Nu urca `node_modules`, `.venv`, `.env` sau fisiere cu chei secrete. `.gitignore` le exclude la folosirea Git.

## 2. Porneste totul pe calculator (Windows)

Instaleaza **Node.js 20+** si **Python 3.11**. Pentru backend cu PyTorch este recomandat un PC cu minimum 8 GB RAM.

1. Dublu click `01_INSTALARE_BACKEND_WINDOWS.bat` (doar prima data).
2. Dublu click `02_PORNESTE_BACKEND_WINDOWS.bat` si lasa fereastra DESCHISA.
3. Deschide `http://127.0.0.1:8000/api/health`. Trebuie sa arate `"status":"ok"`.
4. Dublu click `03_PORNESTE_FRONTEND_WINDOWS.bat` si lasa a doua fereastra DESCHISA.
5. Intra in `http://127.0.0.1:5173`. In panoul Wi-Fi CSI seteaza adresa `http://127.0.0.1:8000`, apasa **Salveaza URL**, apoi **Conecteaza** si selecteaza **Simulat**.
6. Porneste camera cand browserul iti cere permisiunea. Simulatorul va arata **256 canale virtuale**, inclusiv daca backend-ul este oprit.

**Alternativa manuala, in 2 terminale:**

```powershell
# terminal 1: din folderul proiectului
cd neurovision_benchmark
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt -r requirements-server.txt
.\.venv\Scripts\python.exe -m uvicorn server.app:app --host 127.0.0.1 --port 8000
```

```powershell
# terminal 2: din radacina proiectului
npm ci
npm run dev
```

## 3. Ce trebuie sa stii daca publici frontend-ul pe Vercel

GitHub pastreaza codul, dar **NU porneste automat serverul Python**. Vercel porneste doar site-ul, iar `localhost` in browserul altui utilizator inseamna **calculatorul acelui utilizator**, nu al tau.

Daca vrei sa folosesti Vercel **impreuna cu backend-ul local** in aceeasi sesiune, site-ul Vercel este HTTPS si are nevoie de un backend accesibil prin HTTPS/WSS (de exemplu un tunel HTTPS autorizat catre serverul tau), nu de `http://localhost:8000`. Pentru demo rapid, foloseste ambele la adresele locale de mai sus. Pentru utilizatori de la distanta, va trebui sa publici backend-ul separat.

**Nu iti trebuie nicio baza de date pentru acest demo.** Datele de test si modelele se afla in `neurovision_benchmark`. Datele CSI in modul `Simulat` sunt generate artificial. Nu trimite portul 8000 direct in internet fara autentificare si protectie suplimentara.

## 4. Daca apare eroare

- **Backend refused connection:** verifica fereastra `02_PORNESTE_BACKEND_WINDOWS.bat` si URL-ul /api/health.
- **Port 8000 ocupat:** opreste serverul anterior; implicit frontend-ul cauta portul 8000.
- **Nu porneste Python / PyTorch:** verifica versiunea Python si ruleaza inca o data scriptul de instalare; e posibil sa dureze la prima pornire, cand lipsesc modelele de test.
- **`npm` nu este recunoscut:** instaleaza Node.js si redeschide terminalul.
- **Camera nu merge:** accepta permisiunea in browser; functioneaza pe `localhost` sau HTTPS.
- **Pe Vercel nu se conecteaza backend-ul local:** nu e o eroare de baza de date; ai nevoie de backend HTTPS/WSS public.

## 5. Verificari pentru dezvoltatori

```powershell
npm ci
npm test
npm run build
cd neurovision_benchmark
.\.venv\Scripts\python.exe -m pytest tests -q
```

**Nota:** 256 inseamna canale de fosfene **simulate in software**, nu electrozi reali, impulsuri neurologice sau echipament medical.
