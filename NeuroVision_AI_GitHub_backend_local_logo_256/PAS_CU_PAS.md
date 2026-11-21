# NeuroVision AI — pașii de pornire (Windows / Vercel)

## A. Pe calculator, fără Vercel (cel mai rapid pentru demo)

1. Dezarhivează ZIP-ul; deschide folderul `NeuroVision_AI` în VS Code.
2. Deschide un terminal în acel folder și rulează:

   ```powershell
   npm install
   npm run dev
   ```

3. Deschide `http://localhost:5173` în Chrome/Edge. Apasă **Pornește camera**, acceptă permisiunea. Fosfenele au **exact 256 canale virtuale 16 × 16**, inclusiv când backend-ul este oprit. Pentru a obține vectorul de 256 de valori, apasă **Export JSON**.
4. Pentru fuziune CSI deschide **un al doilea terminal**, din folderul `NeuroVision_AI/neurovision_benchmark`:

   ```powershell
   py -3.11 -m venv .venv
   .\.venv\Scripts\python.exe -m pip install -r requirements.txt -r requirements-server.txt
   .\.venv\Scripts\python.exe -m uvicorn server.app:app --host 127.0.0.1 --port 8000
   ```

   Dacă `py -3.11` nu există, folosește `python -m venv .venv`, cu Python 3.11+ instalat.
   Instalează o singură dată. PyTorch poate fi mare și antrenarea de test la primul start poate dura.
5. Testează în browser `http://localhost:8000/api/health`. Ar trebui să vezi `"status": "ok"`. Dacă nu, citește mesajul de eroare din al doilea terminal.
6. În UI, în panoul **Wi-Fi CSI**: adresa `http://localhost:8000` → **Salvează URL** → **Conectează**. Pentru demo alege **Simulat**. `ESP32` necesită hardware compatibil real și port serial disponibil.

## B. Dacă frontend-ul este pe Vercel

1. Publică proiectul ca Vite (root folder `NeuroVision_AI` dacă repository-ul conține un folder părinte; dacă `package.json` este direct în root, nu schimba root). Build: `npm run build`, Output: `dist`.
2. Găzduiește **separat** folderul `neurovision_benchmark` pe un server Python cu suficiente resurse pentru PyTorch și suport WebSocket; comanda este `python -m uvicorn server.app:app --host 0.0.0.0 --port $PORT` pe Linux, cu portul platformei. Instalează `requirements.txt` și `requirements-server.txt`. Datele generate în `runs/` trebuie păstrate sau regenerate la boot; evită hosting fără suficient RAM/disk.
3. În mediul backend setează variabila `NV_CORS_ORIGINS=https://domeniul-tau.vercel.app` (originea completă, fără slash final). Pentru un domeniu custom, poți pune mai multe origini separate prin virgulă.
4. Verifică `https://adresa-backend/api/health` — trebuie să răspundă cu JSON. HTTPS este obligatoriu pentru frontend-ul Vercel; WebSocket-ul devine automat `wss://adresa-backend/ws/live`.
5. În Vercel, **Settings → Environment Variables** adaugă `VITE_BACKEND_URL=https://adresa-backend` și fă **Redeploy**. Alternativ, introdu URL-ul în aplicație la Wi-Fi CSI → Salvează URL (se reține local în acel browser).
6. În aplicație apasă **Conectează**. Dacă apare CORS sau WebSocket error, compară exact domeniul din browser cu `NV_CORS_ORIGINS` și verifică logurile backend-ului. Schimbarea unei variabile de mediu pe Vercel necesită un nou build.

## C. De ce nu se conecta

- Codul inițial avea `http://localhost:8000` implicit **chiar în producție**. Pe Vercel, acesta indică spre PC-ul vizitatorului.
- Vercel nu pornește singur serverul Python al acestui proiect.
- Backend-ul accepta CORS doar de la `localhost`; domeniul Vercel nu era inclus.
- Pe un site HTTPS, o conexiune la un backend HTTP/WS este blocată de browser (mixed content).
- Backend-ul poate lipsi, poate încă să instaleze/antreneze modelele sau poate opri pornirea din cauza dependențelor.

## D. Verificări

```powershell
npm run typecheck
npm test
npm run build
```

În consola browserului (F12 → Network): caută `/api/status` și `/ws/live`. Un HTTP 200 pe `/api/health` indică doar că API-ul răspunde; conexiunea reală necesită `ws/live` deschis și starea `Conectat` în UI.

**Notă de siguranță:** 256 sunt canale virtuale pentru simulare, nu electrozi, parametri de curent sau un sistem de stimulare neuronală. Proiectul nu este un dispozitiv medical și nu este adecvat navigării cu ochii închiși.

## E. Logo personalizat

Logo-ul original trimis este inclus acum in `public/branding/neurovision-logo-original.png`. Simbolul transparent folosit in antet se afla in `public/branding/neurovision-icon.png`, iar iconita tab-ului este `public/favicon.png`. Nu trebuie sa mai copiezi nimic.

Pentru pasii de upload GitHub si backend local vezi `GITHUB_SI_BACKEND_LOCAL.md`.
