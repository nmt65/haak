import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { App } from "./app/App";
import "@fontsource-variable/atkinson-hyperlegible-next"; // bundled locally, works offline
import "./styles/globals.css";

const rootElement = document.getElementById("root");
if (!rootElement) throw new Error("Elementul #root lipsește din index.html");

createRoot(rootElement).render(
  <StrictMode>
    <App />
  </StrictMode>,
);
