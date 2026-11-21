/** @type {import('tailwindcss').Config} */

/*
 * Design tokens — monochrome research instrument.
 * Token names describe roles: chassis = page, paper = panels, raised = controls,
 * ink = text, cobalt = primary accent. Components never hard-code colours.
 */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        chassis: "#F4F4F2", // warm-white page
        paper: "#FFFFFF", // clean panels
        raised: "#EEEEEB", // elevated controls
        ink: "#0A0A0A", // near-black primary text
        graphite: "#666666", // secondary text
        rule: "#D6D6D2", // subtle dividers
        cobalt: { DEFAULT: "#111111", soft: "#E8E8E5", deep: "#000000" }, // monochrome primary
        violet: "#444444", // monochrome secondary
        window: "#000000", // optical windows
        fluor: "#FFFFFF", // detections on optical windows
        phos: "#FFFFFF", // phosphene light
        live: "#111111", // live camera indicator, errors
        caution: { DEFAULT: "#3F3F3F", soft: "rgba(0, 0, 0, 0.045)" },
        ok: "#111111",
      },
      fontFamily: {
        sans: ["'Atkinson Hyperlegible Next Variable'", "Segoe UI", "system-ui", "sans-serif"],
        mono: ["ui-monospace", "Cascadia Mono", "Consolas", "monospace"], // numerical telemetry
      },
      fontSize: {
        // Type scale (px): 12 · 14 · 16 · 20 · 28 · 40
        xs: ["0.75rem", { lineHeight: "1.1rem" }],
        sm: ["0.875rem", { lineHeight: "1.35rem" }],
        base: ["1rem", { lineHeight: "1.55rem" }],
        lg: ["1.25rem", { lineHeight: "1.75rem" }],
        xl: ["1.75rem", { lineHeight: "2.1rem" }],
        "2xl": ["2.5rem", { lineHeight: "2.7rem" }],
      },
      borderRadius: {
        window: "20px",
      },
    },
  },
  plugins: [],
};
