/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,jsx}"],
  theme: {
    extend: {
      colors: {
        clinical: {
          50: "#f3fbff",
          100: "#deeff8",
          200: "#b7dced",
          300: "#88c2dd",
          400: "#4b9fc4",
          500: "#2a7fa8",
          600: "#1f6687",
          700: "#1c536d",
          800: "#1d465a",
          900: "#1d3a4a"
        },
        critical: "#d94141",
        warning: "#f0b429",
        stable: "#1ea672",
        ink: "#0f1724"
      },
      boxShadow: {
        panel: "0 20px 45px rgba(23, 50, 79, 0.08)",
        glow: "0 0 0 1px rgba(82, 164, 214, 0.12), 0 24px 60px rgba(32, 90, 132, 0.18)"
      },
      backgroundImage: {
        "clinical-grid":
          "linear-gradient(rgba(44,127,168,0.06) 1px, transparent 1px), linear-gradient(90deg, rgba(44,127,168,0.06) 1px, transparent 1px)"
      },
      fontFamily: {
        display: ["'Space Grotesk'", "sans-serif"],
        body: ["'Manrope'", "sans-serif"]
      },
      keyframes: {
        pulseSoft: {
          "0%, 100%": { opacity: "0.95", transform: "translateY(0px)" },
          "50%": { opacity: "1", transform: "translateY(-1px)" }
        },
        sweep: {
          "0%": { transform: "translateX(-120%)" },
          "100%": { transform: "translateX(120%)" }
        }
      },
      animation: {
        "pulse-soft": "pulseSoft 2.5s ease-in-out infinite",
        sweep: "sweep 2.4s linear infinite"
      }
    },
  },
  plugins: [],
};
