/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,ts,jsx,tsx}"],
  theme: {
    extend: {
      colors: {
        // Single accent: teal (technical, premium, not purple/indigo).
        brand: {
          50:  '#effcf9',
          100: '#c9f5ec',
          200: '#97e9da',
          300: '#5fd6c3',
          400: '#2fbcaa',
          500: '#14a08f',
          600: '#0d8175',
          700: '#10665e',
          800: '#11514c',
          900: '#11433f',
        },
        // Warm signal accent, used sparingly for active/live "instrument" states.
        signal: {
          400: '#f4b860',
          500: '#e89d3c',
        },
      },
      fontFamily: {
        sans: ['"Space Grotesk"', 'system-ui', 'sans-serif'],
        mono: ['"JetBrains Mono"', 'Fira Code', 'monospace'],
      },
      boxShadow: {
        glass: '0 1px 0 0 rgba(255,255,255,0.04) inset, 0 12px 40px -12px rgba(0,0,0,0.6)',
        'glass-lg': '0 1px 0 0 rgba(255,255,255,0.05) inset, 0 24px 70px -20px rgba(0,0,0,0.7)',
      },
      keyframes: {
        'fade-up': {
          '0%': { opacity: '0', transform: 'translateY(8px)' },
          '100%': { opacity: '1', transform: 'translateY(0)' },
        },
        shimmer: {
          '100%': { transform: 'translateX(100%)' },
        },
      },
      animation: {
        'fade-up': 'fade-up 0.4s cubic-bezier(0.16,1,0.3,1) both',
      },
    },
  },
  plugins: [],
}
