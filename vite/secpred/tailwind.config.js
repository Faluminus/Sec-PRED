/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      fontFamily: {
        ApercuRegular: ['ApercuRegular', 'sans-serif'],
        ApercuMedium: ['ApercuMedium', 'sans-serif'],
        ApercuLight: ['ApercuLight', 'sans-serif'],
        ApercuBold: ['ApercuBold', 'sans-serif'],
      },
    },
    
  },
  plugins: [],
}