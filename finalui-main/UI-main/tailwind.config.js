/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,ts,jsx,tsx}'],
  theme: {
    extend: {
      colors: {
        'confluence-blue': '#0052CC',
        'confluence-light-blue': '#2684FF',
      }
    },
  },
  plugins: [
    require('tailwind-scrollbar'),
  ],
};