import './globals.css';

export const metadata = {
  title: 'Systematic Review Screening MVP',
  description: 'A lightweight screening assistant for systematic reviews using Gemini AI.'
};

export default function RootLayout({ children }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
