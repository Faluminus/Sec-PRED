import type { Metadata } from "next";
import "./globals.css";
import "./style.scss"
import localFont from 'next/font/local'
import Footer_component from './components/footer_component';
import Loading from "./loading";
import { Suspense } from "react";



export const metadata: Metadata = {
  title: "SecPRED",
  description: "Protein secondary prediction and visualization tool",
  
};

const appercu = localFont({
  src: [
    {
      path: '../public/fonts/apercu/apercu-bold.otf',
      weight: '700',
      style:'bold'
    },
    {
      path:'../public/fonts/apercu/apercu-light.otf',
      weight:'200',
      style:'light'
    },
    {
      path: '../public/fonts/apercu/apercu-regular.otf',
      weight: '400',
      style:'regular'
    },
  ],
  variable: '--font-appercu'
})



export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html className={appercu.className} lang="en">
      <head>
        <link rel="icon" href="/favicon.png" sizes="180x180"/>
      </head>
      <body className="h-screen w-screen bg-slate-200">
        <Suspense fallback={<Loading/>}>
         
            {children}
         
        </Suspense>
        <Footer_component/>
      </body>
    </html>
  );
}
