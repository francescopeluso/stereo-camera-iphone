import type { Metadata, Viewport } from "next";
import { Analytics } from "@vercel/analytics/next";
import { Funnel_Display, Funnel_Sans, Martian_Mono } from "next/font/google";
import "katex/dist/katex.min.css";
import "./globals.css";

const display = Funnel_Display({ subsets: ["latin"], variable: "--font-display" });
const sans = Funnel_Sans({ subsets: ["latin"], variable: "--font-sans" });
const mono = Martian_Mono({ subsets: ["latin"], variable: "--font-mono", weight: ["400"] });

const description =
  "StereoVision turns the wide and ultra-wide cameras of an iPhone into a real-time stereo camera: live depth, object segmentation with distances, and nine ways to look at a scene.";

export const metadata: Metadata = {
  metadataBase: process.env.NEXT_PUBLIC_SITE_URL ? new URL(process.env.NEXT_PUBLIC_SITE_URL) : undefined,
  title: "StereoVision: stereo depth on iPhone",
  description,
  openGraph: { title: "StereoVision", description, type: "website" },
  twitter: { card: "summary_large_image", title: "StereoVision", description },
};

export const viewport: Viewport = { themeColor: "#10162E" };

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en" className={`${display.variable} ${sans.variable} ${mono.variable}`}>
      <body>
        {children}
        <Analytics />
      </body>
    </html>
  );
}
