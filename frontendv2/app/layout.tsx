
import type { Metadata } from "next";
import { Outfit } from "next/font/google"; // Outfit is a good free alternative to Google Sans
import "./globals.css";
import { Sidebar } from "@/components/layout/sidebar";

const outfit = Outfit({ subsets: ["latin"] });

export const metadata: Metadata = {
  title: "CHIRANJEEVAI - Medical Assistant",
  description: "AI Medical Assistant",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body className={`${outfit.className} bg-surface text-on-surface flex h-screen w-screen overflow-hidden`}>
        {/* Sidebar (Fixed/Sticky) */}
        <Sidebar />
        
        {/* Main Content Area */}
        <main className="flex-1 h-full overflow-hidden relative flex flex-col transition-all duration-300">
          {children}
        </main>
      </body>
    </html>
  );
}
