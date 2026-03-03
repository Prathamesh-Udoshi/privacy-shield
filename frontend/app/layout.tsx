import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
    title: "Privacy Shield | Differential Privacy Anonymization",
    description:
        "Industrial-grade differential privacy for everyone. Upload a CSV and get a provably anonymous dataset in seconds.",
    keywords: ["differential privacy", "data anonymization", "GDPR", "privacy"],
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
    return (
        <html lang="en">
            <head>
                <link rel="preconnect" href="https://fonts.googleapis.com" />
                <link rel="preconnect" href="https://fonts.gstatic.com" crossOrigin="anonymous" />
                <link
                    href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap"
                    rel="stylesheet"
                />
            </head>
            <body suppressHydrationWarning>
                <div className="mesh-bg" />
                {children}
            </body>
        </html>
    );
}
