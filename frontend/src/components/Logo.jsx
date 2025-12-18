import React from 'react';

const Logo = ({ className = "h-10" }) => (
  <svg 
    xmlns="http://www.w3.org/2000/svg" 
    viewBox="0 0 200 40" 
    fill="none" 
    className={className}
    aria-label="LitScout Logo"
  >
    {/* 1. TEXT: "Lit" (Dark Blue) */}
    <text x="0" y="30" fontFamily="Arial, sans-serif" fontWeight="800" fontSize="32" fill="#1E3A8A">Lit</text>

    {/* 2. TEXT: "Sc" (Teal) */}
    <text x="42" y="30" fontFamily="Arial, sans-serif" fontWeight="800" fontSize="32" fill="#0D9488">Sc</text>

    {/* 3. ICON: The Compass (Replacing 'O') */}
    {/* Center X: 100, Center Y: 20 */}
    
    {/* Outer Ring */}
    <circle cx="100" cy="21" r="12" stroke="#0D9488" strokeWidth="3" fill="none" />
    
    {/* The Needle (Diamond Shape) */}
    <path d="M100 11 L105 21 L100 31 L95 21 Z" fill="#0D9488" />
    
    {/* Small white dot in center for realism */}
    <circle cx="100" cy="21" r="2" fill="white" />

    {/* 4. TEXT: "ut" (Teal) */}
    <text x="118" y="30" fontFamily="Arial, sans-serif" fontWeight="800" fontSize="32" fill="#0D9488">ut</text>
  </svg>
);

export default Logo;