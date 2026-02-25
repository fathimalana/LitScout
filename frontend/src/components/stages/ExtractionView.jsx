// src/components/stages/ExtractionView.jsx
import React from 'react';

const ExtractionView = ({ data }) => {
  // 1. Identify data source
  const summary = data?.summary || data || {};
  const papers = data?.papers || [];

  // 2. Logic: Dynamic Calculation to fix the "15% vs 18.7%" issue
  const attempted = summary.total_attempted || 0;
  const extracted = summary.successfully_extracted || 0;
  
  // Calculate dynamic rate: (extracted / attempted) * 100
  const dynamicRate = attempted > 0 
    ? ((extracted / attempted) * 100).toFixed(1) + '%' 
    : '0%';

  // Use dynamicRate if the backend sends '15.0%' as a placeholder or is missing the rate
  const displayRate = (summary.success_rate && summary.success_rate !== '15.0%') 
    ? summary.success_rate 
    : dynamicRate;
  
  // If we don't have the specific stats yet, show loading
  if (!summary || !summary.total_attempted) {
    return (
      <div className="p-10 text-center">
        <div className="animate-spin h-8 w-8 border-4 border-blue-500 border-t-transparent rounded-full mx-auto mb-4"></div>
        <p className="text-gray-500 italic">Processing PDFs and calculating success rates...</p>
      </div>
    );
  }

  return (
    <div className="space-y-6 animate-in fade-in slide-in-from-bottom-2 duration-500">
      {/* Metrics Grid */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <div className="bg-slate-50 p-4 rounded-xl border border-slate-200 shadow-sm">
          <p className="text-[10px] font-bold text-slate-500 uppercase tracking-wider mb-1">Iterations</p>
          {/* Removed the hardcoded || 5 to show real backend value */}
          <p className="text-2xl font-black text-slate-800">{summary.iterations || 0}</p>
        </div>
        
        <div className="bg-blue-50 p-4 rounded-xl border border-blue-100 shadow-sm">
          <p className="text-[10px] font-bold text-blue-500 uppercase tracking-wider mb-1">Attempted</p>
          <p className="text-2xl font-black text-blue-700">{attempted}</p>
        </div>
        
        <div className="bg-emerald-50 p-4 rounded-xl border border-emerald-100 shadow-sm">
          <p className="text-[10px] font-bold text-emerald-500 uppercase tracking-wider mb-1">Extracted</p>
          <p className="text-2xl font-black text-emerald-700">{extracted}</p>
        </div>
        
        <div className="bg-amber-50 p-4 rounded-xl border border-amber-100 shadow-sm">
          <p className="text-[10px] font-bold text-amber-500 uppercase tracking-wider mb-1">Success Rate</p>
          {/* Now using the displayRate calculation to fix the discrepancy */}
          <p className="text-2xl font-black text-amber-700">{displayRate}</p>
        </div>
      </div>

      {/* Samples List */}
      {papers.length > 0 && (
        <div className="mt-4">
          <h4 className="text-xs font-bold text-gray-500 uppercase mb-3 flex items-center gap-2">
            <span className="w-2 h-2 bg-blue-600 rounded-full animate-pulse"></span>
            Live Extraction Feed
          </h4>
          <div className="space-y-3 max-h-[250px] overflow-y-auto pr-2 custom-scrollbar">
            {papers.map((paper, i) => (
              <div key={i} className="p-4 bg-white border border-gray-100 rounded-lg shadow-sm hover:border-blue-200 transition-colors">
                <p className="font-bold text-sm text-gray-800 mb-1 line-clamp-1">{paper.title}</p>
                <div className="text-xs text-gray-600 bg-gray-50 p-3 rounded italic border-l-2 border-blue-400 font-mono">
                  "{paper.text_preview?.substring(0, 200)}..."
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};

export default ExtractionView;