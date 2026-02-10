import React, { useState, useEffect } from 'react';

const ScreeningView = ({ data }) => {
  const summary = data?.summary || {};
  const papers = data?.papers || [];
  
  const [selectedPaperIndex, setSelectedPaperIndex] = useState(0);
  const [showAll, setShowAll] = useState(false);

  // --- DEBUG: Check backend data ---
  useEffect(() => {
    if (papers.length > 0) {
      console.log("🔍 Screening Data:", papers[0].title, "Status:", papers[0].status);
    }
  }, [papers]);

  // --- LOGIC: Define what "Kept" means ---
  const isPaperKept = (status) => {
    if (!status) return true; // Default to kept if missing
    const s = status.toString().toLowerCase().trim();
    return !['rejected', 'excluded', 'failed', 'irrelevant', 'no', 'drop'].includes(s);
  };

  // --- STATS ---
  const total = summary.total || papers.length || 100;
  const passedCount = papers.filter(p => isPaperKept(p.status)).length;
  const passedPercent = total > 0 ? ((passedCount / total) * 100).toFixed(1) : 0;
  const rejectedPercent = (100 - parseFloat(passedPercent)).toFixed(1);

  const visiblePapers = showAll ? papers : papers.filter(p => isPaperKept(p.status));
  // Fallback: if selected list is empty, show all to prevent blank screen
  const safeVisiblePapers = visiblePapers.length > 0 ? visiblePapers : papers;
  const activePaper = safeVisiblePapers[selectedPaperIndex];

  return (
    <div className="flex flex-col h-full animate-fadeIn gap-4">
      
      {/* 1. TOP SECTION: FILTRATION FUNNEL ONLY */}
      <div className="flex-shrink-0 bg-white border p-4 rounded-xl shadow-sm">
          <div className="flex justify-between items-center mb-3">
            <h4 className="text-xs font-bold text-gray-500 uppercase tracking-wider">Filtration Statistics</h4>
            <span className="text-xs font-bold text-blue-600 bg-blue-50 px-2 py-1 rounded-full">{passedCount} Papers Selected</span>
          </div>
          <div className="space-y-3">
              {/* Input Bar */}
              <div>
                  <div className="flex justify-between text-[10px] mb-1 text-gray-400 font-medium">
                      <span>Total Scanned</span>
                      <span>{total}</span>
                  </div>
                  <div className="w-full bg-gray-100 h-1.5 rounded-full overflow-hidden">
                     <div className="h-full bg-gray-300 w-full"></div>
                  </div>
              </div>
              
              {/* Outcome Bar */}
              <div>
                  <div className="flex justify-between text-[10px] mb-1 text-gray-400 font-medium">
                      <span>Selection Outcome</span>
                      <span>{passedPercent}% Acceptance Rate</span>
                  </div>
                  <div className="w-full h-6 rounded-md flex overflow-hidden text-[10px] text-white font-bold text-center leading-6 shadow-inner">
                      <div style={{ width: `${passedPercent}%` }} className="bg-emerald-500 shadow-sm transition-all duration-1000">Selected</div>
                      <div style={{ width: `${rejectedPercent}%` }} className="bg-slate-200 text-slate-400">Dropped</div>
                  </div>
              </div>
          </div>
      </div>

      {/* 2. MIDDLE SECTION: FILTER BAR */}
      <div className="flex justify-between items-center bg-gray-50 p-2 rounded-lg border border-gray-200 flex-shrink-0">
          <div className="text-xs font-bold text-gray-600 pl-2">
             Results List
          </div>
          <div className="flex items-center gap-2">
             <button 
                onClick={() => { setShowAll(false); setSelectedPaperIndex(0); }}
                className={`text-[10px] px-3 py-1 rounded transition-colors ${!showAll ? 'bg-white border border-green-200 text-green-700 font-bold shadow-sm' : 'text-gray-500 hover:bg-gray-200'}`}
             >
                Selected Only
             </button>
             <button 
                onClick={() => { setShowAll(true); setSelectedPaperIndex(0); }}
                className={`text-[10px] px-3 py-1 rounded transition-colors ${showAll ? 'bg-gray-800 text-white font-bold shadow-sm' : 'text-gray-500 hover:bg-gray-200'}`}
             >
                View All
             </button>
          </div>
      </div>

      {/* 3. BOTTOM SECTION: MASTER-DETAIL LIST */}
      {safeVisiblePapers.length > 0 ? (
        <div className="flex-grow flex border border-gray-200 rounded-xl overflow-hidden shadow-sm min-h-0 bg-white">
            
            {/* LEFT: LIST */}
            <div className="w-1/3 border-r border-gray-100 bg-gray-50 flex flex-col min-w-[200px]">
                <div className="flex-grow overflow-y-auto custom-scrollbar">
                    {safeVisiblePapers.map((p, i) => {
                        const isSelected = i === selectedPaperIndex;
                        return (
                            <button
                                key={i}
                                onClick={() => setSelectedPaperIndex(i)}
                                className={`w-full text-left p-3 border-b border-gray-100 transition-all hover:bg-white group relative
                                    ${isSelected ? 'bg-white z-10 shadow-sm' : 'bg-transparent text-gray-500'}
                                `}
                            >
                                {isSelected && <div className="absolute left-0 top-0 bottom-0 w-1 bg-blue-500"></div>}
                                <div className={`text-xs font-semibold leading-tight mb-1 line-clamp-2 ${isSelected ? 'text-gray-800' : 'text-gray-600'}`}>
                                    {p.title || "Untitled Paper"}
                                </div>
                                <div className="flex justify-between items-center mt-1">
                                    <span className="text-[10px] bg-gray-200 text-gray-600 px-1.5 rounded">{p.year || "N/A"}</span>
                                    <span className={`text-[9px] font-bold ${p.score > 0.7 ? 'text-green-600' : 'text-amber-500'}`}>
                                        {p.score ? `Score: ${(p.score * 100).toFixed(0)}%` : ''}
                                    </span>
                                </div>
                            </button>
                        );
                    })}
                </div>
            </div>

            {/* RIGHT: DETAILS */}
            <div className="w-2/3 flex flex-col bg-white overflow-hidden">
                {activePaper ? (
                    <div className="flex-grow overflow-y-auto p-6 custom-scrollbar">
                        <div className="flex justify-between items-start mb-4">
                            <span className={`px-2 py-1 rounded text-[10px] font-bold border uppercase tracking-wide ${
                                isPaperKept(activePaper.status) 
                                ? 'bg-green-50 text-green-700 border-green-200' 
                                : 'bg-red-50 text-red-700 border-red-200'
                            }`}>
                                {activePaper.status || "Auto-Selected"}
                            </span>
                        </div>

                        <h2 className="text-lg font-bold text-gray-900 leading-snug mb-2">{activePaper.title}</h2>

                        <div className="flex flex-wrap gap-3 text-xs text-gray-500 mb-6 pb-4 border-b border-gray-100">
                             {activePaper.year && <span>📅 {activePaper.year}</span>}
                             {activePaper.venue && <span>🏛 {activePaper.venue}</span>}
                             {activePaper.authors && <span>✍️ {activePaper.authors.map(a => a.name).join(", ")}</span>}
                        </div>

                        <div className="bg-blue-50 border border-blue-100 rounded-lg p-4 mb-6">
                            <h4 className="text-[10px] text-blue-500 font-bold uppercase tracking-wider mb-2">🤖 AI Reasoning</h4>
                            <p className="text-sm text-blue-900 leading-relaxed">
                                {activePaper.reason || activePaper.abstract?.slice(0, 150) + "..." || "Paper selected based on relevance criteria."}
                            </p>
                        </div>

                        <div>
                            <h4 className="text-[10px] text-gray-400 font-bold uppercase tracking-wider mb-2">Abstract</h4>
                            <p className="text-sm text-gray-700 leading-relaxed whitespace-pre-line text-justify">
                                {activePaper.abstract || "No abstract available."}
                            </p>
                        </div>
                        
                        <div className="mt-8 pt-4 border-t border-gray-100">
                            <a href={activePaper.url || `https://scholar.google.com/scholar?q=${encodeURIComponent(activePaper.title)}`} target="_blank" rel="noreferrer" className="text-xs font-bold text-blue-600 hover:underline">
                                Read Full Paper ↗
                            </a>
                        </div>
                    </div>
                ) : (
                    <div className="flex flex-col items-center justify-center h-full text-gray-400">
                        <p className="text-sm">Select a paper to view details</p>
                    </div>
                )}
            </div>
        </div>
      ) : (
        <div className="flex-grow flex flex-col items-center justify-center border border-dashed border-gray-300 rounded-xl bg-gray-50/50">
            <p className="text-gray-500 text-sm">No papers available to display.</p>
        </div>
      )}
    </div>
  );
};

export default ScreeningView;